import ast
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import partial
from pathlib import Path
from queue import Queue
import random
import sqlite3
import time
from typing import Any

import numpy as np
import pufferlib.vector
import wandb

from pokemonred_puffer.data.moves import Moves
from pokemonred_puffer.data.species import Species
from pokemonred_puffer.eval import make_pokemon_red_overlay
from pokemonred_puffer.wrappers.sqlite import SqliteStateResetWrapper


@dataclass
class StateManager:
    sqlite_db: str | None
    env_send_queues: list[Queue]
    env_recv_queues: list[Queue]
    archive_states: bool = False
    archive_path: Path = Path(datetime.now().strftime("%Y%m%d-%H%M%S"))
    states: dict = field(default_factory=lambda: defaultdict(partial(deque, maxlen=1)))
    event_tracker: dict = field(default_factory=lambda: {})
    early_stop: dict[str, int] = field(default_factory=lambda: {})
    async_wrapper: bool = False
    sqlite_wrapper: bool = False
    swarm_enabled: bool = False
    save_overlay: bool = False
    overlay_interval: int = 10
    required_rate: float | bool = False
    wandb_enabled: bool = False
    uptime: int = 0

    def __post_init__(self):
        if self.archive_states:
            if not self.archive_path.exists():
                self.archive_path.mkdir(parents=True, exist_ok=True)
            else:
                print(
                    f"Warning: Archive path {self.archive_path} already exists. States will be appended."
                )
        self.start = datetime.now()

    def process_stats(self, stats: dict[str, Any], epoch: int) -> defaultdict:
        env_ids = stats["env_ids"]
        for k, v in stats.items():
            if k.startswith("state/"):
                key_str = k.split("/")[-1]
                key: tuple[str] = ast.literal_eval(key_str)
                self.states[key].append(v)
                if self.archive_states:
                    state_dir = self.archive_path / str(hash(key))
                    if not state_dir.exists():
                        state_dir.mkdir(exist_ok=True)
                        with open(state_dir / "desc.txt", "w") as f:
                            f.write(str(key))
                    with open(state_dir / f"{hash(v)}.state", "wb") as f:
                        f.write(v)
            elif "stats/required_count" == k:
                for count, eid in zip(v, env_ids):
                    self.event_tracker[eid] = max(self.event_tracker.get(eid, 0), count)

        stats = defaultdict(list, stats)
        # Add Pokemon specific stats
        for k, v in stats.items():
            # Moves into models... maybe. Definitely moves.
            # You could also just return infos and have it in demo
            if "pokemon_exploration_map" in k and self.save_overlay is True:
                if epoch % self.overlay_interval == 0:
                    overlay = make_pokemon_red_overlay(np.stack(stats[k], axis=0))
                    if self.wandb_enabled:
                        stats["Media/aggregate_exploration_map"] = wandb.Image(overlay)
            elif any(s in k for s in ["state", "env_id", "species", "levels", "moves"]):
                continue
            else:
                try:  # TODO: Better checks on log data types
                    stats[k] = np.mean(v)
                except:  # noqa: E722
                    continue

        if (
            all(k in stats.keys() for k in ["env_ids", "species", "levels", "moves"])
            and self.wandb_enabled is not None
        ):
            table = {}
            # The infos are in order of when they were received so this _should_ work
            for env_id, species, levels, moves in zip(
                stats["env_ids"],
                stats["species"],
                stats["levels"],
                stats["moves"],
            ):
                table[env_id] = [
                    f"{Species(_species).name} @ {level} w/ {[Moves(move).name for move in _moves if move]}"
                    if _species
                    else ""
                    for _species, level, _moves in zip(species, levels, moves)
                ]

            stats["party/agents"] = wandb.Table(
                columns=["env_id"] + [str(v) for v in range(6)],
                data=[[str(k)] + v for k, v in table.items()],
            )

        return stats

    def check_early_stop(self) -> bool:
        if not self.early_stop:
            return False
        if not self.states:
            return False
        to_delete = []
        early_stop = False
        td = self.start - datetime.now()
        for event, minutes in self.early_stop.items():
            if any(event in key for key in self.states.keys()):
                to_delete.append(event)
            elif td > timedelta(minutes=minutes) and all(
                event not in key for key in self.states.keys()
            ):
                print(
                    f"Early stopping. In {td.total_seconds() // 60} minutes, "
                    f"Event {event} was not found in any states within its"
                    f"{minutes} minutes time limit"
                )
                early_stop = True
                break
            else:
                print(
                    f"Early stopping check. In {td.total_seconds() // 60} minutes, "
                    f"Event {event} was not found in any states within its"
                    f"{minutes} minutes time limit"
                )
        for event in to_delete:
            print(
                f"Satisified early stopping constraint for {event} within "
                f"{self.early_stop[event]} minutes. "
                f"Event found n {td.total_seconds() // 60} minutes"
            )
            del self.early_stop[event]
        return early_stop

    def swarm(
        self,
        stats: dict[str, Any],
        vecenv: pufferlib.vector.Multiprocessing | pufferlib.vector.Serial,
    ):
        required_counts = stats.get("stats/required_count", [])
        # Update the required completion rate in the sqlite db. Also a bit tricky
        # but doesn't require the manual async reset so that's good
        if (self.async_wrapper or self.sqlite_wrapper) and self.required_rate and required_counts:
            # calculate the average required_count
            required_rate = np.mean(list(self.event_tracker.values()))
            # now update via the async wrapper or sqlite wrapper
            if self.sqlite_db:
                with SqliteStateResetWrapper.DB_LOCK:
                    with sqlite3.connect(self.sqlite_db) as conn:
                        cur = conn.cursor()
                        cur.execute(
                            """
                            UPDATE states
                            SET required_rate=:required_rate
                            """,
                            {"required_rate": required_rate},
                        )
            if self.async_wrapper:
                for key in self.event_tracker.keys():
                    self.env_recv_queues[key].put(f"REQUIRED_RATE{required_rate}".encode())
                for key in self.event_tracker.keys():
                    # print(f"\tWaiting for message from env-id {key}")
                    self.env_send_queues[key].get()

            print("Required rate update - completed")

        # now for a tricky bit:
        # if we have swarm_frequency, we will migrate the bottom
        # % of envs in the batch (by required events count)
        # and migrate them to a new state at random.
        # Now this has a lot of gotchas and is really unstable
        # E.g. Some envs could just constantly be on the bottom since they're never
        # progressing
        # env id in async queues is the index within self.infos - self.config.num_envs + 1
        if (
            (self.async_wrapper or self.sqlite_wrapper)
            and self.swarm_enabled
            and required_counts
            and self.states
        ):
            """
            # V1 implementation - 
            #     collect the top swarm_keep_pct % of the envs in the batch
            #     migrate the envs not in the largest keep pct to one of the top states
            largest = [
                x[1][0]
                for x in heapq.nlargest(
                    math.ceil(len(self.event_tracker) * self.config.swarm_keep_pct),
                    enumerate(self.event_tracker.items()),
                    key=lambda x: x[1][0],
                )
            ]

            to_migrate_keys = set(self.event_tracker.keys()) - set(largest)
            print(f"Migrating {len(to_migrate_keys)} states:")
            for key in to_migrate_keys:
                # we store states in a weird format
                # pull a list of states corresponding to a required event completion state
                new_state_key = random.choice(list(self.states.keys()))
                # pull a state within that list
                new_state = random.choice(self.states[new_state_key])
            """

            # V2 implementation
            # check if we have a new highest required_count with N save states available
            # If we do, migrate 100% of states to one of the states
            max_event_count = 0
            new_state_key = ""
            max_state = None
            for key in self.states.keys():
                candidate_max_state: deque = self.states[key]
                if (
                    len(key) > max_event_count
                    and len(candidate_max_state) == candidate_max_state.maxlen
                ):
                    max_event_count = len(key)
                    new_state_key = key
                    max_state = candidate_max_state
            if max_event_count > self.max_event_count and max_state:
                self.max_event_count = max_event_count

                # Need a way not to reset the env id counter for the driver env
                # Until then env ids are 1-indexed
                print(f"\tNew events ({len(new_state_key)}): {new_state_key}")
                new_states = [
                    state
                    for state in random.choices(
                        self.states[new_state_key], k=len(self.event_tracker.keys())
                    )
                ]
                if self.sqlite_db:
                    with SqliteStateResetWrapper.DB_LOCK:
                        with sqlite3.connect(self.sqlite_db) as conn:
                            cur = conn.cursor()
                            cur.executemany(
                                """
                                UPDATE states
                                SET pyboy_state=:state,
                                    reset=:reset
                                WHERE env_id=:env_id
                                """,
                                tuple(
                                    [
                                        {"state": state, "reset": 1, "env_id": env_id}
                                        for state, env_id in zip(
                                            new_states, self.event_tracker.keys()
                                        )
                                    ]
                                ),
                            )
                    vecenv.async_reset()
                    # drain any INFO
                    key_set = self.event_tracker.keys()
                    while True:
                        # We connect each time just in case we block the wrappers
                        with SqliteStateResetWrapper.DB_LOCK:
                            with sqlite3.connect(self.sqlite_db) as conn:
                                cur = conn.cursor()
                                resets = cur.execute(
                                    """
                                    SELECT reset, env_id
                                    FROM states
                                    """,
                                ).fetchall()
                        if all(not reset for reset, env_id in resets if env_id in key_set):
                            break
                        time.sleep(0.5)
                if self.async_wrapper:
                    for key, state in zip(self.event_tracker.keys(), new_states):
                        self.env_recv_queues[key].put(state)
                    for key in self.event_tracker.keys():
                        # print(f"\tWaiting for message from env-id {key}")
                        self.env_send_queues[key].get()

                print(f"State migration to {self.archive_path}/{str(hash(new_state_key))} complete")
