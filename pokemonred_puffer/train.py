import functools
import importlib
import sqlite3
import time
import uuid
from contextlib import nullcontext
from enum import Enum
from multiprocessing import Queue
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Annotated, Any, Callable

import gymnasium
import pufferlib
import pufferlib.emulation
import pufferlib.vector
import torch
import typer
from omegaconf import DictConfig, OmegaConf
from pufferlib.pufferl import PuffeRL, WandbLogger
from torch import nn

from pokemonred_puffer.environment import RedGymEnv
from pokemonred_puffer.info_handler import StateManager
from pokemonred_puffer.rewards.baseline import BaselineRewardEnv
from pokemonred_puffer.wrappers.async_io import AsyncWrapper
from pokemonred_puffer.wrappers.sqlite import SqliteStateResetWrapper

app = typer.Typer(pretty_exceptions_enable=False)

DEFAULT_CONFIG = "config.yaml"
DEFAULT_POLICY = "multi_convolutional.MultiConvolutionalPolicy"
DEFAULT_REWARD = "baseline.ObjectRewardRequiredEventsMapIdsFieldMoves"
DEFAULT_WRAPPER = "stream_only"
DEFAULT_ROM = Path("red.gb")


class Vectorization(Enum):
    multiprocessing = "multiprocessing"
    serial = "serial"
    ray = "ray"


def make_policy(env: RedGymEnv, policy_name: str, config: DictConfig) -> nn.Module:
    policy_module_name, policy_class_name = policy_name.split(".")
    policy_module = importlib.import_module(f"pokemonred_puffer.policies.{policy_module_name}")
    policy_class = getattr(policy_module, policy_class_name)

    policy = policy_class(env, **config.policies[policy_name].policy)
    if config.train.use_rnn:
        rnn_config = config.policies[policy_name].rnn
        policy_class = getattr(policy_module, rnn_config.name)
        policy = policy_class(env, policy, **rnn_config.args)

    return policy.to(config.train.device)


def load_from_config(config: DictConfig, debug: bool) -> DictConfig:
    default_keys = ["env", "train", "policies", "rewards", "wrappers", "wandb"]
    defaults = OmegaConf.create({key: config.get(key, {}) for key in default_keys})

    # Package and subpackage (environment) configs
    debug_config = config.get("debug", OmegaConf.create({})) if debug else OmegaConf.create({})

    defaults.merge_with(debug_config)
    return defaults


def make_env_creator(
    wrapper_classes: list[tuple[str, type[gymnasium.Env]]],
    reward_class: type[BaselineRewardEnv],
    async_wrapper: bool = False,
    sqlite_wrapper: bool = False,
    puffer_wrapper: bool = True,
) -> Callable[..., pufferlib.emulation.GymnasiumPufferEnv | gymnasium.Env]:
    def env_creator(
        env_config: DictConfig,
        wrappers_config: list[dict[str, Any]],
        reward_config: DictConfig,
        async_config: dict[str, list[Queue]] | None = None,
        sqlite_config: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> pufferlib.emulation.GymnasiumPufferEnv | gymnasium.Env:
        env = reward_class(env_config, reward_config)
        for cfg, (_, wrapper_class) in zip(wrappers_config, wrapper_classes):
            dict_config = OmegaConf.create([x for x in cfg.values()][0])
            assert isinstance(dict_config, DictConfig)
            env = wrapper_class(env, dict_config)  # type: ignore
        if async_wrapper and async_config:
            env = AsyncWrapper(env, async_config["send_queues"], async_config["recv_queues"])  # type: ignore
        if sqlite_wrapper and sqlite_config:
            env = SqliteStateResetWrapper(env, sqlite_config["database"])  # type: ignore
        if puffer_wrapper:
            env = pufferlib.emulation.GymnasiumPufferEnv(env=env, **kwargs)
            # env.is_observation_checked = True
        return env

    return env_creator


def setup_agent(
    wrappers: list[dict[str, Any]],
    reward_name: str,
    async_wrapper: bool = False,
    sqlite_wrapper: bool = False,
    puffer_wrapper: bool = True,
) -> Callable[..., pufferlib.emulation.GymnasiumPufferEnv | gymnasium.Env]:
    # TODO: Make this less dependent on the name of this repo and its file structure
    wrapper_classes: list[tuple[str, type[gymnasium.Env]]] = [
        (
            k,
            getattr(
                importlib.import_module(f"pokemonred_puffer.wrappers.{k.split('.')[0]}"),
                k.split(".")[1],
            ),
        )
        for wrapper_dicts in wrappers
        for k in wrapper_dicts.keys()
    ]
    reward_module, reward_class_name = reward_name.split(".")
    reward_class = getattr(
        importlib.import_module(f"pokemonred_puffer.rewards.{reward_module}"), reward_class_name
    )
    # NOTE: This assumes reward_module has RewardWrapper(RedGymEnv) class
    env_creator = make_env_creator(
        wrapper_classes, reward_class, async_wrapper, sqlite_wrapper, puffer_wrapper
    )

    return env_creator


def setup(
    config: DictConfig,
    debug: bool,
    wrappers_name: str,
    reward_name: str,
    rom_path: Path,
    track: bool,
    puffer_wrapper: bool = True,
) -> tuple[DictConfig, Callable[..., pufferlib.emulation.GymnasiumPufferEnv | gymnasium.Env]]:
    config.train.exp_id = f"pokemon-red-{str(uuid.uuid4())[:8]}"
    config.env.gb_path = rom_path
    config.track = track
    if debug:
        config.vectorization = Vectorization.serial

    async_wrapper = config.train.get("async_wrapper", False)
    sqlite_wrapper = config.train.get("sqlite_wrapper", False)
    env_creator = setup_agent(
        config.wrappers[wrappers_name], reward_name, async_wrapper, sqlite_wrapper, puffer_wrapper
    )
    return config, env_creator


# @app.command()
# def evaluate(
#     config: Annotated[
#         DictConfig, typer.Option(help="Base configuration", parser=OmegaConf.load)
#     ] = DEFAULT_CONFIG,
#     checkpoint_path: Path | None = None,
#     policy_name: Annotated[
#         str,
#         typer.Option(
#             "--policy-name",
#             "-p",
#             help="Policy module to use in policies.",
#         ),
#     ] = DEFAULT_POLICY,
#     reward_name: Annotated[
#         str,
#         typer.Option(
#             "--reward-name",
#             "-r",
#             help="Reward module to use in rewards",
#         ),
#     ] = DEFAULT_REWARD,
#     wrappers_name: Annotated[
#         str,
#         typer.Option(
#             "--wrappers-name",
#             "-w",
#             help="Wrappers to use _in order of instantion_",
#         ),
#     ] = DEFAULT_WRAPPER,
#     rom_path: Path = DEFAULT_ROM,
# ):
#     config, env_creator = setup(
#         config=config,
#         debug=False,
#         wrappers_name=wrappers_name,
#         reward_name=reward_name,
#         rom_path=rom_path,
#         track=False,
#     )
#     env_kwargs = {
#         "env_config": config.env,
#         "wrappers_config": config.wrappers[wrappers_name],
#         "reward_config": config.rewards[reward_name]["reward"],
#         "async_config": {},
#     }
#     try:
#         cleanrl_puffer.rollout(
#             env_creator,
#             env_kwargs,
#             model_path=checkpoint_path,
#             device=config.train.device,
#         )
#     except KeyboardInterrupt:
#         os._exit(0)


# @app.command()
# def autotune(
#     config: Annotated[
#         DictConfig, typer.Option(help="Base configuration", parser=OmegaConf.load)
#     ] = DEFAULT_CONFIG,
#     policy_name: Annotated[
#         str,
#         typer.Option(
#             "--policy-name",
#             "-p",
#             help="Policy module to use in policies.",
#         ),
#     ] = DEFAULT_POLICY,
#     reward_name: Annotated[
#         str,
#         typer.Option(
#             "--reward-name",
#             "-r",
#             help="Reward module to use in rewards",
#         ),
#     ] = DEFAULT_REWARD,
#     wrappers_name: Annotated[
#         str,
#         typer.Option(
#             "--wrappers-name",
#             "-w",
#             help="Wrappers to use _in order of instantion_",
#         ),
#     ] = "empty",
#     rom_path: Path = DEFAULT_ROM,
# ):
#     config = load_from_config(config, False)
#     config.vectorization = "multiprocessing"
#     config, env_creator = setup(
#         config=config,
#         debug=False,
#         wrappers_name=wrappers_name,
#         reward_name=reward_name,
#         rom_path=rom_path,
#         track=False,
#     )
#     env_kwargs = {
#         "env_config": config.env,
#         "wrappers_config": config.wrappers[wrappers_name],
#         "reward_config": config.rewards[reward_name]["reward"],
#         "async_config": {},
#     }
#     pufferlib.vector.autotune(
#         functools.partial(env_creator, **env_kwargs), batch_size=config.train.batch_size
#     )


@app.command()
def debug(
    config: Annotated[
        DictConfig, typer.Option(help="Base configuration", parser=OmegaConf.load)
    ] = DEFAULT_CONFIG,
    reward_name: Annotated[
        str,
        typer.Option(
            "--reward-name",
            "-r",
            help="Reward module to use in rewards",
        ),
    ] = DEFAULT_REWARD,
    wrappers_name: Annotated[
        str,
        typer.Option(
            "--wrappers-name",
            "-w",
            help="Wrappers to use _in order of instantion_",
        ),
    ] = "empty",
    rom_path: Path = DEFAULT_ROM,
):
    config = load_from_config(config, True)
    config.env.gb_path = rom_path
    config, env_creator = setup(
        config=config,
        debug=True,
        wrappers_name=wrappers_name,
        reward_name=reward_name,
        rom_path=rom_path,
        track=False,
        puffer_wrapper=False,
    )
    env = env_creator(
        config.env, config.wrappers[wrappers_name], config.rewards[reward_name]["reward"]
    )
    env.reset()
    while True:
        env.step(5)
        time.sleep(0.2)
    env.close()


@app.command()
def train(
    config: Annotated[
        DictConfig, typer.Option(help="Base configuration", parser=OmegaConf.load)
    ] = DEFAULT_CONFIG,
    policy_name: Annotated[
        str,
        typer.Option(
            "--policy-name",
            "-p",
            help="Policy module to use in policies.",
        ),
    ] = DEFAULT_POLICY,
    reward_name: Annotated[
        str,
        typer.Option(
            "--reward-name",
            "-r",
            help="Reward module to use in rewards",
        ),
    ] = DEFAULT_REWARD,
    wrappers_name: Annotated[
        str,
        typer.Option(
            "--wrappers-name",
            "-w",
            help="Wrappers to use _in order of instantion_",
        ),
    ] = DEFAULT_WRAPPER,
    exp_name: Annotated[str | None, typer.Option(help="Resume from experiment")] = None,
    rom_path: Path = DEFAULT_ROM,
    track: Annotated[bool, typer.Option(help="Track on wandb.")] = False,
    debug: Annotated[bool, typer.Option(help="debug")] = False,
    vectorization: Annotated[
        Vectorization, typer.Option(help="Vectorization method")
    ] = "multiprocessing",
):
    config = load_from_config(config, debug)
    config.vectorization = vectorization
    config, env_creator = setup(
        config=config,
        debug=debug,
        wrappers_name=wrappers_name,
        reward_name=reward_name,
        rom_path=rom_path,
        track=track,
    )

    vec = config.vectorization
    if vec == Vectorization.serial:
        vec = pufferlib.vector.Serial
    elif vec == Vectorization.multiprocessing:
        vec = pufferlib.vector.Multiprocessing
    else:
        vec = pufferlib.vector.Multiprocessing

    # TODO: Remove the +1 once the driver env doesn't permanently increase the env id
    env_send_queues: list[Queue] = []
    env_recv_queues: list[Queue] = []
    if config.train.get("async_wrapper", False):
        env_send_queues = [Queue() for _ in range(2 * config.train.num_envs + 1)]
        env_recv_queues = [Queue() for _ in range(2 * config.train.num_envs + 1)]

    sqlite_context = nullcontext
    if config.train.get("sqlite_wrapper", False):
        sqlite_context = functools.partial(NamedTemporaryFile, suffix="sqlite")

    with sqlite_context() as sqlite_db:
        db_filename = None
        if config.train.get("sqlite_wrapper", False):
            assert sqlite_db
            db_filename = sqlite_db.name
            with sqlite3.connect(db_filename) as conn:
                cur = conn.cursor()
                cur.execute(
                    "CREATE TABLE states(env_id INT PRIMARY_KEY, pyboy_state BLOB, reset BOOLEAN, required_rate REAL, pid INT);"
                )

        vecenv: pufferlib.vector.Multiprocessing | pufferlib.vector.Serial = pufferlib.vector.make(
            env_creator,
            env_kwargs={
                "env_config": config.env,
                "wrappers_config": config.wrappers[wrappers_name],
                "reward_config": config.rewards[reward_name]["reward"],
                "async_config": {
                    "send_queues": env_send_queues,
                    "recv_queues": env_recv_queues,
                },
                "sqlite_config": {"database": db_filename},
            },
            num_envs=config.train.num_envs,
            num_workers=config.train.num_workers,
            batch_size=config.train.batch_size,
            zero_copy=config.train.zero_copy,
            backend=vec,  # type: ignore
        )
        policy = make_policy(vecenv.driver_env, policy_name, config)  # type: ignore

        config.train.env = "Pokemon Red"
        logger = None
        if track:
            logger = WandbLogger(
                args=dict(config.wandb)
                | {
                    "config": {
                        "cleanrl": config.train,
                        "env": config.env,
                        "reward_module": reward_name,
                        "policy_module": policy_name,
                        "reward": config.rewards[reward_name],
                        "policy": config.policies[policy_name],
                        "wrappers": config.wrappers[wrappers_name],
                        "rnn": "rnn" in config.policies[policy_name],
                    },
                    "name": exp_name,
                    "monitor_gym": True,
                    "save_code": True,
                },
            )
        trainer = PuffeRL(config=config.train, vecenv=vecenv, policy=policy, logger=logger)
        if config.train.get("compile", False):
            trainer.policy.forward_eval = torch.compile(  # type: ignore
                trainer.policy.forward_eval,  # type: ignore
                mode=config.train["compile_mode"],
                fullgraph=config.train["compile_fullgraph"],
            )
        state_manager = StateManager(
            sqlite_db=db_filename,
            env_send_queues=env_send_queues,  # type: ignore
            env_recv_queues=env_recv_queues,  # type: ignore
            archive_states=config.train.get("archive_states", False),
            early_stop=config.train.get("early_stop", False),
            async_wrapper=config.train.get("async_wrapper", False),
            sqlite_wrapper=config.train.get("sqlite_wrapper", False),
            swarm_enabled=config.train.get("swarm", False),
            save_overlay=config.train.get("save_overlay", False),
            overlay_interval=config.train.get("overlay_interval", 1000),
            required_rate=config.train.get("required_rate", False),
            wandb_enabled=track,
        )
        epoch = 0
        while trainer.global_step < config.train["total_timesteps"]:
            if config.train.get("compile", False):
                for k in trainer.lstm_h:
                    trainer.lstm_h[k] = trainer.lstm_h[k].clone()
                    trainer.lstm_c[k] = trainer.lstm_c[k].clone()
                torch.compiler.cudagraph_mark_step_begin()
            stats = trainer.evaluate()
            stats = state_manager.process_stats(stats, epoch)
            trainer.stats = stats
            state_manager.check_early_stop()
            state_manager.swarm(stats, vecenv)
            if config.train.get("compile", False):
                for k in trainer.lstm_h:
                    trainer.lstm_h[k] = trainer.lstm_h[k].clone()
                    trainer.lstm_c[k] = trainer.lstm_c[k].clone()
                torch.compiler.cudagraph_mark_step_begin()
            trainer.train()
            epoch += 1

        model_path = trainer.close()
        if logger:
            logger.close(model_path)

        print("Done training")


if __name__ == "__main__":
    app()
