import functools
import importlib
import os
import sqlite3
from tempfile import NamedTemporaryFile
import time
import uuid
from contextlib import contextmanager, nullcontext
from enum import Enum
from multiprocessing import Queue
from pathlib import Path
from types import ModuleType
from typing import Annotated, Any, Callable

import gymnasium
import pufferlib
import pufferlib.emulation
import pufferlib.vector
import typer
from omegaconf import DictConfig, OmegaConf
from torch import nn

import wandb
from pokemonred_puffer import cleanrl_puffer
from pokemonred_puffer.cleanrl_puffer import CleanPuffeRL
from pokemonred_puffer.environment import RedGymEnv
from pokemonred_puffer.wrappers.async_io import AsyncWrapper
from pokemonred_puffer.wrappers.sqlite import SqliteStateResetWrapper

app = typer.Typer(pretty_exceptions_enable=False)

DEFAULT_CONFIG = "config.yaml"
DEFAULT_POLICY = "multi_convolutional.MultiConvolutionalPolicy"
DEFAULT_REWARD = "baseline.ObjectRewardRequiredEventsMapIdsFieldMoves"
DEFAULT_WRAPPER = "stream_only"
DEFAULT_ROM = "red.gb"


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
        policy = pufferlib.frameworks.cleanrl.RecurrentPolicy(policy)
    else:
        policy = pufferlib.frameworks.cleanrl.Policy(policy)

    return policy.to(config.train.device)


def load_from_config(config: DictConfig, debug: bool) -> DictConfig:
    default_keys = ["env", "train", "policies", "rewards", "wrappers", "wandb"]
    defaults = OmegaConf.create({key: config.get(key, {}) for key in default_keys})

    # Package and subpackage (environment) configs
    debug_config = config.get("debug", OmegaConf.create({})) if debug else OmegaConf.create({})

    defaults.merge_with(debug_config)
    return defaults


def make_env_creator(
    wrapper_classes: list[tuple[str, ModuleType]],
    reward_class: RedGymEnv,
    async_wrapper: bool = False,
    sqlite_wrapper: bool = False,
    puffer_wrapper: bool = True,
) -> Callable[[DictConfig, DictConfig], pufferlib.emulation.GymnasiumPufferEnv | gymnasium.Env]:
    def env_creator(
        env_config: DictConfig,
        wrappers_config: list[dict[str, Any]],
        reward_config: DictConfig,
        async_config: dict[str, Queue] | None = None,
        sqlite_config: dict[str, str] | None = None,
    ) -> pufferlib.emulation.GymnasiumPufferEnv | gymnasium.Env:
        env = reward_class(env_config, reward_config)
        for cfg, (_, wrapper_class) in zip(wrappers_config, wrapper_classes):
            env = wrapper_class(env, OmegaConf.create([x for x in cfg.values()][0]))
        if async_wrapper and async_config:
            env = AsyncWrapper(env, async_config["send_queues"], async_config["recv_queues"])
        if sqlite_wrapper and sqlite_config:
            env = SqliteStateResetWrapper(env, sqlite_config["database"])
        if puffer_wrapper:
            env = pufferlib.emulation.GymnasiumPufferEnv(env=env)
        return env

    return env_creator


def setup_agent(
    wrappers: list[str],
    reward_name: str,
    async_wrapper: bool = False,
    sqlite_wrapper: bool = False,
    puffer_wrapper: bool = True,
) -> Callable[[DictConfig, DictConfig], pufferlib.emulation.GymnasiumPufferEnv]:
    # TODO: Make this less dependent on the name of this repo and its file structure
    wrapper_classes = [
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


@contextmanager
def init_wandb(
    config: DictConfig,
    exp_name: str,
    reward_name: str,
    policy_name: str,
    wrappers_name: str,
    resume: bool = True,
):
    if not config.track:
        yield None
    else:
        assert config.wandb.project is not None, "Please set the wandb project in config.yaml"
        assert config.wandb.entity is not None, "Please set the wandb entity in config.yaml"
        wandb_kwargs = {
            "id": exp_name or wandb.util.generate_id(),
            "project": config.wandb.project,
            "entity": config.wandb.entity,
            "group": config.wandb.group,
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
            "resume": resume,
        }
        client = wandb.init(**wandb_kwargs)
        yield client
        client.finish()


def setup(
    config: DictConfig,
    debug: bool,
    wrappers_name: str,
    reward_name: str,
    rom_path: Path,
    track: bool,
    puffer_wrapper: bool = True,
) -> tuple[DictConfig, Callable[[DictConfig, DictConfig], pufferlib.emulation.GymnasiumPufferEnv]]:
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


@app.command()
def evaluate(
    config: Annotated[
        DictConfig, typer.Option(help="Base configuration", parser=OmegaConf.load)
    ] = DEFAULT_CONFIG,
    checkpoint_path: Path | None = None,
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
    rom_path: Path = DEFAULT_ROM,
):
    config, env_creator = setup(
        config=config,
        debug=False,
        wrappers_name=wrappers_name,
        reward_name=reward_name,
        rom_path=rom_path,
        track=False,
    )
    env_kwargs = {
        "env_config": config.env,
        "wrappers_config": config.wrappers[wrappers_name],
        "reward_config": config.rewards[reward_name]["reward"],
        "async_config": {},
    }
    try:
        cleanrl_puffer.rollout(
            env_creator,
            env_kwargs,
            model_path=checkpoint_path,
            device=config.train.device,
        )
    except KeyboardInterrupt:
        os._exit(0)


@app.command()
def autotune(
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
    ] = "empty",
    rom_path: Path = DEFAULT_ROM,
):
    config = load_from_config(config, False)
    config.vectorization = "multiprocessing"
    config, env_creator = setup(
        config=config,
        debug=False,
        wrappers_name=wrappers_name,
        reward_name=reward_name,
        rom_path=rom_path,
        track=False,
    )
    env_kwargs = {
        "env_config": config.env,
        "wrappers_config": config.wrappers[wrappers_name],
        "reward_config": config.rewards[reward_name]["reward"],
        "async_config": {},
    }
    pufferlib.vector.autotune(
        functools.partial(env_creator, **env_kwargs), batch_size=config.train.env_batch_size
    )


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
    with init_wandb(
        config=config,
        exp_name=exp_name,
        reward_name=reward_name,
        policy_name=policy_name,
        wrappers_name=wrappers_name,
    ) as wandb_client:
        vec = config.vectorization
        if vec == Vectorization.serial:
            vec = pufferlib.vector.Serial
        elif vec == Vectorization.multiprocessing:
            vec = pufferlib.vector.Multiprocessing
        elif vec == Vectorization.ray:
            vec = pufferlib.vector.Ray
        else:
            vec = pufferlib.vector.Multiprocessing

        # TODO: Remove the +1 once the driver env doesn't permanently increase the env id
        env_send_queues = []
        env_recv_queues = []
        if config.train.get("async_wrapper", False):
            env_send_queues = [Queue() for _ in range(2 * config.train.num_envs + 1)]
            env_recv_queues = [Queue() for _ in range(2 * config.train.num_envs + 1)]

        sqlite_context = nullcontext
        if config.train.get("sqlite_wrapper", False):
            sqlite_context = functools.partial(NamedTemporaryFile, suffix="sqlite")

        with sqlite_context() as sqlite_db:
            db_filename = None
            if config.train.get("sqlite_wrapper", False):
                db_filename = sqlite_db.name
                with sqlite3.connect(db_filename) as conn:
                    cur = conn.cursor()
                    cur.execute(
                        "CREATE TABLE states(env_id INT PRIMARY_KEY, pyboy_state BLOB, reset BOOLEAN, required_rate REAL, pid INT);"
                    )

            vecenv = pufferlib.vector.make(
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
                batch_size=config.train.env_batch_size,
                zero_copy=config.train.zero_copy,
                backend=vec,
            )
            policy = make_policy(vecenv.driver_env, policy_name, config)

            config.train.env = "Pokemon Red"
            with CleanPuffeRL(
                exp_name=exp_name,
                config=config.train,
                vecenv=vecenv,
                policy=policy,
                env_recv_queues=env_recv_queues,
                env_send_queues=env_send_queues,
                sqlite_db=db_filename,
                wandb_client=wandb_client,
            ) as trainer:
                while not trainer.done_training():
                    trainer.evaluate()
                    trainer.train()

            print("Done training")


if __name__ == "__main__":
    app()
