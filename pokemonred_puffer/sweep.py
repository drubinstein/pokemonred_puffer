import json
import math
import os
from typing import Annotated

import carbs.utils
import sweeps
from sweeps import RunState
import typer
from carbs import (
    CARBS,
    Param,
    ParamDictType,
    ParamType,
    CARBSParams,
    WandbLoggingParams,
    ObservationInParam,
)
from omegaconf import DictConfig, OmegaConf

import wandb
from pokemonred_puffer import train

app = typer.Typer(pretty_exceptions_enable=False)


def sweep_config_to_params(sweep_config: DictConfig, prefix: str = "") -> list[Param]:
    res = []
    for k, v in sweep_config.items():
        # A little hacky. Maybe I should not make this all config based
        if k.startswith("carbs.utils"):
            param_class = getattr(carbs.utils, k.split(".")[-1])
            res += [
                Param(
                    prefix.removesuffix("-").removeprefix("-"),
                    param_class(**v),
                    (v["max"] + v["min"]) // 2
                    if v.get("is_integer", False)
                    else math.sqrt(v["max"] * v["min"]),
                )
            ]
        elif isinstance(v, DictConfig):
            res += sweep_config_to_params(v, prefix=prefix + "-" + k)
        else:
            print(type(v))
    return res


def update_base_config_by_key(base_config: DictConfig, key: str, value: ParamType) -> DictConfig:
    new_config = base_config.copy()
    keys = key.split("-", 1)
    if len(keys) == 1:
        new_config[keys[0]] = value
    else:
        new_config[keys[0]] = update_base_config_by_key(new_config[keys[0]], keys[1], value)
    return new_config


def update_base_config(base_config: DictConfig, suggestion: ParamDictType) -> DictConfig:
    new_config = base_config.copy()
    for k, v in suggestion.items():
        new_config = update_base_config_by_key(new_config, k, v)
    return new_config


@app.command()
def launch_sweep(
    base_config: Annotated[
        DictConfig, typer.Option(help="Base configuration", parser=OmegaConf.load)
    ] = "config.yaml",
    sweep_config: Annotated[
        DictConfig,
        typer.Option(
            help="CARBS sweep config. settings must match base config.", parser=OmegaConf.load
        ),
    ] = "sweep-config.yaml",
    sweep_name: Annotated[str, typer.Option(help="Sweep name")] = "PokeSweep",
):
    config = CARBSParams(
        better_direction_sign=1,
        is_wandb_logging_enabled=False,
        wandb_params=WandbLoggingParams(project_name="Pokemon", run_name="Pokemon"),
    )
    params = sweep_config_to_params(sweep_config)
    carbs = CARBS(config=config, params=params)
    sweep_id = wandb.sweep(
        sweep={
            "name": sweep_name,
            "controller": {"type": "local"},
            "parameters": {},
            "command": ["${args_json}"],
        },
        entity=base_config.wandb.entity,
        project=base_config.wandb.project,
    )

    import pprint

    pprint.pprint(params)
    sweep = wandb.controller(sweep_id)

    print(f"Beginning sweep with id {sweep_id}")
    print(f"On all nodes please run python -m pokemonred_puffer.sweep launch-agent {sweep_id}")
    finished = set([])
    while not sweep.done():
        # Taken from sweep.schedule. Limits runs to only one at a time.
        # if not (sweep._controller and sweep._controller.get("schedule")):
        # Only one run will be scheduled at a time
        suggestion = carbs.suggest()
        run = sweeps.SweepRun(config={"x": {"value": suggestion.suggestion}})
        sweep._step()
        sweep.schedule(run)
        # without this nothing updates...
        sweep_obj = sweep._sweep_obj
        if runs := sweep_obj["runs"]:
            for run in runs:
                if run["state"] == RunState.running.value:
                    pass
                elif (
                    run["state"]
                    in [
                        RunState.failed.value,
                        RunState.finished.value,
                        RunState.crashed.value,
                    ]
                    and run["name"] not in finished
                ):
                    finished.add(run["name"])
                    summary_metrics = json.loads(run["summaryMetrics"])
                    obs_in = ObservationInParam(
                        input=json.loads(run["config"])["x"]["value"],
                        output=summary_metrics["environment/reward_sum"],
                        cost=summary_metrics["performance/uptime"],
                    )
                    carbs.observe(obs_in)
                elif run["state"] == RunState.pending:
                    print(f"PENDING RUN FOUND {run['name']}")
        sweep.print_status()


@app.command()
def launch_agent(
    sweep_id: str,
    base_config: Annotated[
        DictConfig,
        typer.Option(help="Base configuration. MUST MATCH PRIMARY NODE.", parser=OmegaConf.load),
    ] = "config.yaml",
):
    def _fn():
        agent_config: DictConfig = OmegaConf.load(os.environ["WANDB_SWEEP_PARAM_PATH"]).x.value
        agent_config = update_base_config(base_config, agent_config)
        train.train(config=agent_config, debug=True, track=True)

    wandb.agent(
        sweep_id=sweep_id,
        entity=base_config.wandb.entity,
        project=base_config.wandb.project,
        function=_fn,
        count=99999,
    )


if __name__ == "__main__":
    app()
