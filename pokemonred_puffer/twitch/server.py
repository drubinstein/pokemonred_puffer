import base64
from multiprocessing import Queue
from typing import Literal

import numpy as np
from fastapi import FastAPI, WebSocket
from pydantic import BaseModel


from pokemonred_puffer.environment import RedGymEnv


class EnvironmentModel(BaseModel):
    name: Literal["e"] = "e"
    env_id: int
    config: dict[str, str | int | float]


class ActionModel(BaseModel):
    name: Literal["a"] = "a"
    env_id: int
    action: str


class SaveStateModel(BaseModel):
    name: Literal["s"] = "s"
    env_id: int
    save_state: bytes


Messages = ActionModel | SaveStateModel

app = FastAPI()
envs: dict[int, RedGymEnv] = {}
env_frames: dict[int, np.ndarray] = {}
envs_queue: Queue[tuple[int, np.ndarray]]


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        match await websocket.receive_text():
            case EnvironmentModel(env_id=env_id, config=config):
                envs[env_id] = RedGymEnv(**config)
            case ActionModel(env_id=env_id, action=action):
                envs[env_id].step(action)
            case SaveStateModel(env_id=env_id, save_state=save_state):
                envs[env_id].reset(options={"state": base64.b64decode(save_state)})


def stream_process():
    global envs_queue
    # TODO: Setup OBS and ffmpeg stream

    while True:
        env_id, frame = envs_queue.get()
        # construct the image. Something like NxM grid with
        # the center as the current game
        # publish frame to OBS
