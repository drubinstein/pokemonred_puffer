import argparse
import base64
import os

import mediapy
from omegaconf import OmegaConf
from tqdm import tqdm

from pokemonred_puffer.environment import RedGymEnv
from pokemonred_puffer.rewards.baseline import ObjectRewardRequiredEventsMapIdsFieldMoves

CHUNK_SIZE = 100 * 1024  # 100 KB


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("rom_path")
    parser.add_argument("actions_dir")
    parser.add_argument("output_dir")
    parser.add_argument("--n-steps", type=int, default=None, help="Number of steps to render.")
    parser.add_argument("--actions-file", default="", help="Only render this file in actions_dir")
    args = parser.parse_args()

    # steps is a list of lists where the index maps to the step count
    os.makedirs(args.output_dir, exist_ok=True)
    for path in os.listdir(str(args.actions_dir)):
        if args.actions_file and args.actions_file != path:
            continue
        if not path.endswith("actions.nsv"):
            continue
        env_id, _ = path.split("-")
        # The config must match what was used for training
        output_file = os.path.join(args.output_dir, f"actions-{env_id}.mp4")

        # read the file once to get the total number of lines
        n_lines = 0
        with open(os.path.join(args.actions_dir, path)) as f:
            for line in f:
                n_lines += 1
        with (
            open(os.path.join(args.actions_dir, path)) as f,
            mediapy.VideoWriter(output_file, (144, 160), fps=60) as writer,
        ):
            print(f"Writing output {output_file}")
            config = OmegaConf.load("config.yaml")
            config.env.gb_path = args.rom_path
            config.env.log_frequency = None
            env = ObjectRewardRequiredEventsMapIdsFieldMoves(
                config.env,
                config.rewards.get("baseline.ObjectRewardRequiredEventsMapIdsFieldMoves").reward,
            )
            env.reset()
            writer.add_image(env.render()[:, :])
            # Read lines so we can get an estimate of the line count
            for line in tqdm(f, total=n_lines):
                line = line.strip()
                if len(line) == 1:
                    process_action(env, int(line))
                else:
                    process_state(env, base64.b64decode(line))
                writer.add_image(env.render()[:, :])

    os.sync()


def process_state(env: RedGymEnv, state: bytes):
    env.reset(options={"state": state})


def process_action(env: RedGymEnv, action: int):
    env.step(action)


if __name__ == "__main__":
    main()
