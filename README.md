# Pokemon Red (RL Edition)

![Tests](https://github.com/thatguy11325/pokemonred_puffer/actions/workflows/workflow.yml/badge.svg)

This repo is designed as a library to be used for Pokemon Red RL development. It contains some convenience functions that should not be used in a library setting and should be forked. In the future, those convenience functions will be migrated so no forking is needed.

## Quickstart

### 🧰 Requirements

- **Python 3.10/3.11**
  *(Python 3.12 is **not** supported)*
- `pip` (Python package installer)
- `virtualenv` (optional but recommended)
- System build tools and Python development headers

---

### 🛠 System Dependencies

Native extensions require Python development headers and basic build tools. Install them using your system’s package manager.

#### 🐧 Ubuntu / Debian

You will need to add a PPA for Python versions which are not the latest and update your package list before being able to install Python3.10/Python3.11.

```bash
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.11-dev build-essential
```

> Replace `python3.11-dev` with the appropriate version for your system if using a different Python version. E.g Python3.10

#### 🐧 Arch Linux / Manjaro

Arch Linux provides only the latest version of Python, so you must use the **AUR** (Arch User Repository) to install Python 3.10/Python 3.11.

Install an AUR helper like `yay` if you haven’t already:

```bash
sudo pacman -S --needed git base-devel
git clone https://aur.archlinux.org/yay.git
cd yay
makepkg -si
```

Install the aur package [Python 3.11](https://aur.archlinux.org/packages/python311)
```bash
yay -S python311
```
---

### 🔧 Fix: antlr4-python3-runtime Build Issues

Before installing, **upgrade `pip` and `setuptools`** to avoid errors when building some dependencies (like `antlr4-python3-runtime`):

```bash
pip install --upgrade pip setuptools
```

This resolves errors such as:

```
AttributeError: install_layout. Did you mean: 'install_platlib'?
```

> ⚠️ You may still see a warning about `setup.py install` being deprecated — this can usually be ignored if installation completes.

---

### 🐍 (Optional) Virtual Environment Setup & Sourcing

Creating a virtual environment helps isolate dependencies, and prevent system-wide conflicts. The following will create a folder `.venv` for your dependencies. You should source this everytime.

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Ensure `python3` points to Python 3.11 or earlier.

---

### 📦 Install the Project

Install the project in **editable mode** to reflect changes immediately:

```bash
pip install -e .
```

This will install all dependencies listed in the `pyproject.toml`.

---

### Running

Below are commands that use default arguments in some cases. Please run `python3 -m pokemonred_puffer.train --help` if you are unsure how to use the commandline actions associated with this repo. Some commands may not have been tested recently so please make an issue if you have one. 

After installation you can start training by running:

```sh
# Run before training to test what num_envs value you should use
python3 -m pokemonred_puffer.train autotune
# Default
python3 -m pokemonred_puffer.train train
```

### Multinode Hyperparameter Sweeps (in progress)

If you want to run hyperparameter sweeps, you can do so by installing related packages and launching two commands:

```sh
pip3 install -e '.[sweep]'
python3 -m pokemonred_puffer.sweep launch-sweep
python3 -m pokemonred_puffer.sweep launch-agent <sweep-id>
```

The sweep id will be printed when launching the sweep. To resume a sweep, you can relaunch your sweep with

```sh
python3 -m pokemonred_puffer.sweep launch-sweep --sweep-id <sweep-id>
```

The sweeps can be configured with a sweep configuration (defaulted to `sweep-config.yaml`) and base configuration (defaulted to `config.yaml`). The hyperparamter sweep sets bounds using the sweep config and centers the hyperparamters at paramters in the base config. To learn more about the hyperparamter algorithm, you can visit [Imbue's CARBS repo](https://github.com/imbue-ai/carbs/tree/main).

N.B. Currently single node sweeps are not supported. If this is a desired feature, please make an issue.

### Modifying for Training

So you have a run going, but you want to mess with it, what do you do?

You have a few options:

1. Start altering parameters in `config.yaml`
2. Start modifying the code directly (more on that later).
3. Use this repo as a library and make your own wrappers.

### Debugging
If you want to test your changes you can run 

```sh
python3 -m pokemonred_puffer.train --config config.yaml --debug
```

In emergency cases, it is advised to remove the `send_input` function calls from `environment.py` so you can test the rewards on your own schedule and not the model's.

## Directory Structure

This repo is intended to eventually be used as a library. All source files should be under the `pokemonred_puffer` directory. If you want to add a module with a `__main__`, feel free to, but under the `pokemonred_puffer` directory. Afterwards, you should be to run your main with `python3 -m pokemonred_puffer.<your-module>`

Within the `pokemonred_puffer` directory there are the following files and directories:

- `policies`: A directory for different policies to run the model with.
- `rewards`: A directory of `gym.Env` classes that keep track of rewards for a `RedGymEnv` (gym environment for Pokemon Red) object
- `wrappers`: A directory of wrappers that you may want to use, e.g. logging to the [Pokemon Red Map Visualizer](https://pwhiddy.github.io/pokerl-map-viz/).
- `cleanrl_puffer.py` - Responsible for running the actual training logic
- `environment.py` - The core logic of the Pokemon Red Gym Environment.
- `eval.py` - For generating visualizations for logging during training.
- `kanto_map_dsv.png` - A high resolution map of the Kanto region.
- `train.py` - A script and entrypoint to start training with.

## Making Changes

For simple changes, you can update `config.yaml` directly. `config.yaml` has a few important rules. For `wrappers`, `rewards` and `policies`, the wrapper, reward or policy _must_ be keyed by `module_name.class_name`. These sections can hold multiple types of `wrappers`, `rewards` or `policies`. The general layout is `label : arguments for the class`. This is until a better way with less indirection is figured out.

### Adding Wrappers

To add wrappers, add a new class that inherits from `gym.Wrapper` to the `wrappers` directory. Then update the `wrappers` section of `config.yaml`. The wrappers wrap the base environment in order, from top to bottom. The wrappers list is _not_ keyed by the class path. It is a unique name that distinguishes the collection of wrappers.

### Adding Rewards

To add rewards, add a new class to the `rewards` directory. Then update the `rewards` section of `config.yaml`. A reward section is keyed by the class path.

### Adding Policies

To add policies, add a new class to the `policies` directory. Then update the `policies` section of `config.yaml`. A policy section is keyed by the class path. It is assumed that a recurrent policy will live in the same module as the policy it wraps.

## Development

This repo uses [pre-commit](https://pre-commit.com/) to enforce formatting and linting. For development, please install this repo with:

```sh
pip3 install -e '.[dev]'
pre-commit install
```

For any changes, please submit a PR.

## Authors

[David Rubinstein](https://github.com/drubinstein), [Keelan Donovan](https://github.com/leanke), [Daniel Addis](https://github.com/xinpw8), Kyoung Whan Choe, [Joseph Suarez](https://puffer.ai/), [Peter Whidden](https://peterwhidden.webflow.io/)

<a href="https://star-history.com/#drubinstein/pokemonred_puffer&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=drubinstein/pokemonred_puffer&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=drubinstein/pokemonred_puffer&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=drubinstein/pokemonred_puffer&type=Date" />
 </picture>
</a>

[![](assets/puffer.png)](https://puffer.ai)