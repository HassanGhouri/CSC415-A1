import os
os.environ["MUJOCO_GL"] = "egl"

import sys
import subprocess
from pathlib import Path
import pathlib
import dmc2gym
import numpy as np
import gym

# Establish enviroment.
PROJECT_DIR = Path("/Users/hassanghouri/Desktop/CSC415-A1")
os.chdir(PROJECT_DIR)

runs_dir = PROJECT_DIR / "runs"
runs_dir.mkdir(parents=True, exist_ok=True)



# dmc2gym patch: Rewrites np.int and np.bool to fix compatibility issues with NumPy.
pkg = pathlib.Path(dmc2gym.__file__).parent
wrap_path = pkg / "wrappers.py"
content = wrap_path.read_text()
content = content.replace("np.int(", "int(").replace("np.bool", "bool")
wrap_path.write_text(content)

# gym/numpy compatibility patch: Fixes bool8 vs bool compatibility issue between NumPy and gym.
if not hasattr(np, 'bool8'):
    np.bool8 = bool

gym_pkg = pathlib.Path(gym.__file__).parent
files_to_patch = [gym_pkg / "utils/passive_env_checker.py", gym_pkg / "spaces/box.py"]

for file_path in files_to_patch:
    if file_path.exists():
        content = file_path.read_text()
        new_content = content.replace("np.bool8", "bool").replace("np.int(", "int(")
        file_path.write_text(new_content)

# re-apply dmc2gym patch again (Did not work without it for some reason)
dmc_pkg = pathlib.Path(dmc2gym.__file__).parent / "wrappers.py"
if dmc_pkg.exists():
    content = dmc_pkg.read_text()
    dmc_pkg.write_text(content.replace("np.int(", "int("))

# Function to run the traning loop for SAC model with specified Augmentation.
def run_train(work_dir, data_augs):
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    cmd = [sys.executable, "train.py", "--domain_name", "cartpole", "--task_name", "swingup", "--encoder_type", "pixel",
           "--agent", "rad_sac", "--work_dir", str(work_dir), "--action_repeat", "8", "--frame_stack", "3",
           "--pre_transform_image_size", "100", "--image_size", "84", "--data_augs", data_augs, "--seed", "1",
           "--critic_lr", "1e-3", "--actor_lr", "1e-3", "--batch_size", "128", "--eval_freq", "10000",
           "--num_eval_episodes", "10", "--num_train_steps", "50000",]
    subprocess.run(cmd, check=True)


run_train(runs_dir / "repro_crop", "crop")
run_train(runs_dir / "ablation_noaug", "no_aug")