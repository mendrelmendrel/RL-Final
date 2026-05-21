"""
A2C/PPO trainer for Flappy Bird using stable-baselines3.

Usage:
    python train_a2c.py --config config/simple_a2c.yml
    python train_a2c.py --config config/simple_a2c.yml --test --model saved_models/best_model.zip
"""

import time
import os
import yaml
import argparse
import torch
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from shimmy import GymV21CompatibilityV0
from gym.wrappers import GrayScaleObservation, ResizeObservation
from gym import Wrapper

from flappy_bird_gym.flappy_bird_gym.envs import FlappyBirdEnvSimple, FlappyBirdEnvRGB


def create_simple_env(train):
    """Create simple environment (low-dim state: coordinates only)."""
    env = FlappyBirdEnvSimple()
    env = LoggingWrapper(env)
    env = GymV21CompatibilityV0(env=env)
    env = Monitor(env)
    env = DummyVecEnv([lambda: env for _ in range(1)])

    eval_env = None
    if train:
        eval_env = FlappyBirdEnvSimple()
        eval_env = LoggingWrapper(eval_env)
        eval_env = GymV21CompatibilityV0(env=eval_env)
        eval_env = Monitor(eval_env)
        eval_env = DummyVecEnv([lambda: eval_env for _ in range(1)])

    return env, eval_env


def create_rgb_env(train, frame_stack=4):
    """Create RGB environment with preprocessing and frame stacking."""
    env = FlappyBirdEnvRGB()
    env = GrayScaleObservation(env, keep_dim=True)
    env = ResizeObservation(env, (84, 84))
    env = LoggingWrapper(env)
    env = GymV21CompatibilityV0(env=env)
    env = Monitor(env)
    env = DummyVecEnv([lambda: env for _ in range(1)])
    env = VecFrameStack(env, frame_stack, channels_order="last")

    eval_env = None
    if train:
        eval_env = FlappyBirdEnvRGB()
        eval_env = GrayScaleObservation(eval_env, keep_dim=True)
        eval_env = ResizeObservation(eval_env, (84, 84))
        eval_env = LoggingWrapper(eval_env)
        eval_env = GymV21CompatibilityV0(env=eval_env)
        eval_env = Monitor(eval_env)
        eval_env = DummyVecEnv([lambda: eval_env for _ in range(1)])
        eval_env = VecFrameStack(eval_env, frame_stack, channels_order="last")

    return env, eval_env


class LoggingWrapper(Wrapper):
    """Wrapper to log episode scores."""
    def __init__(self, env):
        super().__init__(env)
        self.episode_scores = []

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.episode_scores.append(int(info.get("score", 0)))
        return obs, reward, done, info


def train_model(config_dict, config_path):
    """Train a model using the provided config."""
    # Extract config
    env_type = config_dict["type"]
    algorithm = config_dict["hyperparameter"]["algorithm"]
    policy = config_dict["hyperparameter"]["policy"]
    learning_rate = float(config_dict["hyperparameter"]["learning_rate"])
    gamma = float(config_dict["hyperparameter"]["gamma"])
    total_timesteps = int(config_dict["total_timesteps"])
    eval_freq = int(config_dict["eval_freq"])
    name_prefix = config_dict["checkpoints"]["prefix"]
    frame_stack = config_dict.get("frame_stack", 4)
    if isinstance(frame_stack, str):
        frame_stack = int(frame_stack) if frame_stack.isdigit() else 4

    # Create directories
    timestamp = time.strftime("%Y-%m-%d-%H_%M_%S")
    log_dir = f"./logs_a2c/{timestamp}/"
    model_dir = f"./saved_models_a2c/{timestamp}/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # Create environment
    if env_type == "simple":
        env, eval_env = create_simple_env(train=True)
    elif env_type == "rgb":
        env, eval_env = create_rgb_env(train=True, frame_stack=frame_stack)
    else:
        raise ValueError(f"Unknown env type: {env_type}")

    # Setup callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=eval_freq,
        save_path=model_dir,
        name_prefix=name_prefix
    )
    eval_callback = EvalCallback(
        eval_env=eval_env,
        best_model_save_path=model_dir + "best_models/",
        log_path=log_dir,
        eval_freq=eval_freq,
        verbose=1
    )

    # Setup logger
    logger = configure(log_dir, ["stdout", "csv", "json"])

    # Create and train model
    AlgorithmClass = {"A2C": A2C, "PPO": PPO}[algorithm]
    model = AlgorithmClass(
        policy=policy,
        env=env,
        verbose=1,
        learning_rate=learning_rate,
        gamma=gamma,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    model.set_logger(logger)

    print(f"Training {algorithm} with {policy} on {env_type} environment")
    print(f"Config: {config_path}")
    print(f"Total timesteps: {total_timesteps}")
    model.learn(total_timesteps=total_timesteps, callback=[checkpoint_callback, eval_callback])

    # Save final model
    final_model_path = os.path.join(model_dir, f"{name_prefix}final")
    model.save(final_model_path)
    print(f"Training complete. Final model saved to {final_model_path}.zip")


def test_model(config_dict, model_path):
    """Test a trained model."""
    env_type = config_dict["type"]
    frame_stack = config_dict.get("frame_stack", 4)
    if isinstance(frame_stack, str):
        frame_stack = int(frame_stack) if frame_stack.isdigit() else 4

    # Create environment
    if env_type == "simple":
        env, _ = create_simple_env(train=False)
    elif env_type == "rgb":
        env, _ = create_rgb_env(train=False, frame_stack=frame_stack)
    else:
        raise ValueError(f"Unknown env type: {env_type}")

    # Load and run model
    algorithm = config_dict["hyperparameter"]["algorithm"]
    AlgorithmClass = {"A2C": A2C, "PPO": PPO}[algorithm]
    model = AlgorithmClass.load(model_path)

    obs = env.reset()
    print(f"Testing model from {model_path}")
    while True:
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        env.render()
        time.sleep(1/60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config YAML file")
    parser.add_argument("--test", action="store_true", help="Test a trained model")
    parser.add_argument("--model", help="Path to trained model for testing")
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)

    if args.test:
        if not args.model:
            print("ERROR: --model required for testing")
            return
        test_model(config_dict, args.model)
    else:
        train_model(config_dict, args.config)


if __name__ == "__main__":
    main()
