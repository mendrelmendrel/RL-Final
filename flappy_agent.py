"""
DQN trainer for FlappyBird (Gymnasium `FlappyBird-v0`).

- Replay buffer, target network, epsilon-greedy exploration
- Optional reward shaping
- Periodic deterministic evaluation and checkpoints

"""

import argparse
import math
import random
from collections import namedtuple, deque

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym

try:
    import flappy_bird_gymnasium  # registers FlappyBird-v0
except Exception:
    pass


Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class ReplayMemory:
    def __init__(self, capacity: int):
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size: int):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class NatureDQN(nn.Module):
    def __init__(self, in_channels: int, n_actions: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        with torch.no_grad():
            tensor = torch.zeros(1, in_channels, 84, 84)
            tensor = F.relu(self.conv1(tensor))
            tensor = F.relu(self.conv2(tensor))
            tensor = F.relu(self.conv3(tensor))
            linear_dim = int(np.prod(tensor.shape))

        self.fc1 = nn.Linear(linear_dim, 512)
        self.head = nn.Linear(512, n_actions)

    def forward(self, value):
        value = value / 255.0
        value = F.relu(self.conv1(value))
        value = F.relu(self.conv2(value))
        value = F.relu(self.conv3(value))
        value = value.view(value.size(0), -1)
        value = F.relu(self.fc1(value))
        return self.head(value)


class MLP(nn.Module):
    def __init__(self, input_dim: int, n_actions: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.head = nn.Linear(128, n_actions)

    def forward(self, value):
        value = F.relu(self.fc1(value))
        value = F.relu(self.fc2(value))
        return self.head(value)


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    image = Image.fromarray(frame)
    image = image.convert("L").resize((84, 84))
    return np.array(image, dtype=np.uint8)


def stack_frames(frames, device: str = DEVICE):
    arr = np.stack(frames, axis=0)
    return torch.tensor(arr, dtype=torch.float32, device=device).unsqueeze(0)


def epsilon_by_step(step: int, eps_start: float, eps_end: float, eps_decay: int) -> float:
    return eps_end + (eps_start - eps_end) * math.exp(-1.0 * step / eps_decay)


def select_action(state, policy_net, n_actions: int, epsilon: float, device: str) -> int:
    if random.random() > epsilon:
        with torch.no_grad():
            return int(policy_net(state).argmax(dim=1).item())
    return random.randrange(n_actions)


def make_state_from_obs(obs, is_vector: bool, frame_buffer, device: str):
    if is_vector:
        return torch.tensor(np.asarray(obs, dtype=np.float32), device=device).unsqueeze(0)

    frame = preprocess_frame(obs)
    if frame_buffer is None:
        frame_buffer = deque([frame.copy() for _ in range(4)], maxlen=4)
    else:
        frame_buffer.append(frame)
    return stack_frames(list(frame_buffer), device=device), frame_buffer


def evaluate(policy_net, is_vector: bool, eval_episodes: int, device: str):
    env = gym.make("FlappyBird-v0")
    rewards = []

    for _ in range(eval_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        frame_buffer = None

        if is_vector:
            state = torch.tensor(np.asarray(obs, dtype=np.float32), device=device).unsqueeze(0)
        else:
            first = preprocess_frame(obs)
            frame_buffer = deque([first.copy() for _ in range(4)], maxlen=4)
            state = stack_frames(list(frame_buffer), device=device)

        while not done:
            with torch.no_grad():
                action = int(policy_net(state).argmax(dim=1).item())
            obs_next, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_reward += reward

            if not done:
                if is_vector:
                    state = torch.tensor(np.asarray(obs_next, dtype=np.float32), device=device).unsqueeze(0)
                else:
                    frame_buffer.append(preprocess_frame(obs_next))
                    state = stack_frames(list(frame_buffer), device=device)

        rewards.append(ep_reward)

    env.close()
    return float(np.mean(rewards)) if rewards else 0.0


def train(args):
    env = gym.make("FlappyBird-v0")
    obs, _ = env.reset()
    n_actions = env.action_space.n

    is_vector = isinstance(obs, (list, tuple, np.ndarray)) and np.asarray(obs).ndim == 1

    if is_vector:
        obs_size = int(np.prod(np.asarray(obs).shape))
        policy_net = MLP(obs_size, n_actions).to(args.device)
        target_net = MLP(obs_size, n_actions).to(args.device)
    else:
        policy_net = NatureDQN(4, n_actions).to(args.device)
        target_net = NatureDQN(4, n_actions).to(args.device)

    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.AdamW(policy_net.parameters(), lr=args.lr)
    memory = ReplayMemory(args.memory_cap)

    steps_done = 0
    total_steps = 0
    all_rewards = []

    print(f"Training started | device={args.device} | obs={'vector' if is_vector else 'pixel'}")

    for ep in range(1, args.episodes + 1):
        obs, _ = env.reset()
        frame_buffer = None

        if is_vector:
            state = torch.tensor(np.asarray(obs, dtype=np.float32), device=args.device).unsqueeze(0)
        else:
            # Explicitly re-initialize frame stack every episode.
            first = preprocess_frame(obs)
            frame_buffer = deque([first.copy() for _ in range(4)], maxlen=4)
            state = stack_frames(list(frame_buffer), device=args.device)

        ep_reward = 0.0
        done = False
        ep_steps = 0

        while not done:
            epsilon = epsilon_by_step(steps_done, args.eps_start, args.eps_end, args.eps_decay)
            action = select_action(state, policy_net, n_actions, epsilon, args.device)

            obs_next, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_steps += 1
            steps_done += 1
            total_steps += 1

            raw_reward = reward
            if args.reward_shaping:
                if 0 < reward < 1.0:
                    reward = reward + args.survival_penalty
                if reward < 0:
                    reward = reward * args.crash_penalty_multiplier

            ep_reward += raw_reward

            if not done:
                if is_vector:
                    next_state = torch.tensor(np.asarray(obs_next, dtype=np.float32), device=args.device).unsqueeze(0)
                else:
                    frame_buffer.append(preprocess_frame(obs_next))
                    next_state = stack_frames(list(frame_buffer), device=args.device)
            else:
                next_state = None

            memory.push(
                state,
                torch.tensor([[action]], dtype=torch.long, device=args.device),
                next_state,
                torch.tensor([reward], dtype=torch.float32, device=args.device),
            )

            state = next_state

            if len(memory) >= args.batch_size:
                transitions = memory.sample(args.batch_size)
                batch = Transition(*zip(*transitions))

                non_final_mask = torch.tensor(
                    [entry is not None for entry in batch.next_state],
                    dtype=torch.bool,
                    device=args.device,
                )

                if non_final_mask.any().item():
                    non_final_next_states = torch.cat([entry for entry in batch.next_state if entry is not None])
                else:
                    non_final_next_states = None

                state_batch = torch.cat(batch.state)
                action_batch = torch.cat(batch.action)
                reward_batch = torch.cat(batch.reward)

                state_action_values = policy_net(state_batch).gather(1, action_batch)
                next_state_values = torch.zeros(args.batch_size, device=args.device)

                with torch.no_grad():
                    if non_final_next_states is not None:
                        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]

                expected_state_action_values = reward_batch + args.gamma * next_state_values
                loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 10)
                optimizer.step()

                if total_steps % args.target_update == 0:
                    target_net.load_state_dict(policy_net.state_dict())

        all_rewards.append(ep_reward)

        if ep % args.log_interval == 0 or ep == 1:
            avg_window = min(args.log_interval, len(all_rewards))
            avg_recent = float(np.mean(all_rewards[-avg_window:]))
            current_eps = epsilon_by_step(steps_done, args.eps_start, args.eps_end, args.eps_decay)
            print(
                f"Episode {ep}/{args.episodes} | steps={ep_steps} | reward={ep_reward:.2f} "
                f"| avg{avg_window}={avg_recent:.2f} | epsilon={current_eps:.3f}"
            )

        if ep % args.eval_every == 0:
            eval_avg = evaluate(policy_net, is_vector, args.eval_episodes, args.device)
            print(f"Eval @ episode {ep}: avg_reward={eval_avg:.2f} (epsilon=0)")

        if ep % args.checkpoint_interval == 0:
            checkpoint_path = f"flappy_dqn_ep{ep}.pth"
            torch.save(policy_net.state_dict(), checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

    final_path = "flappy_dqn_final.pth"
    torch.save(policy_net.state_dict(), final_path)
    env.close()
    print(f"Training complete. Model saved to {final_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train a DQN agent on FlappyBird-v0")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--memory-cap", type=int, default=100000)

    parser.add_argument("--eps-start", type=float, default=1.0)
    parser.add_argument("--eps-end", type=float, default=0.01)
    parser.add_argument("--eps-decay", type=int, default=5000)

    parser.add_argument("--target-update", type=int, default=500)
    parser.add_argument("--checkpoint-interval", type=int, default=10)
    parser.add_argument("--log-interval", type=int, default=5)

    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--eval-episodes", type=int, default=5)

    parser.add_argument("--reward-shaping", action="store_true")
    parser.add_argument("--survival-penalty", type=float, default=-0.01)
    parser.add_argument("--crash-penalty-multiplier", type=float, default=2.0)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=DEVICE, choices=["cpu", "cuda"])

    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    set_seed(args.seed)
    train(args)
