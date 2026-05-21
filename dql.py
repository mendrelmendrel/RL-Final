import gymnasium as gym
import math
import random
import numpy as np
from collections import namedtuple, deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


class DQLAgent:
    def __init__(self, env):
        self.env = env
        n_actions = env.action_space.n
        state, _ = env.reset()
        n_observations = len(state)

        self.policy_net = DQN(n_observations, n_actions)
        self.target_net = DQN(n_observations, n_actions)
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def _select_action(self, state, epsilon):
        if random.random() > epsilon:
            with torch.no_grad():
                return self.policy_net(state).max(1).indices.view(1, 1)
        else:
            return torch.tensor([[self.env.action_space.sample()]], dtype=torch.long)

    def _optimize(self, memory, optimizer, batch_size, gamma):
        if len(memory) < batch_size:
            return

        transitions = memory.sample(batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(
            tuple(s is not None for s in batch.next_state), dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        state_batch  = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(batch_size)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values

        expected_state_action_values = reward_batch + gamma * next_state_values

        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        optimizer.step()

    def train(self, neps=600, batch_size=128, gamma=0.99,
              eps_start=0.9, eps_end=0.05, eps_decay=1000,
              tau=0.005, lr=1e-4, memory_size=10000, report_every=10):
        memory     = ReplayMemory(memory_size)
        optimizer  = optim.Adam(self.policy_net.parameters(), lr=lr, amsgrad=True)
        steps_done = 0

        for episode in range(neps):
            state, _ = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            total_reward = 0.0

            while True:
                epsilon = eps_end + (eps_start - eps_end) * math.exp(-steps_done / eps_decay)
                action = self._select_action(state, epsilon)
                steps_done += 1

                obs, reward, terminated, truncated, _ = self.env.step(action.item())
                total_reward += reward
                done = terminated or truncated

                next_state = None if terminated else torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                memory.push(state, action, next_state, torch.tensor([reward], dtype=torch.float32))
                state = next_state

                self._optimize(memory, optimizer, batch_size, gamma)

                target_sd = self.target_net.state_dict()
                policy_sd = self.policy_net.state_dict()
                for key in policy_sd:
                    target_sd[key] = tau * policy_sd[key] + (1 - tau) * target_sd[key]
                self.target_net.load_state_dict(target_sd)

                if done:
                    break

            if episode % report_every == 0:
                print(f"Episode {episode:03d} / {neps}: total reward = {total_reward:.1f}")

    def play(self, max_iters=2000):

        state, _ = self.env.reset()
        score = 0

        for _ in range(max_iters):
            self.env.render()
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                action = self.policy_net(state_tensor).max(1).indices.item()

            state, reward, terminated, truncated, _ = self.env.step(action)

            if reward >= 1:
                score += int(reward)
                print(f"Pipe passed! Score: {score}")

            if terminated or truncated:
                break

        print(f"\n--- Game Over | Final Score: {score} pipes passed ---")
        return score