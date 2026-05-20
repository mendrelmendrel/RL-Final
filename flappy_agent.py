"""
Simple DQN trainer for FlappyBird (gymnasium FlappyBird-v0).
- Detects whether the env observation is low-dim or pixel frames.
- Uses a small MLP for vector observations, or a tiny CNN for image observations.

Usage:
    pip install -r requirements.txt
    python flappy_agent.py

Defaults: 300 episodes (change NUM_EPISODES)
"""

import math, random, time
from collections import namedtuple, deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym

# ensure the flappy package is available
try:
    import flappy_bird_gymnasium  # registers env
except Exception:
    """
    Upgraded FlappyBird trainer using Nature DQN preprocessing.

    Features:
    - RGB -> grayscale, resize to 84x84, stack 4 frames
    - Nature DQN convolutional network
    - Replay buffer, epsilon-greedy, target network, periodic checkpoints

    Run:
        python flappy_agent.py

    Adjust `NUM_EPISODES` and `CHECKPOINT_INTERVAL` below.
    """

    import math, random, time
    from collections import deque, namedtuple
    import numpy as np
    from PIL import Image
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    import gymnasium as gym

    try:
        import flappy_bird_gymnasium
    except Exception:
        pass

    Transition = namedtuple('Transition', ('state','action','next_state','reward'))

    class ReplayMemory:
        def __init__(self, capacity):
            self.memory = deque(maxlen=capacity)
        def push(self, *args):
            self.memory.append(Transition(*args))
        def sample(self, batch_size):
            return random.sample(self.memory, batch_size)
        def __len__(self):
            return len(self.memory)


    def preprocess_frame(frame):
        # frame: H,W,3 uint8 RGB
        img = Image.fromarray(frame)
        img = img.convert('L').resize((84,84))
        arr = np.array(img, dtype=np.uint8)
        return arr


    class NatureDQN(nn.Module):
        def __init__(self, in_channels, n_actions):
            super().__init__()
            self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
            self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
            # compute linear input dim
            with torch.no_grad():
                t = torch.zeros(1, in_channels, 84, 84)
                t = F.relu(self.conv1(t))
                t = F.relu(self.conv2(t))
                t = F.relu(self.conv3(t))
                linear_dim = int(np.prod(t.size()))
            self.fc1 = nn.Linear(linear_dim, 512)
            self.head = nn.Linear(512, n_actions)
        def forward(self, x):
            x = x / 255.0
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc1(x))
            return self.head(x)


    class MLP(nn.Module):
        def __init__(self, input_dim, n_actions):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, 128)
            self.fc2 = nn.Linear(128, 128)
            self.head = nn.Linear(128, n_actions)
        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            return self.head(x)


    # Hyperparameters
    NUM_EPISODES = 500
    BATCH_SIZE = 32
    GAMMA = 0.99
    LR = 1e-4
    MEMORY_CAP = 50000
    EPS_START = 1.0
    EPS_END = 0.01
    EPS_DECAY = 20000
    TARGET_UPDATE = 1000  # steps
    CHECKPOINT_INTERVAL = 50  # episodes
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


    def stack_frames(frames):
        # frames: list of 4 (84,84) uint8 -> returns (4,84,84) float32 tensor
        arr = np.stack(frames, axis=0)
        return torch.tensor(arr, dtype=torch.float32, device=DEVICE).unsqueeze(0)


    def train():
        env = gym.make('FlappyBird-v0')
        obs, info = env.reset()
        n_actions = env.action_space.n

        # detect observation type: vector (1D) or image (H,W,3)
        is_vector = isinstance(obs, (list, tuple, np.ndarray)) and np.asarray(obs).ndim == 1

        # build networks
        if is_vector:
            obs_size = int(np.prod(np.asarray(obs).shape))
            policy_net = MLP(obs_size, n_actions).to(DEVICE)
            target_net = MLP(obs_size, n_actions).to(DEVICE)
        else:
            policy_net = NatureDQN(4, n_actions).to(DEVICE)
            target_net = NatureDQN(4, n_actions).to(DEVICE)
        target_net.load_state_dict(policy_net.state_dict())
        optimizer = optim.AdamW(policy_net.parameters(), lr=LR)
        memory = ReplayMemory(MEMORY_CAP)

        steps_done = 0
        total_steps = 0

        # main training loop
        all_rewards = []
        for ep in range(1, NUM_EPISODES + 1):
            obs, info = env.reset()
            if is_vector:
                state = torch.tensor(np.asarray(obs, dtype=np.float32), device=DEVICE).unsqueeze(0)
            else:
                # preprocess initial frame and init frame stack
                frame = preprocess_frame(obs)
                frames = deque([frame.copy() for _ in range(4)], maxlen=4)
                state = stack_frames(list(frames))
            ep_reward = 0.0
            done = False

            while not done:
                # epsilon
                eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
                steps_done += 1
                total_steps += 1
                if random.random() > eps_threshold:
                    with torch.no_grad():
                        q = policy_net(state)
                        action = int(q.argmax(dim=1).item())
                else:
                    action = int(env.action_space.sample())

                obs_next, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                ep_reward += reward

                if is_vector:
                    next_state = torch.tensor(np.asarray(obs_next, dtype=np.float32), device=DEVICE).unsqueeze(0) if not done else None
                else:
                    next_frame = preprocess_frame(obs_next)
                    frames.append(next_frame)
                    next_state = stack_frames(list(frames)) if not done else None

                memory.push(state, torch.tensor([[action]], device=DEVICE, dtype=torch.long), next_state, torch.tensor([reward], device=DEVICE))

                state = next_state

                # optimize
                if len(memory) >= BATCH_SIZE:
                    transitions = memory.sample(BATCH_SIZE)
                    batch = Transition(*zip(*transitions))
                    non_final_mask = torch.tensor([s is not None for s in batch.next_state], device=DEVICE, dtype=torch.bool)
                    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None]) if any(non_final_mask) else torch.empty((0,), device=DEVICE)
                    state_batch = torch.cat(batch.state)
                    action_batch = torch.cat(batch.action)
                    reward_batch = torch.cat(batch.reward).to(DEVICE)

                    state_action_values = policy_net(state_batch).gather(1, action_batch)
                    next_state_values = torch.zeros(BATCH_SIZE, device=DEVICE)
                    with torch.no_grad():
                        if any(non_final_mask):
                            next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
                    expected = (next_state_values * GAMMA) + reward_batch
                    loss = F.smooth_l1_loss(state_action_values, expected.unsqueeze(1))
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 10)
                    optimizer.step()

                    # periodic target update by steps
                    if total_steps % TARGET_UPDATE == 0:
                        target_net.load_state_dict(policy_net.state_dict())

            all_rewards.append(ep_reward)
            if ep % CHECKPOINT_INTERVAL == 0:
                ckpt = f'flappy_dqn_ep{ep}.pth'
                torch.save(policy_net.state_dict(), ckpt)
                print(f'Episode {ep}/{NUM_EPISODES} | recent avg {np.mean(all_rewards[-50:]):.2f} | saved {ckpt}')
            elif ep % 10 == 0:
                print(f'Episode {ep}/{NUM_EPISODES} | recent avg {np.mean(all_rewards[-10:]):.2f}')

        torch.save(policy_net.state_dict(), 'flappy_dqn_final.pth')
        env.close()
        print('Training complete. Model saved to flappy_dqn_final.pth')


    if __name__ == '__main__':
        train()
