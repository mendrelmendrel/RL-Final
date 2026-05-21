import gymnasium as gym
import flappy_bird_gymnasium
from dql import DQLAgent

NEPS = 600

def main():
    train_env = gym.make("FlappyBird-v0", use_lidar=False)
    play_env  = gym.make("FlappyBird-v0", use_lidar=False, render_mode="human")

    agent = DQLAgent(train_env)

    # Play before training
    print("=== Untrained agent ===")
    agent.env = play_env
    agent.play()

    # Train
    print(f"\n=== Training for {NEPS} episodes ===")
    agent.env = train_env
    agent.train(neps=NEPS)

    # Play after training
    print("\n=== Trained agent ===")
    agent.env = play_env
    agent.play()

    train_env.close()
    play_env.close()

if __name__ == "__main__":
    main()