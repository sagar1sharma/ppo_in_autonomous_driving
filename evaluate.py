import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
import highway_env

def evaluate(episodes=10):
    env = gym.make("highway-v0", render_mode="human")
    env.unwrapped.configure({
        "vehicles_density": 0.4,
        "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
    })

    model = PPO.load('ppo_stage_1.zip', device='cpu')

    successes, collisions = 0, 0
    rewards = []

    for _ in range(episodes):
        obs, info = env.reset()
        total_reward = 0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward

            if info.get("crashed", False):
                collisions += 1
                break

        rewards.append(total_reward)
        if not info.get("crashed", False):
            successes += 1

    print({"success_rate": successes / episodes})
    print({"collision_rate": collisions / episodes})
    print({"average_reward": np.mean(rewards)})

if __name__ == "__main__":
    evaluate(episodes=5)
