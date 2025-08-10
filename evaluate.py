import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

def evaluate(model_path, episodes=10):
    env = gym.make("highway-v0", render_mode="human")
    env.unwrapped.configure({
        "vehicles_density": 0.4,
        "other_vehicles_type": "idm",
    })
    env.reset()
    model = PPO.load(model_path)

    success, collisions, rewards = 0, 0, 0
    for x in range(episodes):
        obs = env.reset()
        total_reward = 0
        done = False
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, info = env.step(action)
            print({"action": action, "reward": reward, "info": info})
            total_reward += reward
            if info.get("crashed", False):
                collisions += 1
                break
        rewards.append(total_reward)
        if not info.get("crashed", False):
            success += 1

    print({"success rate": success / episodes})
    print({"collisions_rate": collisions / episodes})
    print({"Average reward": np.mean(rewards)})
