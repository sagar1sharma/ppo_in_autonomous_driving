import gymnasium as gym
import highway_env
from stable_baselines3 import PPO


from stable_baselines3.common.vec_env import DummyVecEnv

env = DummyVecEnv([lambda: gym.make("highway-v0", render_mode=None, config={
    "observation": {
        "type": "Kinematics",
        "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
        "features_range": {
            "x": [-100, 100],
            "y": [-100, 100],
            "vx": [-10, 10],
            "vy": [-10, 10]
        },
        "grid_size": [[-27.5, 27.5], [-27.5, 27.5]],
        "grid_step": [5, 5],
        "absolute": False,
    }
})])
env.reset()


model = PPO("MlpPolicy", env, verbose=1,  device="cpu",learning_rate=3e-4,
    n_steps=1024,
    batch_size=32,
    n_epochs=3,
    gamma=0.99)


model.learn(total_timesteps=100_000)


model.save("ppo_highway_baseline")

obs = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    env.render()
    if done:
        obs, info = env.reset()


