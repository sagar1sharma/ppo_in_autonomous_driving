import gymnasium as gym
import highway_env
from stable_baselines3 import PPO


env = gym.make(
    "highway-v0", 
    render_mode="human", 
    config={
            "observation": {
            "type": "Kinematics",
            "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
            "features_range": {
                "x": [-100, 100],
                "y": [-100, 100],
                "vx": [-10, 10],  # Reduced velocity range
                "vy": [-10, 10]   # Reduced velocity range
            },
            "grid_size": [[-27.5, 27.5], [-27.5, 27.5]],
            "grid_step": [5, 5],
            "absolute": False
        }
})
env.reset()


model = PPO("MlpPolicy", env, verbose=1, learning_rate=3e-4,
    n_steps=4096,
    batch_size=64,
    n_epochs=5,
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


