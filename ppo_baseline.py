import gymnasium as gym
import highway_env
from stable_baselines3 import PPO


env = gym.make(
    "highway-v0", 
    render_mode="human", 
    config={
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 15,
            "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
            "features_range": {
                "x": [-100, 100],
                "y": [-100, 100],
                "vx": [-20, 20],
                "vy": [-20, 20]
            },
            "grid_size": [[-27.5, 27.5], [-27.5, 27.5]],
            "grid_step": [5, 5],
            "absolute": False,
            "vehicles_count": 15
        }
})
env.reset()


model = PPO("MlpPolicy", env, verbose=1, learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99)


model.learn(total_timesteps=200_000)


model.save("ppo_highway_baseline")

obs = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    env.render()
    if done:
        obs = env.reset()

