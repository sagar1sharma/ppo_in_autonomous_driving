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


model = PPO("MlpPolicy", env, verbose=1)


model.learn(total_timesteps=200_000)


model.save("ppo_highway_baseline")

obs = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs)
    obs, reward, done, _ = env.step(action)
    env.render()
    if done:
        obs = env.reset()

