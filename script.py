import gymnasium as gym
import matplotlib.pyplot as plt
import highway_env

env = gym.make(
    'highway-v0',
    render_mode="rgb_array",
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
    }
)
obs = env.reset()
env.render()

for _ in range(100):
    obs, reward, done, truncated, info = env.step(env.action_space.sample())
