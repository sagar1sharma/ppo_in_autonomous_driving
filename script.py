import gymnasium as gym
import highway_env

env = gym.make("highway-v0",render_mode="rgb_array", config={
    "simulation_frequency": 15,
    "duration": 40})
obs = env.reset()
env.render()
