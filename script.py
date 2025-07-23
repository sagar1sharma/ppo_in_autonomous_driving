import gymnasium as gym
import highway_env

env = gym.make("highway-v0", render_mode="rgb_array")
obs = env.reset()
env.render()
