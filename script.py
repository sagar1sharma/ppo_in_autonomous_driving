import gymnasium as gym
import highway_env

env = gym.make("highway-v0")
env.configure({
    "render_mode":"rgb_array",
    "simulation_frequency":15,
    "duration":40})
obs = env.reset()
env.render()
