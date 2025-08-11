import gymnasium as gym
import highway_env
from stable_baselines3 import PPO


def train_baseline(total_timesteps=250_000):
    env = gym.make("highway-v0")
    env.unwrapped.configure({
        "vehicles_density": 0.3,  # Fixed density for baseline
        "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
        "speed_limit": [10, 20],
        "controlled_vehicle": {
            "type": "highway_env.vehicle.behavior.IDMVehicle",
            "initial_speed": 10
        }
    })
    env.reset()

    model = PPO("MlpPolicy", env, device="cpu", verbose=1, learning_rate=3e-4,
                n_steps=2048, batch_size=64, gamma=0.99)

    model.learn(total_timesteps=total_timesteps)

    model.save("ppo_baseline")
    print("Baseline training completed!")


if __name__ == "__main__":
    train_baseline()


