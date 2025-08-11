import gymnasium as gym
import highway_env
from stable_baselines3 import PPO
from curriculum_utils import get_curriculum_stage
import numpy as np
import itertools
from stable_baselines3.common.vec_env import SubprocVecEnv

# Logging metrics
def log_metrics(stage, rewards, collisions, successes):
    print(f"Stage {stage} Metrics:")
    print(f"  Average Reward: {np.mean(rewards):.2f}")
    print(f"  Collision Rate: {np.mean(collisions):.2f}")
    print(f"  Success Rate: {np.mean(successes):.2f}")

def train_curriculum(total_stages=5, timesteps_per_stage=25_000):
    model = None

    for stage in range(total_stages):
        print(f"\n=== Training Stage {stage} ===")

        config = get_curriculum_stage(stage)

        # Use parallel environments
        env = SubprocVecEnv([lambda: gym.make("highway-v0", render_mode=None) for _ in range(4)])
        env.env_method("configure", config)
        env.reset()

        if model is None:
            model = PPO("MlpPolicy", env, device="cpu", verbose=1, learning_rate=1e-4,  # Optimized hyperparameters
                        n_steps=4096, batch_size=128, gamma=0.99)
        else:
            model.set_env(env)

        # Track metrics during training
        rewards, collisions, successes = [], [], []

        def custom_callback(_locals, _globals):
            info = _locals['infos']
            for i in info:
                rewards.append(i.get('reward', 0))
                collisions.append(i.get('collision', 0))
                successes.append(i.get('success', 0))
            return True

        model.learn(total_timesteps=timesteps_per_stage, callback=custom_callback)

        # Log metrics for the stage
        log_metrics(stage, rewards, collisions, successes)

        # Save model checkpoint for this stage
        model.save(f"ppo_stage_{stage}")

    print("\nCurriculum training completed!")

def hyperparameter_tuning():
    learning_rates = [3e-4, 1e-4]
    n_steps_list = [1024, 2048]
    batch_sizes = [64, 128]

    best_reward = -float('inf')
    best_params = None

    for lr, n_steps, batch_size in itertools.product(learning_rates, n_steps_list, batch_sizes):
        print(f"\nTesting with learning_rate={lr}, n_steps={n_steps}, batch_size={batch_size}")

        model = None
        for stage in range(5):
            config = get_curriculum_stage(stage)

            env = gym.make("highway-v0")
            env.unwrapped.configure(config)
            env.reset()

            if model is None:
                model = PPO("MlpPolicy", env, device="cpu", verbose=0, learning_rate=lr,
                            n_steps=n_steps, batch_size=batch_size, gamma=0.99)
            else:
                model.set_env(env)

            model.learn(total_timesteps=50_000)

        # Evaluate the model
        total_reward = 0
        for _ in range(10):
            obs = env.reset()
            done = False
            while not done:
                action, _ = model.predict(obs)
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                done = terminated or truncated

        avg_reward = total_reward / 10
        print(f"Average Reward: {avg_reward}")

        if avg_reward > best_reward:
            best_reward = avg_reward
            best_params = (lr, n_steps, batch_size)

    print(f"\nBest Parameters: learning_rate={best_params[0]}, n_steps={best_params[1]}, batch_size={best_params[2]}")
    print(f"Best Average Reward: {best_reward}")

if __name__ == "__main__":
    train_curriculum()
    hyperparameter_tuning()
