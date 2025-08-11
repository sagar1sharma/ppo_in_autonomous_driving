import gymnasium as gym
import highway_env
from stable_baselines3 import PPO
from curriculum_utils import get_curriculum_stage
import numpy as np
from stable_baselines3.common.vec_env import SubprocVecEnv

# Logging metrics
def log_metrics(stage, rewards, collisions, successes):
    print(f"Stage {stage} Metrics:")
    print(f"  Average Reward: {np.mean(rewards):.2f}")
    print(f"  Collision Rate: {np.mean(collisions):.2f}")
    print(f"  Success Rate: {np.mean(successes):.2f}")

def train_curriculum(total_stages=5, timesteps_per_stage=10_000):
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


if __name__ == "__main__":
    train_curriculum()
