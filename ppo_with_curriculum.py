import gymnasium as gym
import highway_env
from stable_baselines3 import PPO
from curriculum_utils import get_curriculum_stage

def train_curriculum(total_stages=5, timesteps_per_stage=10_000):
    model = None

    for stage in range(total_stages):
        print(f"\n=== Training Stage {stage} ===")

        config = get_curriculum_stage(stage)

        env = gym.make("highway-v0")
        env.unwrapped.configure(config)
        env.reset()

        if model is None:
            model = PPO("MlpPolicy", env,device="cpu", verbose=1,learning_rate=3e-4,
            n_steps=1024,
            batch_size=32,
            gamma=0.99)
        else:
            model.set_env(env)

        model.learn(total_timesteps=timesteps_per_stage)

        # Save model checkpoint for this stage
        model.save(f"ppo_stage_{stage}")

    print("\nCurriculum training completed!")

if __name__ == "__main__":
    train_curriculum()