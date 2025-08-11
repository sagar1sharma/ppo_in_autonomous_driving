for seed in [0, 42, 100]:
    env = gym.make("highway-v0")
    env.configure({...})
    env.reset()

    model = PPO("MlpPolicy", env, verbose=1, seed=seed, tensorboard_log="./ppo_logs/")
    model.learn(total_timesteps=200_000)
    model.save(f"ppo_baseline_seed_{seed}")
