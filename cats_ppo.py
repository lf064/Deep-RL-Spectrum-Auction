from stable_baselines3 import PPO
from cont_env import CATSAuctionEnv
from cont_config import get_cats_config
from norm_wrapper import CATSActionNormalizationWrapper
from test_script import CATSEvalCallback
import wandb


def train_cats_ppo(
    filepath: str = '0000.txt',
    num_bidders: int = 10,
    timesteps: int = 5000000,
    use_normalization: bool = True,
    seed: int = 42,
    seed_range: tuple = (0, 10000),
    eval_freq: int = 50000,
    **hyperparams
):
    """
    PPO training executor for CATS combinatorial auctions.
    
    Clearable-seed filtering happens at runtime: each env.reset() samples a
    random seed, runs the subgradient check, and keeps trying until it finds
    a clearable instance. Nothing to configure here.
    
    Args:
        filepath: Path to CATS .txt file
        num_bidders: Number of bidders
        timesteps: Total training timesteps
        use_normalization: Use action normalization wrapper
        seed: Random seed for reproducibility
        seed_range: Range of seeds the env samples from each reset
        eval_freq: Evaluate every N timesteps
        **hyperparams: PPO hyperparameter overrides
    """
    # Initial config (used for logging and action space setup)
    config = get_cats_config(filepath, num_bidders=num_bidders, seed=seed)

    # Environment — resampling + clearable filtering handled internally
    env = CATSAuctionEnv(
        config,
        resample_bidders=True,
        cats_filepath=filepath,
        seed_range=seed_range
    )
    
    if use_normalization:
        env = CATSActionNormalizationWrapper(env)
        env_type = "normalized"
    else:
        env_type = "raw"
    
    # WandB setup
    wandb.init(
        project="cats-auction",
        name=f"{env_type}-{num_bidders}b-{config.num_items}i-{timesteps//1000}k-s{seed}",
        config={
            "filepath": filepath,
            "num_bidders": num_bidders,
            "num_items": config.num_items,
            "timesteps": timesteps,
            "env_type": env_type,
            "seed": seed,
            "seed_range": seed_range,
            "eval_freq": eval_freq,
            "clearable_seeds": "runtime_subgradient_check",
            "price_range": f"${config.min_price:.2f}-${config.max_price:.2f}",
            **hyperparams
        }
    )
    
    # PPO hyperparameters
    params = {
        'learning_rate': 3e-4,
        'n_steps': 2048,
        'batch_size': 64,
        'n_epochs': 10,
        'gamma': 0.99,
        'clip_range': 0.2,
        'ent_coef': 0.01,
        'vf_coef': 0.5,
        'verbose': 1,
        'seed': seed,
        **hyperparams
    }
    
    # Info
    print("=" * 70)
    print(f"CATS AUCTION PPO TRAINING")
    print("=" * 70)
    print(f"Config: {filepath} | {num_bidders} bidders, {config.num_items} items")
    print(f"Price range: ${config.min_price:.2f} - ${config.max_price:.2f}")
    print(f"Training: {timesteps:,} timesteps | Seed: {seed}")
    print(f"Seed range: {seed_range} | Eval freq: every {eval_freq:,} steps")
    print(f"Clearable filtering: runtime subgradient check each reset")
    print("=" * 70)
    
    # Model
    model = PPO("MlpPolicy", env, **params)
    
    # Evaluation callback
    eval_callback = CATSEvalCallback(
        eval_env=env,
        config=config,
        use_normalization=use_normalization,
        eval_freq=eval_freq,
        n_eval_episodes=100,
        verbose=1
    )
    
    # Train
    model.learn(
        total_timesteps=timesteps,
        callback=eval_callback
    )
    
    # Save model
    model_path = f"cats_ppo_{num_bidders}b_{config.num_items}i_s{seed}"
    model.save(model_path)
    
    # Log artifact
    artifact = wandb.Artifact(
        f"model_{num_bidders}b_{config.num_items}i_s{seed}", 
        type="model"
    )
    artifact.add_file(f"{model_path}.zip")
    wandb.log_artifact(artifact)
    
    wandb.finish()
    print(f"\n✅ Complete! Model: {model_path}")
    
    return model, config


if __name__ == "__main__":
    experiments = [
        {'num_bidders': 10, 'timesteps': 5000000},
    ]
    
    for exp in experiments:
        print(f"\n{'='*70}")
        print(f"STARTING EXPERIMENT: {exp['num_bidders']} BIDDERS")
        print(f"{'='*70}\n")
        
        train_cats_ppo(
            filepath='0000.txt',
            num_bidders=exp['num_bidders'],
            timesteps=exp['timesteps'],
            use_normalization=True,
            seed_range=(0, 10000),
            seed=86,
            eval_freq=50000
        )
        
        print(f"\n{'='*70}")
        print(f"COMPLETED: {exp['num_bidders']} BIDDERS")
        print(f"{'='*70}\n")