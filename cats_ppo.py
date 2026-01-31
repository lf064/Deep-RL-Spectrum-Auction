from stable_baselines3 import PPO
from cont_env import CATSAuctionEnv
from cont_config import get_cats_config
from norm_wrapper import CATSActionNormalizationWrapper
from test_script import CATSEvalCallback  # CHANGED: Import callback instead of wrapper
import wandb


def train_cats_ppo(
    filepath: str = '0000.txt',
    num_bidders: int = 5,
    timesteps: int = 1000000,
    use_normalization: bool = True,
    seed: int = 42,
    resample_bidders: bool = True,
    seed_range: tuple = (0, 1000),
    clearable_seeds_file: str = None,  # NEW
    eval_freq: int = 50000,  # ADDED: Evaluation frequency in steps
    **hyperparams
):
    """
    Simple PPO training executor
    
    Args:
        filepath: Path to CATS .txt file
        num_bidders: Number of bidders
        timesteps: Total training timesteps
        use_normalization: Use action normalization
        seed: Random seed for reproducibility
        eval_freq: Evaluate every N timesteps
        **hyperparams: PPO hyperparameters override
    """
    # Setup - CATSParser handles its own seeding via seed parameter
    
    config = get_cats_config(filepath, num_bidders=num_bidders, seed=seed)
    env = CATSAuctionEnv(
        config,
        resample_bidders=True,           # NEW: Enable resampling
        cats_filepath=filepath,          # NEW: Path for parser
        seed_range=(0, 1000)   , 
        clearable_seeds_file=clearable_seeds_file  # NEW
         # NEW: Seed range
    )
    
    if use_normalization:
        env = CATSActionNormalizationWrapper(env)
        env_type = "normalized"
    else:
        env_type = "raw"
    
   
    # Evaluation  handled by CATSEvalCallback 
    
    # WandB setup
    wandb.init(
        project="cats-auction",
        name=f"{env_type}-{num_bidders}b-{config.num_items}i-{timesteps//1000}k-s{seed}-resample{resample_bidders}",
        config={
            "filepath": filepath,
            "num_bidders": num_bidders,
            "num_items": config.num_items,
            "timesteps": timesteps,
            "env_type": env_type,
            "seed": seed,
            "eval_freq": eval_freq,  # ADDED: Log eval frequency
            "price_range": f"${config.min_price:.2f}-${config.max_price:.2f}",
            **hyperparams
        }
    )
    
    # PPO hyperparameters - seed parameter handles all internal seeding
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
        'seed': seed,  # Seeds PPO's internal randomness (numpy, torch, action sampling)
        **hyperparams
    }
    
    # Info
    print("=" * 70)
    print(f"CATS AUCTION PPO TRAINING")
    print("=" * 70)
    print(f"Config: {filepath} | {num_bidders} bidders, {config.num_items} items")
    print(f"Price range: ${config.min_price:.2f} - ${config.max_price:.2f}")
    print(f"Training: {timesteps:,} timesteps | Seed: {seed}")
    print(f"Eval freq: Every {eval_freq:,} steps")  # ADDED: Show eval frequency
    print("=" * 70)
    
    # Create model
    model = PPO("MlpPolicy", env, **params)
    
    # ADDED: Create evaluation callback
    # This replaces the non-functional CATSTrainingWrapper
    eval_callback = CATSEvalCallback(
        eval_env=env,
        config=config,
        use_normalization=use_normalization,
        eval_freq=eval_freq,
        n_eval_episodes=100,
        verbose=1
    )
    
    # CHANGED: Train with callback for evaluation
    # Before: model.learn(total_timesteps=timesteps)
    # After: model.learn(total_timesteps=timesteps, callback=eval_callback)
    model.learn(
        total_timesteps=timesteps,
        callback=eval_callback  # Callback handles evaluation during training
    )
    
    # Save
    model_path = f"cats_ppo_{num_bidders}b_{config.num_items}i_s{seed}-resample{resample_bidders}"
    model.save(model_path)
    
    # Artifact
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
        {'num_bidders': 7, 'timesteps': 4000000, 'clearable_seeds_file': 'clearable_seeds_7b_subgrad.npy'},

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
            resample_bidders=True,      
            seed_range=(0, 1000),       
            clearable_seeds_file=exp.get('clearable_seeds_file'),  # Optional
            seed=86,
            eval_freq=50000
        )
        
        print(f"\n{'='*70}")
        print(f"COMPLETED: {exp['num_bidders']} BIDDERS")
        print(f"{'='*70}\n")