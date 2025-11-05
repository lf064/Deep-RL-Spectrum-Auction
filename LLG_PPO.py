from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from cont_env import ContinuousLLGAuctionEnv
from cont_config import get_continuous_llg_config
from norm_wrapper import ActionNormalizationWrapper
import numpy as np
from collections import defaultdict
import wandb


class WandbCallback(BaseCallback):
    """
    Custom callback to log PPO training metrics to WandB
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        
    def _on_step(self) -> bool:
        # Log episode metrics when episode ends
        # Check if any episode finished in this step
        if len(self.locals.get('dones', [])) > 0 and self.locals['dones'][0]:
            # Get info dict which contains episode statistics
            infos = self.locals.get('infos', [])
            if len(infos) > 0 and 'episode' in infos[0]:
                # stable-baselines3 automatically tracks episode stats in Monitor wrapper
                episode_info = infos[0]['episode']
                wandb.log({
                    "episode/reward": episode_info['r'],
                    "episode/length": episode_info['l'],
                    "episode/time": episode_info['t']
                })
        
        return True
    
    def _on_rollout_end(self) -> None:
        """Log training metrics after each rollout"""
        # Access the logger which contains all training metrics
        if len(self.logger.name_to_value) > 0:
            # Log all available training metrics
            metrics_to_log = {}
            
            # Core PPO metrics
            if 'train/value_loss' in self.logger.name_to_value:
                metrics_to_log['train/value_loss'] = self.logger.name_to_value['train/value_loss']
            if 'train/policy_gradient_loss' in self.logger.name_to_value:
                metrics_to_log['train/policy_loss'] = self.logger.name_to_value['train/policy_gradient_loss']
            if 'train/entropy_loss' in self.logger.name_to_value:
                metrics_to_log['train/entropy_loss'] = self.logger.name_to_value['train/entropy_loss']
            if 'train/approx_kl' in self.logger.name_to_value:
                metrics_to_log['train/approx_kl'] = self.logger.name_to_value['train/approx_kl']
            if 'train/clip_fraction' in self.logger.name_to_value:
                metrics_to_log['train/clip_fraction'] = self.logger.name_to_value['train/clip_fraction']
            if 'train/explained_variance' in self.logger.name_to_value:
                metrics_to_log['train/explained_variance'] = self.logger.name_to_value['train/explained_variance']
            if 'train/learning_rate' in self.logger.name_to_value:
                metrics_to_log['train/learning_rate'] = self.logger.name_to_value['train/learning_rate']
            
            # Log to wandb
            if metrics_to_log:
                wandb.log(metrics_to_log)

def train_continuous_llg(timesteps=10000, use_normalization=True, **hyperparams):
    """
    Train PPO on continuous LLG environment with proper WandB tracking
    
    Args:
        timesteps: Total training timesteps
        use_normalization: Whether to use action normalization wrapper
        **hyperparams: PPO hyperparameters
    
    Tracks:
    - Value loss, Policy loss, Entropy
    - Episode rewards
    - Custom clearing rate metrics
    """
    config = get_continuous_llg_config()
    
    # Create continuous environment
    continuous_env = ContinuousLLGAuctionEnv(config)
    
    # Optionally wrap with normalization
    if use_normalization:
        env = ActionNormalizationWrapper(continuous_env)
        env_type = "continuous_normalized"
    else:
        env = continuous_env
        env_type = "continuous_raw"

    # Initialize wandb
    wandb.init(
        project="llg-auction",
        name=f"{env_type}-{timesteps//1000}k-steps",
        config={
            "timesteps": timesteps,
            "env_type": env_type,
            "use_normalization": use_normalization,
            "action_space": str(env.action_space),
            "observation_space": str(env.observation_space),
            "price_range": f"[{config.min_price}, {config.max_price}]",
            **hyperparams
        }
    )
    
    # Default hyperparameters for PPO
    default_params = {
        'learning_rate': 3e-4,
        'n_steps': 2048,
        'batch_size': 64,
        'n_epochs': 10,
        'gamma': 0.99,
        'clip_range': 0.2,
        'ent_coef': 0.01,
        'vf_coef': 0.5,
        'verbose': 1  # Changed to 1 to see training progress
    }
    
    params = {**default_params, **hyperparams}
    
    print("=" * 50)
    print(f"{env_type.upper()} LLG TRAINING")
    print("=" * 50)
    print(f"Training for {timesteps:,} timesteps")
    print(f"Action space: {env.action_space}")
    if use_normalization:
        print(f"Normalized: [-1, 1] → [{config.min_price}, {config.max_price}]")
    print("Progress evaluations:")
    
    # Create model with WandB callback
    model = PPO("MlpPolicy", env, **params)
    callback = WandbCallback()
    
    # Train with periodic evaluation
    eval_points = [0.25, 0.5, 0.75, 1.0]
    
    for i, checkpoint in enumerate(eval_points):
        # Train to this checkpoint
        steps_to_train = int(timesteps * checkpoint) - (int(timesteps * eval_points[i-1]) if i > 0 else 0)
        
        if steps_to_train > 0:
            model.learn(
                total_timesteps=steps_to_train, 
                reset_num_timesteps=False,
                callback=callback
            )
        
        # Evaluation
        cleared_instances = 0
        total_rounds_cleared_only = 0
        allocation_types = defaultdict(int)
        scenario_results = defaultdict(lambda: {'cleared': 0, 'total': 0, 'rounds_cleared': []})
        eval_episodes = 100
        
        for _ in range(eval_episodes):
            obs, info = env.reset()
            
            # Get valuations from unwrapped environment
            valuations = env.unwrapped.bidder_valuations
            global_val = valuations[2]
            scenario = 'global_low' if global_val == 4 else 'global_high'
            
            rounds = 0
            done = False
            
            while not done and rounds < config.max_rounds:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                rounds += 1
                done = terminated or truncated
            
            # Track results
            instance_cleared = info['successful_allocation']
            if instance_cleared:
                cleared_instances += 1
                total_rounds_cleared_only += rounds
                scenario_results[scenario]['rounds_cleared'].append(rounds)
            
            allocation_types[info['allocation_type']] += 1
            scenario_results[scenario]['total'] += 1
            if instance_cleared:
                scenario_results[scenario]['cleared'] += 1
        
        # Calculate metrics
        clearing_rate = cleared_instances / eval_episodes * 100
        avg_rounds_to_clear = total_rounds_cleared_only / cleared_instances if cleared_instances > 0 else 0
        
        print(f"  {checkpoint*100:3.0f}%: Clearing Rate {clearing_rate:5.1f}% ({cleared_instances}/{eval_episodes}) | Avg Rounds to Clear {avg_rounds_to_clear:.2f}")
        
        # Log to wandb
        wandb.log({
            "eval/training_progress": checkpoint,
            "eval/clearing_rate": clearing_rate,
            "eval/cleared_instances": cleared_instances,
            "eval/avg_rounds_to_clear": avg_rounds_to_clear,
            "eval/total_episodes": eval_episodes,
            **{f"eval/allocation_{k}": v/eval_episodes*100 for k, v in allocation_types.items()}
        })
        
        # Detailed breakdown at final checkpoint
        if checkpoint == 1.0:
            print(f"\n    Allocation types:")
            for alloc_type, count in allocation_types.items():
                pct = count / eval_episodes * 100
                print(f"      {alloc_type}: {pct:.1f}%")
            
            print(f"\n    Performance by scenario:")
            scenario_breakdown = {}
            for scenario, results in scenario_results.items():
                if results['total'] > 0:
                    scenario_clearing_rate = results['cleared'] / results['total'] * 100
                    scenario_avg_rounds = np.mean(results['rounds_cleared']) if results['rounds_cleared'] else 0
                    scenario_desc = "Global=$4" if scenario == 'global_low' else "Global=$10"
                    print(f"      {scenario_desc}: {scenario_clearing_rate:.1f}% cleared ({results['cleared']}/{results['total']}) | Avg rounds: {scenario_avg_rounds:.1f}")
                    
                    scenario_breakdown[f"eval/final_{scenario}_clearing_rate"] = scenario_clearing_rate
                    scenario_breakdown[f"eval/final_{scenario}_avg_rounds"] = scenario_avg_rounds
            
            wandb.log(scenario_breakdown)
    
    model_path = f"{env_type}_llg_ppo_model"
    model.save(model_path)
    print(f"\n✅ Training complete! Model saved to {model_path}")
    
    # Log model as artifact
    model_artifact = wandb.Artifact("llg_ppo_model", type="model")
    model_artifact.add_file(f"{model_path}.zip")
    wandb.log_artifact(model_artifact)
    
    wandb.finish()
    return model


if __name__ == "__main__":
    # Train with normalization
    train_continuous_llg(timesteps=2000000, use_normalization=True)