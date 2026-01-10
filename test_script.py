import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from cont_env import CATSAuctionEnv
from norm_wrapper import CATSActionNormalizationWrapper
import wandb


class CATSEvalCallback(BaseCallback):
    """Callback for periodic evaluation and metrics logging"""
    
    def __init__(self, eval_env, config, use_normalization=True,
                 eval_freq=50000, n_eval_episodes=100, verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.config = config
        self.use_normalization = use_normalization
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        
        self.evaluations_timesteps = []
        self.evaluations_results = []
        
        self.recent_successes = []
        self.recent_lengths = []
        
        # Track when we last evaluated
        self.last_eval_step = 0
        
    def _on_step(self) -> bool:
        # Log training episode metrics when episode ends
        if self.locals.get('dones', [False])[0]:
            infos = self.locals.get('infos', [])
            if infos:
                info = infos[0]
                
                # Episode outcome
                cleared = info.get('successful_allocation', False)
                self.recent_successes.append(1 if cleared else 0)
                
                # Episode metrics
                if 'episode' in info:
                    length = info['episode'].get('l', 0)
                    reward = info['episode'].get('r', 0)
                    self.recent_lengths.append(length)
                    
                    wandb.log({
                        'train/episode_reward': reward,
                        'train/episode_length': length,
                        'train/episode_cleared': 1 if cleared else 0,
                    }, step=self.num_timesteps)
                
                # Rolling averages (last 100 episodes)
                if len(self.recent_successes) >= 100:
                    wandb.log({
                        'train/rolling_clearing_rate': np.mean(self.recent_successes[-100:]) * 100,
                        'train/rolling_avg_length': np.mean(self.recent_lengths[-100:]),
                    }, step=self.num_timesteps)
        
        # Periodic evaluation - FIXED trigger
        if self.num_timesteps - self.last_eval_step >= self.eval_freq:
            self._evaluate()
            self.last_eval_step = self.num_timesteps
        
        return True
    
    def _evaluate(self):
        """Run evaluation episodes and log metrics"""
        print(f"\n{'='*70}")
        print(f"Evaluation @ {self.num_timesteps:,} steps")
        print(f"{'='*70}")
        
        # Create fresh eval environment
        eval_env = CATSAuctionEnv(self.config)
        if self.use_normalization:
            eval_env = CATSActionNormalizationWrapper(eval_env)
        
        # Metrics
        cleared = 0
        total_rounds_cleared = 0
        total_revenue = 0.0
        total_value = 0.0
        total_efficiency = 0.0
        all_rounds = []
        sample_data = None
        
        # Run episodes
        for ep in range(self.n_eval_episodes):
            obs, info = eval_env.reset()
            done = False
            rounds = 0
            episode_history = []
            
            while not done and rounds < self.config.max_rounds:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = eval_env.step(action)
                rounds += 1
                done = terminated or truncated
                
                if ep == 0 and rounds <= 15:
                    episode_history.append({
                        'round': rounds,
                        'prices': info['prices'][:3].copy(),
                        'demands': info['demands'].copy(),
                        'clearing': info['market_clearing'],
                        'reward': reward
                    })
            
            all_rounds.append(rounds)
            if info.get('successful_allocation', False):
                cleared += 1
                total_rounds_cleared += rounds
                total_revenue += info.get('revenue', 0.0)
                total_value += info.get('total_value', 0.0)
                total_efficiency += info.get('efficiency', 0.0)
            
            if ep == 0:
                sample_data = {
                    'history': episode_history,
                    'cleared': info.get('successful_allocation', False),
                    'total_rounds': rounds
                }
        
        # Calculate metrics
        clearing_rate = cleared / self.n_eval_episodes * 100
        avg_rounds_cleared = total_rounds_cleared / cleared if cleared > 0 else 0
        avg_revenue = total_revenue / cleared if cleared > 0 else 0
        avg_efficiency = total_efficiency / cleared if cleared > 0 else 0  # FIXED: was 'clead'
        progress = self.num_timesteps / self.model._total_timesteps if hasattr(self.model, '_total_timesteps') else 0
        
        # Print
        print(f"Clearing: {clearing_rate:.1f}% ({cleared}/{self.n_eval_episodes})")
        print(f"Rounds (cleared): {avg_rounds_cleared:.2f}")
        print(f"Revenue: ${avg_revenue:.2f}")
        print(f"Efficiency: {avg_efficiency:.3f}")
        
        # Show sample episode at end
        if progress >= 0.95 and sample_data:
            print(f"\nSample Episode:")
            print(f"{'Rnd':<4} {'Prices':<20} {'Demands':<15} {'Clear':<6} {'Reward':<7}")
            print("-" * 55)
            
            for step in sample_data['history']:
                prices = step['prices']
                price_str = f"[{prices[0]:.1f}, {prices[1]:.1f}, {prices[2]:.1f}]"
                demands_str = str(step['demands'])
                clear_str = '✓' if step['clearing'] else '✗'
                reward_str = f"{step['reward']:+.1f}"
                
                print(f"{step['round']:<4} {price_str:<20} {demands_str:<15} "
                      f"{clear_str:<6} {reward_str:<7}")
            
            if sample_data['total_rounds'] > 15:
                print(f"... +{sample_data['total_rounds'] - 15} more rounds")
            
            print(f"Result: {'CLEARED ✓' if sample_data['cleared'] else 'FAILED ✗'}\n")
        
        # Log to WandB (ONLY ONCE)
        wandb.log({
            'eval/clearing_rate': clearing_rate,
            'eval/cleared_instances': cleared,
            'eval/avg_rounds_cleared': avg_rounds_cleared,
            'eval/avg_revenue': avg_revenue,
            'eval/avg_efficiency': avg_efficiency,
        }, step=self.num_timesteps)
        
        print(f"✓ Logged to WandB at step {self.num_timesteps}")
        
        return True
    
    def _on_rollout_end(self) -> None:
        """Log training metrics after each rollout"""
        if len(self.logger.name_to_value) > 0:
            metrics = {}
            for key in ['train/value_loss', 'train/policy_gradient_loss',
                       'train/entropy_loss', 'train/approx_kl',
                       'train/clip_fraction', 'train/explained_variance']:
                if key in self.logger.name_to_value:
                    log_key = key.replace('policy_gradient_loss', 'policy_loss')
                    metrics[log_key] = self.logger.name_to_value[key]
            
            if metrics:
                wandb.log(metrics, step=self.num_timesteps)