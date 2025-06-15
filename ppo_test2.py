from stable_baselines3 import PPO
from SimpleSettingEnv import MultiRoundAuctionEnv
from config import get_config
import numpy as np
from collections import defaultdict

def train_baseline(config_name='multi_round', timesteps=100000, **hyperparams):
    """Train PPO baseline with periodic evaluation"""
    config = get_config(config_name)
    env = MultiRoundAuctionEnv(config)
    
    # Default hyperparameters
    default_params = {
        'learning_rate': 3e-4,
        'n_steps': 2048,
        'batch_size': 64,
        'n_epochs': 10,
        'gamma': 0.99,
        'clip_range': 0.2,
        'ent_coef': 0.01,
        'vf_coef': 0.5,
        'verbose': 0
    }
    
    # Update with any provided hyperparameters
    params = {**default_params, **hyperparams}
    
    print(f"Training {config_name} with MlpPolicy for {timesteps:,} timesteps...")
    print("Progress evaluations:")
    
    model = PPO("MlpPolicy", env, **params)
    
    # Train with periodic evaluation
    eval_points = [0.25, 0.5, 0.75, 1.0]  # 25%, 50%, 75%, 100%
    
    for i, checkpoint in enumerate(eval_points):
        # Train to this checkpoint
        steps_to_train = int(timesteps * checkpoint) - (int(timesteps * eval_points[i-1]) if i > 0 else 0) # steps_to_train = 50000 - int(100000 * 0.25) when i = 1 and checkpoint = 0.25
        
        if steps_to_train > 0:
            model.learn(total_timesteps=steps_to_train, reset_num_timesteps=False) # reset_num_timesteps equals false 
        
        # Quick evaluation
        success_count = 0
        total_rounds = 0
        eval_episodes = 100
        
        for _ in range(eval_episodes):
            obs, info = env.reset()
            rounds = 0
            done = False
            
            while not done and rounds < config.max_rounds:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                rounds += 1
                done = terminated or truncated
            
            if info['successful_allocation']:
                success_count += 1
            total_rounds += rounds
        
        success_rate = success_count / eval_episodes * 100
        avg_rounds = total_rounds / eval_episodes
        
        print(f"  {checkpoint*100:3.0f}%: Success Rate {success_rate:5.1f}% | Avg Rounds {avg_rounds:.2f}")
    
    model_path = f"baseline_{config_name}"
    model.save(model_path)
    print(f"\n✅ Training complete! Model saved to {model_path}")
    
    return model

def analyze_policy(config_name='multi_round', model_path=None, episodes=1000):
    """Analyze learned policy and compute expected rounds"""
    
    # Load model and environment
    if model_path is None:
        model_path = f"baseline_{config_name}"
    
    model = PPO.load(model_path)
    config = get_config(config_name)
    env = MultiRoundAuctionEnv(config)
    
    # Get all possible scenarios
    scenarios = []
    for b1_cfg in config.bidder_configs[0].valuation_low, config.bidder_configs[0].valuation_high:
        for b2_cfg in config.bidder_configs[1].valuation_low, config.bidder_configs[1].valuation_high:
            scenarios.append((int(b1_cfg), int(b2_cfg)))
    
    # Track policy for each scenario
    scenario_data = defaultdict(lambda: {
        'first_actions': [],
        'obs_action_counts': defaultdict(int),
        'rounds': [],
        'total_episodes': 0
    })
    
    # Run episodes
    for episode in range(episodes):
        obs, info = env.reset()
        scenario = tuple(int(v) for v in info['valuations'])
        
        rounds = 0
        done = False
        scenario_data[scenario]['total_episodes'] += 1
        
        while not done and rounds < config.max_rounds:
            action, _ = model.predict(obs, deterministic=True)
            
            # Track first action
            if rounds == 0:
                scenario_data[scenario]['first_actions'].append(int(action))
            
            # Track obs-action pairs
            obs_key = f"[{obs[0]},{obs[1]}]"
            action_price = config.price_options[int(action)]
            pair_key = f"{obs_key}->${action_price}"
            scenario_data[scenario]['obs_action_counts'][pair_key] += 1
            
            obs, reward, terminated, truncated, info = env.step(action)
            rounds += 1
            done = terminated or truncated
        
        scenario_data[scenario]['rounds'].append(rounds)
    
    # Print summary statistics
    print(f"\n=== POLICY ANALYSIS: {config_name.upper()} ===")
    
    total_expected_rounds = 0
    
    for scenario in scenarios:
        if scenario in scenario_data:
            data = scenario_data[scenario]
            avg_rounds = np.mean(data['rounds'])
            
            print(f"\nScenario: Valuations {scenario[0]} and {scenario[1]} | rounds: {avg_rounds:.1f}")
            
            # Most common first action
            if data['first_actions']:
                first_action_counts = defaultdict(int)
                for action in data['first_actions']:
                    first_action_counts[action] += 1
                
                most_common_first = max(first_action_counts, key=first_action_counts.get)
                first_freq = first_action_counts[most_common_first] / len(data['first_actions'])
                first_price = config.price_options[most_common_first]
                
                print(f"  Most common first action: ${first_price} ({first_freq:.0%} of episodes)")
            
            # Most common observation-action patterns
            print(f"  Common patterns:")
            sorted_patterns = sorted(data['obs_action_counts'].items(), key=lambda x: x[1], reverse=True)
            for pattern, count in sorted_patterns[:3]:  # Top 3 patterns
                freq = count / data['total_episodes']
                print(f"    {pattern} ({freq:.0%})")
            
            total_expected_rounds += avg_rounds * 0.25  # Each scenario has 25% probability
        else:
            print(f"\nScenario: Valuations {scenario[0]} and {scenario[1]} | rounds: No data")
    
    print(f"\nExpected # of rounds: {total_expected_rounds:.2f}")
    
    return scenario_data

def run_experiment(config_name, **hyperparams):
    """Run complete experiment: train and analyze"""
    print(f"\n{'='*50}")
    print(f"EXPERIMENT: {config_name.upper()}")
    print(f"{'='*50}")
    
    # Train
    model = train_baseline(config_name, timesteps=50000, **hyperparams)
    
    # Analyze
    results = analyze_policy(config_name)
    
    return results

if __name__ == "__main__":
    # Run both experiments

        run_experiment('multi_round')

    #for config in ['multi_round', 'simple_price']:
     #   run_experiment(config)
        
    # Example with custom hyperparameters:
    # run_experiment('multi_round', learning_rate=1e-3, ent_coef=0.02)