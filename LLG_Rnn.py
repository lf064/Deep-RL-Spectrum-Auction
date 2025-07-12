from sb3_contrib import RecurrentPPO
from LLG_ENV import StandardLLGAuctionEnv
from LLG_config import get_standard_llg_config
import numpy as np
from collections import defaultdict

def train_standard_llg_rnn(timesteps=10000, **hyperparams):
    """
    Train RecurrentPPO (LSTM) on standard LLG environment
    
    Metrics:
    1. Clearing rate: % of instances where clearing prices are found within max_rounds
    2. Average rounds to clear: Mean rounds needed, computed only over cleared instances
    """
    config = get_standard_llg_config()
    env = StandardLLGAuctionEnv(config)
    
    # Default hyperparameters for RecurrentPPO
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
    
    print("=" * 50)
    print("STANDARD LLG RNN TRAINING")
    print("=" * 50)
    print(f"Training for {timesteps:,} timesteps")
    print(f"Model: RecurrentPPO with LSTM")
    print(f"Action space: Discrete {env.action_space}")
    print(f"Price options: {env.price_options}")
    print("Progress evaluations:")
    
    model = RecurrentPPO("MlpLstmPolicy", env, **params)
    
    # Train with periodic evaluation
    eval_points = [0.25, 0.5, 0.75, 1.0]
    
    for i, checkpoint in enumerate(eval_points):
        # Train to this checkpoint
        steps_to_train = int(timesteps * checkpoint) - (int(timesteps * eval_points[i-1]) if i > 0 else 0)
        
        if steps_to_train > 0:
            model.learn(total_timesteps=steps_to_train, reset_num_timesteps=False)
        
        # Evaluation
        cleared_instances = 0
        total_rounds_cleared_only = 0
        allocation_types = defaultdict(int)
        scenario_results = defaultdict(lambda: {'cleared': 0, 'total': 0, 'rounds_cleared': []})
        eval_episodes = 100
        
        for _ in range(eval_episodes):
            obs, info = env.reset()
            
            # Reset LSTM states for new episode
            lstm_states = None
            episode_starts = np.ones((1,), dtype=bool)
            
            # Determine scenario: global_low ($4) vs global_high ($10)
            global_val = info['valuations'][2]
            scenario = 'global_low' if global_val == 4 else 'global_high'
            
            rounds = 0
            done = False
            
            while not done and rounds < config.max_rounds:
                action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_starts = np.array([False])  # Only first step is episode start
                rounds += 1
                done = terminated or truncated
            
            # Track results - only count as cleared if successful within max_rounds
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
        
        # Show detailed breakdown at final checkpoint
        if checkpoint == 1.0:
            print(f"\n    Allocation types:")
            for alloc_type, count in allocation_types.items():
                pct = count / eval_episodes * 100
                print(f"      {alloc_type}: {pct:.1f}%")
            
            print(f"\n    Performance by scenario:")
            for scenario, results in scenario_results.items():
                if results['total'] > 0:
                    scenario_clearing_rate = results['cleared'] / results['total'] * 100
                    scenario_avg_rounds = np.mean(results['rounds_cleared']) if results['rounds_cleared'] else 0
                    scenario_desc = "Global=$4" if scenario == 'global_low' else "Global=$10"
                    print(f"      {scenario_desc}: {scenario_clearing_rate:.1f}% cleared ({results['cleared']}/{results['total']}) | Avg rounds: {scenario_avg_rounds:.1f}")
    
    model_path = "standard_llg_rnn_model"
    model.save(model_path)
    print(f"\n✅ Training complete! Model saved to {model_path}")
    
    return model

def analyze_standard_llg_rnn_policy(model_path=None, episodes=1000):
    """
    Analyze learned standard LLG RecurrentPPO policy
    
    Comprehensive evaluation over 1000 episodes:
    1. Clearing rate: % of instances where clearing prices are found within max_rounds
    2. Average rounds to clear: Mean rounds needed, computed only over cleared instances
    3. Action-observation correspondence analysis
    4. LSTM memory usage patterns
    """
    
    # Load model and environment
    if model_path is None:
        model_path = "standard_llg_rnn_model"
    
    model = RecurrentPPO.load(model_path)
    config = get_standard_llg_config()
    env = StandardLLGAuctionEnv(config)
    
    print("=" * 60)
    print("STANDARD LLG RNN POLICY ANALYSIS")
    print("=" * 60)
    print(f"Model: {model_path} (RecurrentPPO with LSTM)")
    print(f"Evaluation episodes: {episodes}")
    
    # Track comprehensive metrics
    action_patterns = defaultdict(int)
    successful_patterns = defaultdict(int)
    scenario_actions = defaultdict(lambda: defaultdict(int))
    observation_action_pairs = defaultdict(lambda: defaultdict(int))
    
    # Metrics tracking
    cleared_instances = 0
    total_rounds_cleared_only = 0
    allocation_types = defaultdict(int)
    scenario_results = defaultdict(lambda: {'cleared': 0, 'total': 0, 'rounds_cleared': []})
    episode_details = []
    
    # Run episodes and collect data
    for episode in range(episodes):
        obs, info = env.reset()
        valuations = info['valuations']
        global_val = valuations[2]
        scenario = 'global_low' if global_val == 4 else 'global_high'
        
        # Reset LSTM states for new episode
        lstm_states = None
        episode_starts = np.ones((1,), dtype=bool)
        
        step = 0
        done = False
        episode_steps = []
        
        while not done and step < config.max_rounds:
            action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
            
            # Convert action to prices
            prices = [env.price_options[action[i]] for i in range(config.num_items)]
            price_a, price_b = prices[0], prices[1]
            total_price = price_a + price_b
            
            # Store step details
            step_info = {
                'obs': obs.copy(),
                'action': action.copy(),
                'price_a': price_a,
                'price_b': price_b,
                'total_price': total_price
            }
            episode_steps.append(step_info)
            
            obs, reward, terminated, truncated, info = env.step(action)
            episode_starts = np.array([False])  # Only first step is episode start
            step += 1
            done = terminated or truncated
            
            # Track action patterns
            action_key = f"[{action[0]},{action[1]}]"
            action_patterns[action_key] += 1
            scenario_actions[scenario][action_key] += 1
            
            # Track observation-action correspondence
            obs_key = f"[{step_info['obs'][0]},{step_info['obs'][1]},{step_info['obs'][2]}]"
            observation_action_pairs[obs_key][action_key] += 1
            
            if done:
                if info['successful_allocation']:
                    successful_patterns[action_key] += 1
                break
        
        # Track metrics
        instance_cleared = info['successful_allocation']
        if instance_cleared:
            cleared_instances += 1
            total_rounds_cleared_only += step
            scenario_results[scenario]['rounds_cleared'].append(step)
        
        allocation_types[info['allocation_type']] += 1
        scenario_results[scenario]['total'] += 1
        if instance_cleared:
            scenario_results[scenario]['cleared'] += 1
        
        # Store episode details (save last 10 for detailed output)
        if episode >= episodes - 10:
            episode_detail = {
                'episode': episode,
                'scenario': scenario,
                'valuations': valuations,
                'steps': episode_steps,
                'rounds': step,
                'success': instance_cleared,
                'allocation_type': info['allocation_type'],
                'final_decisions': info['decisions'],
                'market_clearing': info['market_clearing'],
                'revenue': info.get('revenue', 0),
                'allocation_result': info.get('allocation_result', {})
            }
            episode_details.append(episode_detail)
    
    # Calculate final metrics
    clearing_rate = cleared_instances / episodes * 100
    avg_rounds_to_clear = total_rounds_cleared_only / cleared_instances if cleared_instances > 0 else 0
    
    print(f"\n=== FINAL RESULTS ({episodes} episodes) ===")
    print(f"Clearing Rate: {clearing_rate:.1f}% ({cleared_instances}/{episodes})")
    print(f"Average Rounds to Clear: {avg_rounds_to_clear:.2f} (excluding failed instances)")
    
    print(f"\nAllocation types:")
    for alloc_type, count in allocation_types.items():
        pct = count / episodes * 100
        print(f"  {alloc_type}: {pct:.1f}%")
    
    print(f"\nPerformance by scenario:")
    for scenario, results in scenario_results.items():
        if results['total'] > 0:
            scenario_clearing_rate = results['cleared'] / results['total'] * 100
            scenario_avg_rounds = np.mean(results['rounds_cleared']) if results['rounds_cleared'] else 0
            scenario_desc = "Global=$4" if scenario == 'global_low' else "Global=$10"
            print(f"  {scenario_desc}: {scenario_clearing_rate:.1f}% cleared ({results['cleared']}/{results['total']}) | Avg rounds: {scenario_avg_rounds:.1f}")
    
    # Action pattern analysis
    print(f"\n=== ACTION PATTERNS ===")
    print("Most common actions:")
    sorted_actions = sorted(action_patterns.items(), key=lambda x: x[1], reverse=True)
    for action_str, count in sorted_actions[:5]:
        success_count = successful_patterns.get(action_str, 0)
        success_rate = (success_count / count * 100) if count > 0 else 0
        
        action_idx = eval(action_str)
        price_a = env.price_options[action_idx[0]]
        price_b = env.price_options[action_idx[1]]
        total_price = price_a + price_b
        
        print(f"  Action {action_str}: ${price_a}+${price_b}=${total_price} | Used {count} times | {success_rate:.1f}% success")
    
    # Actions by scenario
    print(f"\n=== ACTIONS BY SCENARIO ===")
    for scenario in ['global_low', 'global_high']:
        if scenario in scenario_actions:
            scenario_desc = "Global=$4" if scenario == 'global_low' else "Global=$10"
            print(f"{scenario_desc}:")
            scenario_sorted = sorted(scenario_actions[scenario].items(), key=lambda x: x[1], reverse=True)
            for action_str, count in scenario_sorted[:3]:  # Top 3 actions for this scenario
                action_idx = eval(action_str)
                price_a = env.price_options[action_idx[0]]
                price_b = env.price_options[action_idx[1]]
                print(f"    Action {action_str}: ${price_a}+${price_b}=${price_a + price_b} | {count} times")
    
    # Observation-action correspondence
    print(f"\n=== OBSERVATION-ACTION CORRESPONDENCE ===")
    print("Most common observation-action patterns:")
    for obs_key, action_dict in observation_action_pairs.items():
        if sum(action_dict.values()) >= 10:  # Only show frequent patterns
            most_common_action = max(action_dict, key=action_dict.get)
            frequency = action_dict[most_common_action]
            total_for_obs = sum(action_dict.values())
            percentage = frequency / total_for_obs * 100
            
            action_idx = eval(most_common_action)
            price_a = env.price_options[action_idx[0]]
            price_b = env.price_options[action_idx[1]]
            
            print(f"  Obs {obs_key} → Action {most_common_action} (${price_a}+${price_b}) | {percentage:.1f}% of time ({frequency}/{total_for_obs})")
    
    # Show last 10 episodes in detail
    print(f"\n=== LAST 10 EPISODES DETAILS ===")
    for episode_detail in episode_details:
        episode = episode_detail['episode']
        scenario = episode_detail['scenario']
        valuations = episode_detail['valuations']
        rounds = episode_detail['rounds']
        
        print(f"Episode {episode}: Scenario {scenario}, Valuations {valuations}")
        
        for step_idx, step_info in enumerate(episode_detail['steps']):
            print(f"  Step {step_idx}: obs={step_info['obs']}, action={step_info['action']} → ${step_info['price_a']}+${step_info['price_b']}=${step_info['total_price']}")
        
        if episode_detail['success']:
            print(f"  → Success! Market cleared in {rounds} rounds, Revenue: ${episode_detail['revenue']}")
        else:
            print(f"  → Failed after {rounds} rounds, Type: {episode_detail['allocation_type']}")
        
        if episode < episode_details[-1]['episode']:
            print()
    
    return action_patterns, successful_patterns

def run_standard_llg_rnn_experiment(timesteps=10000):
    """Run complete standard LLG RNN experiment: train and analyze"""
    print("=" * 70)
    print("STANDARD LLG RNN COMPLETE EXPERIMENT")
    print("=" * 70)
    
    # Train
    model = train_standard_llg_rnn(timesteps=timesteps)
    
    # Analyze
    action_patterns, successful_patterns = analyze_standard_llg_rnn_policy(episodes=1000)
    
    return model, action_patterns, successful_patterns

if __name__ == "__main__":
    # Run complete experiment
    run_standard_llg_rnn_experiment(timesteps=100000)

