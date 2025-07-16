import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from LLG_std_config import get_continuous_llg_config  # Fixed import
from LLG_std_env import SimpleContinuousLLGEnv  # Fixed import and typo

def train_and_evaluate():
    """Minimal training and evaluation"""
    
    # Setup
    config = get_continuous_llg_config()  # Fixed: removed empty string parameter
    env = SimpleContinuousLLGEnv(config)  # Fixed: corrected class name typo
    
    # Check environment before training
    print("Checking environment...")
    try:
        check_env(env)
        print("✓ Environment check passed!")
    except Exception as e:
        print(f"✗ Environment check failed: {e}")
        return
    
    print("Training PPO...")
    model = PPO("MlpPolicy", env, verbose= True, learning_rate=3e-4, ent_coef=0.05)  # Added verbose for progress
    model.learn(total_timesteps=600000)
    
    print("\nEvaluating 100 episodes...")
    
    # Evaluate
    cleared_count = 0
    total_rounds_cleared = 0
    high_val_episodes = 0
    high_val_cleared = 0
    low_val_episodes = 0
    low_val_cleared = 0
    
    for ep in range(100):
        obs, info = env.reset()
        done = False
        rounds = 0

        # New condition: classify episode based on value comparison
        local_valuation_sum = info['valuations'][0] + info['valuations'][1]
        global_valuation = info['valuations'][2]
        is_high_val = global_valuation >= local_valuation_sum

        if is_high_val:
            high_val_episodes += 1
        else:
            low_val_episodes += 1

        while not done and rounds < config.max_rounds:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            rounds += 1
            done = terminated or truncated

        if info.get('market_clearing', False):
            cleared_count += 1
            total_rounds_cleared += rounds

            if is_high_val:
                high_val_cleared += 1
            else:
                low_val_cleared += 1

        # Print some sample episodes for debugging
        if ep < 5:
            print(f"Episode {ep+1}: Local val sum=${local_valuation_sum:.1f}, "
                f"Global val=${global_valuation}, "
                f"Final prices={info['prices']}, "
                f"Cleared={info['market_clearing']}, "
                f"Rounds={rounds}")
   
    for obs_type in [[0,0,0], [1,1,1], [2,2,2], [1,2,1], [2,1,2], [1,2,2]]:
        action, _ = model.predict(np.array(obs_type), deterministic=True)
        print(f"Obs: {obs_type} -> Action: {action}")


    
    # Results
    success_rate = (cleared_count / 100) * 100
    avg_rounds = total_rounds_cleared / cleared_count if cleared_count > 0 else 0
    
    high_val_rate = (high_val_cleared / high_val_episodes) * 100 if high_val_episodes > 0 else 0
    low_val_rate = (low_val_cleared / low_val_episodes) * 100 if low_val_episodes > 0 else 0
    
    print(f"\n{'='*50}")
    print(f"TRAINING RESULTS")
    print(f"{'='*50}")
    print(f"Overall Success Rate: {success_rate:.1f}% ({cleared_count}/100)")
    print(f"Average Rounds to Clear: {avg_rounds:.2f}")
    print()
    print(f"High Value Episodes (Global ≥ Locals sum): {high_val_episodes}")
    print(f"High Value Success Rate: {high_val_rate:.1f}% ({high_val_cleared}/{high_val_episodes})")
    print()
    print(f"Low Value Episodes (Global < Locals sum): {low_val_episodes}")
    print(f"Low Value Success Rate: {low_val_rate:.1f}% ({low_val_cleared}/{low_val_episodes})")

    
    # Test a few manual episodes to see learned behavior
    print(f"\n{'='*50}")
    print(f"SAMPLE LEARNED BEHAVIOR")
    print(f"{'='*50}")
    
    for test_ep in range(3):
        obs, info = env.reset()
        global_val = info['valuations'][2]
        print(f"\nTest Episode {test_ep+1}: Global valuation = ${global_val}")
        print(f"Valuations: {info['valuations']}")
        
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"Learned prices: {info['prices']}")
        print(f"Bidder decisions: {info['decisions']}")
        print(f"Market clears: {info['market_clearing']}")
        print(f"Revenue: ${info['revenue']:.2f}")

def quick_test():
    """Quick test to verify environment works"""
    print("Quick Environment Test:")
    print("=" * 30)
    
    config = get_continuous_llg_config()
    env = SimpleContinuousLLGEnv(config)
    
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    # Test reset
    obs, info = env.reset()
    print(f"Initial obs: {obs}")
    print(f"Valuations: {info['valuations']}")
    
    # Test step with normalized actions
    action = np.array([0.0, 0.0])  # Middle of normalized range → should map to $5 each
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"After normalized action {action} (maps to ~$5 each):")
    print(f"  Obs: {obs}")
    print(f"  Reward: {reward}")
    print(f"  Market clears: {info['market_clearing']}")
    print(f"  Actual prices: {info['prices']}")

if __name__ == "__main__":
    # First run quick test
    quick_test()
    print("\n" + "="*60 + "\n")
    
    # Then run full training
    train_and_evaluate()