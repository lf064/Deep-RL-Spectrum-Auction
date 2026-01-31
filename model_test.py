import numpy as np
from stable_baselines3 import PPO
from cont_env import CATSAuctionEnv
from cont_config import get_cats_config
from norm_wrapper import CATSActionNormalizationWrapper


def diagnose_model(model_path='cats_ppo_3b_12i_s86-resampleTrue'):
    """
    Diagnose why the model isn't finding clearing prices
    """
    print("=" * 70)
    print("MODEL DIAGNOSTIC")
    print("=" * 70)
    
    # Load model
    try:
        model = PPO.load(model_path)
        print(f"✓ Model loaded successfully from: {model_path}")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Create environment
    config = get_cats_config('0000.txt', num_bidders=3, seed=764)
    env = CATSAuctionEnv(config)
    env = CATSActionNormalizationWrapper(env)
    
    print(f"\n📋 ENVIRONMENT CONFIGURATION:")
    print(f"{'─'*70}")
    print(f"Items: {config.num_items}")
    print(f"Bidders: {len(config.bidder_configs)}")
    print(f"Max rounds: {config.max_rounds}")
    print(f"Price range: ${config.min_price:.2f} - ${config.max_price:.2f}")
    
    print(f"\n👥 BIDDER DETAILS:")
    print(f"{'─'*70}")
    for i, (val, bundle) in enumerate(zip(env.env.bidder_valuations, env.env.bidder_bundles)):
        items_str = "{" + ", ".join(str(x) for x in sorted(bundle)) + "}"
        per_item = val / len(bundle)
        print(f"Bidder {i}: ${val:6.2f} for {items_str:<20} (${per_item:5.2f}/item)")
    
    # Check for overlapping bundles (common issue)
    print(f"\n🔍 BUNDLE OVERLAP ANALYSIS:")
    print(f"{'─'*70}")
    bundles = env.env.bidder_bundles
    for i in range(len(bundles)):
        for j in range(i+1, len(bundles)):
            overlap = bundles[i] & bundles[j]
            if overlap:
                print(f"⚠️  Bidder {i} and Bidder {j} both want items: {sorted(overlap)}")
    
    # Run one episode with detailed output
    print(f"\n🎯 RUNNING DETAILED TEST EPISODE:")
    print(f"{'─'*70}")
    
    obs, info = env.reset()
    
    for round_num in range(min(10, config.max_rounds)):  # Show first 5 rounds
        print(f"\n--- Round {round_num + 1} ---")
        
        # Get model's action
        action, _ = model.predict(obs, deterministic=True)
        
        # Denormalize
        actual_prices = env.denormalize_action(action)
        
        print(f"Model's prices (first 12 items): {actual_prices[:12]}")
        
        # Take step
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Show demand
        print(f"Bidder demands: {info['demands']}")
        
        # Show item-level demand
        excess_demand_items = []
        no_demand_items = []
        single_demand_items = []
        
        for item_id in range(config.num_items):
            bidders = info['item_demands'][item_id]
            if len(bidders) == 0:
                no_demand_items.append(item_id)
            elif len(bidders) == 1:
                single_demand_items.append(item_id)
            else:
                excess_demand_items.append(item_id)
        
        if excess_demand_items:
            print(f"⚠️  Excess demand on items: {excess_demand_items}")
        if no_demand_items:
            print(f"❌ No demand on items: {no_demand_items}")
        if single_demand_items:
            print(f"✓ Single demand on items: {single_demand_items}")
        
        # Show clearing status
        print(f"Market clearing: {'✓ YES' if info['market_clearing'] else '✗ NO'}")
        print(f"Reward: {reward:+.1f}")
        
        # Check price adjustment
        if info.get('price_adjusted', False):
            print(f"🔧 Prices were auto-adjusted!")
            adjusted_items = np.where(info['prices'] != info['original_prices'])[0]
            print(f"   Items set to $0: {list(adjusted_items)}")
        
        if terminated or truncated:
            print(f"\n{'='*70}")
            if info['market_clearing']:
                print(f"✓ EPISODE CLEARED in {round_num + 1} rounds!")
                print(f"Revenue: ${info['revenue']:.2f}")
                print(f"Allocation: {info['allocation_result']}")
            else:
                print(f"✗ Episode terminated without clearing")
                if truncated:
                    print(f"   Reason: Reached max rounds ({config.max_rounds})")
            break
    else:
        # Loop completed without termination
        print(f"\n{'='*70}")
        print(f"⚠️  Episode didn't terminate in first 5 rounds")
        print(f"Continuing until max_rounds={config.max_rounds}...")
        
        # Fast-forward to end
        while round_num < config.max_rounds - 1:
            round_num += 1
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                print(f"\nTerminated at round {round_num + 1}")
                print(f"Market clearing: {'✓ YES' if info['market_clearing'] else '✗ NO'}")
                break
    
    # Test with random prices to see if ANY clearing is possible
    print(f"\n{'='*70}")
    print("TESTING RANDOM PRICES (checking if clearing is possible):")
    print(f"{'─'*70}")
    
    obs, info = env.reset()
    found_clearing = False
    
    for attempt in range(100):
        # Random normalized action
        random_action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(random_action)
        
        if info['market_clearing']:
            found_clearing = True
            actual_prices = env.denormalize_action(random_action)
            print(f"✓ Found clearing prices after {attempt + 1} random attempts!")
            print(f"Clearing prices: {actual_prices}")
            print(f"Demands: {info['demands']}")
            print(f"Allocation: {info['allocation_result']}")
            break
        
        # Reset for next attempt
        obs, info = env.reset()
    
    if not found_clearing:
        print(f"❌ No clearing found in 100 random attempts")
        print(f"\nThis suggests:")
        print(f"  1. The auction instance may be very difficult to clear")
        print(f"  2. Multiple bidders may compete for same items")
        print(f"  3. The clearing condition may be too strict")
    
    # Test with zero prices
    print(f"\n{'─'*70}")
    print("TESTING ZERO PRICES:")
    print(f"{'─'*70}")
    
    obs, info = env.reset()
    zero_action = np.ones(config.num_items) * (-1.0)  # -1 in normalized space = min price
    obs, reward, terminated, truncated, info = env.step(zero_action)
    
    actual_prices = env.denormalize_action(zero_action)
    print(f"Prices: {actual_prices[:6]} (all zeros)")
    print(f"Demands: {info['demands']}")
    print(f"Market clearing: {'✓ YES' if info['market_clearing'] else '✗ NO'}")
    
    if not info['market_clearing']:
        print(f"\nWhy zero prices don't clear:")
        for item_id in range(min(6, config.num_items)):
            bidders = info['item_demands'][item_id]
            if len(bidders) > 1:
                print(f"  Item {item_id}: {len(bidders)} bidders competing → {bidders}")
    
    print(f"\n{'='*70}")


if __name__ == "__main__":
    diagnose_model('cats_ppo_3b_12i_s86-resampleTrue')