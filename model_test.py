import numpy as np
from stable_baselines3 import PPO
from cont_env import CATSAuctionEnv
from cont_config import get_cats_config
from norm_wrapper import CATSActionNormalizationWrapper
from typing import List, Dict
import pandas as pd


def diagnose_single_seed(model, seed: int, num_bidders: int = 7, 
                         verbose: bool = False, max_rounds_display: int = 20):
    """
    Diagnose model performance on a single seed.
    
    Args:
        model: Trained PPO model
        seed: CATS seed to test
        num_bidders: Number of bidders
        verbose: If True, print detailed round-by-round info
        max_rounds_display: Maximum rounds to show in verbose mode
    
    Returns:
        dict with diagnostic results
    """
    config = get_cats_config('0000.txt', num_bidders=num_bidders, seed=seed)
    env = CATSAuctionEnv(config)
    env = CATSActionNormalizationWrapper(env)
    
    result = {
        'seed': seed,
        'num_bidders': num_bidders,
        'num_items': config.num_items,
        'cleared': False,
        'rounds': 0,
        'revenue': 0.0,
        'welfare': 0.0,
        'efficiency': 0.0,
        'max_overlap': 0,  # Max items overlapping between any two bundles
        'total_overlaps': 0,  # Total number of pairwise overlaps
    }
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"SEED {seed} - DETAILED DIAGNOSTIC")
        print(f"{'='*70}")
        print(f"\n📋 ENVIRONMENT:")
        print(f"Items: {config.num_items}, Bidders: {num_bidders}")
        
        print(f"\n👥 BIDDERS:")
        for i, (val, bundle) in enumerate(zip(env.env.bidder_valuations, env.env.bidder_bundles)):
            items_str = "{" + ", ".join(str(x) for x in sorted(bundle)) + "}"
            print(f"  {i}: ${val:6.2f} for {items_str}")
    
    # Analyze bundle overlaps
    bundles = env.env.bidder_bundles
    overlaps = []
    for i in range(len(bundles)):
        for j in range(i+1, len(bundles)):
            overlap = bundles[i] & bundles[j]
            if overlap:
                overlaps.append(len(overlap))
                if verbose:
                    print(f"  ⚠️  Bidders {i},{j} overlap on {len(overlap)} items: {sorted(overlap)}")
    
    result['total_overlaps'] = len(overlaps)
    result['max_overlap'] = max(overlaps) if overlaps else 0
    
    # Run episode
    obs, info = env.reset()
    
    for round_num in range(config.max_rounds):
        if verbose and round_num < max_rounds_display:
            print(f"\n--- Round {round_num + 1} ---")
        
        action, _ = model.predict(obs, deterministic=True)
        
        if verbose and round_num < max_rounds_display:
            actual_prices = env.denormalize_action(action)
            print(f"Prices (first 6): {actual_prices[:11]}")
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if verbose and round_num < max_rounds_display:
            print(f"Demands: {info['demands']}")
            print(f"Clearing: {'✓' if info['market_clearing'] else '✗'}, Reward: {reward:+.1f}")
        
        if terminated or truncated:
            result['cleared'] = info['market_clearing']
            result['rounds'] = round_num + 1
            result['revenue'] = info.get('revenue', 0.0)
            result['welfare'] = info.get('total_value', 0.0)
            result['efficiency'] = info.get('efficiency', 0.0)
            
            if verbose:
                print(f"\n{'─'*70}")
                if info['market_clearing']:
                    print(f"✓ CLEARED in {result['rounds']} rounds")
                    print(f"  Revenue: ${result['revenue']:.2f}")
                    print(f"  Welfare: ${result['welfare']:.2f}")
                    print(f"  Efficiency: {result['efficiency']:.1%}")
                else:
                    print(f"✗ FAILED after {result['rounds']} rounds")
            break
    
    return result


def diagnose_model(model_path='cats_ppo_7b_12i_s86', 
                   seeds: List[int] = None,
                   num_bidders: int = 7,
                   verbose: bool = False,
                   summary: bool = True,
                   save_csv: str = None):
    """
    Diagnose model on multiple seeds.
    
    Args:
        model_path: Path to saved model
        seeds: List of seeds to test. If None, tests seed 764 only
        num_bidders: Number of bidders per instance
        verbose: If True, print detailed diagnostics for each seed
        summary: If True, print summary statistics at the end
        save_csv: If provided, save results to this CSV file
    
    Returns:
        pandas DataFrame with results for all seeds
    """
    print("=" * 70)
    print("MODEL DIAGNOSTIC - MULTI-SEED EVALUATION")
    print("=" * 70)
    
    # Load model
    try:
        model = PPO.load(model_path)
        print(f"✓ Model loaded: {model_path}")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None
    
    # Default to single seed if none provided
    if seeds is None:
        seeds = [764]
    
    print(f"\n📊 Testing {len(seeds)} seeds with {num_bidders} bidders each")
    print(f"{'─'*70}")
    
    # Run diagnostics on all seeds
    results = []
    for i, seed in enumerate(seeds):
        if not verbose:
            # Progress indicator
            if (i + 1) % 10 == 0 or i == 0:
                print(f"Processing seed {i+1}/{len(seeds)}... (seed={seed})", end='\r')
        
        result = diagnose_single_seed(
            model, 
            seed, 
            num_bidders=num_bidders,
            verbose=verbose,
            max_rounds_display=20
        )
        results.append(result)
    
    if not verbose:
        print()  # New line after progress
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Save to CSV if requested
    if save_csv:
        df.to_csv(save_csv, index=False)
        print(f"\n💾 Results saved to: {save_csv}")
    
    # Print summary
    if summary:
        print(f"\n{'='*70}")
        print("SUMMARY STATISTICS")
        print(f"{'='*70}")
        
        cleared_df = df[df['cleared'] == True]
        
        print(f"\n📈 OVERALL PERFORMANCE:")
        print(f"  Total instances: {len(df)}")
        print(f"  Cleared: {len(cleared_df)} ({len(cleared_df)/len(df)*100:.1f}%)")
        print(f"  Failed: {len(df) - len(cleared_df)} ({(len(df)-len(cleared_df))/len(df)*100:.1f}%)")
        
        if len(cleared_df) > 0:
            print(f"\n📊 CLEARED INSTANCES:")
            print(f"  Median rounds: {cleared_df['rounds'].median():.0f}")
            print(f"  Mean rounds: {cleared_df['rounds'].mean():.1f}")
            print(f"  Min rounds: {cleared_df['rounds'].min():.0f}")
            print(f"  Max rounds: {cleared_df['rounds'].max():.0f}")
            print(f"  Std rounds: {cleared_df['rounds'].std():.1f}")
            
            print(f"\n💰 PERFORMANCE METRICS:")
            print(f"  Avg revenue: ${cleared_df['revenue'].mean():.2f}")
            print(f"  Avg welfare: ${cleared_df['welfare'].mean():.2f}")
            print(f"  Avg efficiency: {cleared_df['efficiency'].mean():.1%}")
        
        failed_df = df[df['cleared'] == False]
        if len(failed_df) > 0:
            print(f"\n❌ FAILED INSTANCES:")
            print(f"  Avg rounds to failure: {failed_df['rounds'].mean():.1f}")
            print(f"  Avg max overlap: {failed_df['max_overlap'].mean():.1f} items")
            print(f"  Avg total overlaps: {failed_df['total_overlaps'].mean():.1f}")
        
        print(f"\n🔍 BUNDLE OVERLAP ANALYSIS:")
        print(f"  Avg max overlap (all): {df['max_overlap'].mean():.1f} items")
        print(f"  Avg total overlaps (all): {df['total_overlaps'].mean():.1f}")
        
        if len(cleared_df) > 0 and len(failed_df) > 0:
            print(f"\n📉 CLEARED vs FAILED:")
            print(f"  Overlap (cleared): {cleared_df['max_overlap'].mean():.1f} items")
            print(f"  Overlap (failed): {failed_df['max_overlap'].mean():.1f} items")
    
    print(f"\n{'='*70}")
    
    return df


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    
    """# Example 1: Test single seed with verbose output
    print("\n" + "="*70)
    print("EXAMPLE 1: Single seed, verbose")
    print("="*70)
    df = diagnose_model(
        model_path='cats_ppo_7b_12i_s86',
        seeds=[764],
        num_bidders=7,
        verbose=True,
        summary=True
    )  """
    
    # Example 2: Test 100 seeds quietly
    print("\n\n" + "="*70)
    print("EXAMPLE 2: 100 seeds, quiet mode")
    print("="*70)
    test_seeds = list(range(100, 200))  # Test set
    df = diagnose_model(
        model_path='cats_ppo_7b_12i_s86-resampleTrue',
        seeds=test_seeds,
        num_bidders=7,
        verbose=False,
        summary=True,
        save_csv='ppo_diagnostic_results.csv' 
    )
    
    # Example 3: Test specific problematic seeds
    print("\n\n" + "="*70)
    print("EXAMPLE 3: Specific seeds with verbose output")
    print("="*70)
    problematic_seeds = [764, 1234, 5678]
    df = diagnose_model(
        model_path='cats_ppo_7b_12i_s86',
        seeds=problematic_seeds,
        num_bidders=7,
        verbose=True,
        summary=True
    )
    
    # Can also access the DataFrame for further analysis
    if df is not None:
        print("\n" + "="*70)
        print("DataFrame sample:")
        print(df.head()) 