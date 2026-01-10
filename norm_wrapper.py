import numpy as np
import gymnasium as gym
from gymnasium import spaces

class CATSActionNormalizationWrapper(gym.Wrapper):
    """
    Normalizes action space to [-1, 1] for neural network compatibility.
    
    The NN outputs actions in [-1, 1], and this wrapper:
    1. Denormalizes them to the actual price range [min_price, max_price]
    2. Passes denormalized actions to the underlying environment
    
    Works with CATSAuctionEnv (continuous action space with Box actions).
    """
    
    def __init__(self, env):
        """
        Args:
            env: CATSAuctionEnv instance
        """
        super().__init__(env)
        
        # Verify this is wrapping a continuous environment
        if not isinstance(env.action_space, spaces.Box):
            raise ValueError(
                "ActionNormalizationWrapper only works with continuous action spaces (Box). "
                "You're trying to wrap an environment with action space: {}".format(type(env.action_space))
            )
        
        # Get price range from config
        if hasattr(env, 'unwrapped'):
            base_env = env.unwrapped
        else:
            base_env = env
            
        self.min_price = base_env.config.min_price
        self.max_price = base_env.config.max_price
        self.num_items = base_env.config.num_items
        
        # Store original action space for reference
        self._original_action_space = env.action_space
        
        # Override action space to normalized range [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.num_items,),
            dtype=np.float32
        )
        
        # Observation space remains unchanged
        self.observation_space = env.observation_space
    
    def denormalize_action(self, normalized_action):
        """
        Convert normalized action [-1, 1] to actual price range [min_price, max_price]
        
        Direct formula: actual = min + (normalized + 1) / 2 * (max - min)
        
        Args:
            normalized_action: Array in [-1, 1]
            
        Returns:
            denormalized_action: Array in [min_price, max_price]
        """
        # Clip to ensure we're in [-1, 1] (safety check)
        normalized_action = np.clip(normalized_action, -1.0, 1.0)
        
        # Direct denormalization
        denormalized = self.min_price + (normalized_action + 1.0) / 2.0 * (self.max_price - self.min_price)
        
        return denormalized.astype(np.float32)
    
    def normalize_action(self, denormalized_action):
        """
        Convert actual price to normalized action (inverse operation, useful for debugging)
        
        Formula: normalized = 2 * (actual - min) / (max - min) - 1
        
        Args:
            denormalized_action: Array in [min_price, max_price]
            
        Returns:
            normalized_action: Array in [-1, 1]
        """
        normalized = 2.0 * (denormalized_action - self.min_price) / (self.max_price - self.min_price) - 1.0
        
        return np.clip(normalized, -1.0, 1.0).astype(np.float32)
    
    def step(self, normalized_action):
        """
        Step with normalized action from neural network
        
        Args:
            normalized_action: Action in [-1, 1] range
            
        Returns:
            Standard gym step returns (obs, reward, terminated, truncated, info)
        """
        # Denormalize action to actual price range
        denormalized_action = self.denormalize_action(normalized_action)
        
        # Pass to underlying environment (Monitor wrapper handles reset state)
        # PPO's vectorized environment will auto-reset if needed
        return self.env.step(denormalized_action)
    
    def reset(self, **kwargs):
        """Reset environment (no changes needed)"""
        return self.env.reset(**kwargs)


def test_normalization_wrapper():
    """Test the normalization wrapper"""
    from cont_env import CATSAuctionEnv
    from cont_config import get_cats_config
    
    print("Testing CATS Action Normalization Wrapper")
    print("=" * 50)
    
    # Create base environment and wrap it
    config = get_cats_config('0000.txt', num_bidders=3, seed=42)
    base_env = CATSAuctionEnv(config)
    wrapped_env = CATSActionNormalizationWrapper(base_env)
    
    print(f"Original action space: {base_env.action_space}")
    print(f"Wrapped action space:  {wrapped_env.action_space}")
    print(f"Price range: ${wrapped_env.min_price} - ${wrapped_env.max_price}")
    
    # Test denormalization (using first few items as examples)
    num_test_items = min(3, config.num_items)
    print(f"\nTesting denormalization (using {num_test_items} items as examples):")
    test_normalized_actions = [
        np.array([-1.0] * num_test_items, dtype=np.float32),  # Should map to min_price
        np.array([0.0] * num_test_items, dtype=np.float32),    # Should map to mid_price
        np.array([1.0] * num_test_items, dtype=np.float32),    # Should map to max_price
        np.array([-0.5, 0.0, 0.5][:num_test_items], dtype=np.float32),   # Mixed values
    ]
    
    for norm_action in test_normalized_actions:
        denorm_action = wrapped_env.denormalize_action(norm_action)
        print(f"  {norm_action} → {denorm_action}")
    
    # Test round-trip (normalize then denormalize)
    print(f"\nTesting round-trip (should get back original, using {num_test_items} items):")
    mid_price = (wrapped_env.min_price + wrapped_env.max_price) / 2.0
    test_prices = [
        np.array([wrapped_env.min_price] * num_test_items, dtype=np.float32),
        np.array([mid_price] * num_test_items, dtype=np.float32),
        np.array([wrapped_env.max_price] * num_test_items, dtype=np.float32),
    ]
    
    for price in test_prices:
        normalized = wrapped_env.normalize_action(price)
        denormalized = wrapped_env.denormalize_action(normalized)
        print(f"  {price} → {normalized} → {denormalized}")
    
    # Test in actual episode
    print("\nTesting in episode:")
    obs, info = wrapped_env.reset(seed=42)
    print(f"  Bidders: {len(config.bidder_configs)}, Items: {config.num_items}")
    print(f"  Valuations: {[b.valuation for b in config.bidder_configs]}")
    
    # Use normalized actions (what NN would output)
    # Create normalized action for all items (mid-range for all)
    normalized_action = np.zeros(config.num_items, dtype=np.float32)  # Mid-range
    obs, reward, terminated, truncated, info = wrapped_env.step(normalized_action)
    
    print(f"  Normalized action (first 5): {normalized_action[:5]}")
    print(f"  Actual prices set (first 5): {info['prices'][:5]}")
    print(f"  Demands: {info['demands']}")
    print(f"  Market clearing: {info['market_clearing']}")
    print(f"  Success: {info['successful_allocation']}")


if __name__ == "__main__":
    test_normalization_wrapper()