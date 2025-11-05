import numpy as np
import gymnasium as gym
from gymnasium import spaces
from cont_env import ContinuousLLGAuctionEnv
from cont_config import get_continuous_llg_config

class DiscreteLLGWrapper(gym.Wrapper):
    """
    Wrapper that converts continuous LLG environment to discrete actions
    
    This wrapper:
    1. Takes discrete action indices as input
    2. Maps them to continuous price values  
    3. Passes continuous actions to the underlying continuous environment
    4. Returns the same observations and rewards
    
    Useful for:
    - Comparing discrete vs continuous action policies
    - Using discrete RL algorithms on continuous environment
    - Easier interpretation and analysis of learned strategies
    """
    
    def __init__(self, env: ContinuousLLGAuctionEnv, price_options=None):
        """
        Initialize discrete wrapper
        
        Args:
            env: ContinuousLLGAuctionEnv instance
            price_options: List of discrete price values to use
                          If None, uses [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        """
        super().__init__(env)
        
        # Set price options (discrete values to choose from)
        if price_options is None:
            self.price_options = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        else:
            self.price_options = price_options
        
        # Override action space to be discrete
        # MultiDiscrete([n, n]) where n = number of price options for each item
        num_items = env.config.num_items
        self.action_space = spaces.MultiDiscrete([len(self.price_options)] * num_items)
        
        # Observation space stays the same
        self.observation_space = env.observation_space
        
        # Store original continuous action space for reference
        self._continuous_action_space = env.action_space
        
    def action(self, discrete_action):
        """
        Convert discrete action to continuous action
        
        Args:
            discrete_action: Array of indices into price_options
            
        Returns:
            continuous_action: Array of actual price values
        """
        # Convert action indices to actual prices
        continuous_action = np.array([
            self.price_options[discrete_action[i]] 
            for i in range(len(discrete_action))
        ], dtype=np.float32)
        
        return continuous_action
    
    def step(self, discrete_action):
        """
        Step with discrete action (converted to continuous internally)
        """
        # Convert discrete action to continuous
        continuous_action = self.action(discrete_action)
        
        # Pass to underlying continuous environment
        return super().step(continuous_action)
    
    def get_price_for_action(self, discrete_action):
        """
        Helper method to see what prices correspond to discrete actions
        """
        return [self.price_options[discrete_action[i]] for i in range(len(discrete_action))]


class FlexibleDiscreteLLGWrapper(DiscreteLLGWrapper):
    """
    More flexible discrete wrapper with different discretization options
    """
    
    def __init__(self, env: ContinuousLLGAuctionEnv, discretization_method="uniform", num_options=10, price_range=None):
        """
        Initialize flexible discrete wrapper
        
        Args:
            env: ContinuousLLGAuctionEnv instance
            discretization_method: "uniform", "logarithmic", "custom"
            num_options: Number of discrete price options
            price_range: (min_price, max_price) tuple, defaults to env's range
        """
        
        if price_range is None:
            price_range = (env.config.min_price, env.config.max_price)
        
        min_price, max_price = price_range
        
        # Generate price options based on method
        if discretization_method =zewcf= "uniform":
            # Evenly spaced prices
            price_options = np.linspace(min_price, max_price, num_options).tolist()
            
        elif discretization_method == "logarithmic":
            # Log-spaced prices (more options at lower prices)
            if min_price <= 0:
                min_price = 0.1  # Avoid log(0)
            price_options = np.logspace(np.log10(min_price), np.log10(max_price), num_options).tolist()
            
        elif discretization_method == "custom":
            # Standard price options like original discrete environment
            price_options = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            
        else:
            raise ValueError(f"Unknown discretization method: {discretization_method}")
        
        # Round to reasonable precision
        price_options = [round(p, 2) for p in price_options]
        
        super().__init__(env, price_options)
        self.discretization_method = discretization_method
        self.price_range = price_range


def create_discrete_llg_env(discretization_method="custom", num_options=10, **env_kwargs):
    """
    Factory function to create discrete LLG environment
    
    Args:
        discretization_method: "uniform", "logarithmic", "custom"
        num_options: Number of discrete price options
        **env_kwargs: Additional arguments for the base environment
        
    Returns:
        Wrapped discrete environment
    """
    # Create base continuous environment
    config = get_continuous_llg_config()
    continuous_env = ContinuousLLGAuctionEnv(config)
    
    # Wrap with discrete wrapper
    discrete_env = FlexibleDiscreteLLGWrapper(
        continuous_env, 
        discretization_method=discretization_method,
        num_options=num_options
    )
    
    return discrete_env


def test_discrete_wrapper():
    """Test the discrete wrapper functionality"""
    print("Testing Discrete Wrapper")
    print("=" * 40)
    
    # Create wrapped environment
    env = create_discrete_llg_env(discretization_method="custom")
    
    print(f"Wrapped action space: {env.action_space}")
    print(f"Original continuous action space: {env._continuous_action_space}")
    print(f"Price options: {env.price_options}")
    print(f"Observation space: {env.observation_space}")
    
    # Test episode
    obs, info = env.reset(seed=42)
    print(f"\nInitial observation: {obs}")
    print(f"Valuations: {info['valuations']}")
    
    # Test different discrete actions
    test_actions = [
        [4, 4],  # Index 4 = $5, so $5+$5=$10
        [3, 3],  # Index 3 = $4, so $4+$4=$8  
        [5, 5],  # Index 5 = $6, so $6+$6=$12
    ]
    
    for discrete_action in test_actions:
        env.reset(seed=42)  # Reset to same state
        
        # Get corresponding prices
        prices = env.get_price_for_action(discrete_action)
        
        # Take step
        obs, reward, terminated, truncated, info = env.step(discrete_action)
        
        print(f"\nDiscrete action {discrete_action} → Prices {prices} → Total ${sum(prices)}")
        print(f"  Observation: {obs}")
        print(f"  Success: {info['successful_allocation']}")
        print(f"  Allocation type: {info['allocation_type']}")


def compare_discretization_methods():
    """Compare different discretization methods"""
    print("\nComparing Discretization Methods")
    print("=" * 50)
    
    methods = [
        ("custom", 10),
        ("uniform", 10), 
        ("uniform", 20),
        ("logarithmic", 10)
    ]
    
    for method, num_options in methods:
        env = create_discrete_llg_env(discretization_method=method, num_options=num_options)
        
        print(f"\nMethod: {method}, Options: {num_options}")
        print(f"Price options: {[round(p, 2) for p in env.price_options[:8]]}{'...' if len(env.price_options) > 8 else ''}")
        print(f"Range: ${min(env.price_options):.2f} - ${max(env.price_options):.2f}")


if __name__ == "__main__":
    test_discrete_wrapper()
    compare_discretization_methods()