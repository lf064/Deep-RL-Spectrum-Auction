import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple
from LLG_config import StandardLLGAuctionConfig, get_standard_llg_config

class StandardLLGAuctionEnv(gym.Env):
    """
    Standard Local-Local-Global (LLG) Auction Environment
    
    Features:
    - Discrete valuations: Locals=$4 fixed, Global=$4 or $10 (random)
    - Discrete actions: 10 price options [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    - Action space: MultiDiscrete([10, 10]) for two items
    - Observation: [local1_decision, local2_decision, global_decision]
      - 0 = neutral, 1 = reject, 2 = accept
    - Reward: +10 for market clearing, -1 otherwise
    """
    
    def __init__(self, config: StandardLLGAuctionConfig):
        super().__init__()
        self.config = config
        
        # Validate setup
        assert len(config.bidder_configs) >= 2, "Need at least 2 bidders"
        assert config.num_items >= 1, "Need at least 1 item"
        
        # Set price options
        self.price_options = config.price_options
        
        # Action space: price index for each item
        self.action_space = spaces.MultiDiscrete([len(self.price_options)] * config.num_items)
        
        # Observation space: decision for each bidder
        self.observation_space = spaces.MultiDiscrete([3] * len(config.bidder_configs))
        
        # State variables
        self.bidder_valuations = []
        self.round_number = 0
        self.successful_allocation = False
        self.total_revenue = 0
        self.current_prices = [0] * config.num_items
    
    def reset(self, seed=None, options=None):
        """Reset environment for new episode"""
        super().reset(seed=seed)
        
        # Sample fresh valuations for this episode
        self.bidder_valuations = []
        for cfg in self.config.bidder_configs:
            # Random choice between low and high valuation
            valuation = self.np_random.choice([cfg.valuation_low, cfg.valuation_high])
            self.bidder_valuations.append(valuation)
        
        # Reset state
        self.round_number = 0
        self.successful_allocation = False
        self.total_revenue = 0
        self.current_prices = [0] * self.config.num_items
        
        # Initial observation: all neutral
        initial_obs = np.array([0] * len(self.config.bidder_configs), dtype=np.int32)
        return initial_obs, self._get_info()
    
    def step(self, action):
        """Execute one auction round"""
        # Convert action indices to actual prices
        self.current_prices = [
            self.price_options[np.clip(int(action[i]), 0, len(self.price_options) - 1)]
            for i in range(self.config.num_items)
        ]
        
        self.round_number += 1
        
        # Get bidder decisions and calculate demand
        bidder_decisions, item_demands = self._calculate_demand()
        
        # Check market clearing condition
        is_market_clearing, allocation_result = self._check_market_clearing(item_demands)
        
        # Create observation: map True->2, False->1
        obs = np.array([2 if d else 1 for d in bidder_decisions], dtype=np.int32)
        
        # Calculate reward and termination
        if is_market_clearing:
            reward = 10.0
            terminated = True
            self.successful_allocation = True
            self.total_revenue = self._calculate_revenue(allocation_result)
        else:
            reward = -1.0
            terminated = False
        
        # End episode if max rounds reached
        if self.round_number >= self.config.max_rounds:
            terminated = True
        
        return obs, reward, terminated, False, self._get_info()
    
    def _calculate_demand(self) -> Tuple[List[bool], Dict[int, List[int]]]:
        """Calculate which bidders want to buy and aggregate demand per item"""
        bidder_decisions = []
        item_demands = {i: [] for i in range(self.config.num_items)}
        
        for bidder_id, (bidder_config, valuation) in enumerate(zip(self.config.bidder_configs, self.bidder_valuations)):
            
            if len(bidder_config.interested_items) == 1:
                # Local bidder: wants single item
                item_id = bidder_config.interested_items[0]
                item_price = self.current_prices[item_id]
                wants_to_buy = item_price <= valuation
                
                if wants_to_buy:
                    item_demands[item_id].append(bidder_id)
                    
            else:
                # Global bidder: wants bundle of items
                bundle_price = sum(self.current_prices[item_id] for item_id in bidder_config.interested_items)
                wants_to_buy = bundle_price <= valuation
                
                if wants_to_buy:
                    # Add to demand for each item in the bundle
                    for item_id in bidder_config.interested_items:
                        item_demands[item_id].append(bidder_id)
            
            bidder_decisions.append(wants_to_buy)
        
        return bidder_decisions, item_demands
    
    def _check_market_clearing(self, item_demands: Dict[int, List[int]]) -> Tuple[bool, Dict]:
        """Check if supply equals demand for all items (market clearing condition)"""
        allocation_result = {}
        
        for item_id in range(self.config.num_items):
            demanding_bidders = item_demands.get(item_id, [])
            supply = 1  # One unit of each item available
            demand = len(demanding_bidders)
            
            if demand == supply:
                # Perfect match: exactly one bidder wants this item
                allocation_result[item_id] = demanding_bidders[0]
            else:
                # Either no demand (demand=0) or excess demand (demand>1)
                return False, {}
        
        # Market clears if all items have exactly one buyer
        return True, allocation_result
    
    def _calculate_revenue(self, allocation_result: Dict[int, int]) -> float:
        """Calculate total revenue from allocation"""
        revenue = 0
        for item_id, winner_bidder_id in allocation_result.items():
            if winner_bidder_id is not None:
                revenue += self.current_prices[item_id]
        return revenue
    
    def _get_info(self) -> Dict:
        """Get episode information"""
        # Calculate current decisions if prices are set
        if any(p > 0 for p in self.current_prices):
            decisions, item_demands = self._calculate_demand()
            is_clearing, allocation = self._check_market_clearing(item_demands)
        else:
            decisions = [False] * len(self.config.bidder_configs)
            item_demands = {i: [] for i in range(self.config.num_items)}
            is_clearing = False
            allocation = {}
        
        return {
            'round': self.round_number,
            'valuations': self.bidder_valuations.copy(),
            'prices': self.current_prices.copy(),
            'decisions': decisions,
            'item_demands': item_demands,
            'market_clearing': is_clearing,
            'allocation_result': allocation if is_clearing else {},
            'successful_allocation': self.successful_allocation,
            'revenue': self.total_revenue,
            'max_rounds': self.config.max_rounds,
            'allocation_type': self._get_allocation_type(decisions, is_clearing, allocation)
        }
    
    def _get_allocation_type(self, decisions: List[bool], is_clearing: bool, allocation: Dict) -> str:
        """Classify the type of allocation"""
        if not is_clearing:
            if not any(decisions):
                return "no_demand"
            else:
                return "excess_demand"
        
        # Market clearing case - check who won
        winners = set(allocation.values())
        
        # Find global bidder (bidder with multiple interested items)
        global_bidder = None
        for i, bidder_config in enumerate(self.config.bidder_configs):
            if len(bidder_config.interested_items) > 1:
                global_bidder = i
                break
        
        if global_bidder is not None and global_bidder in winners:
            # Count how many items global bidder won
            global_wins = sum(1 for winner in allocation.values() if winner == global_bidder)
            if global_wins == len(self.config.bidder_configs[global_bidder].interested_items):
                return "global_wins"
        
        # Check if locals split the items
        local_bidders = [i for i, cfg in enumerate(self.config.bidder_configs) if len(cfg.interested_items) == 1]
        if all(bidder in winners for bidder in local_bidders):
            return "local_split"
        
        return "partial_allocation"

def test_standard_llg_environment():
    """Test the standard LLG environment"""
    print("Testing Standard LLG Environment")
    print("=" * 40)
    
    config = get_standard_llg_config()
    env = StandardLLGAuctionEnv(config)
    
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    print(f"Price options: {env.price_options}")
    
    # Test a few episodes
    for episode in range(3):
        print(f"\nEpisode {episode + 1}:")
        obs, info = env.reset(seed=42 + episode)
        print(f"  Valuations: {info['valuations']}")
        
        # Test different actions
        test_actions = [[4, 4], [3, 3], [5, 5]]  # $5+$5, $4+$4, $6+$6
        
        for action in test_actions:
            env.reset(seed=42 + episode)  # Reset to same state
            obs, reward, terminated, truncated, info = env.step(action)
            
            prices = [env.price_options[action[i]] for i in range(2)]
            print(f"    Action {action} → Prices ${prices[0]}+${prices[1]}=${sum(prices)}: "
                  f"Success={info['successful_allocation']}, Type={info['allocation_type']}")

if __name__ == "__main__":
    test_standard_llg_environment()