import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple
from LLG_std_config import ContinuousLLGAuctionConfig, get_continuous_llg_config

class SimpleContinuousLLGEnv(gym.Env):
    """
    Simple Continuous LLG Environment - Clean Implementation
    
    Features:
    - Normal distribution valuations: Locals N(4, 0.1²), Global N(10, global_std²)
    - Continuous action space: Direct price setting
    - Clear observation encoding: 0=neutral, 1=reject, 2=accept
    """
    def _denormalize_action(self, action_normalized: float) -> float:
        """Convert normalized action [-1, 1] to actual price range [min_price, max_price]."""
        action_clipped = np.clip(action_normalized, -1.0, 1.0)
        return self.min_price + (action_clipped + 1.0) / 2.0 * (self.max_price - self.min_price)

    
    def __init__(self, config: ContinuousLLGAuctionConfig):
        super().__init__()
        self.config = config
        
        # Price bounds - reasonable for LLG domain
        self.min_price = 2.0
        self.max_price = 8.0
        
        # Action space: direct price setting for each item
        self.action_space = spaces.Box(
            low=-1,
            high=1,
            shape=(config.num_items,),
            dtype=np.float32
        )
        
        # Observation space: decision for each bidder (0=neutral, 1=reject, 2=accept)
        self.observation_space = spaces.MultiDiscrete([3] * len(config.bidder_configs))
        
        # State variables
        self.bidder_valuations = []
        self.current_prices = []
        self.round_number = 0
        
    def reset(self, seed=None, options=None):
        """Reset for new episode"""
        super().reset(seed=seed)
        
        # Sample valuations from normal distributions
        self.bidder_valuations = []
        for bidder_config in self.config.bidder_configs:
            val = self.np_random.normal(bidder_config.valuation_mean, bidder_config.valuation_std)
            val = max(0.0, val)  # Ensure non-negative
            self.bidder_valuations.append(round(val, 1))  # Round to 1 decimal
        
        self.current_prices = [0.0] * self.config.num_items
        self.round_number = 0
        
        # Initial observation: all neutral (0)
        obs = np.array([0] * len(self.config.bidder_configs), dtype=np.int32)
        return obs, self._get_info()
    
    def step(self, action):
        """Execute one step"""
        # Set prices from action (clip to valid range)
        self.current_prices = [
            self._denormalize_action(float(action[i]))
            for i in range(self.config.num_items)
        ]

        
        self.round_number += 1
        
        # Calculate bidder decisions
        bidder_decisions = self._get_bidder_decisions()
        
        # Calculate item demands
        item_demands = self._get_item_demands(bidder_decisions)
        
        # Check if market clears
        market_clears, allocation = self._check_market_clearing(item_demands)
        
        # Create observation
        obs = np.array([2 if decision else 1 for decision in bidder_decisions], dtype=np.int32)
        
        # Calculate reward
        if market_clears:
            reward = 10.0
            terminated = True
        else:
            reward = -1.0
            terminated = False
        
        # Check max rounds
        if self.round_number >= self.config.max_rounds:
            terminated = True
        
        return obs, reward, terminated, False, self._get_info()
    
    def _get_bidder_decisions(self) -> List[bool]:
        """Get each bidder's decision (accept/reject)"""
        decisions = []
        
        for i, (bidder_config, valuation) in enumerate(zip(self.config.bidder_configs, self.bidder_valuations)):
            if len(bidder_config.interested_items) == 1:
                # Local bidder: single item
                item_id = bidder_config.interested_items[0]
                price = self.current_prices[item_id]
                decision = price <= valuation
            else:
                # Global bidder: bundle
                bundle_price = sum(self.current_prices[item_id] for item_id in bidder_config.interested_items)
                decision = bundle_price <= valuation
            
            decisions.append(decision)
        
        return decisions
    
    def _get_item_demands(self, bidder_decisions: List[bool]) -> Dict[int, List[int]]:
        """Calculate demand for each item"""
        item_demands = {i: [] for i in range(self.config.num_items)}
        
        for bidder_id, (bidder_config, decision) in enumerate(zip(self.config.bidder_configs, bidder_decisions)):
            if decision:  # Bidder wants to buy
                for item_id in bidder_config.interested_items:
                    item_demands[item_id].append(bidder_id)
        
        return item_demands
    
    def _check_market_clearing(self, item_demands: Dict[int, List[int]]) -> Tuple[bool, Dict[int, int]]:
        """Check if market clears (supply = demand for all items)"""
        allocation = {}
        
        for item_id in range(self.config.num_items):
            demand = len(item_demands.get(item_id, []))
            supply = 1  # One unit per item
            
            if demand == supply:
                # Exactly one bidder wants this item
                allocation[item_id] = item_demands[item_id][0]
            else:
                # Market doesn't clear
                return False, {}
        
        return True, allocation
    
    def _get_info(self) -> Dict:
        """Get episode info"""
        if self.round_number > 0:
            decisions = self._get_bidder_decisions()
            item_demands = self._get_item_demands(decisions)
            market_clears, allocation = self._check_market_clearing(item_demands)
        else:
            decisions = [False] * len(self.config.bidder_configs)
            item_demands = {i: [] for i in range(self.config.num_items)}
            market_clears = False
            allocation = {}
        
        return {
            'round': self.round_number,
            'valuations': self.bidder_valuations.copy(),
            'prices': self.current_prices.copy(),
            'decisions': decisions,
            'item_demands': item_demands,
            'market_clearing': market_clears,
            'allocation': allocation,
            'revenue': sum(self.current_prices[item_id] for item_id in allocation.keys()) if market_clears else 0
        }