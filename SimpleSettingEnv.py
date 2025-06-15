import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple
from dataclasses import dataclass

@dataclass
class BidderConfig:
    """Configuration for a single bidder"""
    valuation_low: float
    valuation_high: float
    name: str = ""

@dataclass
class AuctionConfig:
    """Configuration for the auction environment"""
    bidder_configs: List[BidderConfig]
    max_rounds: int = 20
    price_options: List[float] = None

class MultiRoundAuctionEnv(gym.Env):
    """
    Multi-Round Auction Environment
    
    Observation: [bidder1_decision, bidder2_decision] where:
    - 0 = neutral (no decision yet)
    - 1 = reject (doesn't want to buy)  
    - 2 = accept (wants to buy)
    
    Reward: +10 for exactly 1 buyer (success), -1 otherwise
    """
    
    def __init__(self, config: AuctionConfig):
        super().__init__()
        self.config = config
        
        # Set default price options if not provided
        self.price_options = config.price_options or [1, 2, 3, 4, 5]
        
        # Spaces
        self.action_space = spaces.Discrete(len(self.price_options))
        self.observation_space = spaces.MultiDiscrete([3, 3])  # 0=neutral, 1=reject, 2=accept
        
        # State variables (initialized in reset)
        self.bidder_valuations = []
        self.round_number = 0
        self.successful_allocation = False
        self.total_revenue = 0
    
    def reset(self, seed=None, options=None):
        """Reset environment for new episode"""
        super().reset(seed=seed)
        
        # Sample fresh valuations for this episode
        self.bidder_valuations = [
            self.np_random.choice([cfg.valuation_low, cfg.valuation_high])
            for cfg in self.config.bidder_configs
        ]
        
        # Reset state
        self.round_number = 0
        self.successful_allocation = False
        self.total_revenue = 0
        
        return np.array([0, 0], dtype=np.int32), self._get_info()
    
    def step(self, action):
        """Execute one auction priceround"""
        # Convert action to price
        price = self.price_options[np.clip(action, 0, len(self.price_options) - 1)]     # np.clip(2, 0, 4) = 2  #   | action = 0 → price = self.price_options[0] = $1  -> index
        self.round_number += 1
        
        # Get bidder decisions
        decisions = [price <= val for val in self.bidder_valuations]
        num_buyers = sum(decisions)
        
        # Create observation: map True->2, False->1
        obs = np.array([2 if d else 1 for d in decisions], dtype=np.int32)    # for d in decisions loops through each boolean defined in decisions

        
        # Calculate reward and termination
        if num_buyers == 1:
            # Success: exactly one buyer
            reward = 10.0
            terminated = True
            self.successful_allocation = True
            self.total_revenue = price
        else:
            # Failure: 0 or 2 buyers
            reward = -1.0
            terminated = False
        
        # End episode if max rounds reached
        if self.round_number >= self.config.max_rounds:
            terminated = True
        
        return obs, reward, terminated, False, self._get_info()
    
    def _get_info(self):
        """Get episode information"""
        return {
            'round': self.round_number,
            'valuations': self.bidder_valuations.copy(),
            'successful_allocation': self.successful_allocation,
            'revenue': self.total_revenue,
            'max_rounds': self.config.max_rounds
        }

# Configuration factory functions
def create_multi_round_config():
    """Bidder1 {2,4} vs Bidder2 {1,3}, 20 rounds"""
    return AuctionConfig(
        bidder_configs=[
            BidderConfig(2, 4, "Bidder1_2_4"),
            BidderConfig(1, 3, "Bidder2_1_3")
        ],
        max_rounds=20,
        price_options=[1, 2, 3, 4, 5]
    )

def create_simple_price_config():
    """Bidder1 {1,2} vs Bidder2 {4,5}, 10 rounds"""
    return AuctionConfig(
        bidder_configs=[
            BidderConfig(1, 2, "Bidder1_1_2"),
            BidderConfig(4, 5, "Bidder2_4_5")
        ],
        max_rounds=10,
        price_options=[1, 2, 3, 4, 5]
    )