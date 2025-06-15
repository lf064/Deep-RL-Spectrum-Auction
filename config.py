import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class SimpleBidderConfig:
    """Configuration for a single bidder with uniform valuation range"""
    valuation_low: float
    valuation_high: float
    name: str = ""

@dataclass
class SimpleAuctionConfig:
    """Configuration for the simple spectrum auction"""
    bidder_configs: List[SimpleBidderConfig]
    max_rounds: int = 5
    price_options: List[float] = None

# Configuration 1: Original multi-round config
def create_multi_round_config():
    """Create configuration for multi-round auction: Bidder1 {2,4} vs Bidder2 {1,3}"""
    return SimpleAuctionConfig(
        bidder_configs=[
            SimpleBidderConfig(valuation_low=2, valuation_high=4, name="Bidder1_2_4"),
            SimpleBidderConfig(valuation_low=1, valuation_high=3, name="Bidder2_1_3")
        ],
        max_rounds=10,
        price_options=[1, 2, 3, 4, 5]
    )

# Configuration 2: Simple price config
def create_simple_price_config():
    """Create configuration for simple price auction: Bidder1 {1,2} vs Bidder2 {4,5}"""
    return SimpleAuctionConfig(
        bidder_configs=[
            SimpleBidderConfig(valuation_low=1, valuation_high=2, name="Bidder1_1_2"),
            SimpleBidderConfig(valuation_low=4, valuation_high=5, name="Bidder2_4_5")
        ],
        max_rounds=10,
        price_options=[1, 2, 3, 4, 5]
    )

# Configuration registry
AUCTION_CONFIGS = {
    'multi_round': create_multi_round_config,
    'simple_price': create_simple_price_config
}

def get_config(config_name: str):
    """Get configuration by name"""
    if config_name not in AUCTION_CONFIGS:
        available = list(AUCTION_CONFIGS.keys())
        raise ValueError(f"Unknown config '{config_name}'. Available: {available}")
    
    return AUCTION_CONFIGS[config_name]()

def list_configs():
    """List all available configurations"""
    print("Available auction configurations:")
    
    print("\n1. 'multi_round':")
    config = create_multi_round_config()
    print(f"   - Bidder1: {{{config.bidder_configs[0].valuation_low}, {config.bidder_configs[0].valuation_high}}}")
    print(f"   - Bidder2: {{{config.bidder_configs[1].valuation_low}, {config.bidder_configs[1].valuation_high}}}")
    print(f"   - Max rounds: {config.max_rounds}")
    print(f"   - Price options: {config.price_options}")
    
    print("\n2. 'simple_price':")
    config = create_simple_price_config()
    print(f"   - Bidder1: {{{config.bidder_configs[0].valuation_low}, {config.bidder_configs[0].valuation_high}}}")
    print(f"   - Bidder2: {{{config.bidder_configs[1].valuation_low}, {config.bidder_configs[1].valuation_high}}}")
    print(f"   - Max rounds: {config.max_rounds}")
    print(f"   - Price options: {config.price_options}")







# Usage examples:
if __name__ == "__main__":
    # List all available configs
    list_configs()
    
    # Test both configs
    print("\n" + "="*50)
    print("TESTING CONFIGURATIONS")
    
    for config_name in ['multi_round', 'simple_price']:
        print(f"\n--- Testing {config_name} ---")
        config = get_config(config_name)
        
        # Show config details
        print(f"Config: {config_name}")
        print(f"Bidder1 valuations: {{{config.bidder_configs[0].valuation_low}, {config.bidder_configs[0].valuation_high}}}")
        print(f"Bidder2 valuations: {{{config.bidder_configs[1].valuation_low}, {config.bidder_configs[1].valuation_high}}}")
        print(f"Max rounds: {config.max_rounds}")
        print(f"Price options: {config.price_options}")
        
        # Test scenarios for this config
        print("Possible scenarios:")
        for b1_val in [config.bidder_configs[0].valuation_low, config.bidder_configs[0].valuation_high]:
            for b2_val in [config.bidder_configs[1].valuation_low, config.bidder_configs[1].valuation_high]:
                print(f"  ({int(b1_val)}, {int(b2_val)})")