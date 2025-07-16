import numpy as np
from dataclasses import dataclass
from typing import List

@dataclass
class ContinuousLLGBidderConfig:
    """Configuration for continuous LLG bidders with normal distributions"""
    valuation_mean: float
    valuation_std: float
    interested_items: List[int]  # Which items this bidder wants (0=Item A, 1=Item B, etc.)
    name: str = ""

@dataclass
class ContinuousLLGAuctionConfig:
    """Configuration for continuous LLG auction environment"""
    bidder_configs: List[ContinuousLLGBidderConfig]
    num_items: int = 2
    max_rounds: int = 100
    global_std: float = 1.0  # Standard deviation for global bidder

def create_continuous_llg_config(global_std=1.0):
    """
    Create continuous LLG configuration with normal distributions
    
    Setup:
    - Local 1: wants only Item A (item 0), values at N(4, 0.1²) 
    - Local 2: wants only Item B (item 1), values at N(4, 0.1²)
    - Global: wants both items (items 0,1), values bundle at N(10, global_std²)
    
    Features:
    - Normal distribution valuations (rounded to 1 decimal)
    - Continuous action space for pricing
    - Adjustable global bidder uncertainty via global_std
    
    Args:
        global_std: Standard deviation for global bidder valuation (1, 5, 10, 15, 20)
    """
    return ContinuousLLGAuctionConfig(
        bidder_configs=[
            ContinuousLLGBidderConfig(
                valuation_mean=4.0,
                valuation_std=0.1,  # Fixed low std for locals
                interested_items=[0],  # Only Item A (index 0)
                name="Local1_ItemA"
            ),
            ContinuousLLGBidderConfig(
                valuation_mean=4.0,
                valuation_std=0.1,  # Fixed low std for locals
                interested_items=[1],  # Only Item B (index 1)
                name="Local2_ItemB"
            ),
            ContinuousLLGBidderConfig(
                valuation_mean=10.0,
                valuation_std=global_std,  # Adjustable std for global
                interested_items=[0, 1],  # Both items A and B (bundle)
                name="Global_Both"
            )
        ],
        num_items=2,
        max_rounds=100,
        global_std=global_std  # Store for reference
    )

# Predefined configurations for common standard deviations
CONTINUOUS_LLG_CONFIGS = {
    'std_1': lambda: create_continuous_llg_config(global_std=1.0),
    'std_5': lambda: create_continuous_llg_config(global_std=5.0),
    'std_10': lambda: create_continuous_llg_config(global_std=10.0),
    'std_15': lambda: create_continuous_llg_config(global_std=15.0),
    'std_20': lambda: create_continuous_llg_config(global_std=20.0)
}

def get_continuous_llg_config(config_name='std_1', global_std=None):
    """
    Get continuous LLG configuration
    
    Args:
        config_name: Predefined config ('std_1', 'std_5', 'std_10', 'std_15', 'std_20')
        global_std: Override global standard deviation (takes precedence over config_name)
    
    Returns:
        ContinuousLLGAuctionConfig
    """
    if global_std is not None:
        # Direct specification overrides config_name
        return create_continuous_llg_config(global_std=global_std)
    
    if config_name not in CONTINUOUS_LLG_CONFIGS:
        available = list(CONTINUOUS_LLG_CONFIGS.keys())
        raise ValueError(f"Unknown config '{config_name}'. Available: {available}")
    
    return CONTINUOUS_LLG_CONFIGS[config_name]()

def list_continuous_llg_configs():
    """List all available continuous LLG configurations"""
    print("Continuous LLG Configurations:")
    print("=" * 45)
    
    for config_name in CONTINUOUS_LLG_CONFIGS.keys():
        config = get_continuous_llg_config(config_name)
        std_val = config.global_std
        
        print(f"\n{config_name.upper()}: (global_std={std_val})")
        print(f"  Items: {config.num_items}")
        print(f"  Max rounds: {config.max_rounds}")
        print(f"  Action space: Continuous Box([0.1, 15.0] x [0.1, 15.0])")
        print(f"  Bidders:")
        
        for bidder in config.bidder_configs:
            items_str = ", ".join([f"Item {chr(65+item)}" for item in bidder.interested_items])
            print(f"    {bidder.name}: {items_str}, valuation N({bidder.valuation_mean}, {bidder.valuation_std}²)")
    


if __name__ == "__main__":
    list_continuous_llg_configs()
