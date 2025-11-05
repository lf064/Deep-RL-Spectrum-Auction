import numpy as np
from dataclasses import dataclass
from typing import List

@dataclass
class ContinuousLLGBidderConfig:
    """Configuration for continuous LLG bidders"""
    valuation_low: float
    valuation_high: float
    interested_items: List[int]  # Which items this bidder wants (0=Item A, 1=Item B, etc.)
    name: str = ""

@dataclass
class ContinuousLLGAuctionConfig:
    """Configuration for continuous LLG auction environment"""
    bidder_configs: List[ContinuousLLGBidderConfig]
    num_items: int = 2
    max_rounds: int = 100
    min_price: float = 0.0
    max_price: float = 15.0  # Extended range for continuous pricing

def create_continuous_llg_config():
    """
    Create the continuous LLG configuration
    
    Setup:
    - Local 1: wants only Item A (item 0), values at $4 (fixed)
    - Local 2: wants only Item B (item 1), values at $4 (fixed)
    - Global: wants both items (items 0,1), values bundle at $4 or $10 (random each episode)
    
    Efficient allocation: Global should win both items when valuation is $10
    Market clearing: Supply=1 per item, demand must equal 1 per item
    """
    return ContinuousLLGAuctionConfig(
        bidder_configs=[
            ContinuousLLGBidderConfig(
                valuation_low=4, 
                valuation_high=4,  # Fixed at $4
                interested_items=[0],  # Only Item A (index 0)
                name="Local1_ItemA"
            ),
            ContinuousLLGBidderConfig(
                valuation_low=4, 
                valuation_high=4,  # Fixed at $4
                interested_items=[1],  # Only Item B (index 1)
                name="Local2_ItemB"
            ),
            ContinuousLLGBidderConfig(
                valuation_low=4,   # Can be $4
                valuation_high=10,  # or $10 (random each episode)
                interested_items=[0, 1],  # Both items A and B (bundle)
                name="Global_Both"
            )
        ],
        num_items=2,
        max_rounds=100,
        min_price=0.0,
        max_price=15.0
    )

def get_continuous_llg_config():
    """Get continuous LLG configuration"""
    return create_continuous_llg_config()

def list_continuous_llg_config():
    """List continuous LLG configuration details"""
    config = get_continuous_llg_config()
    
    print("Continuous LLG Configuration:")
    print("=" * 40)
    print(f"Items: {config.num_items}")
    print(f"Max rounds: {config.max_rounds}")
    print(f"Price range: ${config.min_price} - ${config.max_price}")
    print()
    print("Bidders:")
    for bidder in config.bidder_configs:
        items_str = ", ".join([f"Item {chr(65+item)}" for item in bidder.interested_items])
        if bidder.valuation_low == bidder.valuation_high:
            val_str = f"${bidder.valuation_low}"
        else:
            val_str = f"${{{bidder.valuation_low}, {bidder.valuation_high}}} (random)"
        print(f"  {bidder.name}: {items_str}, valuation {val_str}")

if __name__ == "__main__":
    list_continuous_llg_config()