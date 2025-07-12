import numpy as np
from dataclasses import dataclass
from typing import List

@dataclass
class StandardLLGBidderConfig:
    """Configuration for standard LLG bidders"""
    valuation_low: float
    valuation_high: float
    interested_items: List[int]  # Which items this bidder wants (0=Item A, 1=Item B, etc.)
    name: str = ""

@dataclass
class StandardLLGAuctionConfig:
    """Configuration for standard LLG auction environment"""
    bidder_configs: List[StandardLLGBidderConfig]
    num_items: int = 2
    max_rounds: int = 20
    price_options: List[float] = None

def create_standard_llg_config():
    """
    Create the standard LLG configuration
    
    Setup:
    - Local 1: wants only Item A (item 0), values at $4 (fixed)
    - Local 2: wants only Item B (item 1), values at $4 (fixed)
    - Global: wants both items (items 0,1), values bundle at $4 or $10 (random each episode)
    
    Efficient allocation: Global should win both items when valuation is $10
    Market clearing: Supply=1 per item, demand must equal 1 per item
    """
    return StandardLLGAuctionConfig(
        bidder_configs=[
            StandardLLGBidderConfig(
                valuation_low=4, 
                valuation_high=4,  # Fixed at $4
                interested_items=[0],  # Only Item A (index 0)
                name="Local1_ItemA"
            ),
            StandardLLGBidderConfig(
                valuation_low=4, 
                valuation_high=4,  # Fixed at $4
                interested_items=[1],  # Only Item B (index 1)
                name="Local2_ItemB"
            ),
            StandardLLGBidderConfig(
                valuation_low=4,   # Can be $4
                valuation_high=10,  # or $10 (random each episode)
                interested_items=[0, 1],  # Both items A and B (bundle)
                name="Global_Both"
            )
        ],
        num_items=2,
        max_rounds=20,
        price_options=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    )

def get_standard_llg_config():
    """Get standard LLG configuration"""
    return create_standard_llg_config()

def list_standard_llg_config():
    """List standard LLG configuration details"""
    config = get_standard_llg_config()
    
    print("Standard LLG Configuration:")
    print("=" * 40)
    print(f"Items: {config.num_items}")
    print(f"Max rounds: {config.max_rounds}")
    print(f"Price options: {config.price_options}")
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
    list_standard_llg_config()