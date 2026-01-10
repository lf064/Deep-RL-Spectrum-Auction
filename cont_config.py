import numpy as np
from dataclasses import dataclass
from typing import List, Set
from collections import defaultdict

# Default configuration values
DEFAULT_NUM_BIDDERS = 3

@dataclass
class CATSBidderConfig:
    """Configuration for a CATS bidder (single-minded)"""
    bidder_id: int
    valuation: float
    interested_items: List[int]  # Bundle of items this bidder wants
    name: str = ""

    def __post_init__(self):
        """Auto-generate name if not provided"""
        if not self.name:
            items_str = "_".join(str(i) for i in sorted(self.interested_items))
            self.name = f"Bidder{self.bidder_id}_Items{items_str}"

@dataclass
class CATSAuctionConfig:
    """Configuration for CATS auction environment"""
    bidder_configs: List[CATSBidderConfig]
    num_items: int
    max_rounds: int = 100
    min_price: float = 0.0
    max_price: float = None  # Will be set based on max valuation
    
    def __post_init__(self):
        """Set max_price based on valuations if not specified"""
        if self.max_price is None:
            # Maximum any bidder would rationally pay per item
            self.max_price = 1.5 * (max(
                b.valuation / len(b.interested_items) 
                for b in self.bidder_configs
            ))

def create_cats_config_from_file(
    filepath: str, 
    num_bidders: int = DEFAULT_NUM_BIDDERS,
    seed: int = None,
    max_rounds: int = 100
) -> CATSAuctionConfig:
    """
    Create a CATS auction config by sampling bidders from a CATS file
    
    Args:
        filepath: Path to CATS .txt file
        num_bidders: Number of bidders to sample
        seed: Random seed for reproducibility
        max_rounds: Maximum auction rounds
        
    Returns:
        CATSAuctionConfig with sampled bidders (num_items determined from file)
    """
    from cats_parser import CATSParser
    
    # Parse CATS file
    parser = CATSParser(filepath)
    all_bidders = parser.parse()
    
    # Calculate max_price from FULL population (before sampling)
    max_price_from_file = parser.get_max_price()
    
    # Sample subset of bidders
    sampled_bidders = parser.sample_bidders(n=num_bidders, seed=seed)
    
    # Convert to config format
    bidder_configs = []
    for cats_bidder in sampled_bidders:
        config = CATSBidderConfig(
            bidder_id=cats_bidder.bidder_id,
            valuation=cats_bidder.valuation,
            interested_items=sorted(list(cats_bidder.bundle)),
        )
        bidder_configs.append(config)
    
    if len(bidder_configs) == 0:
        raise ValueError("No bidders found after sampling")
    
    return CATSAuctionConfig(
        bidder_configs=bidder_configs,
        num_items=parser.num_goods,
        max_rounds=max_rounds,
        max_price=max_price_from_file  # CHANGED: Use full population max instead of None
    )

def create_cats_config_manual(
    bidder_configs: List[CATSBidderConfig],
    num_items: int,
    max_rounds: int = 100,
    min_price: float = 0.0,
    max_price: float = None
) -> CATSAuctionConfig:
    """
    Create a CATS auction config manually
    
    Useful for creating custom test instances
    """
    return CATSAuctionConfig(
        bidder_configs=bidder_configs,
        num_items=num_items,
        max_rounds=max_rounds,
        min_price=min_price,
        max_price=max_price
    )

def get_cats_config(
    filepath: str = '0000.txt',
    num_bidders: int = DEFAULT_NUM_BIDDERS,
    seed: int = 42 #42 for unallocated item test
) -> CATSAuctionConfig:
    """
    Convenience function to get a CATS config
    
    Args:
        filepath: Path to CATS file
        num_bidders: Number of bidders to sample
        seed: Random seed
        
    Returns:
        CATSAuctionConfig (num_items determined from file)
    """
    return create_cats_config_from_file(filepath, num_bidders, seed)

def list_cats_config(config: CATSAuctionConfig):
    """List CATS configuration details"""
    print("CATS Auction Configuration:")
    print("=" * 60)
    print(f"Items: {config.num_items}")
    print(f"Bidders: {len(config.bidder_configs)}")
    print(f"Max rounds: {config.max_rounds}")
    print(f"Price range: ${config.min_price:.2f} - ${config.max_price:.2f}")
    print()
    
    # Calculate statistics
    valuations = [b.valuation for b in config.bidder_configs]
    bundle_sizes = [len(b.interested_items) for b in config.bidder_configs]
    
    print("Valuation Statistics:")
    print(f"  Min: ${min(valuations):.2f}")
    print(f"  Max: ${max(valuations):.2f}")
    print(f"  Mean: ${np.mean(valuations):.2f}")
    print(f"  Median: ${np.median(valuations):.2f}")
    print()
    
    print("Bundle Size Statistics:")
    print(f"  Min: {min(bundle_sizes)} items")
    print(f"  Max: {max(bundle_sizes)} items")
    print(f"  Mean: {np.mean(bundle_sizes):.1f} items")
    print(f"  Median: {np.median(bundle_sizes):.0f} items")
    print()
    
    # Show per-item valuation range (useful for understanding price bounds)
    per_item_vals = [b.valuation / len(b.interested_items) for b in config.bidder_configs]
    print("Per-Item Valuation Statistics:")
    print(f"  Min: ${min(per_item_vals):.2f}")
    print(f"  Max: ${max(per_item_vals):.2f} ")
    print(f"  Mean: ${np.mean(per_item_vals):.2f}")
    print(f"  Median: ${np.median(per_item_vals):.2f}")
    print()

    
    print("Bidders:")
    for i, bidder in enumerate(config.bidder_configs):
        items_str = "{" + ", ".join(str(x) for x in bidder.interested_items) + "}"
        per_item = bidder.valuation / len(bidder.interested_items)
        print(f"  {i+1}. Bidder {bidder.bidder_id}: ${bidder.valuation:.2f} for {items_str} (${per_item:.2f}/item)")
        if i >= 9:  # Only show first 10 bidders
            remaining = len(config.bidder_configs) - 10
            if remaining > 0:
                print(f"  ... and {remaining} more bidders")
            break

def test_cats_config():
    """Test the CATS configuration"""
    print("Testing CATS Configuration")
    print("=" * 60)
    
    # Test 1: Load from file
    print("\n### Test 1: Loading from CATS file ###")
    config = get_cats_config('0000.txt', num_bidders=DEFAULT_NUM_BIDDERS, seed=14)
    list_cats_config(config)


if __name__ == "__main__":
    test_cats_config()