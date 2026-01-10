import numpy as np
from dataclasses import dataclass
from typing import List, Set
import re
from collections import defaultdict

@dataclass
class CATSBidder:
    """Represents a single-minded bidder from CATS"""
    bidder_id: int
    valuation: float
    bundle: Set[int]  # Set of item indices (0-indexed)
    
    def __repr__(self):
        items_str = "{" + ", ".join(str(i) for i in sorted(self.bundle)) + "}"
        return f"Bidder {self.bidder_id}: ${self.valuation:.2f} for bundle {items_str}"

class CATSParser:
    """Parse CATS auction files"""
    
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.num_goods = None
        self.num_bids = None
        self.dummy_good_id = None
        self.bidders = []
        
    def parse(self) -> List[CATSBidder]:
        """Parse the CATS file and return list of bidders"""
        with open(self.filepath, 'r') as f:
            lines = f.readlines()
        
        # Parse header
        for line in lines:
            line = line.strip()
            
            # Skip comments and empty lines
            if line.startswith('%') or line.startswith('//') or not line:
                continue
            
            # Parse metadata
            if line.startswith('goods'):
                self.num_goods = int(line.split()[1])
            elif line.startswith('bids'):
                self.num_bids = int(line.split()[1])
            elif line.startswith('dummy'):
                self.dummy_good_id = int(line.split()[1])
            else:
                # Parse bid line
                self._parse_bid_line(line)
        
        print(f"Parsed {len(self.bidders)} bids from CATS file")
        
        # Count unique bidders
        unique_bidders = len(set(b.bidder_id for b in self.bidders))
        print(f"Unique bidders: {unique_bidders}")
        print(f"Goods: {self.num_goods}, Dummy good ID: {self.dummy_good_id}")
        
        return self.bidders
    
    def _parse_bid_line(self, line: str):
        """Parse a single bid line"""
        parts = line.replace('#', '').strip().split()
        
        if len(parts) < 2:
            return
        
        try:
            bid_id = int(parts[0])
            valuation = float(parts[1])
            raw_items = [int(x) for x in parts[2:]]
            
            # Separate real items from dummy goods
            bundle = set()
            dummy_goods = []
            
            for item in raw_items:
                if item < self.num_goods:
                    bundle.add(item)  # Real item
                else:
                    dummy_goods.append(item)  # Dummy good (bidder ID)
            
            # Determine bidder ID
            if len(dummy_goods) == 1:
                # Multi-minded bidder: use dummy good as bidder ID
                bidder_id = dummy_goods[0]
            elif len(dummy_goods) == 0:
                # Single-minded bidder: use negative bid_id to avoid collision with dummy goods
                bidder_id = -bid_id
            else:
                # Multiple dummy goods - skip this bid (shouldn't happen)
                return
            
            if len(bundle) > 0:
                bidder = CATSBidder(
                    bidder_id=bidder_id,
                    valuation=valuation,
                    bundle=bundle
                )
                self.bidders.append(bidder)
                
        except (ValueError, IndexError) as e:
            print(f"Warning: Could not parse line: {line[:50]}... Error: {e}")
    
    def sample_bidders(self, n: int, seed: int = None) -> List[CATSBidder]:
        """
        Sample n bidders randomly, picking one bid per bidder
        
        Args:
            n: Number of unique bidders to sample
            seed: Random seed for reproducibility
            
        Returns:
            List of n sampled bidders (one bid per bidder)
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Group bids by bidder_id
        bidder_groups = defaultdict(list)
        for bidder in self.bidders:
            bidder_groups[bidder.bidder_id].append(bidder)
        
        # Get unique bidder IDs
        unique_bidders = list(bidder_groups.keys())
        
        if n > len(unique_bidders):
            print(f"Warning: Requested {n} bidders but only {len(unique_bidders)} unique bidders available")
            n = len(unique_bidders)
        
        # Sample n unique bidders
        sampled_bidder_ids = np.random.choice(unique_bidders, size=n, replace=False)
        
        # For each sampled bidder, pick their first bid
        sampled = []
        for bidder_id in sampled_bidder_ids:
            bids = bidder_groups[bidder_id]
            chosen_bid = bids[0]  # CHANGED: Always pick first bid instead of random
            sampled.append(chosen_bid)
        
        return sampled

    def get_max_price(self) -> float:
        """Calculate max price from all bidders in file"""
        if not self.bidders:
            raise ValueError("No bidders parsed yet")
        
        # Group by bidder to avoid double-counting multi-minded bidders
        bidder_groups = defaultdict(list)
        for bidder in self.bidders:
            bidder_groups[bidder.bidder_id].append(bidder)
        
        # Get max per-item valuation across all bidders
        max_per_item = 0
        for bidder_id, bids in bidder_groups.items():
            # For multi-minded bidders, take first bid
            bid = bids[0]
            per_item_val = bid.valuation / len(bid.bundle)
            max_per_item = max(max_per_item, per_item_val)
        
        return 1.1 * max_per_item
    
    def get_statistics(self):
        """Print statistics about the parsed bidders"""
        if not self.bidders:
            print("No bidders parsed yet")
            return
        
        # Group by bidder to get unique bidder stats
        bidder_groups = defaultdict(list)
        for bidder in self.bidders:
            bidder_groups[bidder.bidder_id].append(bidder)
        
        # For statistics, take one random bid per multi-minded bidder
        # This represents the TRUE bidder distribution
        representative_bidders = []
        for bidder_id, bids in bidder_groups.items():
            # Pick one random bid from this bidder's substitutes
            chosen_bid = bids[np.random.randint(len(bids))]
            representative_bidders.append(chosen_bid)
        
        # Separate single-minded and multi-minded bidders
        single_minded = [b for b in representative_bidders if b.bidder_id < 0]
        multi_minded = [b for b in representative_bidders if b.bidder_id >= 0]
        
        valuations = [b.valuation for b in representative_bidders]
        bundle_sizes = [len(b.bundle) for b in representative_bidders]
        bids_per_bidder = [len(bids) for bids in bidder_groups.values()]
        
        print(f"\n=== CATS Dataset Statistics ===")
        print(f"Total bids in file: {len(self.bidders)}")
        print(f"Unique bidders: {len(bidder_groups)}")
        print(f"  Single-minded bidders: {len(single_minded)}")
        print(f"  Multi-minded bidders: {len(multi_minded)}")
        print(f"Number of goods: {self.num_goods}")
        print(f"\nBids per bidder:")
        print(f"  Min: {min(bids_per_bidder)}")
        print(f"  Max: {max(bids_per_bidder)}")
        print(f"  Mean: {np.mean(bids_per_bidder):.1f}")
        print(f"  Median: {np.median(bids_per_bidder):.0f}")
        print(f"\nValuations (one per bidder):")
        print(f"  Min: ${min(valuations):.2f}")
        print(f"  Max: ${max(valuations):.2f}")
        print(f"  Mean: ${np.mean(valuations):.2f}")
        print(f"  Median: ${np.median(valuations):.2f}")
        print(f"\nBundle sizes (one per bidder):")
        print(f"  Min: {min(bundle_sizes)} items")
        print(f"  Max: {max(bundle_sizes)} items")
        print(f"  Mean: {np.mean(bundle_sizes):.1f} items")
        print(f"  Median: {np.median(bundle_sizes):.0f} items")


def test_cats_parser(n_bidders: int = 3, seed: int = 42):
    """
    Test the CATS parser
    
    Args:
        n_bidders: Number of unique bidders to sample (default: 3)
        seed: Random seed for reproducibility (default: 42)
    """
    print("Testing CATS Parser")
    print("=" * 50)
    
    # Parse the file
    parser = CATSParser('0000.txt')  # Your CATS file
    all_bidders = parser.parse()
    
    # Show statistics
    parser.get_statistics()
    
    # Sample bidders
    print(f"\n=== Sampling {n_bidders} Unique Bidders ===")
    sampled_bidders = parser.sample_bidders(n=n_bidders, seed=seed)
    
    for bidder in sampled_bidders:
        print(f"  {bidder}")
    
    # Verify no dummy goods in bundles
    print(f"\n=== Verification ===")
    if sampled_bidders and any(len(b.bundle) > 0 for b in sampled_bidders):
        max_item = max(max(b.bundle) for b in sampled_bidders if len(b.bundle) > 0)
        print(f"Maximum item ID in sampled bundles: {max_item}")
        print(f"Number of goods: {parser.num_goods}")
        print(f"All items < num_goods: {max_item < parser.num_goods}")
        
        # Check for duplicate bidders
        bidder_ids = [b.bidder_id for b in sampled_bidders]
        print(f"Sampled bidder IDs: {bidder_ids}")
        print(f"All unique: {len(bidder_ids) == len(set(bidder_ids))}")
    
    return parser, sampled_bidders


if __name__ == "__main__":
    parser, sampled = test_cats_parser()