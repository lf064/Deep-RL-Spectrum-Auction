import numpy as np
from dataclasses import dataclass
from typing import List, Set, Tuple

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
            self.max_price = 1.5 * (max(
                b.valuation / len(b.interested_items) 
                for b in self.bidder_configs
            ))


# ---------------------------------------------------------------------------
# Subgradient clearability check (moved from clearable.py)
# ---------------------------------------------------------------------------

def _check_clearability_subgradient(
    bidder_valuations: List[float],
    bidder_bundles: List[Set[int]],
    num_items: int,
    min_price: float,
    max_price: float,
    num_stepsizes: int = 100,
    max_rounds: int = 200
) -> bool:
    """
    Test whether a single auction instance is clearable via subgradient search.
    
    Pure function — takes instance data directly, no env object needed.
    This lets it run during config-level pre-filtering before any env exists.
    
    Mirrors AuctionStandard.m: tries num_stepsizes step sizes; for each one
    runs subgradient ascent on the dual. If ANY stepsize finds clearing prices,
    the instance is clearable.
    
    The clearing check here is an inline copy of the Bayesian two-stage logic
    (Stage 1: exact match, Stage 2: zero out truly undemanded items) so that
    the definition of "clearable" is exactly consistent with what the env uses.
    """
    max_valuation = max(bidder_valuations)

    for i in range(1, num_stepsizes + 1):
        stepsize = (max_valuation / num_stepsizes) * i

        prices = np.random.uniform(min_price, max_price / 2, num_items).astype(np.float32)

        for round_num in range(max_rounds):
            # --- inline demand calculation ---
            item_demands = {iid: [] for iid in range(num_items)}
            for bid_id, (val, bundle) in enumerate(zip(bidder_valuations, bidder_bundles)):
                bundle_cost = sum(prices[iid] for iid in bundle)
                if bundle_cost <= val:
                    for iid in bundle:
                        item_demands[iid].append(bid_id)

            # --- inline Bayesian two-stage clearing check ---
            # Stage 1: strict supply == demand
            allocation = {}
            has_excess = False
            for iid in range(num_items):
                demanders = item_demands[iid]
                if len(demanders) > 1:
                    has_excess = True
                elif len(demanders) == 1:
                    allocation[iid] = demanders[0]

            if len(allocation) == num_items and not has_excess:
                winning_bidders = set(allocation.values())
                consistent = all(
                    {iid for iid, w in allocation.items() if w == bid} == bidder_bundles[bid]
                    for bid in winning_bidders
                )
                if consistent:
                    return True

            # Stage 2: zero out truly undemanded items and recheck
            truly_undemanded = [
                iid for iid in range(num_items)
                if not any(iid in bundle for bundle in bidder_bundles)
            ]
            if truly_undemanded:
                adj_prices = prices.copy()
                for iid in truly_undemanded:
                    adj_prices[iid] = 0.0

                adj_demands = {iid: [] for iid in range(num_items)}
                for bid_id, (val, bundle) in enumerate(zip(bidder_valuations, bidder_bundles)):
                    cost = sum(adj_prices[iid] for iid in bundle)
                    if cost <= val:
                        for iid in bundle:
                            adj_demands[iid].append(bid_id)

                adj_alloc = {}
                adj_excess = False
                for iid in range(num_items):
                    if iid in truly_undemanded:
                        continue
                    demanders = adj_demands[iid]
                    if len(demanders) > 1:
                        adj_excess = True
                    elif len(demanders) == 1:
                        adj_alloc[iid] = demanders[0]

                if not adj_excess and len(adj_alloc) > 0:
                    winning_bidders = set(adj_alloc.values())
                    consistent = all(
                        {iid for iid, w in adj_alloc.items() if w == bid} == bidder_bundles[bid]
                        for bid in winning_bidders
                    )
                    if consistent:
                        return True

            # --- subgradient update ---
            demand_vec = np.array(
                [len(item_demands[iid]) for iid in range(num_items)], dtype=np.float32
            )
            supply_vec = np.ones(num_items, dtype=np.float32)
            gradient = demand_vec - supply_vec

            lr = stepsize / np.sqrt(round_num + 1)
            prices = np.clip(prices + lr * gradient, min_price, max_price)

    return False


# ---------------------------------------------------------------------------
# Parser cache + clearable seed selection
# ---------------------------------------------------------------------------

# Only the parser is cached — avoids re-reading the CATS file from disk
# on every reset(). Everything else runs fresh each time.
_parser_cache = {}

def _get_parser(filepath: str):
    """Return a parsed CATSParser for filepath, caching it after first load."""
    global _parser_cache
    if filepath not in _parser_cache:
        from cats_parser import CATSParser
        parser = CATSParser(filepath)
        parser.parse()
        _parser_cache[filepath] = parser
        print(f"[cont_config] Parsed {filepath} ({parser.num_goods} goods)")
    return _parser_cache[filepath]


def get_clearable_seed(
    filepath: str,
    num_bidders: int,
    seed_range: Tuple[int, int] = (0, 1000)
) -> int:
    """
    Sample random seeds until one passes the subgradient clearability check.
    
    Called by env.reset() each episode:
      1. Pick a random seed from seed_range
      2. Sample bidders at that seed
      3. Run subgradient check — is this instance clearable?
      4. If yes → return the seed. If no → go back to 1.
    
    With ~70% clearable rate this loops ~1.4 times on average.
    The parser is cached so the CATS file is only read once per process.
    """
    parser = _get_parser(filepath)
    num_items = parser.num_goods
    max_price = parser.get_max_price()

    while True:
        seed = np.random.randint(seed_range[0], seed_range[1])
        sampled = parser.sample_bidders(n=num_bidders, seed=seed)

        valuations = [b.valuation for b in sampled]
        bundles = [b.bundle for b in sampled]

        if _check_clearability_subgradient(
            bidder_valuations=valuations,
            bidder_bundles=bundles,
            num_items=num_items,
            min_price=0.0,
            max_price=max_price
        ):
            return seed


# ---------------------------------------------------------------------------
# Config creation (unchanged public interface)
# ---------------------------------------------------------------------------

def create_cats_config_from_file(
    filepath: str, 
    num_bidders: int = DEFAULT_NUM_BIDDERS,
    seed: int = None,
    max_rounds: int = 100
) -> CATSAuctionConfig:
    """
    Create a CATS auction config by sampling bidders from a CATS file.
    """
    from cats_parser import CATSParser
    
    parser = CATSParser(filepath)
    parser.parse()
    
    max_price_from_file = parser.get_max_price()
    sampled_bidders = parser.sample_bidders(n=num_bidders, seed=seed)
    
    bidder_configs = []
    for cats_bidder in sampled_bidders:
        bidder_configs.append(CATSBidderConfig(
            bidder_id=cats_bidder.bidder_id,
            valuation=cats_bidder.valuation,
            interested_items=sorted(list(cats_bidder.bundle)),
        ))
    
    if len(bidder_configs) == 0:
        raise ValueError("No bidders found after sampling")
    
    return CATSAuctionConfig(
        bidder_configs=bidder_configs,
        num_items=parser.num_goods,
        max_rounds=max_rounds,
        max_price=max_price_from_file
    )


def create_cats_config_manual(
    bidder_configs: List[CATSBidderConfig],
    num_items: int,
    max_rounds: int = 100,
    min_price: float = 0.0,
    max_price: float = None
) -> CATSAuctionConfig:
    """Create a CATS auction config manually (useful for test instances)."""
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
    seed: int = 42
) -> CATSAuctionConfig:
    """
    Convenience function to get a CATS config for a specific seed.
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
        if i >= 9:
            remaining = len(config.bidder_configs) - 10
            if remaining > 0:
                print(f"  ... and {remaining} more bidders")
            break


def test_cats_config():
    """Test the CATS configuration"""
    print("Testing CATS Configuration")
    print("=" * 60)
    
    print("\n### Test 1: Loading from CATS file ###")
    config = get_cats_config('0000.txt', num_bidders=DEFAULT_NUM_BIDDERS, seed=79)
    list_cats_config(config)


if __name__ == "__main__":
    test_cats_config()