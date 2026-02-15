import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Set
from cont_config import CATSAuctionConfig, get_cats_config, get_clearable_seed

class CATSAuctionEnv(gym.Env):
    """
    CATS Combinatorial Auction Environment
    Single-minded bidders with continuous item pricing
    Bayesian-style two-stage clearing
    
    Resampling: when resample_bidders=True, each reset() draws a new seed
    via get_clearable_seed() — the filtering logic lives in cont_config,
    not here.
    """
    
    def __init__(self, config: CATSAuctionConfig,
                 resample_bidders: bool = False,
                 cats_filepath: str = None,
                 seed_range: tuple = (0, 1000)):
        super().__init__()
        self.config = config

        self.resample_bidders = resample_bidders
        self.cats_filepath = cats_filepath
        self.seed_range = seed_range          # forwarded to get_clearable_seed()
        
        assert len(config.bidder_configs) >= 1, "Need at least 1 bidder"
        assert config.num_items >= 1, "Need at least 1 item"

        # Action space: continuous price for each item
        self.action_space = spaces.Box(
            low=config.min_price,
            high=config.max_price,
            shape=(config.num_items,),
            dtype=np.float32
        )
        
        # Observation space: binary decision per bidder (0=reject, 1=accept)
        self.observation_space = spaces.MultiBinary(len(config.bidder_configs))
        
        # State variables
        self.round_number = 0
        self.successful_allocation = False
        self.total_revenue = 0
        self.current_prices = np.zeros(config.num_items, dtype=np.float32)
        
        # Store bidder valuations and bundles
        self.bidder_valuations = [b.valuation for b in config.bidder_configs]
        self.bidder_bundles = [set(b.interested_items) for b in config.bidder_configs]
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.resample_bidders:
            # Get a clearable seed — cont_config handles caching and filtering
            new_seed = get_clearable_seed(
                filepath=self.cats_filepath,
                num_bidders=len(self.config.bidder_configs),
                seed_range=self.seed_range
            )
            
            self.config = get_cats_config(
                filepath=self.cats_filepath,
                num_bidders=len(self.config.bidder_configs),
                seed=new_seed
            )
            self.bidder_valuations = [b.valuation for b in self.config.bidder_configs]
            self.bidder_bundles = [set(b.interested_items) for b in self.config.bidder_configs]

        # Reset state
        self.round_number = 0
        self.successful_allocation = False
        self.total_revenue = 0
        self.current_prices = np.zeros(self.config.num_items, dtype=np.float32)
        
        initial_obs = np.zeros(len(self.config.bidder_configs), dtype=np.int8)
        info = self._get_info()
        return initial_obs, info
    
    def step(self, action):
        """
        Execute one step in the auction.
        
        1. _check_market_clearing returns (is_clearing, allocation, final_prices)
        2. self.current_prices updated with final_prices (may be adjusted by Stage 2)
        3. Observation uses final_prices' decisions, not the agent's raw action
        """
        agent_prices = np.clip(action, self.config.min_price, self.config.max_price)
        self.round_number += 1
        
        bidder_decisions, item_demands = self._calculate_demand(agent_prices)
        
        is_market_clearing, allocation_result, final_prices = self._check_market_clearing(
            agent_prices, item_demands
        )
        
        self.current_prices = final_prices
        
        # Observation reflects state AFTER any Stage 2 price adjustment
        final_decisions, _ = self._calculate_demand(final_prices)
        obs = np.array(final_decisions, dtype=np.int8)
        
        if is_market_clearing:
            reward = 1.0
            terminated = True
            self.successful_allocation = True
            self.total_revenue = self._calculate_revenue(allocation_result)
        else:
            reward = -1.0
            terminated = False
        
        truncated = self.round_number >= self.config.max_rounds
        if truncated:
            terminated = True
        
        info = self._get_info()
        return obs, reward, terminated, truncated, info
    
    def _calculate_demand(self, prices: np.ndarray) -> Tuple[List[bool], Dict[int, List[int]]]:
        """
        Calculate demand given a price vector (PURE FUNCTION — no state mutation).
        Single-minded: bidder wants bundle iff total cost <= valuation.
        """
        bidder_decisions = []
        item_demands = {i: [] for i in range(self.config.num_items)}
        
        for bidder_id, (valuation, bundle) in enumerate(zip(self.bidder_valuations, self.bidder_bundles)):
            bundle_cost = sum(prices[item_id] for item_id in bundle)
            wants_to_buy = bundle_cost <= valuation
            
            if wants_to_buy:
                for item_id in bundle:
                    item_demands[item_id].append(bidder_id)
            
            bidder_decisions.append(wants_to_buy)
        
        return bidder_decisions, item_demands
    
    def _check_market_clearing(
        self, 
        prices: np.ndarray, 
        item_demands: Dict[int, List[int]]
    ) -> Tuple[bool, Dict[int, int], np.ndarray]:
        """
        Bayesian two-stage clearing (PURE FUNCTION).
        
        Stage 1: Check supply == demand for all items.
        Stage 2: Zero out truly undemanded items (not in ANY bundle) and recheck.
        
        Returns:
            (is_clearing, allocation {item_id -> bidder_id}, final_prices)
        """
        # === STAGE 1 ===
        allocation_result = {}
        has_excess_demand = False
        
        for item_id in range(self.config.num_items):
            demanding_bidders = item_demands.get(item_id, [])
            
            if len(demanding_bidders) > 1:
                has_excess_demand = True
            elif len(demanding_bidders) == 1:
                allocation_result[item_id] = demanding_bidders[0]
        
        if len(allocation_result) == self.config.num_items:
            if self._check_bundle_consistency(allocation_result):
                return True, allocation_result, prices
        
        # === STAGE 2 ===
        truly_undemanded_items = [
            item_id for item_id in range(self.config.num_items)
            if not any(item_id in bundle for bundle in self.bidder_bundles)
        ]
        
        if len(truly_undemanded_items) > 0:
            adjusted_prices = prices.copy()
            for item_id in truly_undemanded_items:
                adjusted_prices[item_id] = 0.0
            
            new_bidder_decisions, new_item_demands = self._calculate_demand(adjusted_prices)
            
            new_allocation = {}
            new_has_excess_demand = False
            
            for item_id in range(self.config.num_items):
                if item_id in truly_undemanded_items:
                    continue
                
                demanding_bidders = new_item_demands.get(item_id, [])
                
                if len(demanding_bidders) > 1:
                    new_has_excess_demand = True
                elif len(demanding_bidders) == 1:
                    new_allocation[item_id] = demanding_bidders[0]
            
            if not new_has_excess_demand and len(new_allocation) > 0:
                if self._check_bundle_consistency(new_allocation):
                    return True, new_allocation, adjusted_prices
        
        return False, {}, prices
    
    def _check_bundle_consistency(self, allocation_result: Dict[int, int]) -> bool:
        """
        Verify winning bidders each receive their complete bundle (PURE FUNCTION).
        """
        if len(allocation_result) == 0:
            return False
        
        winning_bidders = set(allocation_result.values())
        
        for bidder_id in winning_bidders:
            bidder_bundle = self.bidder_bundles[bidder_id]
            items_won = {item_id for item_id, winner in allocation_result.items() 
                        if winner == bidder_id}
            
            if items_won != bidder_bundle:
                return False
        
        return True
    
    def _calculate_revenue(self, allocation_result: Dict[int, int]) -> float:
        """Calculate total revenue from winning bidders at current_prices."""
        revenue = 0.0
        for item_id in allocation_result:
            revenue += self.current_prices[item_id]
        return revenue
    
    def _get_info(self) -> Dict:
        """
        Gather episode info. Pure functions only — no state mutation.
        """
        if self.round_number > 0:
            decisions, item_demands = self._calculate_demand(self.current_prices)
            is_clearing, allocation, _ = self._check_market_clearing(
                self.current_prices, item_demands
            )
        else:
            decisions = [False] * len(self.config.bidder_configs)
            item_demands = {i: [] for i in range(self.config.num_items)}
            is_clearing = False
            allocation = {}
        
        total_value = 0.0
        if is_clearing:
            winning_bidders = set(allocation.values())
            total_value = sum(self.bidder_valuations[b] for b in winning_bidders)
        
        max_possible_value = self._calculate_max_possible_value()
        efficiency = total_value / max_possible_value if max_possible_value > 0 else 0.0
        
        return {
            'round': self.round_number,
            'prices': self.current_prices.copy(),
            'demands': decisions,
            'item_demands': item_demands,
            'market_clearing': is_clearing,
            'allocation_result': allocation if is_clearing else {},
            'successful_allocation': self.successful_allocation,
            'revenue': self.total_revenue,
            'total_value': total_value,
            'max_possible_value': max_possible_value,
            'efficiency': efficiency,
        }
    
    def _calculate_max_possible_value(self) -> float:
        """Upper bound on social welfare. TODO: proper WDP for tighter bound."""
        return sum(self.bidder_valuations)


def test_bayesian_clearing():
    """Test suite for Bayesian two-stage clearing."""
    print("=" * 80)
    print("BAYESIAN TWO-STAGE CLEARING TEST SUITE")
    print("=" * 80)
    
    config = get_cats_config('0000.txt', num_bidders=3, seed=42)
    env = CATSAuctionEnv(
        config,
        resample_bidders=True,
        cats_filepath='0000.txt',  
        seed_range=(0, 1000)
    )
    
    print(f"\n📋 INSTANCE DETAILS (seed=42)")
    print(f"{'─'*80}")
    print(f"Items: {config.num_items} | Bidders: {len(config.bidder_configs)}")
    print(f"Price range: ${config.min_price:.2f} - ${config.max_price:.2f}\n")
    
    for i, (val, bundle) in enumerate(zip(env.bidder_valuations, env.bidder_bundles)):
        items_str = "{" + ", ".join(str(x) for x in sorted(bundle)) + "}"
        print(f"Bidder {i}: ${val:6.2f} for {items_str:<35} ({len(bundle):2d} items, ${val/len(bundle):5.2f}/item)")
    
    # Test 1: Pure functions
    print(f"\n{'='*80}\nTEST 1: PURE FUNCTION VERIFICATION\n{'='*80}")
    test_prices = np.array([50.0] * config.num_items, dtype=np.float32)
    orig = test_prices.copy()
    d1, id1 = env._calculate_demand(test_prices)
    d2, id2 = env._calculate_demand(test_prices)
    print(f"✓ Deterministic: {d1 == d2} | Prices unchanged: {np.array_equal(test_prices, orig)}")
    c1, a1, p1 = env._check_market_clearing(test_prices, id1)
    c2, a2, p2 = env._check_market_clearing(test_prices, id1)
    print(f"✓ Clearing deterministic: {c1 == c2} | Prices still unchanged: {np.array_equal(test_prices, orig)}")
    
    # Test 2: Stage 1
    print(f"\n{'='*80}\nTEST 2: STAGE 1 CLEARING\n{'='*80}")
    test_prices = np.ones(config.num_items, dtype=np.float32) * 100.0
    for iid in env.bidder_bundles[0]:
        test_prices[iid] = 40.0
    demands, item_demands = env._calculate_demand(test_prices)
    is_clearing, allocation, final_prices = env._check_market_clearing(test_prices, item_demands)
    print(f"Decisions: {demands} | Clears: {is_clearing}")
    if is_clearing:
        print(f"✓ STAGE 1 SUCCESS | Allocation: {allocation}")
    
    # Test 3: Stage 2
    print(f"\n{'='*80}\nTEST 3: STAGE 2 CLEARING\n{'='*80}")
    test_prices = np.ones(config.num_items, dtype=np.float32) * 20.0
    for iid in env.bidder_bundles[1]:
        test_prices[iid] = 10.0
    demands, item_demands = env._calculate_demand(test_prices)
    is_clearing, allocation, final_prices = env._check_market_clearing(test_prices, item_demands)
    adjusted = not np.array_equal(test_prices, final_prices)
    if adjusted:
        print(f"✓ STAGE 2 TRIGGERED | Adjusted items: {list(np.where(final_prices != test_prices)[0])}")
    print(f"Clears: {is_clearing}")
    
    # Test 4: Excess demand
    print(f"\n{'='*80}\nTEST 4: EXCESS DEMAND\n{'='*80}")
    test_prices = np.zeros(config.num_items, dtype=np.float32)
    demands, item_demands = env._calculate_demand(test_prices)
    is_clearing, _, _ = env._check_market_clearing(test_prices, item_demands)
    excess = sum(1 for b in item_demands.values() if len(b) > 1)
    print(f"Excess demand items: {excess}/{config.num_items} | Clears: {is_clearing}")
    if not is_clearing:
        print(f"✓ CORRECT: blocked by excess demand")
    
    # Test 5: step() integration
    print(f"\n{'='*80}\nTEST 5: STEP() INTEGRATION\n{'='*80}")
    obs, info = env.reset(seed=123)
    for step_num in range(3):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {step_num+1}: cleared={info['market_clearing']} | reward={reward:+.1f}")
        if terminated:
            print(f"  ✓ Terminated"); break
    
    # Test 6: _get_info no mutation
    print(f"\n{'='*80}\nTEST 6: _get_info() NO STATE MUTATION\n{'='*80}")
    env.reset(seed=123)
    env.step(np.ones(config.num_items, dtype=np.float32) * 30.0)
    before = env.current_prices.copy()
    env._get_info()
    after = env.current_prices.copy()
    ok = np.array_equal(before, after)
    print(f"{'✓ CORRECT' if ok else '✗ ERROR'}: prices {'unchanged' if ok else 'CHANGED'}")
    
    # Test 7: Stage 2 demonstration
    print(f"\n{'='*80}\nTEST 7: STAGE 2 CLEARING DEMONSTRATION\n{'='*80}")
    all_items = set(range(config.num_items))
    demanded = set()
    for bundle in env.bidder_bundles:
        demanded.update(bundle)
    undemanded = all_items - demanded
    print(f"Undemanded items (Stage 2 targets): {sorted(undemanded)}")
    
    test_prices = np.ones(config.num_items, dtype=np.float32) * 50.0
    test_prices[3] = 130.0; test_prices[11] = 130.0; test_prices[8] = 112.0
    for iid in env.bidder_bundles[2]:
        if iid not in [3, 11]:
            test_prices[iid] = 100.0
    
    for bid_id, (val, bundle) in enumerate(zip(env.bidder_valuations, env.bidder_bundles)):
        cost = sum(test_prices[iid] for iid in bundle)
        print(f"  Bidder {bid_id}: ${cost:.2f} vs ${val:.2f} → {'ACCEPT ✓' if cost <= val else 'REJECT ✗'}")
    
    demands, item_demands = env._calculate_demand(test_prices)
    is_clearing, allocation, final_prices = env._check_market_clearing(test_prices, item_demands)
    
    if is_clearing:
        winner = list(set(allocation.values()))[0]
        revenue = sum(final_prices[iid] for iid in allocation)
        print(f"\n🎉 STAGE 2 CLEARED | Winner: Bidder {winner} | Revenue: ${revenue:.2f}")
    else:
        print(f"\n✗ Did not clear")
    
    print(f"\n{'='*80}\nALL TESTS COMPLETE\n{'='*80}\n")


if __name__ == "__main__":
    test_bayesian_clearing()