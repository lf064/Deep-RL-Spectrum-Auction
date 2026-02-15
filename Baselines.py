"""
Baseline Evaluation Script for Combinatorial Auctions

Implements and compares three baseline methods:
1. SG-Auction-D: Subgradient with Static Order (Dependent on instance)
2. SG-Auction-I: Subgradient with Instance-specific best stepsize (Oracle)
3. Bayesian Mechanism: EP belief updates + EM price optimization

Based on:
- Brero & Lahaie (2018): "A Bayesian Clearing Mechanism for Combinatorial Auctions"
- Your PPO training setup
"""

import numpy as np
from typing import List, Set, Tuple, Dict, Optional
from dataclasses import dataclass
from scipy.stats import norm
from scipy.optimize import minimize
import time
from collections import defaultdict


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class Bidder:
    """Single-minded bidder with one desired bundle."""
    bidder_id: int
    bundle: Set[int]  # Set of item indices
    valuation: float  # True valuation (unknown to mechanism)
    
    # Bayesian belief state (for Bayesian mechanism)
    mean: float = 0.0  # Posterior mean
    variance: float = 1.0  # Posterior variance
    
    def utility(self, bundle_price: float) -> float:
        """Utility from buying at given price."""
        return max(0.0, self.valuation - bundle_price)
    
    def demands_bundle(self, bundle_price: float) -> bool:
        """Does bidder want to buy at this price?"""
        return self.valuation > bundle_price


# ============================================================================
# BASELINE 1: SG-AUCTION-D (Static Order Subgradient)
# ============================================================================

class SGAuctionD:
    """
    Subgradient auction with STATIC order (SAOr baseline).
    
    Uses a fixed stepsize τ and deterministic agent ordering.
    This is the "Standard Average Round-Optimized" baseline.
    """
    
    def __init__(self,
                 bidders: List[Bidder],
                 num_items: int,
                 stepsize: float,
                 max_rounds: int = 100,
                 agent_order: Optional[List[int]] = None):
        """
        Args:
            bidders: List of bidders
            num_items: Number of items
            stepsize: Fixed stepsize τ
            max_rounds: Maximum auction rounds
            agent_order: Fixed ordering of agents (if None, use natural order)
        """
        self.bidders = bidders
        self.num_items = num_items
        self.stepsize = stepsize
        self.max_rounds = max_rounds
        
        # Static agent order
        if agent_order is None:
            self.agent_order = list(range(len(bidders)))
        else:
            self.agent_order = agent_order
    
    def run(self, verbose: bool = False) -> Dict:
        """
        Run the static-order subgradient auction.
        
        Returns:
            dict with keys: cleared, rounds, prices, allocation, social_welfare
        """
        n_bidders = len(self.bidders)
        prices = np.zeros(self.num_items)
        supply = np.ones(self.num_items)
        
        for round_num in range(1, self.max_rounds + 1):
            # Calculate demand
            demand = np.zeros(self.num_items)
            allocation = [set() for _ in range(n_bidders)]
            
            # Visit agents in FIXED order
            for bidder_idx in self.agent_order:
                bidder = self.bidders[bidder_idx]
                bundle_price = sum(prices[j] for j in bidder.bundle)
                
                if bidder.demands_bundle(bundle_price):
                    for item in bidder.bundle:
                        demand[item] += 1
                    allocation[bidder_idx] = bidder.bundle
            
            # Check Stage 1 clearing
            if np.array_equal(demand, supply):
                if verbose:
                    print(f"[SG-D] Cleared in {round_num} rounds")
                
                return {
                    'cleared': True,
                    'rounds': round_num,
                    'prices': prices.copy(),
                    'allocation': allocation,
                    'social_welfare': self._compute_welfare(allocation)
                }
            
            # Check Stage 2 clearing (zero out undemanded items)
            if np.all(supply >= demand):
                undemanded = demand < supply
                adjusted_prices = prices.copy()
                adjusted_prices[undemanded] = 0.0
                
                # Recompute demand with adjusted prices
                new_demand = np.zeros(self.num_items)
                new_allocation = [set() for _ in range(n_bidders)]
                
                for bidder_idx in self.agent_order:
                    bidder = self.bidders[bidder_idx]
                    bundle_price = sum(adjusted_prices[j] for j in bidder.bundle)
                    
                    if bidder.demands_bundle(bundle_price):
                        for item in bidder.bundle:
                            new_demand[item] += 1
                        new_allocation[bidder_idx] = bidder.bundle
                
                adjusted_supply = supply.copy()
                adjusted_supply[undemanded] = 0.0
                
                if np.array_equal(new_demand, adjusted_supply):
                    if verbose:
                        print(f"[SG-D] Stage 2 cleared in {round_num} rounds")
                    
                    return {
                        'cleared': True,
                        'rounds': round_num,
                        'prices': adjusted_prices,
                        'allocation': new_allocation,
                        'social_welfare': self._compute_welfare(new_allocation)
                    }
            
            # Subgradient update
            stepsize_t = self.stepsize / np.sqrt(round_num)
            gradient = demand - supply
            prices = prices + stepsize_t * gradient
            prices = np.maximum(prices, 0)  # Non-negativity
        
        # Failed to clear
        if verbose:
            print(f"[SG-D] Failed to clear within {self.max_rounds} rounds")
        
        return {
            'cleared': False,
            'rounds': self.max_rounds,
            'prices': prices,
            'allocation': allocation,
            'social_welfare': self._compute_welfare(allocation)
        }
    
    def _compute_welfare(self, allocation: List[Set[int]]) -> float:
        """Compute social welfare of allocation."""
        welfare = 0.0
        for bidder_idx, bundle in enumerate(allocation):
            if bundle:  # If bidder got their bundle
                bidder = self.bidders[bidder_idx]
                if bundle == bidder.bundle:
                    welfare += bidder.valuation
        return welfare


# ============================================================================
# BASELINE 2: SG-AUCTION-I (Instance-Optimized Oracle)
# ============================================================================

class SGAuctionI:
    """
    Subgradient auction with instance-specific best stepsize (SIO baseline).
    
    Tries multiple stepsizes and picks the best result.
    This is the "Standard Instance Optimized" oracle baseline.
    """
    
    def __init__(self,
                 bidders: List[Bidder],
                 num_items: int,
                 num_stepsizes: int = 100,
                 max_rounds: int = 100):
        """
        Args:
            bidders: List of bidders
            num_items: Number of items
            num_stepsizes: Number of stepsizes to try
            max_rounds: Maximum auction rounds per trial
        """
        self.bidders = bidders
        self.num_items = num_items
        self.num_stepsizes = num_stepsizes
        self.max_rounds = max_rounds
    
    def run(self, verbose: bool = False) -> Dict:
        """
        Run multiple subgradient auctions with different stepsizes.
        Pick the best result (oracle).
        
        Returns:
            dict with keys: cleared, rounds, prices, allocation, social_welfare,
                          best_stepsize, best_stepsize_pct, all_results
        """
        max_val = max(b.valuation for b in self.bidders)
        
        best_result = None
        best_rounds = float('inf')
        best_stepsize = None
        all_results = []
        
        for i in range(1, self.num_stepsizes + 1):
            stepsize = (max_val / self.num_stepsizes) * i
            stepsize_pct = 100 * i / self.num_stepsizes
            
            # Run static-order auction with this stepsize
            auction = SGAuctionD(
                bidders=self.bidders,
                num_items=self.num_items,
                stepsize=stepsize,
                max_rounds=self.max_rounds
            )
            
            result = auction.run(verbose=False)
            result['stepsize'] = stepsize
            result['stepsize_pct'] = stepsize_pct
            all_results.append(result)
            
            # Track best result
            if result['cleared'] and result['rounds'] < best_rounds:
                best_rounds = result['rounds']
                best_stepsize = stepsize
                best_result = result
        
        if best_result is None:
            # None cleared, return the first result
            best_result = all_results[0]
            best_stepsize = best_result['stepsize']
        
        if verbose:
            if best_result['cleared']:
                print(f"[SG-I] Best: {best_rounds} rounds with stepsize {best_stepsize:.2f} "
                      f"({best_result['stepsize_pct']:.1f}%)")
            else:
                print(f"[SG-I] Failed to clear with any stepsize")
        
        # Add summary info
        best_result['best_stepsize'] = best_stepsize
        best_result['best_stepsize_pct'] = best_result['stepsize_pct']
        best_result['all_results'] = all_results
        best_result['cleared_count'] = sum(r['cleared'] for r in all_results)
        
        return best_result


# ============================================================================
# BASELINE 3: BAYESIAN MECHANISM
# ============================================================================

class BayesianAuction:
    """
    Bayesian clearing mechanism from Brero & Lahaie (2018).
    
    Alternates between:
    1. Knowledge Update: Update Gaussian beliefs via Expectation Propagation
    2. Price Update: Compute MAP prices via Expectation Maximization
    """
    
    def __init__(self,
                 bidders: List[Bidder],
                 num_items: int,
                 prior_mean: Optional[np.ndarray] = None,
                 prior_variance: Optional[np.ndarray] = None,
                 beta: float = 10.0,
                 max_rounds: int = 100,
                 variance_floor: float = 0.01):
        """
        Args:
            bidders: List of bidders (will initialize beliefs)
            num_items: Number of items
            prior_mean: Prior mean for each bidder (if None, use valuation/2)
            prior_variance: Prior variance for each bidder (if None, use valuation/4)
            beta: Noise parameter (higher = more rational bidders)
            max_rounds: Maximum auction rounds
            variance_floor: Minimum variance to avoid numerical issues
        """
        self.bidders = bidders
        self.num_items = num_items
        self.beta = beta
        self.max_rounds = max_rounds
        self.variance_floor = variance_floor
        
        # Initialize priors
        for i, bidder in enumerate(self.bidders):
            if prior_mean is not None:
                bidder.mean = prior_mean[i]
            else:
                # Uninformed prior: assume halfway to max observed valuation
                max_val = max(b.valuation for b in self.bidders)
                bidder.mean = max_val / 2.0
            
            if prior_variance is not None:
                bidder.variance = prior_variance[i]
            else:
                # Uninformed prior: high uncertainty
                bidder.variance = (bidder.mean / 2.0) ** 2
        
        self.prices = np.zeros(num_items)
    
    def run(self, verbose: bool = False) -> Dict:
        """
        Run the Bayesian auction.
        
        Returns:
            dict with keys: cleared, rounds, prices, allocation, social_welfare
        """
        n_bidders = len(self.bidders)
        supply = np.ones(self.num_items)
        
        for round_num in range(1, self.max_rounds + 1):
            # Price Update: Compute MAP prices via EM
            self.prices = self._price_update_em()
            
            # Query bidders and get demands
            demand = np.zeros(self.num_items)
            allocation = [set() for _ in range(n_bidders)]
            bids = []
            
            for bidder_idx, bidder in enumerate(self.bidders):
                bundle_price = sum(self.prices[j] for j in bidder.bundle)
                accepts = bidder.demands_bundle(bundle_price)
                
                if accepts:
                    for item in bidder.bundle:
                        demand[item] += 1
                    allocation[bidder_idx] = bidder.bundle
                
                bids.append((bidder_idx, bundle_price, accepts))
            
            # Knowledge Update: Update beliefs via EP
            self._knowledge_update_ep(bids)
            
            # Check Stage 1 clearing
            if np.array_equal(demand, supply):
                if verbose:
                    print(f"[Bayesian] Cleared in {round_num} rounds")
                
                return {
                    'cleared': True,
                    'rounds': round_num,
                    'prices': self.prices.copy(),
                    'allocation': allocation,
                    'social_welfare': self._compute_welfare(allocation)
                }
            
            # Check Stage 2 clearing
            if np.all(supply >= demand):
                undemanded = demand < supply
                adjusted_prices = self.prices.copy()
                adjusted_prices[undemanded] = 0.0
                
                # Recompute demand
                new_demand = np.zeros(self.num_items)
                new_allocation = [set() for _ in range(n_bidders)]
                
                for bidder_idx, bidder in enumerate(self.bidders):
                    bundle_price = sum(adjusted_prices[j] for j in bidder.bundle)
                    
                    if bidder.demands_bundle(bundle_price):
                        for item in bidder.bundle:
                            new_demand[item] += 1
                        new_allocation[bidder_idx] = bidder.bundle
                
                adjusted_supply = supply.copy()
                adjusted_supply[undemanded] = 0.0
                
                if np.array_equal(new_demand, adjusted_supply):
                    if verbose:
                        print(f"[Bayesian] Stage 2 cleared in {round_num} rounds")
                    
                    return {
                        'cleared': True,
                        'rounds': round_num,
                        'prices': adjusted_prices,
                        'allocation': new_allocation,
                        'social_welfare': self._compute_welfare(new_allocation)
                    }
        
        # Failed to clear
        if verbose:
            print(f"[Bayesian] Failed to clear within {self.max_rounds} rounds")
        
        return {
            'cleared': False,
            'rounds': self.max_rounds,
            'prices': self.prices,
            'allocation': allocation,
            'social_welfare': self._compute_welfare(allocation)
        }
    
    def _knowledge_update_ep(self, bids: List[Tuple[int, float, bool]]):
        """
        Update Gaussian beliefs via Expectation Propagation.
        
        Args:
            bids: List of (bidder_idx, bundle_price, accepted) tuples
        """
        for bidder_idx, bundle_price, accepted in bids:
            bidder = self.bidders[bidder_idx]
            
            # EP update (from paper, Williams & Rasmussen 2006)
            y = 1.0 if accepted else -1.0
            m = bidder.mean
            s2 = bidder.variance
            
            # Compute z
            z = y * self.beta * (m - bundle_price) / np.sqrt(1 + s2 * self.beta**2)
            
            # Ratio of normal PDF to CDF
            pdf = norm.pdf(z)
            cdf = norm.cdf(z)
            
            if cdf < 1e-10:  # Numerical safety
                n_over_c = 0.0
            else:
                n_over_c = pdf / cdf
            
            # Update mean
            m_new = m + (y * s2 * self.beta * n_over_c) / np.sqrt(1 + s2 * self.beta**2)
            
            # Update variance (always decreases!)
            s2_new = s2 - (s2**2 * self.beta**2 * n_over_c * (z + n_over_c)) / (1 + s2 * self.beta**2)
            
            # Floor the variance
            s2_new = max(s2_new, self.variance_floor)
            
            bidder.mean = m_new
            bidder.variance = s2_new
    
    def _price_update_em(self, max_iter: int = 100, tol: float = 1e-7) -> np.ndarray:
        """
        Compute MAP prices via Expectation-Maximization.
        
        Returns:
            Updated item prices
        """
        # Initialize prices
        overall_mean = np.mean([b.mean for b in self.bidders])
        prices = np.ones(self.num_items) * (2 * overall_mean / self.num_items)
        
        for iteration in range(max_iter):
            prices_old = prices.copy()
            
            # E-step: Compute weights
            weights = []
            for bidder in self.bidders:
                m = bidder.mean
                s2 = bidder.variance
                s = np.sqrt(s2)
                
                bundle_price = sum(prices[j] for j in bidder.bundle)
                
                # Compute weight w_i = P(q_i = 1 | m, s2, prices)
                # Using formula from paper
                term1_numer = norm.cdf((m - s2 - bundle_price) / s) * \
                              np.exp(-(m - s2/2.0 - bundle_price))
                
                term2 = norm.cdf((bundle_price - m) / s)
                
                denom = term1_numer + term2
                
                if denom < 1e-10:
                    w = 0.0
                else:
                    w = term1_numer / denom
                
                weights.append(w)
            
            # M-step: Optimize prices
            def objective(p):
                """Negative log posterior (to minimize)."""
                val = np.sum(p)  # Revenue term: -R(θ) = -Σp_j
                
                for bidder, w in zip(self.bidders, weights):
                    m = bidder.mean
                    s2 = bidder.variance
                    s = np.sqrt(s2)
                    
                    bundle_price = sum(p[j] for j in bidder.bundle)
                    
                    # Log-likelihood terms
                    term1 = norm.logcdf((m - s2 - bundle_price) / s) - (m - s2/2.0 - bundle_price)
                    term2 = norm.logcdf((bundle_price - m) / s)
                    
                    val -= w * term1 + (1 - w) * term2
                
                return val
            
            # Optimize
            result = minimize(
                objective,
                prices,
                method='L-BFGS-B',
                bounds=[(0, None)] * self.num_items
            )
            
            prices = result.x
            
            # Check convergence
            if np.linalg.norm(prices - prices_old, ord=np.inf) / (overall_mean + 1e-10) < tol:
                break
        
        return prices
    
    def _compute_welfare(self, allocation: List[Set[int]]) -> float:
        """Compute social welfare of allocation."""
        welfare = 0.0
        for bidder_idx, bundle in enumerate(allocation):
            if bundle:
                bidder = self.bidders[bidder_idx]
                if bundle == bidder.bundle:
                    welfare += bidder.valuation
        return welfare


# ============================================================================
# EVALUATION FRAMEWORK
# ============================================================================

def evaluate_all_baselines(
    bidders: List[Bidder],
    num_items: int,
    stepsize_saor: Optional[float] = None,
    verbose: bool = True
) -> Dict:
    """
    Run all three baselines on the same instance.
    
    Args:
        bidders: List of bidders
        num_items: Number of items
        stepsize_saor: Fixed stepsize for SG-D (if None, use 50% of max_val)
        verbose: Print progress
    
    Returns:
        dict with results from all three methods
    """
    # Default stepsize for SG-D
    if stepsize_saor is None:
        max_val = max(b.valuation for b in bidders)
        stepsize_saor = max_val * 0.5  # 50% heuristic
    
    results = {}
    
    # 1. SG-Auction-D (Static order, fixed stepsize)
    if verbose:
        print("\n" + "="*70)
        print("Running SG-Auction-D (Static Order Subgradient)")
        print("="*70)
    
    start = time.time()
    sg_d = SGAuctionD(bidders, num_items, stepsize_saor)
    results['sg_d'] = sg_d.run(verbose=verbose)
    results['sg_d']['time'] = time.time() - start
    
    # 2. SG-Auction-I (Instance-optimized oracle)
    if verbose:
        print("\n" + "="*70)
        print("Running SG-Auction-I (Instance-Optimized Oracle)")
        print("="*70)
    
    start = time.time()
    sg_i = SGAuctionI(bidders, num_items)
    results['sg_i'] = sg_i.run(verbose=verbose)
    results['sg_i']['time'] = time.time() - start
    
    # 3. Bayesian Auction
    if verbose:
        print("\n" + "="*70)
        print("Running Bayesian Auction")
        print("="*70)
    
    # Reset bidder beliefs (make fresh copies)
    bidders_copy = [
        Bidder(b.bidder_id, b.bundle.copy(), b.valuation)
        for b in bidders
    ]
    
    start = time.time()
    bayes = BayesianAuction(bidders_copy, num_items)
    results['bayesian'] = bayes.run(verbose=verbose)
    results['bayesian']['time'] = time.time() - start
    
    # Print summary
    if verbose:
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        for method_name, result in results.items():
            cleared_str = "✓ CLEARED" if result['cleared'] else "✗ FAILED"
            print(f"{method_name.upper():15} | {cleared_str:12} | "
                  f"Rounds: {result['rounds']:3d} | "
                  f"Welfare: {result['social_welfare']:.2f} | "
                  f"Time: {result['time']:.3f}s")
    
    return results


# ============================================================================
# INTEGRATION WITH YOUR CATS SETUP
# ============================================================================

def evaluate_on_cats_instance(
    cats_filepath: str,
    num_bidders: int = 7,
    seed: int = 0,
    stepsize_saor: Optional[float] = None
) -> Dict:
    """
    Load a CATS instance and run all baselines.
    
    This integrates with your existing CATS setup.
    
    Args:
        cats_filepath: Path to CATS .txt file
        num_bidders: Number of bidders to sample
        seed: Random seed
        stepsize_saor: Fixed stepsize for SG-D
    
    Returns:
        dict with results from all baselines
    """
    # Import your existing code
    import sys
    sys.path.append('/mnt/user-data/outputs')
    
    try:
        from cats_parser import CATSParser
        from cont_config import get_cats_config
    except ImportError:
        print("ERROR: Could not import your CATS modules.")
        print("Make sure cats_parser.py and cont_config.py are in /mnt/user-data/outputs")
        return None
    
    # Load CATS instance
    config = get_cats_config(cats_filepath, num_bidders, seed)
    
    # Convert to Bidder objects
    bidders = []
    for i, bidder_config in enumerate(config.bidder_configs):
        bidders.append(Bidder(
            bidder_id=i,
            bundle=set(bidder_config.interested_items),
            valuation=bidder_config.valuation
        ))
    
    # Run all baselines
    return evaluate_all_baselines(
        bidders,
        config.num_items,
        stepsize_saor,
        verbose=True
    )


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example 1: Simple synthetic instance
    print("\n" + "="*70)
    print("EXAMPLE 1: Synthetic Instance")
    print("="*70)
    
    bidders = [
        Bidder(0, {0, 1}, 100.0),
        Bidder(1, {1, 2}, 80.0),
        Bidder(2, {2, 3}, 90.0),
        Bidder(3, {0, 3}, 70.0),
    ]
    
    results = evaluate_all_baselines(bidders, num_items=4, verbose=True)
    
    # Example 2: CATS instance (if files available)
    # Uncomment if you have CATS files set up
    """
    print("\n\n" + "="*70)
    print("EXAMPLE 2: CATS Instance")
    print("="*70)
    
    results_cats = evaluate_on_cats_instance(
        cats_filepath='path/to/your/cats/0000.txt',
        num_bidders=7,
        seed=42
    )
    """