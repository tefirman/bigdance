#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   bracket_optimizer.py
@Time    :   2024/02/22
@Author  :   Taylor Firman
@Version :   0.1.0
@Contact :   tefirman@gmail.com
@Desc    :   Optimization and analysis tools for March Madness brackets
'''

from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np
from dancing.cbb_brackets import Bracket, Pool
from dancing.wn_cbb_scraper import Standings
from dancing.dancing_integration import create_teams_from_standings
from dataclasses import dataclass
from scipy.optimize import differential_evolution

@dataclass
class BracketStats:
    """Statistics about a bracket's picks"""
    num_upsets: int
    upset_by_round: Dict[str, int]
    chalk_score: float  # How closely picks follow seeds
    region_winners: List[str]
    final_four: List[str]
    champion: str
    avg_seed_by_round: Dict[str, float]

def analyze_bracket(bracket: Bracket) -> BracketStats:
    """
    Analyze a bracket's picks for various statistics
    
    Args:
        bracket: Completed bracket with winners selected
        
    Returns:
        BracketStats object containing various metrics
    """
    results = bracket.simulate_tournament()
    
    # Track upsets by round
    upsets = {
        "First Round": 0,
        "Second Round": 0,
        "Sweet 16": 0,
        "Elite 8": 0,
        "Final Four": 0,
        "Championship": 0
    }
    
    # Track average seed by round
    avg_seeds = {
        "First Round": [],
        "Second Round": [],
        "Sweet 16": [],
        "Elite 8": [], 
        "Final Four": [],
        "Championship": []
    }
    
    # Analyze each round
    for round_name in upsets.keys():
        round_teams = results[round_name]
        for team in round_teams:
            avg_seeds[round_name].append(team.seed)
            if round_name != "First Round":  # First round has no upsets yet
                prev_round = list(upsets.keys())[list(upsets.keys()).index(round_name)-1]
                prev_teams = results[prev_round]
                # Check if this team beat a better seed
                opponents = [t for t in prev_teams if t.region == team.region and t != team]
                if opponents and min(t.seed for t in opponents) < team.seed:
                    upsets[round_name] += 1
                    
    # Calculate average seeds
    avg_seeds = {k: np.mean(v) if v else 0 for k, v in avg_seeds.items()}
    
    # Calculate chalk score (how closely picks follow seeds)
    # Lower score means more chalk (following seeds)
    chalk_score = sum(
        sum(team.seed for team in results[round_name]) 
        for round_name in results 
        if round_name != "Champion"
    ) / len(results)
    
    return BracketStats(
        num_upsets=sum(upsets.values()),
        upset_by_round=upsets,
        chalk_score=chalk_score,
        region_winners=[t.name for t in results["Elite 8"] if t],
        final_four=[t.name for t in results["Final Four"] if t],
        champion=results["Champion"].name if results["Champion"] else None,
        avg_seed_by_round=avg_seeds
    )

class BracketOptimizer:
    """Class to handle bracket optimization with progress tracking"""
    
    def __init__(self, standings: Standings, pool_size: int, num_sims: int = 1000, 
                 target_chalk: Optional[float] = None, verbose: bool = True):
        self.standings = standings
        self.pool_size = pool_size
        self.num_sims = num_sims
        self.target_chalk = target_chalk
        self.verbose = verbose
        self.iteration = 0
        
        # Create base bracket
        self.base_bracket = create_teams_from_standings(standings)

    def objective(self, x: np.ndarray) -> float:
        """Objective function for optimization"""
        # Create pool with random entries
        pool = Pool(self.base_bracket)
        
        # Add optimized entry
        opt_bracket = create_teams_from_standings(self.standings)
        pool.add_entry("Optimized", opt_bracket)
        
        # Add random entries
        for i in range(self.pool_size - 1):
            entry = create_teams_from_standings(self.standings)
            pool.add_entry(f"Entry_{i+1}", entry)
            
        # Simulate pool
        results = pool.simulate_pool(num_sims=self.num_sims)
        
        # Get stats for optimized bracket
        stats = analyze_bracket(opt_bracket)
        
        # Calculate objective value
        obj_val = -results.loc[results['name'] == 'Optimized', 'win_pct'].values[0]
        
        # Add penalty if chalk score doesn't match target
        if self.target_chalk is not None:
            chalk_penalty = abs(stats.chalk_score - self.target_chalk)
            obj_val += chalk_penalty
            
        return obj_val
        
    def callback(self, xk, convergence):
        """Callback to monitor optimization progress"""
        self.iteration += 1
        if self.verbose and self.iteration % 10 == 0:  # Print every 10 iterations
            print(f"Iteration {self.iteration}: Best objective = {self.objective(xk):.4f}")
        return False  # Don't stop optimization
    
def optimize_bracket(standings: Standings,
                    pool_size: int,
                    num_sims: int = 1000,
                    target_chalk: Optional[float] = None,
                    verbose: bool = True) -> Tuple[Bracket, float]:
    """
    Optimize bracket picks using differential evolution
    
    Args:
        standings: Current team standings/ratings
        pool_size: Number of entries in the pool
        num_sims: Number of simulations per evaluation
        target_chalk: Optional target chalk score (None for pure optimization)
        verbose: Whether to print progress updates
        
    Returns:
        Tuple of (optimized bracket, expected value)
    """
    optimizer = BracketOptimizer(standings, pool_size, num_sims, target_chalk, verbose)
    
    # Define parameter bounds (upset factors for each round)
    bounds = [(0.0, 0.5)] * 6  # One factor per round
    
    # Run optimization
    result = differential_evolution(
        optimizer.objective,
        bounds,
        maxiter=50,
        popsize=20,
        seed=42,
        callback=optimizer.callback,
        updating='deferred',  # More stable convergence
    )
    
    # Create final optimized bracket
    final_bracket = create_teams_from_standings(standings)
    
    return final_bracket, -result.fun

def pool_tendency_analysis(standings: Standings,
                         pool_size: int,
                         num_pools: int = 100) -> pd.DataFrame:
    """
    Analyze winning bracket tendencies across many simulated pools
    
    Args:
        standings: Current team standings/ratings
        pool_size: Number of entries in each pool
        num_pools: Number of pools to simulate
        
    Returns:
        DataFrame with winning bracket statistics
    """
    stats_list = []
    
    for i in range(num_pools):
        # Create and simulate pool
        pool = Pool(create_teams_from_standings(standings))
        
        # Add entries
        for j in range(pool_size):
            entry = create_teams_from_standings(standings)
            pool.add_entry(f"Entry_{j+1}", entry)
            
        # Simulate pool
        results = pool.simulate_pool(num_sims=100)
        
        # Analyze winning bracket
        winner_name = results.iloc[0]['name']
        winner_bracket = next(entry for name, entry in pool.entries if name == winner_name)
        stats = analyze_bracket(winner_bracket)
        
        # Add to list
        stats_list.append({
            'num_upsets': stats.num_upsets,
            'chalk_score': stats.chalk_score,
            'champion_seed': next(t.seed for t in winner_bracket.teams if t.name == stats.champion),
            'avg_ff_seed': np.mean([next(t.seed for t in winner_bracket.teams if t.name == ff) 
                                  for ff in stats.final_four])
        })
        
    return pd.DataFrame(stats_list).agg(['mean', 'std', 'min', 'max'])

def main():
    # Example usage
    standings = Standings()
    
    print("\nAnalyzing pool tendencies...")
    tendencies = pool_tendency_analysis(standings, pool_size=10)
    print(tendencies)
    
    print("\nOptimizing bracket...")
    optimal_bracket, exp_value = optimize_bracket(standings, pool_size=10)
    stats = analyze_bracket(optimal_bracket)
    
    print(f"\nOptimal bracket expected value: {exp_value:.3f}")
    print(f"Number of upsets: {stats.num_upsets}")
    print(f"Chalk score: {stats.chalk_score:.2f}")
    print(f"Final Four: {stats.final_four}")
    print(f"Champion: {stats.champion}")

if __name__ == "__main__":
    main()
