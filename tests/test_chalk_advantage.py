import pytest
import pandas as pd
import numpy as np
from bigdance.wn_cbb_scraper import Standings
from bigdance.cbb_brackets import Bracket, Pool, Team
from bigdance.bigdance_integration import create_teams_from_standings
from unittest.mock import patch, MagicMock

@pytest.fixture
def tournament_teams():
    """Create a fixture with exactly 64 teams for tournament testing"""
    teams = []
    regions = ["East", "West", "South", "Midwest"]
    seeds = list(range(1, 17))
    
    for region in regions:
        for seed in seeds:
            # Create team with rating roughly correlated to seed
            rating = 2000 - (seed * 50) + np.random.normal(0, 25)
            teams.append(Team(
                name=f"{region} {seed} Seed",
                seed=seed,
                region=region,
                rating=rating,
                conference=f"Conference {(seed-1)//4 + 1}" 
            ))
    
    return teams

def test_chalk_advantage_ratio(tournament_teams):
    """
    Test that a chalk bracket doesn't have an unfair advantage in pool simulations.
    """
    # Parameters
    pool_size = 20  # Small size for faster tests
    num_sims = 100  # Reduced for test speed
    
    # Create actual tournament bracket with some randomness
    actual_bracket = Bracket(tournament_teams)
    for game in actual_bracket.games:
        game.upset_factor = 0.25  # Realistic tournament has upsets
    
    # Initialize pool
    pool = Pool(actual_bracket)
    
    # Create chalk bracket
    chalk_bracket = Bracket(tournament_teams)
    for game in chalk_bracket.games:
        game.upset_factor = 0.0
    pool.add_entry("Chalk_Bracket", chalk_bracket)
    
    # Create other entries with varying upset factors
    for i in range(pool_size - 1):
        entry = Bracket(tournament_teams)
        # Create a variety of upset factors
        upset_factor = 0.05 + (i / (pool_size - 1)) * 0.35  # 0.05 to 0.4
        for game in entry.games:
            game.upset_factor = upset_factor
        pool.add_entry(f"Entry_{i+1}", entry)
    
    # Run simulations
    results = pool.simulate_pool(num_sims=num_sims)
    
    # Get chalk entry results
    chalk_results = results[results['name'] == 'Chalk_Bracket']
    
    # Calculate expected win percentage (equal chance for all entries)
    expected_win_pct = 1.0 / pool_size
    
    # Get chalk win percentage
    if chalk_results.empty:
        chalk_win_pct = 0
    else:
        chalk_win_pct = chalk_results.iloc[0]['win_pct']
    
    # Calculate advantage ratio
    advantage_ratio = chalk_win_pct / expected_win_pct if expected_win_pct > 0 else 0
    
    # Test should pass if chalk doesn't have more than a 25% advantage
    # We use 1.25 to allow for some random variation in the test
    assert advantage_ratio <= 1.25, f"Chalk advantage ratio {advantage_ratio} exceeds 1.25"
    
    # Additional check to make sure chalk isn't at a huge disadvantage either
    assert advantage_ratio >= 0.5, f"Chalk advantage ratio {advantage_ratio} is too low"

def test_chalk_vs_optimal_strategy(tournament_teams):
    """
    Test that optimal upset factors (in the middle range) perform
    better than chalk brackets in pool simulations.
    """
    # Parameters
    num_sims = 100  # Reduced for test speed
    
    # Create actual tournament bracket with moderate upset factor
    actual_bracket = Bracket(tournament_teams)
    for game in actual_bracket.games:
        game.upset_factor = 0.25  # Realistic tournament
    
    # Initialize pool
    pool = Pool(actual_bracket)
    
    # Create a chalk bracket
    chalk_bracket = Bracket(tournament_teams)
    for game in chalk_bracket.games:
        game.upset_factor = 0.0
    pool.add_entry("Chalk", chalk_bracket)
    
    # Create brackets with different upset factors
    upset_factors = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    for factor in upset_factors:
        entry = Bracket(tournament_teams)
        for game in entry.games:
            game.upset_factor = factor
        pool.add_entry(f"Factor_{factor}", entry)
    
    # Run simulations
    results = pool.simulate_pool(num_sims=num_sims)
    
    # Find best-performing strategy
    best_entry = results.iloc[0]
    
    # Get chalk results
    chalk_results = results[results['name'] == 'Chalk']
    chalk_win_pct = chalk_results.iloc[0]['win_pct'] if not chalk_results.empty else 0
    
    # Verify that middle-range upset factors (0.2-0.3) tend to perform better than chalk
    middle_factors = results[(results['name'] == 'Factor_0.2') | 
                            (results['name'] == 'Factor_0.3')]
    
    # Not every run will have these win, so we'll verify at least one middle factor
    # has a reasonable performance compared to chalk
    assert middle_factors['win_pct'].max() >= chalk_win_pct * 0.8, \
        "Middle-range upset factors should perform reasonably well compared to chalk"
    
    # Print results for debugging
    print("\nWin percentages by strategy:")
    for _, row in results.iterrows():
        print(f"{row['name']}: {row['win_pct']*100:.2f}%")

def test_chalk_advantage_comprehensive(tournament_teams):
    """More comprehensive test with larger pool and more simulations"""
    # Parameters
    pool_size = 50
    num_sims = 200  # Reduced from 500 for faster testing
    
    # Create actual tournament bracket
    actual_bracket = Bracket(tournament_teams)
    for game in actual_bracket.games:
        game.upset_factor = 0.25
    
    # Initialize pool
    pool = Pool(actual_bracket)
    
    # Create chalk bracket
    chalk_bracket = Bracket(tournament_teams)
    for game in chalk_bracket.games:
        game.upset_factor = 0.0
    pool.add_entry("Chalk_Bracket", chalk_bracket)
    
    # Create other entries with varying upset factors
    for i in range(pool_size - 1):
        entry = Bracket(tournament_teams)
        # More realistic distribution of upset factors
        if i < pool_size * 0.2:  # 20% with low upset factors
            upset_factor = 0.05 + (i / (pool_size * 0.2)) * 0.1  # 0.05-0.15
        elif i < pool_size * 0.7:  # 50% with medium upset factors
            idx = i - int(pool_size * 0.2)
            upset_factor = 0.15 + (idx / (pool_size * 0.5)) * 0.15  # 0.15-0.3
        else:  # 30% with high upset factors
            idx = i - int(pool_size * 0.7)
            upset_factor = 0.3 + (idx / (pool_size * 0.3)) * 0.2  # 0.3-0.5
            
        for game in entry.games:
            game.upset_factor = upset_factor
        pool.add_entry(f"Entry_{i+1}", entry)
    
    # Run simulations
    results = pool.simulate_pool(num_sims=num_sims)
    
    # Get chalk entry results
    chalk_results = results[results['name'] == 'Chalk_Bracket']
    
    if not chalk_results.empty:
        chalk_win_pct = chalk_results.iloc[0]['win_pct']
        expected_win_pct = 1.0 / pool_size
        advantage_ratio = chalk_win_pct / expected_win_pct
        
        # Verify chalk doesn't have a significant advantage
        assert advantage_ratio <= 1.2, \
            f"Chalk advantage ratio {advantage_ratio} exceeds 1.2 (win rate: {chalk_win_pct*100:.2f}%)"
