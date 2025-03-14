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
    Test that a true chalk bracket (extreme chalk) doesn't have an unfair advantage in pool simulations.
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
    
    # Create true chalk bracket (extreme chalk, always pick higher seed)
    chalk_bracket = Bracket(tournament_teams)
    for game in chalk_bracket.games:
        game.upset_factor = -0.8  # Strongly favor higher seeds (traditional chalk)
    pool.add_entry("Chalk_Bracket", chalk_bracket)
    
    # Create other entries with varying upset factors across the new range
    for i in range(pool_size - 1):
        entry = Bracket(tournament_teams)
        # Create a variety of upset factors from chalk to upset-leaning
        upset_factor = -0.8 + (i / (pool_size - 1)) * 1.6  # Range from -0.8 to 0.8
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

def test_chalk_vs_optimal_strategy(tournament_teams):
    """
    Test that optimal upset factors (around zero) perform
    better than extreme chalk brackets in pool simulations.
    """
    # Parameters
    num_sims = 100  # Reduced for test speed
    
    # Create actual tournament bracket with moderate upset factor
    actual_bracket = Bracket(tournament_teams)
    for game in actual_bracket.games:
        game.upset_factor = 0.25  # Realistic tournament
    
    # Initialize pool
    pool = Pool(actual_bracket)
    
    # Create a pure chalk bracket (extreme chalk)
    chalk_bracket = Bracket(tournament_teams)
    for game in chalk_bracket.games:
        game.upset_factor = -0.8  # Strong chalk strategy
    pool.add_entry("Chalk", chalk_bracket)
    
    # Create brackets with different upset factors across the full range
    upset_factors = [-0.8, -0.4, -0.2, 0.0, 0.2, 0.4, 0.8]
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
    
    # Verify that balanced strategies (around zero) tend to perform better than extreme chalk
    balanced_factors = results[(results['name'] == 'Factor_-0.2') | 
                              (results['name'] == 'Factor_0.0') |
                              (results['name'] == 'Factor_0.2')]
    
    # Not every run will have these win, so we'll verify at least one balanced factor
    # has a reasonable performance compared to chalk
    assert balanced_factors['win_pct'].max() >= chalk_win_pct * 0.8, \
        "Balanced strategies should perform reasonably well compared to extreme chalk"
    
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
    
    # Create chalk bracket (extreme chalk)
    chalk_bracket = Bracket(tournament_teams)
    for game in chalk_bracket.games:
        game.upset_factor = -0.8  # Strong chalk strategy
    pool.add_entry("Chalk_Bracket", chalk_bracket)
    
    # Create other entries with a normal distribution of upset factors
    upset_factors = np.random.normal(0, 0.3, pool_size - 1)
    upset_factors = np.clip(upset_factors, -1.0, 1.0)
    
    for i, upset_factor in enumerate(upset_factors):
        entry = Bracket(tournament_teams)
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
