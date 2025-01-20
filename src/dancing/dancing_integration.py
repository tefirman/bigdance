#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   dancing_integration.py
@Time    :   2025/01/19
@Author  :   Taylor Firman
@Version :   0.1.0
@Contact :   tefirman@gmail.com
@Desc    :   Integration module between Warren Nolan scraper and bracket simulator
'''

from typing import List, Dict, Optional
import pandas as pd
from wn_cbb_scraper import Standings
from cbb_brackets import Team, Bracket, Pool

def create_teams_from_standings(standings: Standings, 
                              seeds: Optional[Dict[str, int]] = None,
                              regions: Optional[Dict[str, str]] = None) -> List[Team]:
    """
    Convert Warren Nolan standings into bracket-compatible Team objects.
    Ensures conference champions get automatic bids, then fills remaining spots
    with highest rated teams. Distributes teams across the four tournament regions.
    
    Args:
        standings: Standings object containing team ratings and info
        seeds: Optional dictionary mapping team names to their regional seeds (1-16)
        regions: Optional dictionary mapping team names to their tournament regions
               
    Returns:
        List of Team objects ready for bracket simulation
    """
    TOURNAMENT_REGIONS = ['East', 'West', 'South', 'Midwest']
    
    # First, get the highest rated team from each conference (auto bids)
    auto_bids = (standings.elo.sort_values('ELO', ascending=False)
                 .groupby('Conference').first()
                 .reset_index())
    
    # Get remaining spots after auto bids
    remaining_spots = 64 - len(auto_bids)
    
    # Get highest rated teams not already in auto bids
    at_large_pool = standings.elo[~standings.elo['Team'].isin(auto_bids['Team'])]
    at_large_bids = at_large_pool.sort_values('ELO', ascending=False).head(remaining_spots)
    
    # Combine auto bids and at-large bids
    tournament_teams = pd.concat([auto_bids, at_large_bids], ignore_index=True)
    tournament_teams = tournament_teams.sort_values('ELO', ascending=False)
    
    # If seeds not provided, generate them based on overall rating
    if seeds is None:
        seeds = {}
        # Split teams into 16 groups of 4 for seeding
        for seed_num in range(1, 17):
            seed_group = tournament_teams.iloc[(seed_num-1)*4:seed_num*4]
            for _, team in seed_group.iterrows():
                seeds[team['Team']] = seed_num
    
    # If regions not provided, distribute teams across regions
    if regions is None:
        regions = {}
        # For each seed line (1-16), distribute teams across regions
        for seed_num in range(1, 17):
            seed_teams = tournament_teams[tournament_teams['Team'].map(seeds) == seed_num]
            seed_teams = seed_teams.sort_values('ELO', ascending=False)
            for i, (_, team) in enumerate(seed_teams.iterrows()):
                regions[team['Team']] = TOURNAMENT_REGIONS[i]
    
    # Create Team objects with proper seeds and regions
    teams = []
    for _, row in tournament_teams.iterrows():
        teams.append(Team(
            name=row['Team'],
            seed=seeds.get(row['Team'], 16),  # Default to 16 seed if not found
            region=regions.get(row['Team'], 'East'),  # Default to East if not found
            rating=row['ELO'],
            conference=row['Conference']
        ))
    
    # Validate bracket structure
    team_counts = pd.DataFrame([(t.region, t.seed) for t in teams])
    team_counts = team_counts.groupby([0, 1]).size().reset_index()
    if not all(team_counts[0] == 1):
        raise ValueError("Invalid bracket structure: Each region must have exactly one team of each seed")
        
    return teams

def balance_regions(teams: List[Team]) -> List[Team]:
    """
    Adjust team regions to ensure proper tournament structure.
    
    Args:
        teams: List of Team objects with preliminary region assignments
        
    Returns:
        List of Team objects with balanced regions
    """
    regions = ['East', 'West', 'South', 'Midwest']
    teams_df = pd.DataFrame([{
        'name': t.name,
        'seed': t.seed,
        'rating': t.rating,
        'conference': t.conference,
        'orig_region': t.region
    } for t in teams])
    
    # Sort by seed and rating for assignment
    teams_df = teams_df.sort_values(['seed', 'rating'], ascending=[True, False])
    
    # Assign to regions ensuring even distribution of seeds
    teams_per_region = len(teams) // 4
    for seed in range(1, 17):
        seed_teams = teams_df[teams_df.seed == seed]
        for i, (_, team) in enumerate(seed_teams.iterrows()):
            region = regions[i % 4]
            teams_df.loc[teams_df.name == team.name, 'region'] = region
            
    # Create new balanced Team objects
    balanced_teams = []
    for _, row in teams_df.iterrows():
        balanced_teams.append(Team(
            name=row['name'],
            seed=row['seed'],
            region=row['region'],
            rating=row['rating'],
            conference=row['conference']
        ))
    
    return balanced_teams

def create_bracket_from_standings(standings: Standings,
                                seeds: Optional[Dict[str, int]] = None) -> Bracket:
    """
    Create a tournament bracket from Warren Nolan standings.
    
    Args:
        standings: Standings object containing team ratings and info
        seeds: Optional dictionary mapping team names to their tournament seeds
        
    Returns:
        Bracket object ready for simulation
    """
    # Create teams and balance regions
    teams = create_teams_from_standings(standings, seeds)
    balanced_teams = balance_regions(teams)
    
    # Create and return bracket
    return Bracket(balanced_teams)

def simulate_bracket_pool(standings: Standings,
                        seeds: Dict[str, int],
                        num_entries: int = 100,
                        upset_factors: Optional[List[float]] = None) -> pd.DataFrame:
    """
    Simulate a bracket pool using Warren Nolan ratings.
    
    Args:
        standings: Standings object containing team ratings and info
        seeds: Dictionary mapping team names to their tournament seeds
        num_entries: Number of bracket entries to simulate
        upset_factors: Optional list of upset factors for each entry
        
    Returns:
        DataFrame containing simulation results
    """
    # Create actual results bracket
    actual_bracket = create_bracket_from_standings(standings, seeds)
    
    # Initialize pool
    pool = Pool(actual_bracket)
    
    # Generate upset factors if not provided
    if upset_factors is None:
        upset_factors = [0.1 + (i/num_entries)*0.3 for i in range(num_entries)]
    
    # Create entries with varying upset factors
    for i in range(num_entries):
        entry_bracket = create_bracket_from_standings(standings, seeds)
        entry_name = f"Entry_{i+1}"
        pool.add_entry(entry_name, entry_bracket)
    
    # Simulate pool
    results = pool.simulate_pool(num_sims=1000)
    return results

def main():
    """Example usage of integration module"""
    # Get current standings
    standings = Standings()
    
    # Example tournament seeds (you would provide real seeds)
    example_seeds = {
        team: i+1 for i, team in 
        enumerate(standings.elo.sort_values('ELO', ascending=False)['Team'][:64])
    }
    
    # Create and simulate bracket pool
    results = simulate_bracket_pool(standings, example_seeds, num_entries=10)
    print("\nPool Simulation Results:")
    print(results.to_string(index=False))

if __name__ == "__main__":
    main()
