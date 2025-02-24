#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   bigdance_integration.py
@Time    :   2025/01/19
@Author  :   Taylor Firman
@Version :   0.1.0
@Contact :   tefirman@gmail.com
@Desc    :   Integration module between Warren Nolan scraper and bracket simulator
'''

from typing import List, Dict, Optional
import pandas as pd
from bigdance.wn_cbb_scraper import Standings
from bigdance.cbb_brackets import Team, Bracket, Pool

def create_teams_from_standings(standings: Standings, 
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
    regions = regions or {}

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
    
    seeds = {}
    # Split teams into 16 groups of 4 for seeding
    for seed_num in range(1, 17):
        seed_group = tournament_teams.iloc[(seed_num-1)*4:seed_num*4]
        for _, team in seed_group.iterrows():
            seeds[team['Team']] = seed_num

    # Track region assignments and counts
    region_seed_counts = {region: {seed: 0 for seed in range(1, 17)} for region in TOURNAMENT_REGIONS}
    team_regions = regions.copy()  # Start with provided regions

    # For each seed line (1-16), distribute remaining teams across regions
    for seed_num in range(1, 17):
        seed_teams = tournament_teams[tournament_teams['Team'].map(seeds) == seed_num]
        
        # First, count teams already assigned to regions
        for team_name in seed_teams['Team']:
            if team_name in team_regions:
                region = team_regions[team_name]
                region_seed_counts[region][seed_num] += 1

        # Then assign remaining teams to maintain balance
        for _, team in seed_teams.iterrows():
            if team['Team'] not in team_regions:
                # Find region with fewest teams of this seed
                available_regions = [
                    r for r in TOURNAMENT_REGIONS 
                    if region_seed_counts[r][seed_num] < 1
                ]
                if not available_regions:
                    raise ValueError(f"Cannot find valid region for seed {seed_num}")
                chosen_region = available_regions[0]
                team_regions[team['Team']] = chosen_region
                region_seed_counts[chosen_region][seed_num] += 1

    # Create Team objects with proper seeds and regions
    teams = []
    for _, row in tournament_teams.iterrows():
        teams.append(Team(
            name=row['Team'],
            seed=seeds[row['Team']],
            region=team_regions[row['Team']],
            rating=row['ELO'],
            conference=row['Conference']
        ))
    
    # Validate bracket structure
    team_counts = pd.DataFrame([(t.region, t.seed) for t in teams], columns=['region', 'seed'])
    team_counts = team_counts.groupby(['region', 'seed']).size().reset_index(name='count')
    if not all(team_counts['count'] == 1):
        raise ValueError("Invalid bracket structure: Each region must have exactly one team of each seed")
        
    return Bracket(teams)

def create_bracket_with_picks(teams, picks_by_round):
    """
    Create a bracket with predetermined picks.
    
    Args:
        teams: List of Team objects for the tournament
        picks_by_round: Dictionary with round names as keys and lists of winning team names as values
                       Rounds should be in order: "First Round", "Second Round", "Sweet 16", 
                       "Elite 8", "Final Four", "Championship"
    
    Returns:
        Bracket object with all picks set
    """
    # Create initial bracket
    bracket = Bracket(teams)
    
    # Set winners for each round
    for round_name, winners in picks_by_round.items():
        round_num = {
            "First Round": 1,
            "Second Round": 2,
            "Sweet 16": 3,
            "Elite 8": 4,
            "Final Four": 5,
            "Championship": 6
        }[round_name]
        
        # Find games for this round
        round_games = [g for g in bracket.games if g.round == round_num]
        
        # Set winners for each game
        for game, winner_name in zip(round_games, winners):
            # Find the winning team object
            winner = None
            if game.team1.name == winner_name:
                winner = game.team1
            elif game.team2.name == winner_name:
                winner = game.team2
            else:
                raise ValueError(f"Winner {winner_name} not found in game between {game.team1.name} and {game.team2.name}")
            
            game.winner = winner
    
    # Simulate to populate results dictionary and calculate log probability
    bracket.simulate_tournament()
    
    return bracket

def simulate_bracket_pool(standings: Standings,
                        num_entries: int = 100,
                        upset_factors: Optional[List[float]] = None) -> pd.DataFrame:
    """
    Simulate a bracket pool using Warren Nolan ratings.
    
    Args:
        standings: Standings object containing team ratings and info
        num_entries: Number of bracket entries to simulate
        upset_factors: Optional list of upset factors for each entry
        
    Returns:
        DataFrame containing simulation results
    """
    # Create actual results bracket
    actual_bracket = create_teams_from_standings(standings)
    
    # Initialize pool
    pool = Pool(actual_bracket)
    
    # Generate upset factors if not provided
    if upset_factors is None:
        # upset_factors = [0.1 + (i/num_entries)*0.3 for i in range(num_entries)]
        upset_factors = [0.1]*len(num_entries)
    elif len(upset_factors) != num_entries:
        raise ValueError("Number of upset factors must match number of entries")
    
    # Create entries with varying upset factors
    for i in range(num_entries):
        entry_bracket = create_teams_from_standings(standings)
        # Set upset factor for all games in this entry
        for game in entry_bracket.games:
            game.upset_factor = upset_factors[i]
        entry_name = f"Entry_{i+1}"
        pool.add_entry(entry_name, entry_bracket)
    
    # Simulate pool with single reality per simulation
    results = pool.simulate_pool(num_sims=1000)
    return results

def main():
    """Example usage of integration module"""
    # Get current standings
    standings = Standings()
    
    # Create and simulate bracket pool
    results = simulate_bracket_pool(standings, num_entries=10)
    print("\nPool Simulation Results:")
    print(results.to_string(index=False))

if __name__ == "__main__":
    main()
