#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   bigdance_integration.py
@Time    :   2025/01/19
@Author  :   Taylor Firman
@Version :   0.2.0
@Contact :   tefirman@gmail.com
@Desc    :   Integration module between Warren Nolan scraper and bracket simulator
"""

import argparse
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from bigdance.cbb_brackets import Bracket, Game, Pool, Team
from bigdance.wn_cbb_scraper import Standings


def create_teams_from_standings(
    standings: Standings, regions: Optional[Dict[str, str]] = None
) -> Bracket:
    """
    Convert Warren Nolan standings into bracket-compatible Team objects.
    Ensures conference champions get automatic bids, then fills remaining spots
    with highest rated teams. Distributes teams across the four tournament regions.

    Args:
        standings: Standings object containing team ratings and info
        regions: Optional dictionary mapping team names to their tournament regions
        women: Whether to use women's basketball data (default: False)

    Returns:
        Bracket object with teams ready for simulation
    """
    TOURNAMENT_REGIONS = ["East", "West", "South", "Midwest"]
    regions = regions or {}

    # First, get the highest rated team from each conference (auto bids)
    auto_bids = (
        standings.elo.sort_values("ELO", ascending=False)
        .groupby("Conference")
        .first()
        .reset_index()
    )

    # Get remaining spots after auto bids
    remaining_spots = 64 - len(auto_bids)

    # Get highest rated teams not already in auto bids
    at_large_pool = standings.elo[~standings.elo["Team"].isin(auto_bids["Team"])]
    at_large_bids = at_large_pool.sort_values("ELO", ascending=False).head(
        remaining_spots
    )

    # Combine auto bids and at-large bids
    tournament_teams = pd.concat([auto_bids, at_large_bids], ignore_index=True)
    tournament_teams = tournament_teams.sort_values("ELO", ascending=False)

    seeds = {}
    # Split teams into 16 groups of 4 for seeding
    for seed_num in range(1, 17):
        seed_group = tournament_teams.iloc[(seed_num - 1) * 4 : seed_num * 4]
        for _, team in seed_group.iterrows():
            seeds[team["Team"]] = seed_num

    # Track region assignments and counts
    region_seed_counts = {
        region: {seed: 0 for seed in range(1, 17)} for region in TOURNAMENT_REGIONS
    }
    team_regions = regions.copy()  # Start with provided regions

    # For each seed line (1-16), distribute remaining teams across regions
    for seed_num in range(1, 17):
        seed_teams = tournament_teams[tournament_teams["Team"].map(seeds) == seed_num]

        # First, count teams already assigned to regions
        for team_name in seed_teams["Team"]:
            if team_name in team_regions:
                region = team_regions[team_name]
                region_seed_counts[region][seed_num] += 1

        # Then assign remaining teams to maintain balance
        for _, team in seed_teams.iterrows():
            if team["Team"] not in team_regions:
                # Find region with fewest teams of this seed
                available_regions = [
                    r for r in TOURNAMENT_REGIONS if region_seed_counts[r][seed_num] < 1
                ]
                if not available_regions:
                    raise ValueError(f"Cannot find valid region for seed {seed_num}")
                chosen_region = available_regions[0]
                team_regions[team["Team"]] = chosen_region
                region_seed_counts[chosen_region][seed_num] += 1

    # Create Team objects with proper seeds and regions
    teams = []
    for _, row in tournament_teams.iterrows():
        teams.append(
            Team(
                name=row["Team"],
                seed=seeds[row["Team"]],
                region=team_regions[row["Team"]],
                rating=row["ELO"],
                conference=row["Conference"],
            )
        )

    # Validate bracket structure
    team_counts = pd.DataFrame(
        [(t.region, t.seed) for t in teams], columns=["region", "seed"]
    )
    team_counts = (
        team_counts.groupby(["region", "seed"]).size().reset_index(name="count")
    )
    if not all(team_counts["count"] == 1):
        raise ValueError(
            "Invalid bracket structure: Each region must have exactly one team of each seed"
        )

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

    # Initialize results dictionary
    bracket.results = {}

    # Map round names to round numbers
    round_nums = {
        "First Round": 1,
        "Second Round": 2,
        "Sweet 16": 3,
        "Elite 8": 4,
        "Final Four": 5,
        "Championship": 6,
    }

    # Working set of games for each round
    current_round_games = (
        bracket.games.copy()
    )  # Start with first round games from initialization

    # Process each round in sequence
    for round_name in [
        "First Round",
        "Second Round",
        "Sweet 16",
        "Elite 8",
        "Final Four",
        "Championship",
    ]:
        if round_name not in picks_by_round:
            continue

        round_num = round_nums[round_name]
        winners_for_round = []
        next_round_games = []

        # Filter to games for the current round
        if round_name == "First Round":
            current_round_games = (
                bracket.games
            )  # These are all first round games from initialization

        # For each game in the current round, set the winner
        for i, game in enumerate(current_round_games):
            if i >= len(picks_by_round[round_name]):
                break  # Not enough picks for this round

            winner_name = picks_by_round[round_name][i]

            # Find the team object for this winner
            winner = None
            if game.team1 and game.team1.name == winner_name:
                winner = game.team1
            elif game.team2 and game.team2.name == winner_name:
                winner = game.team2

            if winner:
                game.winner = winner
                winners_for_round.append(winner)

        # Store the winners for this round in the results dictionary
        bracket.results[round_name] = winners_for_round

        # Special case for championship
        if round_name == "Championship" and winners_for_round:
            bracket.results["Champion"] = winners_for_round[0]

        # Create next round matchups for subsequent rounds
        if len(winners_for_round) > 1:
            for j in range(0, len(winners_for_round), 2):
                if j + 1 < len(winners_for_round):
                    team1, team2 = winners_for_round[j], winners_for_round[j + 1]

                    # Create a game for the next round
                    next_game = Game(
                        team1=team1,
                        team2=team2,
                        round=round_num + 1,
                        region=(
                            team1.region
                            if team1.region == team2.region
                            else "Final Four"
                        ),
                    )
                    next_round_games.append(next_game)

        # Update current games for the next iteration
        current_round_games = next_round_games

    # Calculate log probability for this bracket
    bracket.log_probability = bracket.calculate_log_probability()

    return bracket


def simulate_bracket_pool(
    standings: Optional[Standings] = None,
    num_entries: int = 100,
    upset_factors: Optional[List[float]] = None,
    women: bool = False,
) -> pd.DataFrame:
    """
    Simulate a bracket pool using Warren Nolan ratings.

    Args:
        standings: Standings object containing team ratings and info.
                  If None, will create a new Standings object with the specified gender.
        num_entries: Number of bracket entries to simulate
        upset_factors: Optional list of upset factors for each entry
                      Ranges from -1.0 (extreme chalk) to 1.0 (coin flip)
        women: Whether to use women's basketball data (default: False)

    Returns:
        DataFrame containing simulation results
    """
    # Create standings if not provided
    if standings is None:
        standings = Standings(women=women)

    # Create actual results bracket
    actual_bracket = create_teams_from_standings(standings)

    # Apply a moderate upset factor to the actual tournament result
    # This ensures the actual tournament has a realistic amount of upsets
    for game in actual_bracket.games:
        game.upset_factor = 0.25  # Moderate upset factor for actual tournament

    # Initialize pool
    pool = Pool(actual_bracket)

    # Generate upset factors if not provided
    if upset_factors is None:
        # Create a normal distribution centered around 0 with standard deviation 0.3
        # This gives us a realistic mix of chalk-leaning and upset-leaning entries
        upset_factors = np.random.normal(0, 0.3, num_entries)

        # Clip values to stay within -1.0 to 1.0 range
        upset_factors = np.clip(upset_factors, -1.0, 1.0)

        # Ensure we include some extreme values for variety
        if num_entries >= 10:
            # Include at least one strong chalk picker
            upset_factors[0] = -0.8
            # Include at least one strong upset picker
            upset_factors[1] = 0.8
            # Include at least one pure elo-based picker
            upset_factors[2] = 0.0
            # Shuffle to randomize positions
            np.random.shuffle(upset_factors)

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
    """Example usage of integration module with command-line arguments"""
    parser = argparse.ArgumentParser(description="Simulate March Madness bracket pool")
    parser.add_argument(
        "--num_entries", type=int, default=10, help="Number of entries to simulate"
    )
    parser.add_argument(
        "--num_sims", type=int, default=1000, help="Number of simulations to run"
    )
    parser.add_argument(
        "--women",
        action="store_true",
        help="Use women's basketball data instead of men's",
    )
    parser.add_argument(
        "--conference", type=str, default=None, help="Filter by specific conference"
    )
    parser.add_argument(
        "--upset_min",
        type=float,
        default=-0.5,
        help="Minimum upset factor (-1.0 for extreme chalk)",
    )
    parser.add_argument(
        "--upset_max",
        type=float,
        default=0.5,
        help="Maximum upset factor (1.0 for coin flip)",
    )
    parser.add_argument("--verbose", action="store_true", help="Print verbose output")

    # Parse arguments
    args = parser.parse_args()

    if args.verbose:
        print("Running with the following settings:")
        print(f'  Basketball type: {"Women" if args.women else "Men"}')
        print(f"  Number of entries: {args.num_entries}")
        print(f"  Number of simulations: {args.num_sims}")
        if args.conference:
            print(f"  Conference filter: {args.conference}")
        print(f"  Upset factor range: {args.upset_min} to {args.upset_max}")

    # Get current standings with appropriate gender and conference
    standings = Standings(conference=args.conference, women=args.women)

    # Generate upset factors ranging from min to max
    upset_factors = None
    if args.num_entries > 1:
        upset_factors = [
            args.upset_min
            + (i / (args.num_entries - 1)) * (args.upset_max - args.upset_min)
            for i in range(args.num_entries)
        ]
    else:
        upset_factors = [args.upset_min]  # Just use minimum for a single entry

    # Create and simulate bracket pool
    results = simulate_bracket_pool(
        standings=standings,
        num_entries=args.num_entries,
        upset_factors=upset_factors,
    )

    # Display results
    print("\nPool Simulation Results:")
    print(results.to_string(index=False))

    # Print additional statistics if verbose
    if args.verbose:
        # Calculate average win percentage and score
        avg_win_pct = results["win_pct"].mean()
        avg_score = results["avg_score"].mean()
        print(f"\nAverage win percentage: {avg_win_pct:.4f}")
        print(f"Average score: {avg_score:.2f}")

        # Print top 3 entries
        print("\nTop performing entries:")
        for i, row in results.head(3).iterrows():
            print(
                f"{i+1}. {row['name']}: {row['win_pct']:.1%} win rate, {row['avg_score']:.1f} avg score"
            )


if __name__ == "__main__":
    main()
