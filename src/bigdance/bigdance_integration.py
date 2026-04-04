#!/usr/bin/env python
"""
@File    :   bigdance_integration.py
@Time    :   2025/01/19
@Author  :   Taylor Firman
@Version :   0.8.5
@Contact :   tefirman@gmail.com
@Desc    :   Integration module between Warren Nolan scraper and bracket simulator
"""

import argparse
from typing import Optional

import numpy as np
import pandas as pd

from bigdance.cbb_brackets import Bracket, Game, Pool, Team
from bigdance.wn_cbb_scraper import Standings


def create_teams_from_standings(
    standings: Standings, regions: Optional[dict[str, str]] = None
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
    at_large_bids = at_large_pool.sort_values("ELO", ascending=False).head(remaining_spots)

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
    team_counts = pd.DataFrame([(t.region, t.seed) for t in teams], columns=["region", "seed"])
    team_counts = team_counts.groupby(["region", "seed"]).size().reset_index(name="count")
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
    current_round_games = bracket.games.copy()  # Start with first round games from initialization

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
            bracket.results["Champion"] = winners_for_round[0]  # type: ignore[assignment]

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
                        region=(team1.region if team1.region == team2.region else "Final Four"),
                    )
                    next_round_games.append(next_game)

        # Update current games for the next iteration
        current_round_games = next_round_games

    # Calculate log probability for this bracket
    bracket.log_probability = bracket.calculate_log_probability()

    return bracket


def simulate_hypothetical_bracket_pool(
    standings: Optional[Standings] = None,
    num_entries: int = 100,
    upset_factors: Optional[list[float]] = None,
    women: bool = False,
) -> pd.DataFrame:
    """
    Illustrative example of how to create and customize your own bracket pool
    using the framework described in cbb_brackets.

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
        factors_array = np.random.normal(0, 0.3, num_entries)

        # Clip values to stay within -1.0 to 1.0 range
        factors_array = np.clip(factors_array, -1.0, 1.0)

        # Ensure we include some extreme values for variety
        if num_entries >= 10:
            # Include at least one strong chalk picker
            factors_array[0] = -0.8
            # Include at least one strong upset picker
            factors_array[1] = 0.8
            # Include at least one pure elo-based picker
            factors_array[2] = 0.0
            # Shuffle to randomize positions
            np.random.shuffle(factors_array)

        upset_factors = factors_array.tolist()

    elif len(upset_factors) != num_entries:
        raise ValueError("Number of upset factors must match number of entries")

    # Create entries with varying upset factors
    for i in range(num_entries):
        entry_bracket = create_teams_from_standings(standings)
        # Set upset factor for all games in this entry
        for game in entry_bracket.games:
            game.upset_factor = upset_factors[i]
        entry_name = f"Entry_{i + 1}"
        pool.add_entry(entry_name, entry_bracket)

    # Simulate pool with single reality per simulation
    results = pool.simulate_pool(num_sims=1000)
    return results


def simulate_round_probabilities(
    standings: Optional[Standings] = None,
    num_sims: int = 1000,
    upset_factor: float = 0.25,
    women: bool = False,
    bracket: Optional["Bracket"] = None,
) -> pd.DataFrame:
    """
    Simulate the tournament many times and compute each team's probability of
    reaching each round.

    Args:
        standings: Standings object. If None, fetches current standings.
                   Ignored if bracket is provided.
        num_sims: Number of tournament simulations to run.
        upset_factor: Upset factor applied to every game (-1.0 = chalk, 1.0 = coin flip).
        women: Whether to use women's basketball data. Ignored if bracket is provided.
        bracket: Pre-built Bracket object (e.g. from ESPN). If provided, standings
                 and women are ignored and this bracket is used as the template.

    Returns:
        DataFrame with columns: Team, Seed, Region, and one column per round
        showing the percentage chance of reaching that round, sorted by
        championship probability descending.
    """
    import copy

    rounds = ["First Round", "Second Round", "Sweet 16", "Elite 8", "Final Four", "Championship"]
    counts: dict[str, dict[str, int]] = {}

    if bracket is not None:
        template = bracket
    else:
        if standings is None:
            standings = Standings(women=women)
        template = create_teams_from_standings(standings)

    team_info = {t.name: (t.seed, t.region) for t in template.teams}
    for name in team_info:
        counts[name] = {r: 0 for r in rounds}

    for _ in range(num_sims):
        if bracket is not None:
            b = copy.deepcopy(bracket)
        else:
            b = create_teams_from_standings(standings)  # type: ignore[arg-type]
        for game in b.games:
            game.upset_factor = upset_factor
        results = b.simulate_tournament()
        for round_name in rounds:
            for team in results.get(round_name, []):
                counts[team.name][round_name] += 1

    rows = []
    for name, (seed, region) in team_info.items():
        row = {"Team": name, "Seed": seed, "Region": region}
        for r in rounds:
            row[r] = f"{counts[name][r] / num_sims:.1%}"
        rows.append(row)

    df = pd.DataFrame(rows)
    # Sort by championship probability descending, then seed ascending
    df["_champ"] = df["Championship"].str.rstrip("%").astype(float)
    df = df.sort_values(["_champ", "Seed"], ascending=[False, True]).drop(columns="_champ")
    return df.reset_index(drop=True)


def main(argv=None):
    """Simulate tournament outcomes and show each team's probability of reaching each round."""
    parser = argparse.ArgumentParser(
        description="Simulate March Madness and show round-by-round probabilities per team"
    )
    parser.add_argument("--num_sims", type=int, default=1000, help="Number of simulations to run")
    parser.add_argument(
        "--upset_factor",
        type=float,
        default=0.25,
        help="Upset factor for all games (-1.0 = chalk, 1.0 = coin flip, ~0.25 matches history)",
    )
    parser.add_argument(
        "--gender",
        choices=["men", "women"],
        default="men",
        help="Which tournament to use (default: men)",
    )
    parser.add_argument(
        "--conference", type=str, default=None, help="Filter by specific conference"
    )
    parser.add_argument("--top", type=int, default=None, help="Show only the top N teams")

    args = parser.parse_args(argv)
    women = args.gender == "women"

    standings = Standings(conference=args.conference, women=women)

    print(f"\nSimulating {args.num_sims} tournaments (upset_factor={args.upset_factor})...\n")
    df = simulate_round_probabilities(
        standings=standings,
        num_sims=args.num_sims,
        upset_factor=args.upset_factor,
        women=women,
    )

    if args.top is not None:
        df = df.head(args.top)

    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
