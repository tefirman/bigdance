#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   cbb_brackets.py
@Time    :   2024/01/11
@Author  :   Taylor Firman
@Version :   0.2.0
@Contact :   tefirman@gmail.com
@Desc    :   Generalized March Madness bracket simulation package
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class Team:
    """
    Represents a tournament team with relevant attributes.

    Attributes:
        name: Team name
        seed: Tournament seed (1-16)
        region: Tournament region
        rating: Team rating (e.g. Elo, KenPom)
        conference: Conference name

    Raises:
        ValueError: If seed is not between 1 and 16
    """

    name: str
    seed: int
    region: str
    rating: float  # Any rating system (Elo, KenPom, etc.)
    conference: str

    def __post_init__(self):
        """Validate team attributes after initialization"""
        if not isinstance(self.seed, int) or self.seed < 1 or self.seed > 16:
            raise ValueError(
                f"Seed must be an integer between 1 and 16, got {self.seed}"
            )

        if not self.name or not isinstance(self.name, str):
            raise ValueError("Team name must be a non-empty string")

        if not self.region or not isinstance(self.region, str):
            raise ValueError("Region must be a non-empty string")

        if not isinstance(self.rating, (int, float)):
            raise ValueError("Rating must be a numeric value")

        if not self.conference or not isinstance(self.conference, str):
            raise ValueError("Conference must be a non-empty string")


@dataclass
class Game:
    """Represents a tournament game between two teams"""

    team1: Team
    team2: Team
    round: int
    region: str
    upset_factor: float = None
    winner: Optional[Team] = None
    actual_winner: Optional[Team] = None  # For comparing to real results


@dataclass
class Bracket:
    """
    Represents a tournament bracket, either actual results or a contestant's picks
    """

    teams: List[Team]
    games: List[Game] = field(default_factory=list)
    results: Dict[str, List[Team]] = field(default_factory=dict)
    log_probability: float = float("inf")  # Initialize to infinity
    log_probability_by_round: Dict[str, float] = field(default_factory=dict)
    underdogs_by_round: Dict[str, List[Team]] = field(default_factory=dict)

    def __post_init__(self):
        """Called after dataclass auto-generated __init__"""
        self._validate_teams()
        if self.teams:
            self._create_initial_games()

    def _validate_teams(self):
        """Validate tournament teams meet requirements"""
        if len(self.teams) != 64:
            raise ValueError("Tournament must have exactly 64 teams")

        # Validate seeds and regions
        regions = set()
        for team in self.teams:
            if not 1 <= team.seed <= 16:
                raise ValueError(f"Invalid seed {team.seed} for {team.name}")
            regions.add(team.region)

        if len(regions) != 4:
            raise ValueError("Tournament must have exactly 4 regions")

    def _create_initial_games(self):
        """Create first round matchups based on NCAA tournament seeding pattern"""
        if not self.teams:
            return

        # Define first round matchup pattern [(seed1, seed2), ...]
        seed_matchups = [
            (1, 16),
            (8, 9),
            (5, 12),
            (4, 13),
            (6, 11),
            (3, 14),
            (7, 10),
            (2, 15),
        ]

        # Create games for each region
        for region in ["East", "West", "South", "Midwest"]:
            # Get teams in this region
            region_teams = {t.seed: t for t in self.teams if t.region == region}

            # Create games following seed matchup pattern
            for seed1, seed2 in seed_matchups:
                if seed1 not in region_teams or seed2 not in region_teams:
                    raise ValueError(
                        f"Missing seed {seed1} or {seed2} in {region} region"
                    )

                self.games.append(
                    Game(
                        team1=region_teams[seed1],
                        team2=region_teams[seed2],
                        round=1,
                        region=region,
                    )
                )

    def simulate_game(self, game: Game, upset_factor: float = None) -> Team:
        """
        Simulate single game outcome with upset factor adjustment.

        Args:
            game: Game to simulate
            upset_factor: Adjustment to probability (-1.0 to 1.0)
                - If None, uses game.upset_factor if available, otherwise defaults to 0.0
                - negative values: favor chalk (reduce upsets)
                - zero: pure rating-based probability
                - positive values: favor underdogs (increase upsets)

        Returns:
            Winning team
        """
        # Use game's upset factor if available and no override was provided
        if upset_factor is None:
            upset_factor = getattr(game, "upset_factor", 0.0)
        if upset_factor is None:
            upset_factor = 0.0

        # Determine favorite and underdog based on seed (lower seed is better)
        if game.team1.seed < game.team2.seed:
            favorite, underdog = game.team1, game.team2
        elif game.team1.seed > game.team2.seed:
            favorite, underdog = game.team2, game.team1
        else:
            # Equal seeds, determine by rating
            if game.team1.rating > game.team2.rating:
                favorite, underdog = game.team1, game.team2
            else:
                favorite, underdog = game.team2, game.team1

        # Calculate favorite's base win probability using Elo formula
        rating_diff = favorite.rating - underdog.rating

        # Calculate probability with adjusted rating diff
        base_prob = 1 / (1 + 10 ** (-rating_diff / 400))

        # Ensure the upset factor is within bounds
        effective_upset_factor = np.clip(upset_factor, -1.0, 1.0)

        # Apply the upset factor differently based on its sign
        if effective_upset_factor >= 0:
            # Move toward 50-50
            adjusted_prob = base_prob * (1 - effective_upset_factor) + (
                0.5 * effective_upset_factor
            )
        else:
            # Move toward 100% (chalk)
            # Convert to positive for calculation
            chalk_factor = abs(effective_upset_factor)
            # Move from base_prob toward 1.0
            adjusted_prob = base_prob * (1 - chalk_factor) + (1.0 * chalk_factor)

        # Determine winner
        random_value = np.random.random()
        if random_value < adjusted_prob:
            return favorite
        else:
            return underdog

    def advance_round(self, games: List[Game]) -> List[Game]:
        """
        Simulate games and create matchups for next round

        Args:
            games: List of games in current round

        Returns:
            List of games for next round
        """
        next_games = []
        winners = []

        # Get the upset_factor to pass to next round (using first game's value)
        # Assuming all games in current round have the same upset_factor
        upset_factor = getattr(games[0], "upset_factor", None) if games else None

        # Simulate current round's games and collect winners
        for game in games:
            if not game.winner:  # Only simulate if winner not already set
                game.winner = self.simulate_game(game)
            winners.append(game.winner)

        # Create next round matchups
        if len(winners) > 1:  # Not championship game
            for i in range(0, len(winners), 2):
                next_game = Game(
                    team1=winners[i],
                    team2=winners[i + 1],
                    round=games[0].round + 1,
                    region=(
                        winners[i].region
                        if winners[i].region == winners[i + 1].region
                        else "Final Four"
                    ),
                    upset_factor=upset_factor,  # Pass along the upset_factor
                )
                next_games.append(next_game)

        return next_games

    def calculate_game_probability(self, game: Game) -> float:
        """Calculate probability of game outcome based on Elo ratings"""
        rating_diff = game.winner.rating - (
            game.team2.rating if game.winner == game.team1 else game.team1.rating
        )
        return 1 / (1 + 10 ** (-rating_diff / 400))

    def calculate_log_probability(self) -> float:
        """Calculate the negative log probability of the entire bracket outcome"""
        if not self.results:
            return float("inf")

        total_log_prob = 0
        round_names = {
            1: "First Round",
            2: "Second Round",
            3: "Sweet 16",
            4: "Elite 8",
            5: "Final Four",
            6: "Championship",
        }

        # Initialize log probabilities by round
        self.log_probability_by_round = {
            round_name: 0.0 for round_name in round_names.values()
        }

        # Track teams that advance from each round to calculate probabilities
        round_outcomes = {}

        # First round probabilities from initial games
        if "First Round" in self.results:
            first_round_winners = {
                team.name: team for team in self.results["First Round"]
            }
            round_log_prob = 0.0  # Track log probability for this round
            for (
                game
            ) in self.games:  # This only contains first round games from initialization
                if game.winner and game.winner.name in first_round_winners:
                    prob = self.calculate_game_probability(game)
                    game_log_prob = -np.log(prob)
                    round_log_prob += game_log_prob  # Add to round total
                    total_log_prob += game_log_prob
                    # Track teams for next rounds
                    round_outcomes[(game.team1.name, game.team2.name)] = game.winner

            # Store log probability for this round
            self.log_probability_by_round["First Round"] = round_log_prob

        # Calculate probabilities for subsequent rounds using results dictionary
        for round_num in range(2, 7):
            round_name = round_names[round_num]
            if round_name not in self.results:
                continue

            round_winners = {team.name: team for team in self.results[round_name]}
            prev_round_name = round_names[round_num - 1]
            prev_winners = self.results.get(prev_round_name, [])

            round_log_prob = 0.0  # Track log probability for this round

            # Create matchups from previous round winners
            for i in range(0, len(prev_winners), 2):
                if i + 1 < len(prev_winners):
                    team1, team2 = prev_winners[i], prev_winners[i + 1]

                    # Create a temporary game to calculate probability
                    temp_game = Game(
                        team1=team1,
                        team2=team2,
                        round=round_num,
                        region=(
                            team1.region
                            if team1.region == team2.region
                            else "Final Four"
                        ),
                    )

                    # Determine winner based on results
                    if team1.name in round_winners:
                        temp_game.winner = team1
                    elif team2.name in round_winners:
                        temp_game.winner = team2

                    if temp_game.winner:
                        prob = self.calculate_game_probability(temp_game)
                        game_log_prob = -np.log(prob)
                        round_log_prob += game_log_prob  # Add to round total
                        total_log_prob += game_log_prob

            # Store log probability for this round
            self.log_probability_by_round[round_name] = round_log_prob

        return total_log_prob

    def is_underdog(self, team: Team, round_name: str) -> bool:
        """
        Determine if a team is considered an underdog based on seeding expectations for the round.

        Args:
            team: Team to evaluate
            round_name: Tournament round name

        Returns:
            True if the team is an underdog for this round, False otherwise
        """
        seed_thresholds = {
            "First Round": 8,  # Seeds 9-16 are underdogs
            "Second Round": 4,  # Seeds 5-16 are underdogs
            "Sweet 16": 2,  # Seeds 3-16 are underdogs
            "Elite 8": 1,  # Seeds 2-16 are underdogs
            "Final Four": 1,  # Seeds 2-16 are underdogs
            "Championship": 1,  # Seeds 2-16 are underdogs
        }

        threshold = seed_thresholds.get(round_name, 1)
        return team.seed > threshold

    def identify_underdogs(self) -> Dict[str, List[Team]]:
        """
        Identify underdog teams in all rounds of bracket results.
        An underdog is defined as a team with a seed lower than typically expected
        for advancement in a given round.

        Returns:
            Dictionary mapping round names to lists of underdog teams
        """
        if not self.results:
            return {}

        underdogs = {}

        # Check each round
        for round_name, teams in self.results.items():
            if round_name == "Champion":
                continue  # Skip the single champion result

            round_underdogs = [
                team for team in teams if self.is_underdog(team, round_name)
            ]
            if round_underdogs:
                underdogs[round_name] = round_underdogs

        self.underdogs_by_round = underdogs
        return underdogs

    def count_underdogs_by_round(self) -> Dict[str, int]:
        """
        Count the number of underdogs in each round.

        Returns:
            Dictionary mapping round names to count of underdogs
        """
        if not self.underdogs_by_round and self.results:
            self.identify_underdogs()

        return {
            round_name: len(teams)
            for round_name, teams in self.underdogs_by_round.items()
        }

    def total_underdogs(self) -> int:
        """
        Count the total number of underdogs across all rounds.

        Returns:
            Total number of underdogs in the bracket
        """
        if not self.underdogs_by_round and self.results:
            self.identify_underdogs()

        return sum(len(teams) for teams in self.underdogs_by_round.values())

    def simulate_tournament(self) -> Dict[str, List[Team]]:
        """Simulate entire tournament and store results"""
        self.results = {}  # Reset results
        self.underdogs_by_round = {}  # Reset underdogs tracking

        # Use only first round games from the initial setup
        current_games = (
            self.games.copy()
        )  # These are all first round games from initialization

        # First round
        for game in current_games:
            game.winner = self.simulate_game(game)
        self.results["First Round"] = [g.winner for g in current_games]

        # Subsequent rounds
        for round_name in [
            "Second Round",
            "Sweet 16",
            "Elite 8",
            "Final Four",
            "Championship",
        ]:
            current_games = self.advance_round(current_games)
            for game in current_games:
                if not game.winner:
                    game.winner = self.simulate_game(game)

            if round_name == "Championship":
                self.results[round_name] = [current_games[0].winner]
                self.results["Champion"] = current_games[0].winner
            else:
                self.results[round_name] = [g.winner for g in current_games]

        # Calculate log probability of final bracket
        self.log_probability = self.calculate_log_probability()

        # Identify underdogs in the results
        self.identify_underdogs()

        return self.results


class Pool:
    """
    Represents a tournament pool with multiple bracket entries
    """

    def __init__(self, actual_results: Bracket):
        """Initialize pool with actual tournament results for comparison"""
        self.actual_results = actual_results
        self.entries: List[Tuple[str, Bracket, bool]] = (
            []
        )  # name, bracket, simulate_flag
        self.actual_tournament = None  # Store the one true tournament outcome

    def add_entry(self, name: str, bracket: Bracket, simulate: bool = True):
        """
        Add new bracket entry to pool

        Args:
            name: Entry name
            bracket: Bracket to add
            simulate: Whether to simulate this bracket during pool simulation (default: True)
                      Set to False for user-picked brackets that should be preserved
        """
        self.entries.append((name, bracket, simulate))

    def score_bracket(
        self, entry_results: Dict[str, List[Team]], round_values: Dict[str, int] = None
    ) -> int:
        """
        Score a bracket against actual results

        Args:
            bracket: Bracket to score
            round_values: Points per correct pick in each round

        Returns:
            Total points scored
        """
        if self.actual_tournament is None:
            raise ValueError("Must simulate actual tournament before scoring entries")
        if round_values is None:
            round_values = {
                "First Round": 1,
                "Second Round": 2,
                "Sweet 16": 4,
                "Elite 8": 8,
                "Final Four": 16,
                "Championship": 32,
            }

        points = 0
        for round_name, value in round_values.items():
            if round_name in entry_results and round_name in self.actual_tournament:
                entry_winners = set(t.name for t in entry_results[round_name])
                actual_winners = set(t.name for t in self.actual_tournament[round_name])
                points += len(entry_winners & actual_winners) * value

        return points

    def simulate_pool(self, num_sims: int = 1000) -> pd.DataFrame:
        """
        Simulate pool multiple times and calculate winning probabilities

        Args:
            num_sims: Number of simulations to run

        Returns:
            DataFrame with simulation statistics for each entry
        """
        results = []

        # NEW: Additional data to track
        round_log_probs = {
            entry_name: {
                round_name: []
                for round_name in [
                    "First Round",
                    "Second Round",
                    "Sweet 16",
                    "Elite 8",
                    "Final Four",
                    "Championship",
                ]
            }
            for entry_name, _, _ in self.entries
        }

        for sim in range(num_sims):
            # Simulate actual tournament once per simulation
            self.actual_tournament = self.actual_results.simulate_tournament()

            scores = []
            for name, entry, should_simulate in self.entries:
                if should_simulate:
                    # Simulate this entry - typically for computer-generated entries
                    entry_results = entry.simulate_tournament()
                else:
                    # Don't re-simulate - typically for user entries with fixed picks
                    # Just use the existing results
                    entry_results = entry.results

                    # If the entry hasn't been simulated yet, do it once
                    if not entry_results:
                        entry_results = entry.simulate_tournament()

                score = self.score_bracket(entry_results)
                scores.append({"name": name, "score": score})

                # NEW: Record per-round log probabilities
                for round_name, log_prob in entry.log_probability_by_round.items():
                    round_log_probs[name][round_name].append(log_prob)

            # Find winner(s)
            scores_df = pd.DataFrame(scores)
            max_score = scores_df["score"].max()
            winners = scores_df[scores_df["score"] == max_score]["name"].tolist()

            # In case of ties, split the win
            win_share = 1.0 / len(winners)
            for name in winners:
                results.append(
                    {
                        "simulation": sim,
                        "name": name,
                        "score": scores_df[scores_df["name"] == name]["score"].iloc[0],
                        "win_share": win_share,
                    }
                )

        # Aggregate results
        results_df = pd.DataFrame(results)
        summary = (
            results_df.groupby("name")
            .agg({"score": ["mean", "std"], "win_share": "sum"})
            .reset_index()
        )

        summary.columns = ["name", "avg_score", "std_score", "wins"]
        summary["win_pct"] = summary["wins"] / num_sims

        # NEW: Add per-round log probability statistics to the summary
        for round_name in [
            "First Round",
            "Second Round",
            "Sweet 16",
            "Elite 8",
            "Final Four",
            "Championship",
        ]:
            summary[f"{round_name}_avg_log_prob"] = [
                (
                    np.mean(round_log_probs[name][round_name])
                    if round_log_probs[name][round_name]
                    else np.nan
                )
                for name in summary["name"]
            ]

        return summary.sort_values("win_pct", ascending=False, ignore_index=True)


def main():
    """Example usage of bracket simulation"""
    # Create some example teams
    teams = []
    regions = ["East", "West", "South", "Midwest"]
    seeds = list(range(1, 17))

    for region in regions:
        for seed in seeds:
            # Create fictional team with rating roughly correlated to seed
            rating = 2000 - (seed * 50) + np.random.normal(0, 25)
            teams.append(
                Team(
                    name=f"{region} {seed} Seed",
                    seed=seed,
                    region=region,
                    rating=rating,
                    conference="Conference " + str((seed - 1) // 4 + 1),
                )
            )

    # Create actual tournament results
    actual_bracket = Bracket(teams)

    # Create pool with multiple entries
    pool = Pool(actual_bracket)

    # Add some entries
    for i in range(10):
        entry = Bracket(teams)  # Each entry gets fresh simulation
        pool.add_entry(f"Entry {i+1}", entry)

    # Simulate pool
    results = pool.simulate_pool(num_sims=1000)
    print("\nPool Simulation Results:")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
