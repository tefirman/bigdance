#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   cbb_brackets.py
@Time    :   2024/01/11
@Author  :   Taylor Firman
@Version :   0.1.0
@Contact :   tefirman@gmail.com
@Desc    :   Generalized March Madness bracket simulation package
'''

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np

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
            raise ValueError(f"Seed must be an integer between 1 and 16, got {self.seed}")
        
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
            (1, 16), (8, 9), (5, 12), (4, 13),
            (6, 11), (3, 14), (7, 10), (2, 15)
        ]
        
        # Create games for each region
        for region in ["East", "West", "South", "Midwest"]:
            # Get teams in this region
            region_teams = {t.seed: t for t in self.teams if t.region == region}
            
            # Create games following seed matchup pattern
            for seed1, seed2 in seed_matchups:
                if seed1 not in region_teams or seed2 not in region_teams:
                    raise ValueError(f"Missing seed {seed1} or {seed2} in {region} region")
                    
                self.games.append(Game(
                    team1=region_teams[seed1],
                    team2=region_teams[seed2],
                    round=1,
                    region=region
                ))

    def simulate_game(self, game: Game, upset_factor: float = 0.1) -> Team:
        """
        Simulate single game outcome using rating differential and optional upset factor
        
        Args:
            game: Game to simulate
            upset_factor: Factor to increase randomness/upset probability (0-1)
            
        Returns:
            Winning team
        """
        rating_diff = game.team1.rating - game.team2.rating
        base_prob = 1 / (1 + 10**(-rating_diff/400))  # Basic Elo formula
        
        # Apply upset factor to make results less predictable
        prob = (base_prob * (1 - upset_factor)) + (0.5 * upset_factor)
        
        return game.team1 if np.random.random() < prob else game.team2

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
        
        # Simulate current round's games and collect winners
        for game in games:
            if not game.winner:  # Only simulate if winner not already set
                game.winner = self.simulate_game(game)
            winners.append(game.winner)
        
        # Create next round matchups
        if len(winners) > 1:  # Not championship game
            for i in range(0, len(winners), 2):
                next_games.append(Game(
                    team1=winners[i],
                    team2=winners[i+1],
                    round=games[0].round + 1,
                    region=winners[i].region if winners[i].region == winners[i+1].region else "Final Four"
                ))
        
        return next_games

    def simulate_tournament(self) -> Dict[str, List[Team]]:
        """Simulate entire tournament and store results"""
        self.results = {}  # Reset results
        current_games = self.games.copy()
        
        # First round
        for game in current_games:
            game.winner = self.simulate_game(game)
        self.results["First Round"] = [g.winner for g in current_games]
        
        # Subsequent rounds...
        for round_name in ["Second Round", "Sweet 16", "Elite 8", "Final Four", "Championship"]:
            current_games = self.advance_round(current_games)
            for game in current_games:
                if not game.winner:
                    game.winner = self.simulate_game(game)
            
            if round_name == "Championship":
                self.results[round_name] = [current_games[0].winner]
                self.results["Champion"] = current_games[0].winner
            else:
                self.results[round_name] = [g.winner for g in current_games]
        
        return self.results

class Pool:
    """
    Represents a tournament pool with multiple bracket entries
    """
    def __init__(self, actual_results: Bracket):
        """Initialize pool with actual tournament results for comparison"""
        self.actual_results = actual_results
        self.entries: List[Tuple[str, Bracket]] = []
        self.actual_tournament = None  # Store the one true tournament outcome
        
    def add_entry(self, name: str, bracket: Bracket):
        """Add new bracket entry to pool"""
        self.entries.append((name, bracket))
        
    def score_bracket(self, entry_results: Dict[str, List[Team]], round_values: Dict[str, int] = None) -> int:
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
                "Championship": 32
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
        
        for sim in range(num_sims):
            # Simulate actual tournament once per simulation
            self.actual_tournament = self.actual_results.simulate_tournament()
            
            scores = []
            for name, entry in self.entries:
                entry_results = entry.simulate_tournament()
                score = self.score_bracket(entry_results)  # Pass in already simulated results
                scores.append({"name": name, "score": score})
                
            # Find winner(s)
            scores_df = pd.DataFrame(scores)
            max_score = scores_df["score"].max()
            winners = scores_df[scores_df["score"] == max_score]["name"].tolist()
            
            # In case of ties, split the win
            win_share = 1.0 / len(winners)
            for name in winners:
                results.append({
                    "simulation": sim,
                    "name": name,
                    "score": scores_df[scores_df["name"] == name]["score"].iloc[0],
                    "win_share": win_share
                })
                
        # Aggregate results
        results_df = pd.DataFrame(results)
        summary = results_df.groupby("name").agg({
            "score": ["mean", "std"],
            "win_share": "sum"
        }).reset_index()
        
        summary.columns = ["name", "avg_score", "std_score", "wins"]
        summary["win_pct"] = summary["wins"] / num_sims
        
        return summary.sort_values("win_pct", ascending=False)

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
            teams.append(Team(
                name=f"{region} {seed} Seed",
                seed=seed,
                region=region,
                rating=rating,
                conference="Conference " + str((seed - 1) // 4 + 1)
            ))
    
    # Create actual tournament results
    actual_bracket = Bracket(teams)
    actual_results = actual_bracket.simulate_tournament()
    
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
