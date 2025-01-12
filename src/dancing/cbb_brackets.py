#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   march_madness.py
@Time    :   2024/01/11
@Author  :   Taylor Firman
@Version :   0.1.0
@Contact :   tefirman@gmail.com
@Desc    :   Generalized March Madness bracket simulation package
'''

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime

@dataclass
class Team:
    """Represents a tournament team with relevant attributes"""
    name: str
    seed: int
    region: str
    rating: float  # Any rating system (Elo, KenPom, etc.)
    conference: str

@dataclass
class Game:
    """Represents a tournament game between two teams"""
    team1: Team
    team2: Team
    round: int
    region: str
    winner: Optional[Team] = None
    actual_winner: Optional[Team] = None  # For comparing to real results

class Bracket:
    """
    Represents a tournament bracket, either actual results or a contestant's picks
    """
    def __init__(self, teams: List[Team] = None):
        """Initialize bracket with list of tournament teams"""
        self.teams = teams or []
        self.games: List[Game] = []
        self._validate_teams()
        if teams:
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
        """Create first round matchups based on seeding"""
        for region in set(team.region for team in self.teams):
            region_teams = [t for t in self.teams if t.region == region]
            region_teams.sort(key=lambda x: x.seed)
            
            # Create 1v16, 2v15, etc. matchups
            for i in range(8):
                self.games.append(Game(
                    team1=region_teams[i],
                    team2=region_teams[15-i],
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
        
        # Simulate current round's games
        for game in games:
            if not game.winner:  # Don't re-simulate if winner already set
                game.winner = self.simulate_game(game)
            winners.append(game.winner)
        
        # Create next round matchups
        if len(games) > 1:  # Not championship game
            for i in range(0, len(winners), 2):
                next_games.append(Game(
                    team1=winners[i],
                    team2=winners[i+1],
                    round=games[0].round + 1,
                    region=winners[i].region if winners[i].region == winners[i+1].region else "Final Four"
                ))
        
        return next_games

    def simulate_tournament(self) -> Dict[str, List[Team]]:
        """
        Simulate entire tournament
        
        Returns:
            Dictionary mapping round names to lists of advancing teams
        """
        results = {}
        games_by_round = {}
        
        # First round
        games_by_round[1] = self.games.copy()
        for game in games_by_round[1]:
            if not game.winner:
                game.winner = self.simulate_game(game)
        results["First Round"] = [g.winner for g in games_by_round[1]]
        
        # Subsequent rounds
        round_names = ["Second Round", "Sweet 16", "Elite 8", "Final Four", "Championship"]
        current_round = 1
        
        for round_name in round_names:
            current_round += 1
            # Advance winners from previous round into new matchups
            games_by_round[current_round] = self.advance_round(games_by_round[current_round-1])
            results[round_name] = [g.winner for g in games_by_round[current_round-1]]
        results["Champion"] = self.simulate_game(games_by_round[current_round][0])
        return results

class Pool:
    """
    Represents a tournament pool with multiple bracket entries
    """
    def __init__(self, actual_results: Bracket):
        """Initialize pool with actual tournament results for comparison"""
        self.actual_results = actual_results
        self.entries: List[Tuple[str, Bracket]] = []
        
    def add_entry(self, name: str, bracket: Bracket):
        """Add new bracket entry to pool"""
        self.entries.append((name, bracket))
        
    def score_bracket(self, bracket: Bracket, round_values: Dict[str, int] = None) -> int:
        """
        Score a bracket against actual results
        
        Args:
            bracket: Bracket to score
            round_values: Points per correct pick in each round
            
        Returns:
            Total points scored
        """
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
        bracket_results = bracket.simulate_tournament()
        actual_results = self.actual_results.simulate_tournament()
        
        for round_name, value in round_values.items():
            bracket_winners = set(t.name for t in bracket_results[round_name])
            actual_winners = set(t.name for t in actual_results[round_name])
            points += len(bracket_winners & actual_winners) * value
            
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
        
        # Store teams for creating fresh brackets each time
        teams = self.actual_results.teams
        
        for _ in range(num_sims):
            # Create fresh actual results bracket for this simulation
            self.actual_results = Bracket(teams)
            
            scores = []
            for name, _ in self.entries:  # Original bracket not used
                # Create fresh entry bracket for this simulation
                entry_bracket = Bracket(teams)
                score = self.score_bracket(entry_bracket)
                scores.append({"name": name, "score": score})
                
            # Find winner(s)
            scores_df = pd.DataFrame(scores)
            max_score = scores_df["score"].max()
            winners = scores_df[scores_df["score"] == max_score]["name"].tolist()
            
            # In case of ties, split the win
            win_share = 1.0 / len(winners)
            for name in winners:
                results.append({
                    "simulation": _,
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
