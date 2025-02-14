#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   bracket_analysis.py 
@Time    :   2024/02/13
@Author  :   Taylor Firman
@Version :   0.1.0
@Contact :   tefirman@gmail.com
@Desc    :   Analyzing trends in March Madness bracket pool simulations
'''

from typing import List
import pandas as pd
from collections import defaultdict
from dancing.wn_cbb_scraper import Standings
from dancing.cbb_brackets import Bracket, Pool
from dancing.dancing_integration import simulate_bracket_pool

class BracketAnalysis:
    """Class for analyzing trends across multiple bracket pool simulations"""
    
    def __init__(self, standings: Standings, num_pools: int = 100):
        """
        Initialize analysis with standings data
        
        Args:
            standings: Warren Nolan standings data
            num_pools: Number of pools to simulate for analysis
        """
        self.standings = standings
        self.num_pools = num_pools
        self.pools: List[Pool] = []
        self.winning_brackets: List[Bracket] = []
        self.all_results = pd.DataFrame()
        
    def simulate_pools(self, entries_per_pool: int = 100) -> None:
        """
        Simulate multiple bracket pools
        
        Args:
            entries_per_pool: Number of entries in each pool
        """
        for i in range(self.num_pools):
            pool_results = simulate_bracket_pool(
                self.standings,
                num_entries=entries_per_pool
            )
            
            # Store winning bracket from this pool
            winning_entry = pool_results.iloc[0]['name']
            winning_bracket = [entry[1] for entry in self.pools[-1].entries 
                             if entry[0] == winning_entry][0]
            self.winning_brackets.append(winning_bracket)
            
            # Store full results
            pool_results['pool_id'] = i
            self.all_results = pd.concat([self.all_results, pool_results])
            
    def analyze_upsets(self) -> pd.DataFrame:
        """
        Analyze upset patterns in winning brackets
        
        Returns:
            DataFrame containing upset statistics by round
        """
        upset_stats = defaultdict(list)
        
        for bracket in self.winning_brackets:
            results = bracket.simulate_tournament()
            
            for round_name, teams in results.items():
                if round_name != "Champion":  # Skip final result
                    # Count upsets (when lower seed beats higher seed)
                    upsets = sum(1 for team in teams 
                               if any(t for t in bracket.games 
                                    if t.winner == team and t.winner.seed > t.team1.seed))
                    upset_stats[round_name].append(upsets)
        
        # Convert to DataFrame
        stats_df = pd.DataFrame(upset_stats)
        
        # Calculate summary statistics
        summary = pd.DataFrame({
            'round': stats_df.columns,
            'avg_upsets': stats_df.mean(),
            'std_upsets': stats_df.std(),
            'min_upsets': stats_df.min(),
            'max_upsets': stats_df.max()
        })
        
        return summary.sort_values('round')
    
    def find_common_upsets(self) -> pd.DataFrame:
        """
        Identify most common specific upsets in winning brackets
        
        Returns:
            DataFrame containing most frequent specific upsets
        """
        upset_counts = defaultdict(int)
        
        for bracket in self.winning_brackets:
            results = bracket.simulate_tournament()
            
            for round_name, teams in results.items():
                if round_name != "Champion":
                    for team in teams:
                        # Find the game where this team won
                        game = next(g for g in bracket.games 
                                  if g.winner == team)
                        
                        # Check if it was an upset
                        if game.winner.seed > game.team1.seed:
                            key = (round_name, 
                                  f"{game.winner.seed} {game.winner.name}",
                                  f"{game.team1.seed} {game.team1.name}")
                            upset_counts[key] += 1
        
        # Convert to DataFrame
        upsets_df = pd.DataFrame([
            {
                'round': round_name,
                'winner': winner,
                'loser': loser,
                'frequency': count / self.num_pools
            }
            for (round_name, winner, loser), count in upset_counts.items()
        ])
        
        return upsets_df.sort_values('frequency', ascending=False)
    
    def analyze_champion_picks(self) -> pd.DataFrame:
        """
        Analyze championship picks in winning brackets
        
        Returns:
            DataFrame containing champion pick statistics
        """
        champion_counts = defaultdict(int)
        
        for bracket in self.winning_brackets:
            results = bracket.simulate_tournament()
            champion = results['Champion']
            key = (champion.seed, champion.name, champion.conference)
            champion_counts[key] += 1
        
        # Convert to DataFrame
        champions_df = pd.DataFrame([
            {
                'seed': seed,
                'team': team,
                'conference': conf,
                'frequency': count / self.num_pools
            }
            for (seed, team, conf), count in champion_counts.items()
        ])
        
        return champions_df.sort_values('frequency', ascending=False)
    
    def analyze_entry_strategies(self) -> pd.DataFrame:
        """
        Analyze characteristics of winning bracket strategies
        
        Returns:
            DataFrame containing strategy statistics
        """
        strategy_stats = []
        
        for bracket in self.winning_brackets:
            stats = {
                'total_upsets': 0,
                'chalk_picks': 0,  # Number of favorites picked
                'avg_winner_seed': 0,
                'unique_conferences': set()
            }
            
            results = bracket.simulate_tournament()
            
            for round_name, teams in results.items():
                if round_name != "Champion":
                    for team in teams:
                        game = next(g for g in bracket.games 
                                  if g.winner == team)
                        
                        # Count upsets
                        if game.winner.seed > game.team1.seed:
                            stats['total_upsets'] += 1
                        else:
                            stats['chalk_picks'] += 1
                            
                        # Track conferences
                        stats['unique_conferences'].add(team.conference)
                        
                        # Calculate average winner seed
                        stats['avg_winner_seed'] += team.seed
            
            # Finalize calculations
            total_games = len([g for g in bracket.games if g.winner])
            stats['avg_winner_seed'] /= total_games
            stats['conference_diversity'] = len(stats['unique_conferences'])
            del stats['unique_conferences']
            
            strategy_stats.append(stats)
        
        return pd.DataFrame(strategy_stats).describe()

def main():
    """Example usage of bracket analysis"""
    # Get current standings
    standings = Standings()
    
    # Initialize analyzer
    analyzer = BracketAnalysis(standings, num_pools=50)
    
    # Run simulations
    analyzer.simulate_pools(entries_per_pool=100)
    
    # Print various analyses
    print("\nUpset Statistics by Round:")
    print(analyzer.analyze_upsets())
    
    print("\nMost Common Upsets:")
    print(analyzer.find_common_upsets().head(10))
    
    print("\nChampionship Pick Analysis:")
    print(analyzer.analyze_champion_picks().head(10))
    
    print("\nWinning Strategy Statistics:")
    print(analyzer.analyze_entry_strategies())

if __name__ == "__main__":
    main()
