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

from typing import List, Dict
import pandas as pd
from collections import defaultdict
from dancing.wn_cbb_scraper import Standings
from dancing.cbb_brackets import Team, Pool
from dancing.dancing_integration import create_teams_from_standings
from datetime import datetime
import numpy as np

class BracketAnalysis:
    """Class for analyzing trends across multiple bracket pool simulations"""
    
    ROUND_ORDER = [
        "First Round",
        "Second Round", 
        "Sweet 16",
        "Elite 8",
        "Final Four",
        "Championship"
    ]
    
    def __init__(self, standings: Standings, num_pools: int = 100):
        self.standings = standings
        self.num_pools = num_pools
        self.pools: List[Pool] = []
        self.winning_results: List[Dict[str, List[Team]]] = []  # Store results instead of brackets
        self.all_results = pd.DataFrame()
        
    def simulate_pools(self, entries_per_pool: int = 10) -> None:
        """Simulate multiple bracket pools"""
        print(f"Beginning simulation, {datetime.now()}")
        
        # Track log probabilities for all entries
        self.all_log_probs = []
        
        for i in range(self.num_pools):
            if (i + 1)%100 == 0:
                print(f"Simulation {i + 1} out of {self.num_pools}, {datetime.now()}")

            # Create actual bracket for this pool
            actual_bracket = create_teams_from_standings(self.standings)
            pool = Pool(actual_bracket)
            
            # Create entries with varying upset factors
            upset_factors = [0.1 + (j/entries_per_pool)*0.3 for j in range(entries_per_pool)]
            for j, upset_factor in enumerate(upset_factors):
                entry_bracket = create_teams_from_standings(self.standings)
                for game in entry_bracket.games:
                    game.upset_factor = upset_factor
                entry_name = f"Entry_{j+1}"
                pool.add_entry(entry_name, entry_bracket)
            
            self.pools.append(pool)
            
            # Simulate and store results
            pool_results = pool.simulate_pool(num_sims=1000)
            
            # Store winning entry's results and log probability
            winning_entry = pool_results.iloc[0]['name']
            winning_bracket = [entry[1] for entry in pool.entries 
                             if entry[0] == winning_entry][0]
            winning_results = winning_bracket.simulate_tournament()
            self.winning_results.append(winning_results)
            self.all_log_probs.append(winning_bracket.log_probability)
            
            pool_results['pool_id'] = i
            self.all_results = pd.concat([self.all_results, pool_results])
            
    def analyze_upsets(self) -> pd.DataFrame:
        """
        Analyze upset patterns in winning brackets
        
        Returns:
            DataFrame containing upset statistics by round, ordered chronologically
        """
        upset_stats = defaultdict(list)
        
        for results in self.winning_results:
            for round_name, teams in results.items():
                if round_name != "Champion":  # Skip final result
                    # Count upsets by looking at team seeds
                    upsets = 0
                    for team in teams:
                        # In each round, a team is an upset if their seed is higher than expected
                        if round_name == "Second Round" and team.seed > 8:
                            upsets += 1
                        elif round_name == "Sweet 16" and team.seed > 4:
                            upsets += 1
                        elif round_name == "Elite 8" and team.seed > 2:
                            upsets += 1
                        elif round_name in ["Final Four", "Championship"] and team.seed > 1:
                            upsets += 1
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
        
        # Sort by predefined round order
        summary['round_order'] = summary['round'].map({round_name: i for i, round_name in enumerate(self.ROUND_ORDER)})
        summary = summary.sort_values('round_order').drop('round_order', axis=1)
        
        return summary
    
    def find_common_underdogs(self) -> pd.DataFrame:
        """
        Identify most common upset teams by round, where an upset is defined
        as a team advancing further than their seed traditionally would.
        
        Expected seeds by round:
        - First Round: All seeds (no upsets possible)
        - Second Round: Seeds 1-8
        - Sweet 16: Seeds 1-4
        - Elite 8: Seeds 1-2
        - Final Four: Seeds 1
        - Championship: Seeds 1
        
        Returns:
            DataFrame containing most frequent upset teams, grouped by round
        """
        # Define expected maximum seed for each round
        EXPECTED_MAX_SEEDS = {
            "First Round": 16,
            "Second Round": 8,
            "Sweet 16": 4,
            "Elite 8": 2,
            "Final Four": 1,
            "Championship": 1
        }
        
        upset_counts = defaultdict(int)
        
        for results in self.winning_results:  # Now using stored results
            for round_name, teams in results.items():
                if round_name != "Champion":
                    expected_max_seed = EXPECTED_MAX_SEEDS[round_name]
                    for team in teams:
                        if team.seed > expected_max_seed:
                            key = (round_name, team.seed, team.name)
                            upset_counts[key] += 1
        
        # Convert to DataFrame
        upsets_df = pd.DataFrame([
            {
                'round': round_name,
                'seed': seed,
                'team': team,
                'frequency': count / self.num_pools
            }
            for (round_name, seed, team), count in upset_counts.items()
        ])
        
        if upsets_df.empty:
            return pd.DataFrame(columns=['round', 'seed', 'team', 'frequency'])
        
        # Sort chronologically by round, then by frequency within each round
        upsets_df['round_order'] = upsets_df['round'].map(
            {round_name: i for i, round_name in enumerate(self.ROUND_ORDER)}
        )
        upsets_df = upsets_df.sort_values(
            ['round_order', 'frequency'], 
            ascending=[True, False]
        )
        upsets_df = upsets_df.drop('round_order', axis=1)
        upsets_df.rename(columns={"round":"make_it_to"},inplace=True)
        
        return upsets_df
    
    def analyze_champion_picks(self) -> pd.DataFrame:
        """
        Analyze championship picks in winning brackets
        
        Returns:
            DataFrame containing champion pick statistics
        """
        champion_counts = defaultdict(int)
        
        for results in self.winning_results:  # Now using stored results
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
    
    def analyze_bracket_likelihood(self) -> pd.DataFrame:
        """
        Analyze the distribution of bracket likelihoods in winning entries
        
        Returns:
            DataFrame containing statistics about bracket likelihood scores
        """
        if not hasattr(self, 'all_log_probs'):
            raise ValueError("Must run simulations before analyzing likelihoods")
            
        probs_df = pd.DataFrame({
            'log_probability': self.all_log_probs
        })
        
        # Calculate summary statistics
        summary = {
            'mean_log_prob': np.mean(self.all_log_probs),
            'std_log_prob': np.std(self.all_log_probs),
            'min_log_prob': np.min(self.all_log_probs),
            'max_log_prob': np.max(self.all_log_probs),
            'median_log_prob': np.median(self.all_log_probs),
            'q25_log_prob': np.percentile(self.all_log_probs, 25),
            'q75_log_prob': np.percentile(self.all_log_probs, 75)
        }
        
        # # Add interpretation
        # summary['interpretation'] = (
        #     f"Winning brackets tend to have log probabilities "
        #     f"between {summary['q25_log_prob']:.1f} and {summary['q75_log_prob']:.1f}. "
        #     f"The median is {summary['median_log_prob']:.1f}. "
        #     f"Lower values indicate more likely brackets."
        # )
        
        return pd.DataFrame([summary])

def main():
    """Example usage of bracket analysis"""
    # Get current standings
    standings = Standings()
    
    # Initialize analyzer
    analyzer = BracketAnalysis(standings, num_pools=1000)
    
    # Run simulations
    analyzer.simulate_pools(entries_per_pool=10)
    
    # Print various analyses
    print("\nUpset Statistics by Round:")
    print(analyzer.analyze_upsets().to_string(index=False))
    
    print("\nMost Common Underdogs:")
    print(analyzer.find_common_underdogs().groupby("make_it_to").head(10).to_string(index=False))
    
    print("\nChampionship Pick Analysis:")
    print(analyzer.analyze_champion_picks().head(10).to_string(index=False))

    print("\nBracket Likelihood Analysis:")
    print(analyzer.analyze_bracket_likelihood().T.to_string(header=False))

if __name__ == "__main__":
    main()
