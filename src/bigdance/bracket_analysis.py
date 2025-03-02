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

from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from collections import defaultdict
from bigdance.wn_cbb_scraper import Standings
from bigdance.cbb_brackets import Team, Pool
from bigdance.bigdance_integration import create_teams_from_standings
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path

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
    
    def __init__(self, standings: Standings, num_pools: int = 100, output_dir: Optional[str] = None):
        self.standings = standings
        self.num_pools = num_pools
        self.pools: List[Pool] = []
        self.winning_results: List[Dict[str, List[Team]]] = []  # Store results instead of brackets
        self.all_results = pd.DataFrame()
        
        # Set up output directory for saving graphs and data
        if output_dir:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.output_dir = Path("bracket_analysis_output")
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def simulate_pools(self, entries_per_pool: int = 10) -> None:
        """Simulate multiple bracket pools"""
        print(f"Beginning simulation, {datetime.now()}")
        
        # Track log probabilities for all entries
        self.all_log_probs = []
        
        # Track upsets per round for all winning entries
        self.upsets_by_round = {round_name: [] for round_name in self.ROUND_ORDER}
        
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
            
            # Count upsets by round and append to tracking data
            for round_name, teams in winning_results.items():
                if round_name != "Champion":  # Skip final result
                    upsets = 0
                    for team in teams:
                        # In each round, a team is an upset if their seed is higher than expected
                        if round_name == "First Round" and team.seed > 8:
                            upsets += 1
                        elif round_name == "Second Round" and team.seed > 4:
                            upsets += 1
                        elif round_name == "Sweet 16" and team.seed > 2:
                            upsets += 1
                        elif round_name == "Elite 8" and team.seed > 1:
                            upsets += 1
                        elif round_name in ["Final Four", "Championship"] and team.seed > 1:
                            upsets += 1
                    
                    if round_name in self.upsets_by_round:
                        self.upsets_by_round[round_name].append(upsets)
            
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
                        if round_name == "First Round" and team.seed > 8:
                            upsets += 1
                        elif round_name == "Second Round" and team.seed > 4:
                            upsets += 1
                        elif round_name == "Sweet 16" and team.seed > 2:
                            upsets += 1
                        elif round_name == "Elite 8" and team.seed > 1:
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
    
    def plot_upset_distributions(self, save: bool = True) -> Dict[str, plt.Figure]:
        """
        Plot distributions of upsets per round with discrete integer bins
        
        Args:
            save (bool): Whether to save plots to files
                
        Returns:
            Dict of figures for each round
        """
        if not hasattr(self, 'upsets_by_round'):
            raise ValueError("Must run simulations before plotting upset distributions")
        
        # Create a single figure with multiple subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("Distribution of Underdogs by Tournament Round", fontsize=16)
        
        # Flatten axes for easier iteration
        axes = axes.flatten()
        
        # Plot each round
        max_per_round = {"First Round": 32, "Second Round": 16, "Sweet 16": 8,
                        "Elite 8": 4, "Final Four": 2, "Championship": 1}
        for i, round_name in enumerate(self.ROUND_ORDER):
            # Skipping Final Four and Championship, upsets are less meaningful
            if round_name in ["Final Four", "Championship"]:
                continue

            if round_name in self.upsets_by_round and len(self.upsets_by_round[round_name]) > 0:
                ax = axes[i]
                
                # Get max possible for x-axis
                max_possible = max_per_round.get(round_name, 8)
                
                # Get data for this round
                data = self.upsets_by_round[round_name]
                
                # Create integer bins
                bins = np.arange(-0.5, max_possible + 1.5, 1)  # Ensure bins are centered on integers
                
                # Plot histogram with discrete integer bins
                sns.histplot(data, ax=ax, bins=bins, discrete=True, stat="density")
                
                # Add mean line
                mean_value = np.mean(data)
                ax.axvline(mean_value, color='red', linestyle='--', 
                        label=f'Mean: {mean_value:.2f}')
                
                # Customize plot
                ax.set_title(f"{round_name}")
                ax.set_xlabel("Number of Underdogs")
                ax.set_ylabel("Density")
                ax.set_xlim(-0.5, max_possible + 0.5)
                
                # Set x-ticks to be integers
                ax.set_xticks(range(max_possible + 1))
                
                ax.legend()
                
        # Adjust layout and save combined figure
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        if save:
            fig.savefig(self.output_dir / "upsets_all_rounds.png", dpi=300, bbox_inches='tight')
        
        return fig

    def plot_total_upsets_distribution(self, save: bool = True) -> plt.Figure:
        """
        Plot distribution of total number of upsets across all rounds with discrete integer bins
        
        Args:
            save (bool): Whether to save plot to file
                
        Returns:
            Figure object
        """
        if not hasattr(self, 'upsets_by_round'):
            raise ValueError("Must run simulations before plotting total upsets distribution")
        
        # Calculate total upsets for each simulation
        total_upsets = []
        
        for i in range(len(self.winning_results)):
            sim_total = 0
            for round_name in self.ROUND_ORDER:
                if round_name in self.upsets_by_round and i < len(self.upsets_by_round[round_name]):
                    sim_total += self.upsets_by_round[round_name][i]
            total_upsets.append(sim_total)
        
        # Create figure
        fig = plt.figure(figsize=(10, 6))
        
        # Find min and max for bin range
        min_upsets = min(total_upsets)
        max_upsets = max(total_upsets)
        
        # Create integer bins
        bins = np.arange(min_upsets - 0.5, max_upsets + 1.5, 1)
        
        # Plot distribution with discrete bins
        sns.histplot(total_upsets, bins=bins, discrete=True)
        
        # Add mean line
        mean_value = np.mean(total_upsets)
        plt.axvline(mean_value, color='red', linestyle='--', 
                label=f'Mean: {mean_value:.2f}')
        
        # Add percentile lines
        q25 = np.percentile(total_upsets, 25)
        q75 = np.percentile(total_upsets, 75)
        plt.axvline(q25, color='green', linestyle=':', 
                label=f'25th percentile: {q25:.2f}')
        plt.axvline(q75, color='orange', linestyle=':', 
                label=f'75th percentile: {q75:.2f}')
        
        # Customize plot
        plt.title("Distribution of Total Upsets Across All Rounds")
        plt.xlabel("Total Number of Upsets")
        plt.ylabel("Frequency")
        
        # Set x-ticks to be integers
        plt.xticks(range(int(min_upsets), int(max_upsets) + 1))
        
        plt.legend()
        
        # Save figure
        if save:
            plt.savefig(self.output_dir / "total_upsets_distribution.png", dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_log_probability_by_round(self, save: bool = True) -> plt.Figure:
        """
        Plot distributions of log probabilities per round
        
        Args:
            save (bool): Whether to save plots to files
                
        Returns:
            Figure object for the combined plots
        """
        if not hasattr(self, 'winning_results'):
            raise ValueError("Must run simulations before plotting log probabilities by round")
        
        # Create a single figure with multiple subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("Distribution of Log Probabilities by Tournament Round", fontsize=16)
        
        # Flatten axes for easier iteration
        axes = axes.flatten()
        
        # Calculate log probabilities by round
        log_probs_by_round = {round_name: [] for round_name in self.ROUND_ORDER}
        
        for results in self.winning_results:
            # Track previous round winners to calculate matchups
            prev_round_winners = []
            
            for i, round_name in enumerate(self.ROUND_ORDER):
                if round_name == "First Round":
                    # First round games come from the initial bracket setup
                    # We need to match these with the results
                    round_winners = results.get(round_name, [])
                    winner_names = {team.name for team in round_winners}
                    
                    # Calculate log probabilities for first round games
                    round_log_probs = []
                    
                    # This requires accessing the original bracket games
                    # We'll use the actual tournament bracket from the pool
                    for pool in self.pools:
                        for game in pool.actual_results.games:
                            if game.winner and game.winner.name in winner_names:
                                # Calculate and store probability
                                prob = 1 / (1 + 10**((game.team2.rating - game.team1.rating)/400))
                                if game.winner.name == game.team2.name:
                                    prob = 1 - prob
                                if prob > 0:  # Avoid log(0)
                                    round_log_probs.append(-np.log(prob))
                        
                        # Only need to process one pool's games
                        break
                    
                    if round_log_probs:
                        log_probs_by_round[round_name].extend(round_log_probs)
                    
                    # Set up for next round
                    prev_round_winners = round_winners
                    
                elif round_name != "Champion" and round_name in results:
                    # Get current round winners
                    round_winners = results.get(round_name, [])
                    if not round_winners or not prev_round_winners:
                        continue
                        
                    # Create matchups from previous round winners
                    round_log_probs = []
                    
                    # Create temporary games and calculate probabilities
                    for i in range(0, len(prev_round_winners), 2):
                        if i + 1 < len(prev_round_winners):
                            team1, team2 = prev_round_winners[i], prev_round_winners[i + 1]
                            
                            # Find the winner of this matchup in the current round
                            winner = None
                            for team in round_winners:
                                if team.name == team1.name or team.name == team2.name:
                                    winner = team
                                    break
                            
                            if winner:
                                # Calculate probability
                                prob = 1 / (1 + 10**((team2.rating - team1.rating)/400))
                                if winner.name == team2.name:
                                    prob = 1 - prob
                                if prob > 0:
                                    round_log_probs.append(-np.log(prob))
                    
                    if round_log_probs:
                        log_probs_by_round[round_name].extend(round_log_probs)
                    
                    # Set up for next round
                    prev_round_winners = round_winners
        
        # Plot each round
        for i, round_name in enumerate(self.ROUND_ORDER):
            # Skipping Final Four and Championship for better visualization
            if round_name in ["Final Four", "Championship"]:
                continue
                
            if round_name in log_probs_by_round and len(log_probs_by_round[round_name]) > 0:
                ax = axes[i]
                
                # Get data for this round
                data = log_probs_by_round[round_name]
                
                # Plot histogram
                sns.histplot(data, ax=ax, kde=True, bins=30, stat="density")
                
                # Add mean line
                mean_value = np.mean(data)
                ax.axvline(mean_value, color='red', linestyle='--', 
                        label=f'Mean: {mean_value:.2f}')
                
                # Add percentile lines
                q25 = np.percentile(data, 25)
                q75 = np.percentile(data, 75)
                ax.axvline(q25, color='green', linestyle=':', 
                        label=f'25th percentile: {q25:.2f}')
                ax.axvline(q75, color='orange', linestyle=':', 
                        label=f'75th percentile: {q75:.2f}')
                
                # Customize plot
                ax.set_title(f"{round_name}")
                ax.set_xlabel("Negative Log Probability (lower is more likely)")
                ax.set_ylabel("Density")
                ax.legend()
        
        # Adjust layout and save combined figure
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        if save:
            fig.savefig(self.output_dir / "log_probabilities_by_round.png", dpi=300, bbox_inches='tight')
        
        return fig

    def plot_log_probability_distribution(self, save: bool = True, x_margin: float = 5.0, bins: int = 30) -> plt.Figure:
        """
        Plot distribution of log probability scores
        
        Args:
            save (bool): Whether to save plot to file
            x_margin (float): How much to extend the x-axis on each side
            bins (int): Number of bins for the histogram
            
        Returns:
            Figure object
        """
        if not hasattr(self, 'all_log_probs'):
            raise ValueError("Must run simulations before plotting log probability distribution")
            
        # Create figure
        fig = plt.figure(figsize=(10, 6))
        
        # Plot distribution with increased bins
        sns.histplot(self.all_log_probs, kde=True, bins=bins)
        
        # Set extended x-axis limits
        min_val = min(self.all_log_probs) - x_margin
        max_val = max(self.all_log_probs) + x_margin
        plt.xlim(min_val, max_val)
        
        # Add mean line
        mean_value = np.mean(self.all_log_probs)
        plt.axvline(mean_value, color='red', linestyle='--', 
                label=f'Mean: {mean_value:.2f}')
        
        # Add percentile lines
        q25 = np.percentile(self.all_log_probs, 25)
        q75 = np.percentile(self.all_log_probs, 75)
        plt.axvline(q25, color='green', linestyle=':', 
                label=f'25th percentile: {q25:.2f}')
        plt.axvline(q75, color='orange', linestyle=':', 
                label=f'75th percentile: {q75:.2f}')
        
        # Customize plot
        plt.title("Distribution of Bracket Log Probability Scores")
        plt.xlabel("Negative Log Probability (lower is more likely)")
        plt.ylabel("Frequency")
        plt.legend()
        
        # Save figure
        if save:
            plt.savefig(self.output_dir / "log_probability_distribution.png", dpi=300, bbox_inches='tight')
        
        return fig
    
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
        
        # Save data
        upsets_df.to_csv(self.output_dir / "common_underdogs.csv", index=False)
        
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
        
        # Sort by frequency
        champions_df = champions_df.sort_values('frequency', ascending=False)
        
        # Save data
        champions_df.to_csv(self.output_dir / "champion_picks.csv", index=False)
        
        return champions_df
    
    def analyze_bracket_likelihood(self) -> pd.DataFrame:
        """
        Analyze the distribution of bracket likelihoods in winning entries
        
        Returns:
            DataFrame containing statistics about bracket likelihood scores
        """
        if not hasattr(self, 'all_log_probs'):
            raise ValueError("Must run simulations before analyzing likelihoods")
            
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
        
        summary_df = pd.DataFrame([summary])
        
        # Save data
        summary_df.to_csv(self.output_dir / "bracket_likelihood_stats.csv", index=False)
        
        return summary_df

    def save_all_data(self):
        """Save all analysis data to CSV files"""
        if hasattr(self, 'all_log_probs'):
            pd.DataFrame({'log_probability': self.all_log_probs}).to_csv(
                self.output_dir / "log_probability_data.csv", index=False)
        
        if hasattr(self, 'upsets_by_round'):
            pd.DataFrame(self.upsets_by_round).to_csv(
                self.output_dir / "upsets_by_round_data.csv", index=False)
        
        if hasattr(self, 'all_results'):
            self.all_results.to_csv(self.output_dir / "pool_results.csv", index=False)
        
        # Save upset statistics
        upset_stats = self.analyze_upsets()
        upset_stats.to_csv(self.output_dir / "upset_statistics.csv", index=False)

def main():
    """Example usage of bracket analysis"""
    # Get current standings
    standings = Standings()
    
    # Set up output directory
    output_dir = "bracket_analysis_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize analyzer with output directory
    analyzer = BracketAnalysis(standings, num_pools=1000, output_dir=output_dir)
    
    # Run simulations
    analyzer.simulate_pools(entries_per_pool=10)
    
    # Save all data and generate plots
    analyzer.save_all_data()
    
    # Generate original plots
    analyzer.plot_upset_distributions()
    analyzer.plot_log_probability_distribution()
    
    # Generate new plots
    analyzer.plot_log_probability_by_round()
    analyzer.plot_total_upsets_distribution()
    
    # Print various analyses
    print("\nUpset Statistics by Round:")
    print(analyzer.analyze_upsets().to_string(index=False))
    
    print("\nMost Common Underdogs:")
    print(analyzer.find_common_underdogs().groupby("make_it_to").head(10).to_string(index=False))
    
    print("\nChampionship Pick Analysis:")
    print(analyzer.analyze_champion_picks().head(10).to_string(index=False))

    print("\nBracket Likelihood Analysis:")
    print(analyzer.analyze_bracket_likelihood().T.to_string(header=False))
    
    print(f"\nAnalysis data and visualizations have been saved to {output_dir}/")

if __name__ == "__main__":
    main()
