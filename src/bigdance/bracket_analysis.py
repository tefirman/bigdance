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
from bigdance.cbb_brackets import Team, Pool, Bracket
from bigdance.bigdance_integration import create_teams_from_standings
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
import argparse

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
        self.winning_brackets: List[Bracket] = []  # Store the actual winning brackets
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
        
        # Track log probabilities by round for all winning entries
        self.log_probs_by_round = {round_name: [] for round_name in self.ROUND_ORDER}
        
        # Track underdogs by round for all winning entries
        self.underdogs_by_round = {round_name: [] for round_name in self.ROUND_ORDER}
        
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
            self.winning_brackets.append(winning_bracket)
            self.all_log_probs.append(winning_bracket.log_probability)
            
            # Store per-round log probabilities for the winning entry
            for round_name, log_prob in winning_bracket.log_probability_by_round.items():
                if round_name in self.log_probs_by_round:
                    self.log_probs_by_round[round_name].append(log_prob)
            
            # Store underdog counts by round - using new direct methods from Bracket class
            underdog_counts = winning_bracket.count_underdogs_by_round()
            for round_name in self.ROUND_ORDER:
                if round_name in underdog_counts:
                    self.underdogs_by_round[round_name].append(underdog_counts[round_name])
                else:
                    self.underdogs_by_round[round_name].append(0)
            
            pool_results['pool_id'] = i
            self.all_results = pd.concat([self.all_results, pool_results])
            
    def analyze_upsets(self) -> pd.DataFrame:
        """
        Analyze upset patterns in winning brackets
        
        Returns:
            DataFrame containing upset statistics by round, ordered chronologically
        """
        if not hasattr(self, 'underdogs_by_round') or not self.underdogs_by_round:
            raise ValueError("Must run simulations before analyzing upsets")
            
        # Calculate summary statistics from stored underdog counts
        summary = pd.DataFrame({
            'round': list(self.underdogs_by_round.keys()),
            'avg_upsets': [np.mean(counts) if counts else 0 for counts in self.underdogs_by_round.values()],
            'std_upsets': [np.std(counts) if counts else 0 for counts in self.underdogs_by_round.values()],
            'min_upsets': [min(counts) if counts else 0 for counts in self.underdogs_by_round.values()],
            'max_upsets': [max(counts) if counts else 0 for counts in self.underdogs_by_round.values()]
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
        if not hasattr(self, 'underdogs_by_round') or not self.underdogs_by_round:
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

            if round_name in self.underdogs_by_round and len(self.underdogs_by_round[round_name]) > 0:
                ax = axes[i]
                
                # Get max possible for x-axis
                max_possible = max_per_round.get(round_name, 8)
                
                # Get data for this round
                data = self.underdogs_by_round[round_name]
                
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
        if not hasattr(self, 'underdogs_by_round') or not self.underdogs_by_round:
            raise ValueError("Must run simulations before plotting total upsets distribution")
        
        # Calculate total upsets for each simulation
        total_upsets = []
        
        # Use the total_underdogs from each winning bracket
        for bracket in self.winning_brackets:
            total_upsets.append(bracket.total_underdogs())
        
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
    
    def plot_all_rounds_log_probability(self, save: bool = True) -> plt.Figure:
        """
        Plot distributions of log probabilities for all tournament rounds in a single 3x2 grid
        
        Args:
            save (bool): Whether to save plot to file
                    
        Returns:
            Figure object for the combined plots
        """
        if not hasattr(self, 'log_probs_by_round'):
            raise ValueError("Must run simulations before plotting log probabilities by round")
        
        # Create a single figure with a 3x2 grid of subplots
        fig, axes = plt.subplots(3, 2, figsize=(15, 18))
        fig.suptitle("Distribution of Log Probabilities by Tournament Round", fontsize=16)
        
        # Flatten axes for easier iteration
        axes = axes.flatten()
        
        # Plot each round
        for i, round_name in enumerate(self.ROUND_ORDER):  # All 6 rounds
            if round_name in self.log_probs_by_round and len(self.log_probs_by_round[round_name]) > 0:
                ax = axes[i]
                
                # Get data for this round
                data = self.log_probs_by_round[round_name]
                
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
            fig.savefig(self.output_dir / "all_rounds_log_probabilities.png", dpi=300, bbox_inches='tight')
        
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
        as a team advancing through a round despite having a seed lower than typically expected.
        
        Returns:
            DataFrame containing most frequent upset teams, grouped by round they advanced from
        """
        # Map round names to the next round they advance to
        NEXT_ROUND = {
            "First Round": "Second Round",
            "Second Round": "Sweet 16",
            "Sweet 16": "Elite 8",
            "Elite 8": "Final Four",
            "Final Four": "Championship"
        }
        
        upset_counts = defaultdict(int)
        
        # Use the underdogs identified in each bracket
        for bracket in self.winning_brackets:
            for round_name, underdogs in bracket.underdogs_by_round.items():
                for team in underdogs:
                    # Key format: (round advanced through, seed, team name)
                    key = (round_name, team.seed, team.name)
                    upset_counts[key] += 1
        
        # Convert to DataFrame
        upsets_df = pd.DataFrame([
            {
                'advanced_through': round_name,
                'to_reach': NEXT_ROUND.get(round_name, "N/A"),
                'seed': seed,
                'team': team,
                'frequency': count / self.num_pools
            }
            for (round_name, seed, team), count in upset_counts.items()
        ])
        
        if upsets_df.empty:
            return pd.DataFrame(columns=['advanced_through', 'to_reach', 'seed', 'team', 'frequency'])
        
        # Sort chronologically by round, then by frequency within each round
        upsets_df['round_order'] = upsets_df['advanced_through'].map(
            {round_name: i for i, round_name in enumerate(self.ROUND_ORDER)}
        )
        upsets_df = upsets_df.sort_values(
            ['round_order', 'frequency'], 
            ascending=[True, False]
        )
        upsets_df = upsets_df.drop('round_order', axis=1)
        
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

    def analyze_round_log_probabilities(self) -> pd.DataFrame:
        """
        Analyze log probability statistics by round
        
        Returns:
            DataFrame containing log probability statistics by round
        """
        if not hasattr(self, 'log_probs_by_round'):
            raise ValueError("Must run simulations before analyzing log probabilities by round")
        
        stats = []
        for round_name in self.ROUND_ORDER:
            if round_name in self.log_probs_by_round and len(self.log_probs_by_round[round_name]) > 0:
                data = self.log_probs_by_round[round_name]
                stats.append({
                    'round': round_name,
                    'avg_log_prob': np.mean(data),
                    'std_log_prob': np.std(data),
                    'min_log_prob': np.min(data),
                    'q25_log_prob': np.percentile(data, 25),
                    'median_log_prob': np.median(data),
                    'q75_log_prob': np.percentile(data, 75),
                    'max_log_prob': np.max(data),
                    'total_games': len(data)
                })
        
        # Convert to DataFrame
        stats_df = pd.DataFrame(stats)
        
        # Sort by predefined round order
        stats_df['round_order'] = stats_df['round'].map({round_name: i for i, round_name in enumerate(self.ROUND_ORDER)})
        stats_df = stats_df.sort_values('round_order').drop('round_order', axis=1)
        
        # Save to CSV
        stats_df.to_csv(self.output_dir / "round_log_probability_stats.csv", index=False)
        
        return stats_df

    def save_all_data(self):
        """Save all analysis data to CSV files"""
        if hasattr(self, 'all_log_probs'):
            pd.DataFrame({'log_probability': self.all_log_probs}).to_csv(
                self.output_dir / "log_probability_data.csv", index=False)
        
        if hasattr(self, 'log_probs_by_round'):
            # Save per-round log probabilities
            log_probs_df = pd.DataFrame({
                round_name: self.log_probs_by_round.get(round_name, [])
                for round_name in self.ROUND_ORDER
            })
            log_probs_df.to_csv(self.output_dir / "log_probabilities_by_round_data.csv", index=False)
        
        if hasattr(self, 'upsets_by_round'):
            pd.DataFrame(self.upsets_by_round).to_csv(
                self.output_dir / "upsets_by_round_data.csv", index=False)
        
        if hasattr(self, 'all_results'):
            self.all_results.to_csv(self.output_dir / "pool_results.csv", index=False)
        
        # Save upset statistics
        upset_stats = self.analyze_upsets()
        upset_stats.to_csv(self.output_dir / "upset_statistics.csv", index=False)
        
        # Save round log probability statistics
        if hasattr(self, 'log_probs_by_round'):
            round_log_stats = self.analyze_round_log_probabilities()
            round_log_stats.to_csv(self.output_dir / "round_log_probability_stats.csv", index=False)

def main():
    """Example usage of bracket analysis"""
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Analyze March Madness bracket pool simulations')
    parser.add_argument('--num_pools', type=int, default=1000,
                        help='Number of pools to simulate')
    parser.add_argument('--entries_per_pool', type=int, default=10,
                        help='Number of entries per pool')
    parser.add_argument('--women', action='store_true',
                        help='Whether to use women\'s basketball data instead of men\'s')
    args = parser.parse_args()

    # Get current standings
    standings = Standings(women=args.women)
    
    # Set up output directory
    gender = "women" if args.women else "men"
    output_dir = f"bracket_analysis_{args.entries_per_pool}entries_{gender}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize analyzer with output directory
    print(f"Starting analysis for {args.entries_per_pool} entries over {args.num_pools} pools")
    analyzer = BracketAnalysis(standings, num_pools=args.num_pools, output_dir=output_dir)
    
    # Run simulations
    analyzer.simulate_pools(entries_per_pool=args.entries_per_pool)
    
    # Save all data and generate plots
    analyzer.save_all_data()
    
    # Generate original plots
    analyzer.plot_upset_distributions()
    analyzer.plot_all_rounds_log_probability()
    analyzer.plot_log_probability_distribution()
    analyzer.plot_total_upsets_distribution()
    
    # Print various analyses
    print("\nUpset Statistics by Round:")
    print(analyzer.analyze_upsets().to_string(index=False))
    
    print("\nMost Common Underdogs:")
    print(analyzer.find_common_underdogs().groupby("advanced_through").head(10).to_string(index=False))
    
    print("\nChampionship Pick Analysis:")
    print(analyzer.analyze_champion_picks().head(10).to_string(index=False))

    print("\nBracket Likelihood Analysis:")
    print(analyzer.analyze_bracket_likelihood().T.to_string(header=False))
    
    print(f"\nAnalysis data and visualizations have been saved to {output_dir}/")

if __name__ == "__main__":
    main()
