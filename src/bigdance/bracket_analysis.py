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
    
    def plot_upset_distributions(self, save: bool = True) -> plt.Figure:
        """
        Plot distributions of upsets per round and total upsets in a single figure.
        The figure has a 3x2 grid with round-specific plots in the top 2 rows
        and the total distribution spanning the bottom row.
        Also saves the underlying histogram data for use in other applications.
        
        Args:
            save (bool): Whether to save plots and data to files
                
        Returns:
            Figure object
        """
        if not hasattr(self, 'underdogs_by_round') or not self.underdogs_by_round:
            raise ValueError("Must run simulations before plotting upset distributions")
        
        # Create a figure with a 3x2 grid
        fig = plt.figure(figsize=(15, 18))
        gs = plt.GridSpec(3, 2, figure=fig)
        
        # Add main title
        fig.suptitle("Tournament Upset Distributions", fontsize=16)
        
        # Dictionary to store histogram data for each round
        histogram_data = {}
        
        # Plot each round in top two rows (2x2 grid)
        max_per_round = {"First Round": 32, "Second Round": 16, "Sweet 16": 8,
                        "Elite 8": 4, "Final Four": 2, "Championship": 1}
        
        # Define which rounds to plot (skip Final Four and Championship)
        plot_rounds = ["First Round", "Second Round", "Sweet 16", "Elite 8"]
        
        for i, round_name in enumerate(plot_rounds):
            row, col = divmod(i, 2)  # Calculate position in 2x2 grid
            
            if round_name in self.underdogs_by_round and len(self.underdogs_by_round[round_name]) > 0:
                # Create subplot in the corresponding position
                ax = fig.add_subplot(gs[row, col])
                
                # Get data for this round
                data = self.underdogs_by_round[round_name]
                
                # Get max possible for x-axis
                max_possible = max_per_round.get(round_name, 8)
                
                # Create integer bins
                bins = np.arange(-0.5, max_possible + 1.5, 1)  # Ensure bins are centered on integers
                
                # Compute histogram values manually to capture the data
                hist_values, bin_edges = np.histogram(data, bins=bins, density=True)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                
                # Store the histogram data for this round
                histogram_data[round_name] = {
                    'bin_center': bin_centers.tolist(),
                    'bin_start': bin_edges[:-1].tolist(),
                    'bin_end': bin_edges[1:].tolist(),
                    'density': hist_values.tolist(),
                    'count': np.histogram(data, bins=bins, density=False)[0].tolist()
                }
                
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
        
        # Calculate total upsets for each simulation for bottom plot
        total_upsets = [bracket.total_underdogs() for bracket in self.winning_brackets]
        
        # Create subplot that spans the bottom row
        ax_total = fig.add_subplot(gs[2, :])
        
        # Find min and max for bin range
        min_upsets = min(total_upsets)
        max_upsets = max(total_upsets)
        
        # Create integer bins
        bins = np.arange(min_upsets - 0.5, max_upsets + 1.5, 1)
        
        # Compute histogram values manually to capture the data
        hist_values, bin_edges = np.histogram(total_upsets, bins=bins, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Add total upsets to histogram data
        histogram_data["Total Upsets"] = {
            'bin_center': bin_centers.tolist(),
            'bin_start': bin_edges[:-1].tolist(),
            'bin_end': bin_edges[1:].tolist(),
            'density': hist_values.tolist(),
            'count': np.histogram(total_upsets, bins=bins, density=False)[0].tolist()
        }
        
        # Plot distribution with discrete bins
        sns.histplot(total_upsets, bins=bins, discrete=True, ax=ax_total)
        
        # Add mean line
        mean_value = np.mean(total_upsets)
        ax_total.axvline(mean_value, color='red', linestyle='--', 
                    label=f'Mean: {mean_value:.2f}')
        
        # Add percentile lines
        q25 = np.percentile(total_upsets, 25)
        q75 = np.percentile(total_upsets, 75)
        ax_total.axvline(q25, color='green', linestyle=':', 
                    label=f'25th percentile: {q25:.2f}')
        ax_total.axvline(q75, color='orange', linestyle=':', 
                    label=f'75th percentile: {q75:.2f}')
        
        # Customize plot
        ax_total.set_title("Distribution of Total Upsets Across All Rounds")
        ax_total.set_xlabel("Total Number of Upsets")
        ax_total.set_ylabel("Frequency")
        
        # Set x-ticks to be integers
        ax_total.set_xticks(range(int(min_upsets), int(max_upsets) + 1))
        
        ax_total.legend()
        
        # Save the histogram data to a single JSON file
        if save:
            import json
            with open(self.output_dir / "upset_distributions_data.json", 'w') as f:
                json.dump(histogram_data, f, indent=2)
        
        # Adjust layout and save figure
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        if save:
            plt.savefig(self.output_dir / "all_upset_distributions.png", dpi=300, bbox_inches='tight')
        
        # Store the data for later access
        self.upset_histogram_data = histogram_data
        
        return fig

    def plot_log_probability_distributions(self, save: bool = True) -> plt.Figure:
        """
        Plot distributions of log probabilities per round and overall log probability in a single figure.
        The figure has a 4x2 grid with round-specific plots in the top 3 rows
        and the overall distribution spanning the bottom row.
        Also saves the underlying histogram data for use in other applications.
        
        Args:
            save (bool): Whether to save plots and data to files
                
        Returns:
            Figure object
        """
        if not hasattr(self, 'log_probs_by_round'):
            raise ValueError("Must run simulations before plotting log probabilities")
        
        # Create a figure with a 4x2 grid (3 rows for rounds, 1 row for overall)
        fig = plt.figure(figsize=(15, 22))
        gs = plt.GridSpec(4, 2, figure=fig)
        
        # Add main title
        fig.suptitle("Tournament Log Probability Distributions", fontsize=16)
        
        # Dictionary to store histogram data for each round
        log_prob_histogram_data = {}
        
        # Plot each round in top three rows (3x2 grid)
        for i, round_name in enumerate(self.ROUND_ORDER):  # All 6 rounds
            row, col = divmod(i, 2)  # Calculate position in grid
            
            if round_name in self.log_probs_by_round and len(self.log_probs_by_round[round_name]) > 0:
                # Create subplot in the corresponding position
                ax = fig.add_subplot(gs[row, col])
                
                # Get data for this round
                data = self.log_probs_by_round[round_name]
                
                # Calculate histogram bin edges - for log probabilities, use KDE-based binning
                hist_values, bin_edges = np.histogram(data, bins=30, density=True)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                
                # Store the histogram data for this round
                log_prob_histogram_data[round_name] = {
                    'bin_center': bin_centers.tolist(),
                    'bin_start': bin_edges[:-1].tolist(),
                    'bin_end': bin_edges[1:].tolist(),
                    'density': hist_values.tolist(),
                    'count': np.histogram(data, bins=bin_edges, density=False)[0].tolist()
                }
                
                # Plot histogram with KDE
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
        
        # Create subplot for overall log probability, spans the bottom row
        ax_total = fig.add_subplot(gs[3, :])
        
        # Get data for overall log probability
        data = self.all_log_probs
        
        # Calculate bin edges with margin for better visualization
        x_margin = 5.0
        min_val = min(data) - x_margin
        max_val = max(data) + x_margin
        bin_edges = np.linspace(min_val, max_val, 30 + 1)
        
        # Compute histogram values
        hist_values, bin_edges = np.histogram(data, bins=bin_edges, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Add overall log probability to histogram data
        log_prob_histogram_data["Overall"] = {
            'bin_center': bin_centers.tolist(),
            'bin_start': bin_edges[:-1].tolist(),
            'bin_end': bin_edges[1:].tolist(),
            'density': hist_values.tolist(),
            'count': np.histogram(data, bins=bin_edges, density=False)[0].tolist()
        }
        
        # Plot overall distribution with KDE
        sns.histplot(data, kde=True, bins=bin_edges, ax=ax_total)
        
        # Set extended x-axis limits
        ax_total.set_xlim(min_val, max_val)
        
        # Add mean line
        mean_value = np.mean(data)
        ax_total.axvline(mean_value, color='red', linestyle='--', 
                    label=f'Mean: {mean_value:.2f}')
        
        # Add percentile lines
        q25 = np.percentile(data, 25)
        q75 = np.percentile(data, 75)
        ax_total.axvline(q25, color='green', linestyle=':', 
                    label=f'25th percentile: {q25:.2f}')
        ax_total.axvline(q75, color='orange', linestyle=':', 
                    label=f'75th percentile: {q75:.2f}')
        
        # Customize plot
        ax_total.set_title("Distribution of Overall Bracket Log Probability Scores")
        ax_total.set_xlabel("Negative Log Probability (lower is more likely)")
        ax_total.set_ylabel("Frequency")
        ax_total.legend()
        
        # Save the histogram data to a single JSON file
        if save:
            import json
            with open(self.output_dir / "log_probability_distributions_data.json", 'w') as f:
                json.dump(log_prob_histogram_data, f, indent=2)
        
        # Adjust layout and save figure
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        if save:
            plt.savefig(self.output_dir / "all_log_probability_distributions.png", dpi=300, bbox_inches='tight')
        
        # Store the data for later access
        self.log_prob_histogram_data = log_prob_histogram_data
        
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
    
    def analyze_log_probabilities(self) -> pd.DataFrame:
        """
        Analyze log probability statistics by round and overall.
        Combines per-round analysis with overall bracket likelihood.
        
        Returns:
            DataFrame containing log probability statistics by round and overall
        """
        if not hasattr(self, 'log_probs_by_round'):
            raise ValueError("Must run simulations before analyzing log probabilities by round")
        
        if not hasattr(self, 'all_log_probs'):
            raise ValueError("Must run simulations before analyzing overall log probabilities")
        
        stats = []
        
        # First analyze each round
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
        
        # Then analyze overall bracket
        overall_data = self.all_log_probs
        stats.append({
            'round': 'Overall',
            'avg_log_prob': np.mean(overall_data),
            'std_log_prob': np.std(overall_data),
            'min_log_prob': np.min(overall_data),
            'q25_log_prob': np.percentile(overall_data, 25),
            'median_log_prob': np.median(overall_data),
            'q75_log_prob': np.percentile(overall_data, 75),
            'max_log_prob': np.max(overall_data),
            'total_games': len(overall_data)
        })
        
        # Convert to DataFrame
        stats_df = pd.DataFrame(stats)
        
        # Sort with rounds first, then overall at the end
        round_order = {round_name: i for i, round_name in enumerate(self.ROUND_ORDER)}
        round_order['Overall'] = len(self.ROUND_ORDER)  # Put Overall at the end
        
        stats_df['sort_order'] = stats_df['round'].map(round_order)
        stats_df = stats_df.sort_values('sort_order').drop('sort_order', axis=1)
        
        return stats_df

    def save_all_data(self):
        """Save all analysis data to CSV files and generate plots"""
        # # Save log probabilities data (combined overall and by round)
        # if hasattr(self, 'log_probs_by_round') and hasattr(self, 'all_log_probs'):
        #     # Start with per-round log probabilities
        #     log_probs_df = pd.DataFrame({
        #         round_name: pd.Series(self.log_probs_by_round.get(round_name, []))
        #         for round_name in self.ROUND_ORDER
        #     })
            
        #     # Add overall log probability as another column
        #     # Ensure lengths match by padding shorter arrays if needed
        #     max_len = max([len(log_probs_df[col]) for col in log_probs_df.columns] + [len(self.all_log_probs)])
        #     log_probs_df['Overall'] = pd.Series(self.all_log_probs).reindex(range(max_len))
            
        #     # Save combined data
        #     log_probs_df.to_csv(self.output_dir / "log_probabilities.csv", index=False)
        
        # # Save upsets by round data
        # if hasattr(self, 'underdogs_by_round'):
        #     upsets_df = pd.DataFrame({
        #         round_name: pd.Series(counts) 
        #         for round_name, counts in self.underdogs_by_round.items()
        #     })
        #     upsets_df.to_csv(self.output_dir / "upsets_by_round_data.csv", index=False)
        
        # # Save pool results
        # if hasattr(self, 'all_results'):
        #     self.all_results.to_csv(self.output_dir / "pool_results.csv", index=False)
        
        # Save analysis results
        upset_stats = self.analyze_upsets()
        upset_stats.to_csv(self.output_dir / "upset_statistics.csv", index=False)
        
        # Combine round and overall log probability stats
        if hasattr(self, 'log_probs_by_round') and hasattr(self, 'all_log_probs'):
            log_prob_stats = self.analyze_log_probabilities()
            log_prob_stats.to_csv(self.output_dir / "log_probability_stats.csv", index=False)
        
        common_underdogs = self.find_common_underdogs()
        common_underdogs.to_csv(self.output_dir / "common_underdogs.csv", index=False)
        
        champion_picks = self.analyze_champion_picks()
        champion_picks.to_csv(self.output_dir / "champion_picks.csv", index=False)
        
        # Generate plots and histogram data if not already generated
        if not hasattr(self, 'upset_histogram_data'):
            self.plot_upset_distributions(save=True)
            
        if not hasattr(self, 'log_prob_histogram_data'):
            self.plot_log_probability_distributions(save=True)
            
        print(f"All analysis data and visualizations saved to {self.output_dir}/")

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
    analyzer.plot_log_probability_distributions()
    
    # Print various analyses
    print("\nUpset Statistics by Round:")
    print(analyzer.analyze_upsets().to_string(index=False))
    
    print("\nMost Common Underdogs:")
    print(analyzer.find_common_underdogs().groupby("advanced_through").head(10).to_string(index=False))
    
    print("\nChampionship Pick Analysis:")
    print(analyzer.analyze_champion_picks().head(10).to_string(index=False))

    print("\nBracket Likelihood Analysis:")
    print(analyzer.analyze_log_probabilities().T.to_string(header=False))
    
    print(f"\nAnalysis data and visualizations have been saved to {output_dir}/")

if __name__ == "__main__":
    main()
