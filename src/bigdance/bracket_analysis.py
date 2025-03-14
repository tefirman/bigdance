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
import logging

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
        """Simulate multiple bracket pools and track both winning and non-winning brackets"""
        print(f"Beginning simulation, {datetime.now()}")
        
        # Track statistics for all entries, separated by winner status
        self.winning_brackets = []
        self.non_winning_brackets = []
        
        # Track log probabilities for all entries
        self.winning_log_probs = []
        self.non_winning_log_probs = []
        
        # Track log probabilities by round
        self.winning_log_probs_by_round = {round_name: [] for round_name in self.ROUND_ORDER}
        self.non_winning_log_probs_by_round = {round_name: [] for round_name in self.ROUND_ORDER}
        
        # Track underdogs by round
        self.winning_underdogs_by_round = {round_name: [] for round_name in self.ROUND_ORDER}
        self.non_winning_underdogs_by_round = {round_name: [] for round_name in self.ROUND_ORDER}
        
        # Track total underdogs
        self.winning_total_underdogs = []
        self.non_winning_total_underdogs = []
        
        # For backwards compatibility (originally we only tracked winning brackets)
        self.winning_results = []
        self.all_results = pd.DataFrame()
        
        # Keep count of successful simulations
        successful_sims = 0
        
        for i in range(self.num_pools):
            if (i + 1)%10 == 0:
                print(f"Simulation {i + 1} out of {self.num_pools}, {datetime.now()}")

            try:
                # Create actual bracket for this pool
                actual_bracket = create_teams_from_standings(self.standings)
                
                # IMPORTANT: Add moderate upset factor to actual tournament results
                for game in actual_bracket.games:
                    game.upset_factor = 0.0  # More realistic tournament has upsets
                    
                pool = Pool(actual_bracket)
                
                # Create entries with varying upset factors
                # Create a normal distribution centered around 0 with standard deviation 0.3
                upset_factors = np.random.normal(0, 0.3, entries_per_pool)
                
                # Clip values to stay within -1.0 to 1.0 range
                upset_factors = np.clip(upset_factors, -1.0, 1.0)
                
                # Ensure we include some extreme values for variety
                if entries_per_pool >= 10:
                    # Include at least one strong chalk picker
                    upset_factors[0] = -0.8
                    # Include at least one strong upset picker
                    upset_factors[1] = 0.8
                    # Include at least one pure elo-based picker
                    upset_factors[2] = 0.0
                    # Shuffle to randomize positions
                    np.random.shuffle(upset_factors)
                
                # Use upset_factors list instead of the old approach
                for j, upset_factor in enumerate(upset_factors):
                    try:
                        entry_bracket = create_teams_from_standings(self.standings)
                        for game in entry_bracket.games:
                            game.upset_factor = upset_factor
                        entry_name = f"Entry_{j+1}"
                        pool.add_entry(entry_name, entry_bracket)
                    except Exception as e:
                        print(f"Warning: Error creating entry {j+1} in pool {i+1}: {str(e)}")
                        continue
                
                self.pools.append(pool)
                
                # Skip to next pool if no entries were added successfully
                if not pool.entries:
                    print(f"Warning: No valid entries in pool {i+1}, skipping")
                    continue
                
                # Simulate and store results
                try:
                    pool_results = pool.simulate_pool(num_sims=1000)
                    
                    # Find winning entry name(s)
                    top_entries = pool_results.sort_values('win_pct', ascending=False)
                    
                    # Define what counts as a "winner" - only take the top entry
                    # This guarantees we'll have both winners and non-winners
                    winning_entries = set(top_entries.iloc[0:1]['name'].tolist())
                    
                    # Check if we found valid winners
                    if len(winning_entries) == 0:
                        print(f"Warning: No winners found in pool {i+1}, skipping")
                        continue
                    
                    # Track entries that were processed
                    processed_winners = 0
                    processed_non_winners = 0
                    
                    # Process all entries, categorizing as winners or non-winners
                    for entry_name, entry_bracket, _ in pool.entries:
                        # Determine if this entry is a winner (top-performing bracket)
                        is_winner = entry_name in winning_entries
                        
                        # Skip entries with incomplete data
                        if not entry_bracket.results or not hasattr(entry_bracket, 'log_probability_by_round'):
                            continue
                        
                        # Store bracket in appropriate list
                        if is_winner:
                            self.winning_brackets.append(entry_bracket)
                            processed_winners += 1
                            
                            # Also store in original format for backward compatibility
                            self.winning_results.append(entry_bracket.results)
                            
                            # Store log probability
                            if hasattr(entry_bracket, 'log_probability'):
                                self.winning_log_probs.append(entry_bracket.log_probability)
                            
                            # Store per-round log probabilities
                            for round_name, log_prob in entry_bracket.log_probability_by_round.items():
                                if round_name in self.winning_log_probs_by_round:
                                    self.winning_log_probs_by_round[round_name].append(log_prob)
                            
                            # Store underdog counts by round
                            underdog_counts = entry_bracket.count_underdogs_by_round()
                            for round_name in self.ROUND_ORDER:
                                if round_name in underdog_counts:
                                    self.winning_underdogs_by_round[round_name].append(underdog_counts[round_name])
                                else:
                                    self.winning_underdogs_by_round[round_name].append(0)
                            
                            # Store total underdogs
                            self.winning_total_underdogs.append(entry_bracket.total_underdogs())
                        else:
                            self.non_winning_brackets.append(entry_bracket)
                            processed_non_winners += 1
                            
                            # Store log probability
                            if hasattr(entry_bracket, 'log_probability'):
                                self.non_winning_log_probs.append(entry_bracket.log_probability)
                            
                            # Store per-round log probabilities
                            for round_name, log_prob in entry_bracket.log_probability_by_round.items():
                                if round_name in self.non_winning_log_probs_by_round:
                                    self.non_winning_log_probs_by_round[round_name].append(log_prob)
                            
                            # Store underdog counts by round
                            underdog_counts = entry_bracket.count_underdogs_by_round()
                            for round_name in self.ROUND_ORDER:
                                if round_name in underdog_counts:
                                    self.non_winning_underdogs_by_round[round_name].append(underdog_counts[round_name])
                                else:
                                    self.non_winning_underdogs_by_round[round_name].append(0)
                            
                            # Store total underdogs
                            self.non_winning_total_underdogs.append(entry_bracket.total_underdogs())
                    
                    # # Log the counts for this pool
                    # print(f"Pool {i+1}: Processed {processed_winners} winners and {processed_non_winners} non-winners")
                    
                    # Store pool results for later analysis
                    pool_results['pool_id'] = i
                    self.all_results = pd.concat([self.all_results, pool_results])
                    
                    successful_sims += 1
                    
                except Exception as e:
                    print(f"Warning: Error simulating pool {i+1}: {str(e)}")
                    continue
                
            except Exception as e:
                print(f"Warning: Error creating pool {i+1}: {str(e)}")
                continue
        
        print(f"Successfully simulated {successful_sims} out of {self.num_pools} pools")
        print(f"Total brackets processed: {len(self.winning_brackets)} winners and {len(self.non_winning_brackets)} non-winners")
        
        # Handle empty results
        if not self.winning_brackets:
            print("Warning: No winning brackets were tracked. Analysis may be limited.")
        
        if not self.non_winning_brackets:
            print("Warning: No non-winning brackets were tracked. Comparative analysis will not be possible.")
        
        # For backward compatibility (originally we tracked these from winning brackets)
        self.underdogs_by_round = self.winning_underdogs_by_round
        self.log_probs_by_round = self.winning_log_probs_by_round 
        self.all_log_probs = self.winning_log_probs
            
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
    
    def plot_comparative_upset_distributions(self, save: bool = True) -> plt.Figure:
        """
        Plot comparative distributions of upsets per round for winning vs. non-winning brackets.
        
        Args:
            save (bool): Whether to save plots and data to files
                
        Returns:
            Figure object
        """
        if not hasattr(self, 'winning_underdogs_by_round') or not self.winning_underdogs_by_round:
            raise ValueError("Must run simulations before plotting upset distributions")
        
        # Check if we have data in both categories
        if not self.winning_total_underdogs:
            print("Warning: No winning brackets data available for analysis")
            return None
            
        if not self.non_winning_total_underdogs:
            print("Warning: No non-winning brackets data available for analysis")
            return None
        
        # Create a figure with a 3x2 grid
        fig = plt.figure(figsize=(15, 18))
        gs = plt.GridSpec(3, 2, figure=fig)
        
        # Add main title
        fig.suptitle("Tournament Upset Distributions: Winners vs. Non-Winners", fontsize=16)
        
        # Dictionary to store histogram data
        histogram_data = {}
        
        # Define which rounds to plot
        plot_rounds = ["First Round", "Second Round", "Sweet 16", "Elite 8"]
        max_per_round = {"First Round": 32, "Second Round": 16, "Sweet 16": 8,
                        "Elite 8": 4, "Final Four": 2, "Championship": 1}
        
        for i, round_name in enumerate(plot_rounds):
            row, col = divmod(i, 2)  # Calculate position in 2x2 grid
            
            # Get data with checks to ensure both are available and non-empty
            winning_data = self.winning_underdogs_by_round.get(round_name, [])
            non_winning_data = self.non_winning_underdogs_by_round.get(round_name, [])
            
            if not winning_data or not non_winning_data:
                print(f"Warning: Missing data for {round_name}, skipping this round")
                continue
                
            # Create subplot
            ax = fig.add_subplot(gs[row, col])
            
            # Get max possible for x-axis
            max_possible = max_per_round.get(round_name, 8)
            
            # Create integer bins
            bins = np.arange(-0.5, max_possible + 1.5, 1)
            
            # Plot both distributions
            sns.histplot(winning_data, ax=ax, bins=bins, discrete=True, 
                    stat="density", alpha=0.7, label="Winners", color="green")
            sns.histplot(non_winning_data, ax=ax, bins=bins, discrete=True, 
                    stat="density", alpha=0.7, label="Non-Winners", color="red")
            
            # Calculate and store statistics for histogram data
            winning_hist, bin_edges = np.histogram(winning_data, bins=bins, density=True)
            non_winning_hist, _ = np.histogram(non_winning_data, bins=bins, density=True)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            # Store the histogram data
            histogram_data[round_name] = {
                'bin_center': bin_centers.tolist(),
                'bin_start': bin_edges[:-1].tolist(),
                'bin_end': bin_edges[1:].tolist(),
                'winners_density': winning_hist.tolist(),
                'non_winners_density': non_winning_hist.tolist(),
                'winners_count': np.histogram(winning_data, bins=bins, density=False)[0].tolist(),
                'non_winners_count': np.histogram(non_winning_data, bins=bins, density=False)[0].tolist(),
                'winners_mean': np.mean(winning_data),
                'non_winners_mean': np.mean(non_winning_data)
            }
            
            # Add mean lines
            winning_mean = np.mean(winning_data)
            non_winning_mean = np.mean(non_winning_data)
            
            ax.axvline(winning_mean, color='green', linestyle='--', 
                    label=f'Winners Mean: {winning_mean:.2f}')
            ax.axvline(non_winning_mean, color='red', linestyle='--', 
                    label=f'Non-Winners Mean: {non_winning_mean:.2f}')
            
            # Customize plot
            ax.set_title(f"{round_name}")
            ax.set_xlabel("Number of Underdogs")
            ax.set_ylabel("Density")
            ax.set_xlim(-0.5, max_possible + 0.5)
            ax.set_xticks(range(max_possible + 1))
            ax.legend()
        
        # Create subplot for total upsets that spans the bottom row
        ax_total = fig.add_subplot(gs[2, :])
        
        # Compute histograms for total upsets
        winning_total = self.winning_total_underdogs
        non_winning_total = self.non_winning_total_underdogs
        
        # Check data availability - this is where the error was happening
        if not winning_total or not non_winning_total:
            print("Warning: Missing total upsets data, skipping total upsets plot")
            if save:
                import json
                with open(self.output_dir / "comparative_upset_distributions_data.json", 'w') as f:
                    json.dump(histogram_data, f, indent=2)
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            if save:
                plt.savefig(self.output_dir / "comparative_upset_distributions.png", dpi=300, bbox_inches='tight')
            self.comparative_upset_histogram_data = histogram_data
            return fig
        
        # Find min and max for bin range safely
        min_upsets = min(min(winning_total) if winning_total else 0, 
                        min(non_winning_total) if non_winning_total else 0)
        max_upsets = max(max(winning_total) if winning_total else 0, 
                        max(non_winning_total) if non_winning_total else 0)
        
        # Create integer bins
        bins = np.arange(min_upsets - 0.5, max_upsets + 1.5, 1)
        
        # Plot distributions
        sns.histplot(winning_total, bins=bins, discrete=True, 
                stat="density", alpha=0.7, label="Winners", color="green", ax=ax_total)
        sns.histplot(non_winning_total, bins=bins, discrete=True, 
                stat="density", alpha=0.7, label="Non-Winners", color="red", ax=ax_total)
        
        # Compute and store histogram data
        winning_hist, bin_edges = np.histogram(winning_total, bins=bins, density=True)
        non_winning_hist, _ = np.histogram(non_winning_total, bins=bins, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Store the histogram data
        histogram_data["Total Upsets"] = {
            'bin_center': bin_centers.tolist(),
            'bin_start': bin_edges[:-1].tolist(),
            'bin_end': bin_edges[1:].tolist(),
            'winners_density': winning_hist.tolist(),
            'non_winners_density': non_winning_hist.tolist(),
            'winners_count': np.histogram(winning_total, bins=bins, density=False)[0].tolist(),
            'non_winners_count': np.histogram(non_winning_total, bins=bins, density=False)[0].tolist(),
            'winners_mean': np.mean(winning_total),
            'non_winners_mean': np.mean(non_winning_total)
        }
        
        # Add mean lines
        winning_mean = np.mean(winning_total)
        non_winning_mean = np.mean(non_winning_total)
        
        ax_total.axvline(winning_mean, color='green', linestyle='--', 
                    label=f'Winners Mean: {winning_mean:.2f}')
        ax_total.axvline(non_winning_mean, color='red', linestyle='--', 
                    label=f'Non-Winners Mean: {non_winning_mean:.2f}')
        
        # Add percentile lines
        winning_q25 = np.percentile(winning_total, 25)
        winning_q75 = np.percentile(winning_total, 75)
        non_winning_q25 = np.percentile(non_winning_total, 25)
        non_winning_q75 = np.percentile(non_winning_total, 75)
        
        ax_total.axvline(winning_q25, color='green', linestyle=':', 
                    label=f'Winners 25th: {winning_q25:.2f}')
        ax_total.axvline(winning_q75, color='green', linestyle=':', 
                    label=f'Winners 75th: {winning_q75:.2f}')
        ax_total.axvline(non_winning_q25, color='red', linestyle=':', 
                    label=f'Non-Winners 25th: {non_winning_q25:.2f}')
        ax_total.axvline(non_winning_q75, color='red', linestyle=':', 
                    label=f'Non-Winners 75th: {non_winning_q75:.2f}')
        
        # Customize plot
        ax_total.set_title("Distribution of Total Upsets Across All Rounds")
        ax_total.set_xlabel("Total Number of Upsets")
        ax_total.set_ylabel("Density")
        ax_total.set_xticks(range(int(min_upsets), int(max_upsets) + 1))
        ax_total.legend()
        
        # Save the histogram data
        if save:
            import json
            with open(self.output_dir / "comparative_upset_distributions_data.json", 'w') as f:
                json.dump(histogram_data, f, indent=2)
        
        # Adjust layout and save figure
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        if save:
            plt.savefig(self.output_dir / "comparative_upset_distributions.png", dpi=300, bbox_inches='tight')
        
        # Store the data for later access
        self.comparative_upset_histogram_data = histogram_data
        
        return fig

    def plot_comparative_log_probability_distributions(self, save: bool = True) -> plt.Figure:
        """
        Plot comparative distributions of log probabilities for winning vs. non-winning brackets.
        
        Args:
            save (bool): Whether to save plots and data to files
                
        Returns:
            Figure object
        """
        if not hasattr(self, 'winning_log_probs_by_round') or not self.winning_log_probs_by_round:
            raise ValueError("Must run simulations before plotting log probability distributions")
        
        # Check if we have data in both categories
        if not self.winning_log_probs:
            print("Warning: No winning brackets log probability data available for analysis")
            return None
            
        if not self.non_winning_log_probs:
            print("Warning: No non-winning brackets log probability data available for analysis")
            return None
        
        # Create a figure with a 4x2 grid
        fig = plt.figure(figsize=(15, 22))
        gs = plt.GridSpec(4, 2, figure=fig)
        
        # Add main title
        fig.suptitle("Tournament Log Probability Distributions: Winners vs. Non-Winners", fontsize=16)
        
        # Dictionary to store histogram data
        histogram_data = {}
        
        # Plot each round
        for i, round_name in enumerate(self.ROUND_ORDER):
            row, col = divmod(i, 2)
            
            # Get data with checks to ensure both are available and non-empty
            winning_data = self.winning_log_probs_by_round.get(round_name, [])
            non_winning_data = self.non_winning_log_probs_by_round.get(round_name, [])
            
            if not winning_data or not non_winning_data:
                print(f"Warning: Missing log probability data for {round_name}, skipping this round")
                continue
                
            # Create subplot
            ax = fig.add_subplot(gs[row, col])
            
            # Calculate histogram bin edges
            all_data = winning_data + non_winning_data
            bin_min = min(all_data) - 1
            bin_max = max(all_data) + 1
            bins = np.linspace(bin_min, bin_max, 30)
            
            # Plot distributions
            sns.histplot(winning_data, ax=ax, bins=bins, kde=True,
                    stat="density", alpha=0.7, label="Winners", color="green")
            sns.histplot(non_winning_data, ax=ax, bins=bins, kde=True,
                    stat="density", alpha=0.7, label="Non-Winners", color="red")
            
            # Compute histogram data
            winning_hist, bin_edges = np.histogram(winning_data, bins=bins, density=True)
            non_winning_hist, _ = np.histogram(non_winning_data, bins=bins, density=True)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            # Store histogram data
            histogram_data[round_name] = {
                'bin_center': bin_centers.tolist(),
                'bin_start': bin_edges[:-1].tolist(),
                'bin_end': bin_edges[1:].tolist(),
                'winners_density': winning_hist.tolist(),
                'non_winners_density': non_winning_hist.tolist(),
                'winners_count': np.histogram(winning_data, bins=bins, density=False)[0].tolist(),
                'non_winners_count': np.histogram(non_winning_data, bins=bins, density=False)[0].tolist(),
                'winners_mean': np.mean(winning_data),
                'non_winners_mean': np.mean(non_winning_data)
            }
            
            # Add mean lines
            winning_mean = np.mean(winning_data)
            non_winning_mean = np.mean(non_winning_data)
            
            ax.axvline(winning_mean, color='green', linestyle='--', 
                    label=f'Winners Mean: {winning_mean:.2f}')
            ax.axvline(non_winning_mean, color='red', linestyle='--', 
                    label=f'Non-Winners Mean: {non_winning_mean:.2f}')
            
            # Customize plot
            ax.set_title(f"{round_name}")
            ax.set_xlabel("Negative Log Probability (lower is more likely)")
            ax.set_ylabel("Density")
            ax.legend()
        
        # Create subplot for overall log probability
        ax_total = fig.add_subplot(gs[3, :])
        
        # Get overall log probability data
        winning_total = self.winning_log_probs
        non_winning_total = self.non_winning_log_probs
        
        # Check data availability again
        if not winning_total or not non_winning_total:
            print("Warning: Missing total log probability data, skipping overall plot")
            if save:
                import json
                with open(self.output_dir / "comparative_log_probability_distributions_data.json", 'w') as f:
                    json.dump(histogram_data, f, indent=2)
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            if save:
                plt.savefig(self.output_dir / "comparative_log_probability_distributions.png", dpi=300, bbox_inches='tight')
            self.comparative_log_prob_histogram_data = histogram_data
            return fig
        
        # Calculate bin edges
        all_data = winning_total + non_winning_total
        bin_min = min(all_data) - 5
        bin_max = max(all_data) + 5
        bins = np.linspace(bin_min, bin_max, 30)
        
        # Plot distributions
        sns.histplot(winning_total, bins=bins, kde=True,
                stat="density", alpha=0.7, label="Winners", color="green", ax=ax_total)
        sns.histplot(non_winning_total, bins=bins, kde=True,
                stat="density", alpha=0.7, label="Non-Winners", color="red", ax=ax_total)
        
        # Compute histogram data
        winning_hist, bin_edges = np.histogram(winning_total, bins=bins, density=True)
        non_winning_hist, _ = np.histogram(non_winning_total, bins=bins, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Store histogram data
        histogram_data["Overall"] = {
            'bin_center': bin_centers.tolist(),
            'bin_start': bin_edges[:-1].tolist(),
            'bin_end': bin_edges[1:].tolist(),
            'winners_density': winning_hist.tolist(),
            'non_winners_density': non_winning_hist.tolist(),
            'winners_count': np.histogram(winning_total, bins=bins, density=False)[0].tolist(),
            'non_winners_count': np.histogram(non_winning_total, bins=bins, density=False)[0].tolist(),
            'winners_mean': np.mean(winning_total),
            'non_winners_mean': np.mean(non_winning_total)
        }
        
        # Add mean lines
        winning_mean = np.mean(winning_total)
        non_winning_mean = np.mean(non_winning_total)
        
        ax_total.axvline(winning_mean, color='green', linestyle='--', 
                    label=f'Winners Mean: {winning_mean:.2f}')
        ax_total.axvline(non_winning_mean, color='red', linestyle='--', 
                    label=f'Non-Winners Mean: {non_winning_mean:.2f}')
        
        # Add percentile lines
        winning_q25 = np.percentile(winning_total, 25)
        winning_q75 = np.percentile(winning_total, 75)
        non_winning_q25 = np.percentile(non_winning_total, 25)
        non_winning_q75 = np.percentile(non_winning_total, 75)
        
        ax_total.axvline(winning_q25, color='green', linestyle=':', 
                    label=f'Winners 25th: {winning_q25:.2f}')
        ax_total.axvline(winning_q75, color='green', linestyle=':', 
                    label=f'Winners 75th: {winning_q75:.2f}')
        ax_total.axvline(non_winning_q25, color='red', linestyle=':', 
                    label=f'Non-Winners 25th: {non_winning_q25:.2f}')
        ax_total.axvline(non_winning_q75, color='red', linestyle=':', 
                    label=f'Non-Winners 75th: {non_winning_q75:.2f}')
        
        # Customize plot
        ax_total.set_title("Distribution of Overall Bracket Log Probability Scores")
        ax_total.set_xlabel("Negative Log Probability (lower is more likely)")
        ax_total.set_ylabel("Density")
        ax_total.legend()
        
        # Save histogram data
        if save:
            import json
            with open(self.output_dir / "comparative_log_probability_distributions_data.json", 'w') as f:
                json.dump(histogram_data, f, indent=2)
        
        # Adjust layout and save figure
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        if save:
            plt.savefig(self.output_dir / "comparative_log_probability_distributions.png", dpi=300, bbox_inches='tight')
        
        # Store data for later access
        self.comparative_log_prob_histogram_data = histogram_data
        
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

    def _interpret_effect_size(self, d: float) -> str:
        """Interpret Cohen's d effect size magnitude"""
        if abs(d) < 0.2:
            return "Negligible"
        elif abs(d) < 0.5:
            return "Small"
        elif abs(d) < 0.8:
            return "Medium"
        else:
            return "Large"

    def analyze_winning_vs_non_winning_upsets(self) -> pd.DataFrame:
        """
        Analyze differences in upset patterns between winning and non-winning brackets.
        
        Returns:
            DataFrame with comparative upset statistics for each round
        """
        if not hasattr(self, 'winning_underdogs_by_round') or not self.winning_underdogs_by_round:
            raise ValueError("Must run simulations before analyzing upsets")
        
        # Import scipy only when needed to avoid dependency issues
        from scipy.stats import ttest_ind as stats_ttest_ind
        
        # Check if we have data to compare
        if not self.winning_brackets or not self.non_winning_brackets:
            print("Warning: Missing bracket data for comparison. Make sure both winning and non-winning brackets were tracked.")
            return pd.DataFrame(columns=['round', 'winning_avg_upsets', 'non_winning_avg_upsets', 
                                        'difference', 'p_value', 'significant', 
                                        'effect_size', 'effect_magnitude'])
        
        stats = []
        
        # Calculate statistics for each round
        for round_name in self.ROUND_ORDER:
            winning_data = self.winning_underdogs_by_round.get(round_name, [])
            non_winning_data = self.non_winning_underdogs_by_round.get(round_name, [])
            
            # Skip rounds with insufficient data
            if len(winning_data) < 2 or len(non_winning_data) < 2:
                print(f"Warning: Insufficient data for statistical analysis in {round_name}. Skipping.")
                continue
            
            # Calculate basic statistics
            winning_mean = np.mean(winning_data)
            non_winning_mean = np.mean(non_winning_data)
            
            # Handle potential issues with data
            if np.isnan(winning_mean) or np.isnan(non_winning_mean):
                print(f"Warning: NaN values encountered in {round_name}. Skipping.")
                continue
                
            try:
                # Perform t-test to check if difference is statistically significant
                t_stat, p_value = stats_ttest_ind(winning_data, non_winning_data)
                
                # Calculate effect size (Cohen's d)
                winning_std = np.std(winning_data)
                non_winning_std = np.std(non_winning_data)
                
                # Check for zero standard deviations
                if winning_std == 0 and non_winning_std == 0:
                    cohens_d = 0  # No variation in either group
                else:
                    pooled_std = np.sqrt((winning_std**2 + non_winning_std**2) / 2)
                    cohens_d = (winning_mean - non_winning_mean) / pooled_std if pooled_std != 0 else 0
                
                # Add to results
                stats.append({
                    'round': round_name,
                    'winning_avg_upsets': winning_mean,
                    'non_winning_avg_upsets': non_winning_mean,
                    'difference': winning_mean - non_winning_mean,
                    'p_value': p_value,
                    'significant': p_value < 0.05,
                    'effect_size': cohens_d,
                    'effect_magnitude': self._interpret_effect_size(cohens_d)
                })
            except Exception as e:
                print(f"Warning: Error analyzing {round_name}: {str(e)}")
        
        # Add total upsets if available
        if hasattr(self, 'winning_total_underdogs') and hasattr(self, 'non_winning_total_underdogs'):
            if len(self.winning_total_underdogs) >= 2 and len(self.non_winning_total_underdogs) >= 2:
                try:
                    winning_mean = np.mean(self.winning_total_underdogs)
                    non_winning_mean = np.mean(self.non_winning_total_underdogs)
                    
                    t_stat, p_value = stats_ttest_ind(self.winning_total_underdogs, self.non_winning_total_underdogs)
                    
                    winning_std = np.std(self.winning_total_underdogs)
                    non_winning_std = np.std(self.non_winning_total_underdogs)
                    
                    if winning_std == 0 and non_winning_std == 0:
                        cohens_d = 0
                    else:
                        pooled_std = np.sqrt((winning_std**2 + non_winning_std**2) / 2)
                        cohens_d = (winning_mean - non_winning_mean) / pooled_std if pooled_std != 0 else 0
                    
                    stats.append({
                        'round': 'Total',
                        'winning_avg_upsets': winning_mean,
                        'non_winning_avg_upsets': non_winning_mean,
                        'difference': winning_mean - non_winning_mean,
                        'p_value': p_value,
                        'significant': p_value < 0.05,
                        'effect_size': cohens_d,
                        'effect_magnitude': self._interpret_effect_size(cohens_d)
                    })
                except Exception as e:
                    print(f"Warning: Error analyzing total upsets: {str(e)}")
        
        # Return empty DataFrame if no statistics were calculated
        if not stats:
            print("Warning: No valid statistics could be calculated.")
            return pd.DataFrame(columns=['round', 'winning_avg_upsets', 'non_winning_avg_upsets', 
                                        'difference', 'p_value', 'significant', 
                                        'effect_size', 'effect_magnitude'])
        
        # Convert to DataFrame and sort
        stats_df = pd.DataFrame(stats)
        
        # Sort with individual rounds first, then total at the end
        round_order = {round_name: i for i, round_name in enumerate(self.ROUND_ORDER)}
        round_order['Total'] = len(self.ROUND_ORDER)
        
        stats_df['sort_order'] = stats_df['round'].map(round_order)
        stats_df = stats_df.sort_values('sort_order').drop('sort_order', axis=1)
        
        # Save to file
        stats_df.to_csv(self.output_dir / "upset_comparison_statistics.csv", index=False)
        
        return stats_df

    def analyze_winning_vs_non_winning_log_probs(self) -> pd.DataFrame:
        """
        Analyze differences in log probabilities between winning and non-winning brackets.
        
        Returns:
            DataFrame with comparative log probability statistics for each round
        """
        if not hasattr(self, 'winning_log_probs_by_round') or not self.winning_log_probs_by_round:
            raise ValueError("Must run simulations before analyzing log probabilities")
        
        # Import scipy only when needed
        from scipy.stats import ttest_ind as stats_ttest_ind
        
        # Check if we have data to compare
        if not self.winning_log_probs or not self.non_winning_log_probs:
            print("Warning: Missing log probability data for comparison. Make sure both winning and non-winning brackets were tracked.")
            return pd.DataFrame(columns=['round', 'winning_avg_log_prob', 'non_winning_avg_log_prob', 
                                        'difference', 'p_value', 'significant', 
                                        'effect_size', 'effect_magnitude'])
        
        stats = []
        
        # Calculate statistics for each round
        for round_name in self.ROUND_ORDER:
            winning_data = self.winning_log_probs_by_round.get(round_name, [])
            non_winning_data = self.non_winning_log_probs_by_round.get(round_name, [])
            
            # Skip rounds with insufficient data
            if len(winning_data) < 2 or len(non_winning_data) < 2:
                print(f"Warning: Insufficient log probability data for statistical analysis in {round_name}. Skipping.")
                continue
            
            # Calculate basic statistics
            winning_mean = np.mean(winning_data)
            non_winning_mean = np.mean(non_winning_data)
            
            # Handle potential issues with data
            if np.isnan(winning_mean) or np.isnan(non_winning_mean):
                print(f"Warning: NaN values encountered in {round_name} log probabilities. Skipping.")
                continue
                
            try:
                # Perform t-test to check if difference is statistically significant
                t_stat, p_value = stats_ttest_ind(winning_data, non_winning_data)
                
                # Calculate effect size (Cohen's d)
                winning_std = np.std(winning_data)
                non_winning_std = np.std(non_winning_data)
                
                # Check for zero standard deviations
                if winning_std == 0 and non_winning_std == 0:
                    cohens_d = 0  # No variation in either group
                else:
                    pooled_std = np.sqrt((winning_std**2 + non_winning_std**2) / 2)
                    cohens_d = (winning_mean - non_winning_mean) / pooled_std if pooled_std != 0 else 0
                
                # Add to results
                stats.append({
                    'round': round_name,
                    'winning_avg_log_prob': winning_mean,
                    'non_winning_avg_log_prob': non_winning_mean,
                    'difference': winning_mean - non_winning_mean,
                    'p_value': p_value,
                    'significant': p_value < 0.05,
                    'effect_size': cohens_d,
                    'effect_magnitude': self._interpret_effect_size(cohens_d)
                })
            except Exception as e:
                print(f"Warning: Error analyzing {round_name} log probabilities: {str(e)}")
        
        # Add overall log probability if available
        if hasattr(self, 'winning_log_probs') and hasattr(self, 'non_winning_log_probs'):
            if len(self.winning_log_probs) >= 2 and len(self.non_winning_log_probs) >= 2:
                try:
                    winning_mean = np.mean(self.winning_log_probs)
                    non_winning_mean = np.mean(self.non_winning_log_probs)
                    
                    t_stat, p_value = stats_ttest_ind(self.winning_log_probs, self.non_winning_log_probs)
                    
                    winning_std = np.std(self.winning_log_probs)
                    non_winning_std = np.std(self.non_winning_log_probs)
                    
                    if winning_std == 0 and non_winning_std == 0:
                        cohens_d = 0
                    else:
                        pooled_std = np.sqrt((winning_std**2 + non_winning_std**2) / 2)
                        cohens_d = (winning_mean - non_winning_mean) / pooled_std if pooled_std != 0 else 0
                    
                    stats.append({
                        'round': 'Overall',
                        'winning_avg_log_prob': winning_mean,
                        'non_winning_avg_log_prob': non_winning_mean,
                        'difference': winning_mean - non_winning_mean,
                        'p_value': p_value,
                        'significant': p_value < 0.05,
                        'effect_size': cohens_d,
                        'effect_magnitude': self._interpret_effect_size(cohens_d)
                    })
                except Exception as e:
                    print(f"Warning: Error analyzing overall log probabilities: {str(e)}")
        
        # Return empty DataFrame if no statistics were calculated
        if not stats:
            print("Warning: No valid log probability statistics could be calculated.")
            return pd.DataFrame(columns=['round', 'winning_avg_log_prob', 'non_winning_avg_log_prob', 
                                        'difference', 'p_value', 'significant', 
                                        'effect_size', 'effect_magnitude'])
        
        # Convert to DataFrame and sort
        stats_df = pd.DataFrame(stats)
        
        # Sort with individual rounds first, then overall at the end
        round_order = {round_name: i for i, round_name in enumerate(self.ROUND_ORDER)}
        round_order['Overall'] = len(self.ROUND_ORDER)
        
        stats_df['sort_order'] = stats_df['round'].map(round_order)
        stats_df = stats_df.sort_values('sort_order').drop('sort_order', axis=1)
        
        # Save to file
        stats_df.to_csv(self.output_dir / "log_probability_comparison_statistics.csv", index=False)
        
        return stats_df

    def compare_underdog_distributions(self) -> pd.DataFrame:
        """
        Calculate the distribution difference between winning and non-winning brackets
        for each number of upsets in each round.
        
        Returns:
            DataFrame showing the probability difference for each number of upsets
        """
        if not hasattr(self, 'comparative_upset_histogram_data'):
            # Generate the comparative histogram data if not already done
            self.plot_comparative_upset_distributions()
        
        # Initialize results
        results = []
        
        # Process each round
        for round_name, data in self.comparative_upset_histogram_data.items():
            bin_centers = data['bin_center']
            winners_density = data['winners_density']
            non_winners_density = data['non_winners_density']
            
            # Calculate advantage (positive means winners favor this number of upsets)
            for i, bin_center in enumerate(bin_centers):
                # Skip if no data for this bin
                if i >= len(winners_density) or i >= len(non_winners_density):
                    continue
                    
                # Calculate the difference (how much more likely winners are to have this many upsets)
                advantage = winners_density[i] - non_winners_density[i]
                
                # Create row in results
                results.append({
                    'round': round_name,
                    'upsets': int(bin_center) if round_name != "Total Upsets" else bin_center,
                    'winners_density': winners_density[i],
                    'non_winners_density': non_winners_density[i],
                    'advantage': advantage,
                    'relative_advantage': advantage / non_winners_density[i] if non_winners_density[i] > 0 else float('inf')
                })
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        # Add sorting column for rounds
        round_order = {round_name: i for i, round_name in enumerate(self.ROUND_ORDER)}
        round_order['Total Upsets'] = len(self.ROUND_ORDER)
        
        results_df['round_order'] = results_df['round'].map(lambda x: round_order.get(x, 999))
        
        # Sort by round and then by number of upsets
        results_df = results_df.sort_values(['round_order', 'upsets']).drop('round_order', axis=1)
        
        # Save results
        results_df.to_csv(self.output_dir / "upset_distribution_differences.csv", index=False)
        
        return results_df

    def identify_optimal_upset_strategy(self) -> pd.DataFrame:
        """
        Identify the optimal number of upsets in each round based on winning brackets.
        
        Returns:
            DataFrame with optimal upset counts for each round
        """
        if not hasattr(self, 'comparative_upset_histogram_data'):
            # Generate the comparative histogram data if not already done
            self.plot_comparative_upset_distributions()
        
        optimal_strategy = []
        
        # Process each round
        for round_name, data in self.comparative_upset_histogram_data.items():
            bin_centers = data['bin_center']
            winners_density = data['winners_density']
            non_winners_density = data['non_winners_density']
            
            # Find the bin with maximum advantage for winners
            max_advantage_idx = np.argmax(np.array(winners_density) - np.array(non_winners_density))
            max_advantage_upsets = bin_centers[max_advantage_idx]
            max_advantage = winners_density[max_advantage_idx] - non_winners_density[max_advantage_idx]
            
            # Find the bin with maximum density for winners
            max_winners_idx = np.argmax(winners_density)
            max_winners_upsets = bin_centers[max_winners_idx]
            
            # Find the mode of the winning upsets
            if round_name != "Total Upsets":
                if round_name in self.winning_underdogs_by_round:
                    winning_mode = pd.Series(self.winning_underdogs_by_round[round_name]).mode()[0]
                else:
                    winning_mode = None
            else:
                winning_mode = pd.Series(self.winning_total_underdogs).mode()[0]
            
            # Calculate the winner mean
            if round_name != "Total Upsets":
                if round_name in self.winning_underdogs_by_round:
                    winning_mean = np.mean(self.winning_underdogs_by_round[round_name])
                else:
                    winning_mean = None
            else:
                winning_mean = np.mean(self.winning_total_underdogs)
            
            # Add to results
            optimal_strategy.append({
                'round': round_name,
                'max_advantage_upsets': max_advantage_upsets,
                'max_advantage': max_advantage,
                'max_density_upsets': max_winners_upsets,
                'mode_upsets': winning_mode,
                'mean_upsets': winning_mean
            })
        
        # Convert to DataFrame
        strategy_df = pd.DataFrame(optimal_strategy)
        
        # Add sorting column for rounds
        round_order = {round_name: i for i, round_name in enumerate(self.ROUND_ORDER)}
        round_order['Total Upsets'] = len(self.ROUND_ORDER)
        
        strategy_df['round_order'] = strategy_df['round'].map(lambda x: round_order.get(x, 999))
        
        # Sort by round
        strategy_df = strategy_df.sort_values('round_order').drop('round_order', axis=1)
        
        # Save results
        strategy_df.to_csv(self.output_dir / "optimal_upset_strategy.csv", index=False)
        
        return strategy_df

    def compare_champion_distributions(self) -> pd.DataFrame:
        """
        Compare the champion picks between winning and non-winning brackets.
        
        Returns:
            DataFrame with statistics on champion picks by seed and team
        """
        # Count champion picks in winning brackets
        winning_champions = defaultdict(int)
        for bracket in self.winning_brackets:
            if "Champion" in bracket.results:
                champion = bracket.results["Champion"]
                key = (champion.seed, champion.name)
                winning_champions[key] += 1
        
        # Count champion picks in non-winning brackets
        non_winning_champions = defaultdict(int)
        for bracket in self.non_winning_brackets:
            if "Champion" in bracket.results:
                champion = bracket.results["Champion"]
                key = (champion.seed, champion.name)
                non_winning_champions[key] += 1
        
        # Calculate frequencies
        winning_total = len(self.winning_brackets)
        non_winning_total = len(self.non_winning_brackets)
        
        # Merge results
        all_champions = set(list(winning_champions.keys()) + list(non_winning_champions.keys()))
        
        comparison = []
        for seed, team in all_champions:
            winning_count = winning_champions.get((seed, team), 0)
            non_winning_count = non_winning_champions.get((seed, team), 0)
            
            winning_freq = winning_count / winning_total if winning_total > 0 else 0
            non_winning_freq = non_winning_count / non_winning_total if non_winning_total > 0 else 0
            
            # Calculate advantage (how much more likely winners are to pick this team)
            freq_diff = winning_freq - non_winning_freq
            rel_advantage = freq_diff / non_winning_freq if non_winning_freq > 0 else float('inf')
            
            comparison.append({
                'seed': seed,
                'team': team,
                'winning_count': winning_count,
                'non_winning_count': non_winning_count,
                'winning_freq': winning_freq,
                'non_winning_freq': non_winning_freq,
                'freq_diff': freq_diff,
                'relative_advantage': rel_advantage
            })
        
        # Convert to DataFrame
        comparison_df = pd.DataFrame(comparison)
        
        # Sort by frequency difference (most advantageous picks first)
        comparison_df = comparison_df.sort_values('freq_diff', ascending=False)
        
        # Save results
        comparison_df.to_csv(self.output_dir / "champion_pick_comparison.csv", index=False)
        
        return comparison_df

    def save_all_comparative_data(self):
        """Save all comparative analysis data and generate plots, with robust error handling"""
        # Make sure simulations have been run
        if not hasattr(self, 'winning_brackets') or not hasattr(self, 'non_winning_brackets'):
            raise ValueError("Must run simulations before saving comparative data")
        
        # Check data availability 
        if not self.winning_brackets:
            print("Warning: No winning brackets data available. Limited analysis will be performed.")
        if not self.non_winning_brackets:
            print("Warning: No non-winning brackets data available. Comparative analysis will be limited.")
        
        # Generate and save plots with error handling
        print("Generating comparative upset distributions...")
        try:
            upset_plot = self.plot_comparative_upset_distributions(save=True)
            if upset_plot is None:
                print("Failed to generate comparative upset distributions.")
        except Exception as e:
            print(f"Error generating comparative upset distributions: {str(e)}")
        
        print("Generating comparative log probability distributions...")
        try:
            log_prob_plot = self.plot_comparative_log_probability_distributions(save=True)
            if log_prob_plot is None:
                print("Failed to generate comparative log probability distributions.")
        except Exception as e:
            print(f"Error generating comparative log probability distributions: {str(e)}")
        
        # Generate and save statistics with error handling
        print("Analyzing upset patterns...")
        try:
            upset_stats = self.analyze_winning_vs_non_winning_upsets()
            if upset_stats.empty:
                print("No valid upset statistics could be generated.")
        except Exception as e:
            print(f"Error analyzing upset patterns: {str(e)}")
            upset_stats = pd.DataFrame()
        
        print("Analyzing log probabilities...")
        try:
            log_prob_stats = self.analyze_winning_vs_non_winning_log_probs()
            if log_prob_stats.empty:
                print("No valid log probability statistics could be generated.")
        except Exception as e:
            print(f"Error analyzing log probabilities: {str(e)}")
            log_prob_stats = pd.DataFrame()
        
        # Generate and save strategy insights with error handling
        print("Analyzing upset distribution differences...")
        try:
            if hasattr(self, 'comparative_upset_histogram_data') and self.comparative_upset_histogram_data:
                distribution_diff = self.compare_underdog_distributions()
            else:
                print("No comparative upset histogram data available. Skipping upset distribution differences.")
                distribution_diff = pd.DataFrame()
        except Exception as e:
            print(f"Error analyzing upset distribution differences: {str(e)}")
            distribution_diff = pd.DataFrame()
        
        print("Identifying optimal upset strategy...")
        try:
            if hasattr(self, 'comparative_upset_histogram_data') and self.comparative_upset_histogram_data:
                optimal_strategy = self.identify_optimal_upset_strategy()
            else:
                print("No comparative upset histogram data available. Skipping optimal upset strategy.")
                optimal_strategy = pd.DataFrame()
        except Exception as e:
            print(f"Error identifying optimal upset strategy: {str(e)}")
            optimal_strategy = pd.DataFrame()
        
        print("Comparing champion pick distributions...")
        try:
            if self.winning_brackets and self.non_winning_brackets:
                champion_comparison = self.compare_champion_distributions()
            else:
                print("Insufficient data for champion pick comparison.")
                champion_comparison = pd.DataFrame()
        except Exception as e:
            print(f"Error comparing champion distributions: {str(e)}")
            champion_comparison = pd.DataFrame()
        
        # Create a summary report with error handling
        print("Creating comparative summary report...")
        try:
            self._create_comparative_summary_report()
        except Exception as e:
            print(f"Error creating summary report: {str(e)}")
            # Create a basic error report
            with open(self.output_dir / "comparative_analysis_summary.md", 'w') as f:
                f.write("# Comparative Analysis Summary Report\n\n")
                f.write(f"## Date Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
                f.write("## Error Generating Complete Report\n\n")
                f.write(f"An error occurred while generating the complete report: {str(e)}\n\n")
                f.write("Please check the individual data files for available analysis results.\n")
        
        print(f"All comparative analysis data and visualizations saved to {self.output_dir}/")

    def _create_comparative_summary_report(self):
        """Create a summary report of key findings from the comparative analysis, with robust error handling"""
        # Collect key statistics with error handling
        try:
            upset_stats = self.analyze_winning_vs_non_winning_upsets()
            significant_rounds_upsets = upset_stats[upset_stats['significant']]['round'].tolist() if not upset_stats.empty else []
        except Exception:
            upset_stats = pd.DataFrame()
            significant_rounds_upsets = []
        
        try:
            log_prob_stats = self.analyze_winning_vs_non_winning_log_probs()
            significant_rounds_probs = log_prob_stats[log_prob_stats['significant']]['round'].tolist() if not log_prob_stats.empty else []
        except Exception:
            log_prob_stats = pd.DataFrame()
            significant_rounds_probs = []
        
        try:
            optimal_strategy = self.identify_optimal_upset_strategy()
        except Exception:
            optimal_strategy = pd.DataFrame()
        
        try:
            champion_comparison = self.compare_champion_distributions().head(5) if hasattr(self, 'compare_champion_distributions') else pd.DataFrame()
        except Exception:
            champion_comparison = pd.DataFrame()
        
        # Create report
        report = [
            "# Comparative Analysis Summary Report",
            f"## Date Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"## Number of Pools Analyzed: {self.num_pools}",
            f"## Total Brackets Analyzed: {len(self.winning_brackets) + len(self.non_winning_brackets)}",
            f"### Winners: {len(self.winning_brackets)}",
            f"### Non-Winners: {len(self.non_winning_brackets)}",
            "",
            "## Key Findings",
            "",
            "### Significant Differences in Upsets",
        ]
        
        if significant_rounds_upsets:
            report.append("The following rounds showed statistically significant differences in upset patterns between winning and non-winning brackets:")
            for round_name in significant_rounds_upsets:
                row = upset_stats[upset_stats['round'] == round_name].iloc[0]
                report.append(f"- **{round_name}**: Winners avg {row['winning_avg_upsets']:.2f} upsets vs. non-winners {row['non_winning_avg_upsets']:.2f} upsets (p={row['p_value']:.4f}, {row['effect_magnitude']} effect)")
        else:
            report.append("No statistically significant differences in upset patterns were found between winning and non-winning brackets.")
        
        report.extend([
            "",
            "### Significant Differences in Log Probabilities",
        ])
        
        if significant_rounds_probs:
            report.append("The following rounds showed statistically significant differences in log probabilities between winning and non-winning brackets:")
            for round_name in significant_rounds_probs:
                row = log_prob_stats[log_prob_stats['round'] == round_name].iloc[0]
                report.append(f"- **{round_name}**: Winners avg {row['winning_avg_log_prob']:.2f} vs. non-winners {row['non_winning_avg_log_prob']:.2f} (p={row['p_value']:.4f}, {row['effect_magnitude']} effect)")
        else:
            report.append("No statistically significant differences in log probabilities were found between winning and non-winning brackets.")
        
        report.extend([
            "",
            "### Optimal Upset Strategy",
        ])
        
        if not optimal_strategy.empty:
            report.append("Based on the most successful brackets, the following upset strategy is recommended:")
            
            for _, row in optimal_strategy.iterrows():
                try:
                    if row['round'] != "Total Upsets":
                        report.append(f"- **{row['round']}**: {int(row['max_advantage_upsets'])} upsets (mode: {int(row['mode_upsets']) if not pd.isna(row['mode_upsets']) else 'N/A'}, mean: {row['mean_upsets']:.1f})")
                    else:
                        report.append(f"- **Total Across All Rounds**: {int(row['max_advantage_upsets'])} upsets (mode: {int(row['mode_upsets'])}, mean: {row['mean_upsets']:.1f})")
                except Exception as e:
                    report.append(f"- **{row.get('round', 'Unknown Round')}**: Error calculating optimal strategy: {str(e)}")
        else:
            report.append("Insufficient data to determine optimal upset strategy.")
        
        report.extend([
            "",
            "### Top Champion Picks",
        ])
        
        if not champion_comparison.empty:
            report.append("The following champion picks were most advantageous in winning brackets:")
            
            for _, row in champion_comparison.iterrows():
                try:
                    advantage = row['freq_diff'] * 100  # Convert to percentage
                    report.append(f"- **{row['team']} (Seed {row['seed']})**: Winners picked {row['winning_freq']*100:.1f}% vs. non-winners {row['non_winning_freq']*100:.1f}% (advantage: {advantage:.1f}%)")
                except Exception as e:
                    report.append(f"- Error calculating champion advantage: {str(e)}")
        else:
            report.append("Insufficient data to determine top champion picks.")
        
        # Save report to file
        with open(self.output_dir / "comparative_analysis_summary.md", 'w') as f:
            f.write("\n".join(report))
        
        return report

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
    """Example usage of bracket analysis with winning vs non-winning comparisons, with robust error handling"""
    try:
        # Set up argument parser
        parser = argparse.ArgumentParser(description='Analyze March Madness bracket pool simulations')
        parser.add_argument('--num_pools', type=int, default=1000,
                            help='Number of pools to simulate')
        parser.add_argument('--entries_per_pool', type=int, default=10,
                            help='Number of entries per pool')
        parser.add_argument('--women', action='store_true',
                            help='Whether to use women\'s basketball data instead of men\'s')
        parser.add_argument('--comparative', action='store_true',
                            help='Run winning vs non-winning comparative analysis')
        parser.add_argument('--output_dir', type=str, default=None,
                            help='Custom output directory (optional)')
        parser.add_argument('--debug', action='store_true',
                            help='Enable additional debug output')
        args = parser.parse_args()

        # Set up logging
        log_level = logging.DEBUG if args.debug else logging.INFO
        logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')

        try:
            # Get current standings
            logging.info("Retrieving current basketball standings...")
            standings = Standings(women=args.women)
        except Exception as e:
            logging.error(f"Error retrieving standings: {str(e)}")
            print("Error: Could not retrieve basketball standings. Please check your internet connection and try again.")
            return 1
        
        # Set up output directory
        gender = "women" if args.women else "men"
        output_dir = args.output_dir if args.output_dir else f"bracket_analysis_{args.entries_per_pool}entries_{gender}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize analyzer with output directory
        logging.info(f"Starting analysis for {args.entries_per_pool} entries over {args.num_pools} pools")
        analyzer = BracketAnalysis(standings, num_pools=args.num_pools, output_dir=output_dir)
        
        try:
            # Run simulations
            logging.info("Beginning bracket simulations...")
            analyzer.simulate_pools(entries_per_pool=args.entries_per_pool)
        except Exception as e:
            logging.error(f"Error during simulation: {str(e)}")
            print(f"Error during simulation: {str(e)}")
            print("Some analysis may still be available with partial results.")
        
        if args.comparative:
            # Run comparative analysis between winning and non-winning brackets
            logging.info("Running winning vs. non-winning comparative analysis...")
            print("\nRunning winning vs. non-winning comparative analysis...")
            
            try:
                # Save all data and generate summary report
                analyzer.save_all_comparative_data()
                print(f"\nComparative analysis data and visualizations have been saved to {output_dir}/")
                print(f"A summary report has been generated at {output_dir}/comparative_analysis_summary.md")
            except Exception as e:
                logging.error(f"Error during comparative analysis: {str(e)}")
                print(f"Error during comparative analysis: {str(e)}")
                print("Some partial results may still be available.")
        else:
            # Run standard analysis
            logging.info("Running standard analysis...")
            
            try:
                # Save all data and generate plots from original analysis
                analyzer.save_all_data()
                
                # Generate original plots with error handling
                try:
                    analyzer.plot_upset_distributions()
                except Exception as e:
                    logging.error(f"Error generating upset distributions: {str(e)}")
                
                try:
                    analyzer.plot_log_probability_distributions()
                except Exception as e:
                    logging.error(f"Error generating log probability distributions: {str(e)}")
                
                # Print various analyses with error handling
                print("\nUpset Statistics by Round:")
                try:
                    upset_stats = analyzer.analyze_upsets()
                    print(upset_stats.to_string(index=False))
                except Exception as e:
                    logging.error(f"Error analyzing upsets: {str(e)}")
                    print(f"Error analyzing upsets: {str(e)}")
                
                print("\nMost Common Underdogs:")
                try:
                    common_underdogs = analyzer.find_common_underdogs()
                    print(common_underdogs.groupby("advanced_through").head(10).to_string(index=False))
                except Exception as e:
                    logging.error(f"Error finding common underdogs: {str(e)}")
                    print(f"Error finding common underdogs: {str(e)}")
                
                print("\nChampionship Pick Analysis:")
                try:
                    champion_picks = analyzer.analyze_champion_picks()
                    print(champion_picks.head(10).to_string(index=False))
                except Exception as e:
                    logging.error(f"Error analyzing champion picks: {str(e)}")
                    print(f"Error analyzing champion picks: {str(e)}")

                print("\nBracket Likelihood Analysis:")
                try:
                    log_prob_stats = analyzer.analyze_log_probabilities()
                    print(log_prob_stats.T.to_string(header=False))
                except Exception as e:
                    logging.error(f"Error analyzing log probabilities: {str(e)}")
                    print(f"Error analyzing log probabilities: {str(e)}")
                
                print(f"\nAnalysis data and visualizations have been saved to {output_dir}/")
            except Exception as e:
                logging.error(f"Error during standard analysis: {str(e)}")
                print(f"Error during standard analysis: {str(e)}")
        
        return 0
    
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        print(f"An unexpected error occurred: {str(e)}")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())
