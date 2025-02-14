#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   bracket_optimizer.py
@Time    :   2024/02/22
@Author  :   Taylor Firman
@Version :   0.1.0
@Contact :   tefirman@gmail.com
@Desc    :   Optimization and analysis tools for March Madness brackets
'''

from typing import List
import pandas as pd
from dancing.wn_cbb_scraper import Standings
from dancing.cbb_brackets import Bracket

class BracketOptimizer:
    """
    Optimizer class that finds the best bracket picks based on simulation results and optimization criteria.
    
    Key features:
    - Simulates many possible brackets to identify winning strategies
    - Uses ELO ratings and seed matchups to estimate win probabilities 
    - Considers both likelihood of outcomes and potential bracket points
    - Can optimize for different pool sizes and scoring systems
    - Provides confidence levels for each pick
    
    Attributes:
        dancing: DataFrame of tournament teams with seeds and ratings
        num_sims: Number of simulations to run
        scoring_weights: Points awarded for each round
        risk_factor: How much to favor upsets vs. chalk (0-1)
        optimal_picks: DataFrame of optimized picks for each round
        pick_confidence: Confidence scores for each optimized pick
    """
    
    def __init__(self, dancing: pd.DataFrame, num_sims: int = 10000,
                 scoring_weights: List[float] = None, risk_factor: float = 0.5):
        """
        Initialize optimizer with tournament field and parameters.
        
        Args:
            dancing: DataFrame with team seeds and ratings
            num_sims: Number of simulations to run
            scoring_weights: Points per round, defaults to [10,20,40,80,160,320]
            risk_factor: How much to favor upsets (0-1), defaults to 0.5
        """
        self.dancing = dancing
        self.num_sims = num_sims
        self.scoring_weights = scoring_weights or [10, 20, 40, 80, 160, 320]
        self.risk_factor = min(max(risk_factor, 0), 1)  # Bound between 0-1
        
        # Initialize storage for optimization results
        self.optimal_picks = pd.DataFrame()
        self.pick_confidence = pd.DataFrame()
        
        # Run initial optimization
        self._simulate_tournament()
        self._optimize_picks()
        
    def _simulate_tournament(self):
        """Run many tournament simulations to gather outcome probabilities."""
        self.sim_results = []
        
        for _ in range(self.num_sims):
            # Create new bracket for this simulation
            bracket = Bracket(self.dancing.copy())
            
            # Store results
            round_results = []
            for round_num, round_games in enumerate(bracket.rounds):
                round_results.append({
                    'round': round_num,
                    'winners': round_games.pick.tolist(),
                    'win_probs': round_games[['elo_prob1', 'elo_prob2']].values.tolist()
                })
            self.sim_results.append(round_results)
            
    def _optimize_picks(self):
        """Analyze simulation results to determine optimal picks."""
        rounds = ['Round of 64', 'Round of 32', 'Sweet 16', 'Elite 8', 'Final 4', 'Championship']
        
        optimal_picks = []
        pick_confidence = []
        
        # Analyze each round
        for round_num in range(len(rounds)):
            # Collect all winners from this round across simulations
            round_winners = []
            round_probs = []
            for sim in self.sim_results:
                if round_num < len(sim):
                    round_winners.extend(sim[round_num]['winners'])
                    round_probs.extend(sim[round_num]['win_probs'])
            
            # Calculate win frequencies
            win_counts = pd.Series(round_winners).value_counts()
            win_probs = pd.DataFrame(round_probs).mean()
            
            # Calculate expected value for each team
            exp_values = {}
            for team in win_counts.index:
                win_freq = win_counts[team] / len(round_winners)
                
                # Blend frequency with ELO probability using risk factor
                elo_prob = win_probs[self.dancing.Team == team].iloc[0] \
                    if team in self.dancing.Team.values else 0.5
                blended_prob = (win_freq * (1 - self.risk_factor) + 
                              elo_prob * self.risk_factor)
                
                # Calculate expected points
                exp_value = blended_prob * self.scoring_weights[round_num]
                exp_values[team] = exp_value
            
            # Select optimal picks maximizing expected value
            best_picks = pd.Series(exp_values).sort_values(ascending=False)
            optimal_picks.append(best_picks.index.tolist())
            
            # Calculate confidence scores (0-100)
            confidence_scores = (best_picks / best_picks.max() * 100).round(1)
            pick_confidence.append(confidence_scores.to_dict())
        
        # Store results
        self.optimal_picks = pd.DataFrame({
            'round': rounds,
            'picks': optimal_picks
        })
        
        self.pick_confidence = pd.DataFrame({
            'round': rounds,
            'confidence': pick_confidence
        })
    
    def get_optimal_bracket(self) -> Bracket:
        """
        Generate a bracket using the optimized picks.
        
        Returns:
            Bracket object with optimized picks
        """
        # Create new bracket starting with full field
        bracket = Bracket(self.dancing.copy())
        
        # Apply optimal picks for each round
        for round_num, round_picks in enumerate(self.optimal_picks['picks']):
            if round_num < len(bracket.rounds):
                matchups = bracket.rounds[round_num]
                for game_idx, (team1, team2) in enumerate(zip(matchups.Team1, matchups.Team2)):
                    # Pick winner based on optimization
                    if team1 in round_picks:
                        matchups.loc[game_idx, 'pick'] = team1
                    else:
                        matchups.loc[game_idx, 'pick'] = team2
                        
                # Update bracket state
                bracket.update_standings()
        
        return bracket
    
    def get_pick_analysis(self) -> pd.DataFrame:
        """
        Get detailed analysis of optimized picks with confidence levels.
        
        Returns:
            DataFrame with picks and confidence scores for each round
        """
        analysis = pd.DataFrame()
        
        for round_num in range(len(self.optimal_picks)):
            round_name = self.optimal_picks.loc[round_num, 'round']
            picks = self.optimal_picks.loc[round_num, 'picks']
            confidence = self.pick_confidence.loc[round_num, 'confidence']
            
            round_analysis = pd.DataFrame({
                'round': round_name,
                'team': picks,
                'confidence': [confidence[team] for team in picks],
                'seed': [self.dancing[self.dancing.Team == team].tourney_seed.iloc[0] 
                        for team in picks],
                'elo': [self.dancing[self.dancing.Team == team].ELO.iloc[0]
                       for team in picks]
            })
            
            analysis = pd.concat([analysis, round_analysis])
            
        return analysis.reset_index(drop=True)

def main():
    # Get current standings
    standings = Standings(women=False)  # Can use women=True for women's tournament

    # Create optimizer
    optimizer = BracketOptimizer(
        standings=standings,
        num_sims=10000,
        risk_factor=0.5
    )

    # Get analysis
    analysis = optimizer.get_pick_analysis()

    # Print formatted bracket
    optimizer.print_bracket()

if __name__ == "__main__":
    main()
