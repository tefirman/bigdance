#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   server.py
@Time    :   2024/02/20
@Author  :   Taylor Firman
@Version :   1.0
@Contact :   tefirman@gmail.com
@Desc    :   Server logic for March Madness bracket app
'''

from shiny import render, ui
from shiny.types import SilentException
import logging
from typing import Dict, List, Optional, Tuple
from bigdance.bigdance_integration import create_bracket_with_picks
from bigdance.cbb_brackets import Bracket, Pool

from data import teams, actual_bracket

logger = logging.getLogger(__name__)

def get_game_winner(input, game_id: str) -> Optional[str]:
    """Helper function to safely get game winner"""
    try:
        return input[game_id]()
    except SilentException:
        return None
    except Exception as e:
        logger.error(f"Error getting winner for {game_id}: {str(e)}")
        return None

def get_round1_matchups(region: str) -> List[Tuple[Optional[Dict], Optional[Dict]]]:
    """
    Get initial matchups for first round games in a region following NCAA tournament seeding rules:
    1 vs 16, 8 vs 9, 5 vs 12, 4 vs 13, 6 vs 11, 3 vs 14, 7 vs 10, 2 vs 15
    """
    matchups = []
    region_teams = teams[region]
    
    # Define first round matchup pattern [(seed1, seed2), ...]
    seed_matchups = [
        (1, 16), (8, 9), (5, 12), (4, 13),
        (6, 11), (3, 14), (7, 10), (2, 15)
    ]
    
    # Create dictionary mapping seeds to teams
    seed_to_team = {team["Seed"]: team for team in region_teams}
    
    # Create matchups following seed pattern
    for seed1, seed2 in seed_matchups:
        team1 = seed_to_team.get(seed1)
        team2 = seed_to_team.get(seed2)
        matchups.append((team1, team2))
    
    return matchups

def get_second_round_matchups(input, region: str) -> List[Tuple[Optional[Dict], Optional[Dict]]]:
    """
    Get second round matchups based on first round selections.
    Winners of games (1v16 vs 8v9), (5v12 vs 4v13), (6v11 vs 3v14), (7v10 vs 2v15)
    """
    matchups = []
    region_teams = teams[region]
    
    # Define which first round games pair up in second round
    second_round_pairs = [(0, 1), (2, 3), (4, 5), (6, 7)]
    
    for game1_idx, game2_idx in second_round_pairs:
        game1_id = f"{region.lower()}_round1_game_{game1_idx}"
        game2_id = f"{region.lower()}_round1_game_{game2_idx}"
        
        # Get winners from first round if selected
        winner1 = get_game_winner(input, game1_id)
        winner2 = get_game_winner(input, game2_id)
        
        # Find team details for winners
        winner1_details = next((team for team in region_teams if team["Team"] == winner1), None) if winner1 else None
        winner2_details = next((team for team in region_teams if team["Team"] == winner2), None) if winner2 else None
        
        matchups.append((winner1_details, winner2_details))
    
    return matchups

def get_round_matchups(input, region: str, round_num: int) -> List[Tuple[Optional[Dict], Optional[Dict]]]:
    """
    Get matchups for Sweet 16 (round 3) and Elite 8 (round 4) based on previous round selections.
    Sweet 16: Winners of (1v16/8v9 vs 5v12/4v13) and (6v11/3v14 vs 7v10/2v15)
    Elite 8: Winners of Sweet 16 games
    """
    matchups = []
    region_teams = teams[region]
    
    # Calculate number of games in current round
    current_round_games = 2 ** (4 - round_num)  # 2 for Sweet 16, 1 for Elite 8
    
    # Define which previous round games pair up
    game_pairs = [(i*2, i*2+1) for i in range(current_round_games)]
    
    for game1_idx, game2_idx in game_pairs:
        prev_game1_id = f"{region.lower()}_round{round_num-1}_game_{game1_idx}"
        prev_game2_id = f"{region.lower()}_round{round_num-1}_game_{game2_idx}"
        
        winner1 = get_game_winner(input, prev_game1_id)
        winner2 = get_game_winner(input, prev_game2_id)
        
        # Find team details within the specific region first
        winner1_details = next((team for team in region_teams if team["Team"] == winner1), None)
        winner2_details = next((team for team in region_teams if team["Team"] == winner2), None)
        
        # If not found in region (could happen in later rounds), search all regions
        if not winner1_details and winner1:
            winner1_details = next((team for r_teams in teams.values() 
                                  for team in r_teams if team["Team"] == winner1), None)
        if not winner2_details and winner2:
            winner2_details = next((team for r_teams in teams.values() 
                                  for team in r_teams if team["Team"] == winner2), None)
        
        matchups.append((winner1_details, winner2_details))
    
    return matchups

def create_round_ui(input, region: str, round_num: int, matchups: List[Tuple[Dict, Dict]]) -> ui.div:
    """
    Create UI for any round's games with higher seed selected by default,
    while preserving user selections when possible.
    """
    games = []
    for i, (team1, team2) in enumerate(matchups):
        game_id = f"{region.lower()}_round{round_num}_game_{i}"
        choices = {}
        
        if team1:
            choices[team1["Team"]] = f"({team1['Seed']}) {team1['Team']}"
        if team2:
            choices[team2["Team"]] = f"({team2['Seed']}) {team2['Team']}"
        
        if len(choices) == 2:
            # Try to get current selection (if any)
            current_selection = None
            try:
                current_selection = input[game_id]()
            except:
                # No selection yet or other error
                pass
                
            # If current selection exists and is still valid, keep it
            if current_selection and current_selection in choices:
                default_team = current_selection
            else:
                # Otherwise default to higher seed
                default_team = team1["Team"] if team1["Seed"] < team2["Seed"] else team2["Team"]
            
            game = ui.div(
                {"class": "game-container"},
                ui.input_radio_buttons(
                    game_id,
                    f"Game {i + 1}",
                    choices,
                    selected=default_team
                )
            )
        else:
            game = ui.div(
                {"class": "game-container"},
                ui.p("Waiting for previous round selections...")
            )
        
        games.append(game)
    
    return ui.div(
        {"class": "bracket-region"},
        *games
    )

def get_final_four_matchups(input):
    """Get Final Four matchups based on Elite 8 selections"""
    winners = []
    for region in ["east", "west", "south", "midwest"]:
        winner = get_game_winner(input, f"{region}_round4_game_0")  # Elite 8 winner
        if winner:
            winner_details = next((team for region_teams in teams.values() 
                                 for team in region_teams if team["Team"] == winner), None)
            winners.append(winner_details)
    
    # Create Final Four matchups (East vs West, South vs Midwest)
    matchups = []
    if len(winners) >= 2:
        matchups.append((winners[0] if len(winners) > 0 else None, 
                        winners[1] if len(winners) > 1 else None))
    if len(winners) >= 4:
        matchups.append((winners[2] if len(winners) > 2 else None, 
                        winners[3] if len(winners) > 3 else None))
    return matchups

def server(input, output, session):
    """Main server function containing all callbacks and reactive logic"""
    
    # # Track when conference filter changes
    # @reactive.Effect
    # def _():
    #     logger.info(f"Conference changed to: {input.conference()}")

    # # Debug info output
    # @output
    # @render.text
    # def debug_info():
    #     return f"Current Conference: {input.conference()}\nLast Updated: {datetime.now()}"

    # First Round UI Outputs
    @output
    @render.ui
    def east_bracket_round1():
        matchups = get_round1_matchups("East")
        return create_round_ui(input, "East", 1, matchups)

    @output
    @render.ui
    def west_bracket_round1():
        matchups = get_round1_matchups("West")
        return create_round_ui(input, "West", 1, matchups)

    @output
    @render.ui
    def south_bracket_round1():
        matchups = get_round1_matchups("South")
        return create_round_ui(input, "South", 1, matchups)

    @output
    @render.ui
    def midwest_bracket_round1():
        matchups = get_round1_matchups("Midwest")
        return create_round_ui(input, "Midwest", 1, matchups)

    # Second Round UI Outputs
    @output
    @render.ui
    def east_bracket_round2():
        matchups = get_second_round_matchups(input, "East")
        return create_round_ui(input, "East", 2, matchups)

    @output
    @render.ui
    def west_bracket_round2():
        matchups = get_second_round_matchups(input, "West")
        return create_round_ui(input, "West", 2, matchups)

    @output
    @render.ui
    def south_bracket_round2():
        matchups = get_second_round_matchups(input, "South")
        return create_round_ui(input, "South", 2, matchups)

    @output
    @render.ui
    def midwest_bracket_round2():
        matchups = get_second_round_matchups(input, "Midwest")
        return create_round_ui(input, "Midwest", 2, matchups)

    # Third Round UI Outputs
    @output
    @render.ui
    def east_bracket_round3():
        matchups = get_round_matchups(input, "East", 3)
        return create_round_ui(input, "East", 3, matchups)
    
    @output
    @render.ui
    def west_bracket_round3():
        matchups = get_round_matchups(input, "West", 3)
        return create_round_ui(input, "West", 3, matchups)
    
    @output
    @render.ui
    def south_bracket_round3():
        matchups = get_round_matchups(input, "South", 3)
        return create_round_ui(input, "South", 3, matchups)
    
    @output
    @render.ui
    def midwest_bracket_round3():
        matchups = get_round_matchups(input, "Midwest", 3)
        return create_round_ui(input, "Midwest", 3, matchups)

    # Fourth Round UI Outputs
    @output
    @render.ui
    def east_bracket_round4():
        matchups = get_round_matchups(input, "East", 4)
        return create_round_ui(input, "East", 4, matchups)
    
    @output
    @render.ui
    def west_bracket_round4():
        matchups = get_round_matchups(input, "West", 4)
        return create_round_ui(input, "West", 4, matchups)
    
    @output
    @render.ui
    def south_bracket_round4():
        matchups = get_round_matchups(input, "South", 4)
        return create_round_ui(input, "South", 4, matchups)
    
    @output
    @render.ui
    def midwest_bracket_round4():
        matchups = get_round_matchups(input, "Midwest", 4)
        return create_round_ui(input, "Midwest", 4, matchups)

    @output
    @render.ui
    def final_four_games():
        matchups = get_final_four_matchups(input)
        return create_round_ui(input, "final", 5, matchups)

    @output
    @render.ui
    def championship_game():
        # Get Final Four winners
        winner1 = get_game_winner(input, "final_round5_game_0")
        winner2 = get_game_winner(input, "final_round5_game_1")
        
        winner1_details = next((team for region_teams in teams.values() 
                            for team in region_teams if team["Team"] == winner1), None) if winner1 else None
        winner2_details = next((team for region_teams in teams.values() 
                            for team in region_teams if team["Team"] == winner2), None) if winner2 else None
        
        matchups = [(winner1_details, winner2_details)]
        return create_round_ui(input, "final", 6, matchups)

    # Simulation Results Output
    @output
    @render.text
    def simulation_results():
        if input.simulate() == 0:
            return ""
            
        try:
            logger.debug("Starting simulation")
            # Track selections for all rounds
            selections = {
                "First Round": [],
                "Second Round": [],
                "Sweet 16": [],
                "Elite 8": [],
                "Final Four": [],
                "Championship": []
            }
            
            # Check each region's games through Elite 8
            for region in ["east", "west", "south", "midwest"]:
                logger.debug(f"Checking {region} region")
                
                # First round (8 games)
                for i in range(8):
                    game_id = f"{region}_round1_game_{i}"
                    winner = get_game_winner(input, game_id)
                    if winner:
                        selections["First Round"].append(winner)
                
                # Second round (4 games)
                for i in range(4):
                    game_id = f"{region}_round2_game_{i}"
                    winner = get_game_winner(input, game_id)
                    if winner:
                        selections["Second Round"].append(winner)
                
                # Sweet 16 (2 games)
                for i in range(2):
                    game_id = f"{region}_round3_game_{i}"
                    winner = get_game_winner(input, game_id)
                    if winner:
                        selections["Sweet 16"].append(winner)
                
                # Elite 8 (1 game)
                game_id = f"{region}_round4_game_0"
                winner = get_game_winner(input, game_id)
                if winner:
                    selections["Elite 8"].append(winner)
            
            # Final Four games
            for i in range(2):
                game_id = f"final_round5_game_{i}"
                winner = get_game_winner(input, game_id)
                if winner:
                    selections["Final Four"].append(winner)
            
            # Championship game
            championship_winner = get_game_winner(input, "final_round6_game_0")
            if championship_winner:
                selections["Championship"].append(championship_winner)
            
            # Create the user's bracket
            my_bracket = create_bracket_with_picks(actual_bracket.teams, selections)

            # Create a new pool with the actual tournament results
            pool = Pool(actual_bracket)
            
            # Add user's bracket to the pool, setting simulate=False to preserve user picks
            pool.add_entry("Your Bracket", my_bracket, simulate=False)

            # Add computer-generated entries with varying upset factors
            num_entries = int(input.num_entries())
            num_sims = int(input.num_sims())
            
            for i in range(num_entries):
                # Create a fresh bracket from teams for each entry
                entry_bracket = Bracket(actual_bracket.teams)
                
                # Vary the upset factor to create different predictions
                upset_factor = 0.1 + (i / num_entries) * 0.3
                for game in entry_bracket.games:
                    game.upset_factor = upset_factor
                    
                entry_name = f"Entry_{i+1}"
                # These entries will be simulated normally
                pool.add_entry(entry_name, entry_bracket)
            
            # Simulate pool
            results = pool.simulate_pool(num_sims=num_sims)
            
            # Format the results for display
            formatted_results = f"Simulation with {num_entries} entries and {num_sims} simulations:\n\n"
            formatted_results += "Rank | Entry | Win % | Avg Score\n"
            formatted_results += "-" * 40 + "\n"
            
            # Add top 10 results (or all if fewer than 10)
            for i, row in results.iterrows():
                if i >= 10:  # Show only top 10
                    break
                formatted_results += f"{i+1}. {row['name']} | {row['win_pct']:.1%} | {row['avg_score']:.1f}\n"
            
            your_rank = results[results['name'] == 'Your Bracket'].index[0] + 1 if 'Your Bracket' in results['name'].values else "N/A"
            formatted_results += f"\nYour bracket's rank: {your_rank} out of {len(results)}"
            formatted_results += f"\nNumber of upsets: {my_bracket.total_underdogs()}"
            formatted_results += f"\nLog likelihood: {my_bracket.calculate_log_probability()}"
            
            return formatted_results

        except Exception as e:
            logger.error(f"Error in simulation: {str(e)}", exc_info=True)
            return f"Error running simulation: {str(e)}"
