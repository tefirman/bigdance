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

from shiny import render, reactive, ui
from shiny.types import SilentException
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from data import teams

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
    """Get initial matchups for first round games in a region"""
    matchups = []
    region_teams = teams[region]
    
    # Process teams in pairs to create matchups
    for i in range(0, len(region_teams), 2):
        team1 = region_teams[i] if i < len(region_teams) else None
        team2 = region_teams[i+1] if i+1 < len(region_teams) else None
        matchups.append((team1, team2))
    
    return matchups

def get_second_round_matchups(input, region: str) -> List[Tuple[Optional[Dict], Optional[Dict]]]:
    """Get second round matchups based on first round selections"""
    matchups = []
    region_teams = teams[region]
    
    # Process first round games in pairs
    for i in range(0, 8, 2):  # 8 first round games, process 2 at a time
        game1_id = f"{region.lower()}_round1_game_{i}"
        game2_id = f"{region.lower()}_round1_game_{i+1}"
        
        # Get winners from first round if selected
        winner1 = get_game_winner(input, game1_id)
        winner2 = get_game_winner(input, game2_id)
        
        # Find team details for winners
        winner1_details = next((team for team in region_teams if team["Team"] == winner1), None) if winner1 else None
        winner2_details = next((team for team in region_teams if team["Team"] == winner2), None) if winner2 else None
        
        matchups.append((winner1_details, winner2_details))
        
    return matchups

def get_round_matchups(input, region: str, round_num: int) -> List[Tuple[Optional[Dict], Optional[Dict]]]:
    """Get matchups for any round based on previous round selections"""
    matchups = []
    region_teams = teams[region]  # Get teams for specific region
    
    # Calculate number of games in current round
    current_round_games = 2 ** (4 - round_num)  # 2 for Sweet 16, 1 for Elite 8
    
    for i in range(0, current_round_games):
        # For each game in current round, we need to look at two games from previous round
        prev_round_game1 = i * 2
        prev_round_game2 = i * 2 + 1
        
        prev_game1_id = f"{region.lower()}_round{round_num-1}_game_{prev_round_game1}"
        prev_game2_id = f"{region.lower()}_round{round_num-1}_game_{prev_round_game2}"
        
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
    """Create UI for any round's games"""
    games = []
    for i, (team1, team2) in enumerate(matchups):
        game_id = f"{region.lower()}_round{round_num}_game_{i}"
        choices = {}
        
        if team1:
            choices[team1["Team"]] = f"({team1['Seed']}) {team1['Team']}"
        if team2:
            choices[team2["Team"]] = f"({team2['Seed']}) {team2['Team']}"
        
        game = ui.div(
            {"class": "game-container"},
            ui.input_radio_buttons(
                game_id,
                f"Game {i + 1}",
                choices
            ) if len(choices) == 2 else ui.p("Waiting for previous round selections...")
        )
        games.append(game)
    
    return ui.div(
        {"class": "bracket-region"},
        *games
    )

def get_final_four_matchups(input):
    """Get Final Four matchups based on Elite Eight selections"""
    winners = []
    for region in ["east", "west", "south", "midwest"]:
        winner = get_game_winner(input, f"{region}_round4_game_0")  # Elite Eight winner
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
    
    # Track when conference filter changes
    @reactive.Effect
    def _():
        logger.info(f"Conference changed to: {input.conference()}")

    # Debug info output
    @output
    @render.text
    def debug_info():
        return f"Current Conference: {input.conference()}\nLast Updated: {datetime.now()}"

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
                "Round 1": {},
                "Round 2": {},
                "Sweet 16": {},
                "Elite Eight": {},
                "Final Four": {},
                "Championship": {}
            }
            
            # Check each region's games through Elite Eight
            for region in ["east", "west", "south", "midwest"]:
                logger.debug(f"Checking {region} region")
                
                # First round (8 games)
                for i in range(8):
                    game_id = f"{region}_round1_game_{i}"
                    winner = get_game_winner(input, game_id)
                    if winner:
                        selections["Round 1"][game_id] = winner
                
                # Second round (4 games)
                for i in range(4):
                    game_id = f"{region}_round2_game_{i}"
                    winner = get_game_winner(input, game_id)
                    if winner:
                        selections["Round 2"][game_id] = winner
                
                # Sweet 16 (2 games)
                for i in range(2):
                    game_id = f"{region}_round3_game_{i}"
                    winner = get_game_winner(input, game_id)
                    if winner:
                        selections["Sweet 16"][game_id] = winner
                
                # Elite Eight (1 game)
                game_id = f"{region}_round4_game_0"
                winner = get_game_winner(input, game_id)
                if winner:
                    selections["Elite Eight"][game_id] = winner
            
            # Final Four games
            for i in range(2):
                game_id = f"final_round5_game_{i}"
                winner = get_game_winner(input, game_id)
                if winner:
                    selections["Final Four"][game_id] = winner
            
            # Championship game
            championship_winner = get_game_winner(input, "final_round6_game_0")
            if championship_winner:
                selections["Championship"]["final_round6_game_0"] = championship_winner
            
            # Format results message
            result_msg = "Your Picks:\n\n"
            
            # Add picks for each round
            for round_name, games in selections.items():
                if games:  # Only show rounds that have picks
                    result_msg += f"{round_name}:\n"
                    for game_id, winner in sorted(games.items()):
                        if "final_round" in game_id:
                            result_msg += f"Game {int(game_id.split('_')[-1]) + 1}: {winner}\n"
                        else:
                            region = game_id.split("_")[0].title()
                            game_num = int(game_id.split("_")[-1]) + 1
                            result_msg += f"{region} Game {game_num}: {winner}\n"
                    result_msg += "\n"
            
            # Add championship winner announcement if there is one
            if "Championship" in selections and selections["Championship"]:
                winner = list(selections["Championship"].values())[0]
                result_msg += f"\nTournament Champion: {winner}! 🏆\n"
            
            return result_msg
            
        except Exception as e:
            logger.error(f"Error in simulation: {str(e)}", exc_info=True)
            return f"Error running simulation: {str(e)}"
