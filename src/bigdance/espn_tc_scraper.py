#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   espn_tc_scraper.py
@Time    :   2025/03/17
@Author  :   Taylor Firman
@Version :   0.2.0
@Contact :   tefirman@gmail.com
@Desc    :   Extracting bracket matchups from ESPN for March Madness bracket pool simulations
"""

import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import sys
from bs4 import BeautifulSoup
import json
from bigdance import Standings
from bigdance.cbb_brackets import Bracket, Team
import numpy as np

def get_espn_bracket():
    """
    Use Selenium to access the ESPN men's basketball bracket page
    and extract the bracket information.
    """
    # Set up Chrome options
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Run in headless mode (no browser UI)
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")
    
    # Add realistic user agent
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36")
    
    # Initialize the Chrome driver
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
    
    try:
        # Navigate to the ESPN bracket page
        # print("Accessing ESPN bracket page...")
        driver.get("https://www.espn.com/mens-college-basketball/bracket")
        
        # Wait for the page to fully load
        time.sleep(5)
        
        # Save the page source for debugging if needed
        html_content = driver.page_source
        
        # Take a screenshot for debugging if needed
        driver.save_screenshot("espn_bracket.png")
        # print("Screenshot saved as espn_bracket.png")
        
        return html_content        
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
    finally:
        # Close the browser
        driver.quit()

def extract_json_data(html_content):
    """
    Extract the JSON data embedded in the HTML.
    This contains the complete bracket information.
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Find all script tags
    script_tags = soup.find_all('script')
    
    bracket_json = None
    
    # Look for script tags containing bracket data
    for script in script_tags:
        script_text = script.string
        if script_text and '"bracket":' in script_text:
            json_text = script_text.split("window['__espnfitt__']=")[-1]
            decoder = json.JSONDecoder()
            json_data = decoder.raw_decode(json_text)[0]
            bracket_json = json_data["page"]["content"]["bracket"]
    
    return bracket_json

def extract_first_round_from_json(bracket_json):
    """
    Extract teams and matchups from the bracket JSON data
    Returns teams list and games list suitable for CSV output
    """
    # Get regions information if available
    regions = {}
    if 'regions' in bracket_json:
        for region in bracket_json['regions']:
            region_id = region.get('id')
            region_name = region.get('labelPrimary')
            if region_id and region_name:
                regions[region_id] = region_name
        # print(f"Found regions in JSON: {regions}")
    
    first_four = [game for game in bracket_json["matchups"] if game['roundId'] == 0]
    first_round = [game for game in bracket_json["matchups"] if game['roundId'] == 1]

    for ind in range(len(first_round)):
        first_round[ind]["regionId"] = (first_round[ind]["bracketLocation"] - 1)//8 + 1
        first_round[ind]["label"] = regions[first_round[ind]["regionId"]]
        if first_round[ind]["competitorTwo"]["name"] == "TBD":
            play_in = [game for game in first_four if game["label"] == first_round[ind]["label"] \
                and game["competitorOne"]["seed"] == first_round[ind]["competitorTwo"]["seed"]][0]
            first_round[ind]["competitorTwo"] = play_in["competitorOne"] # Just using competitorOne for now...
    return first_round

def convert_espn_to_bigdance(first_round, ratings_source=None):
    """
    Convert ESPN bracket data to bigdance Team objects for simulation.
    
    Parameters:
        espn_json_file (str): Path to the ESPN first round matchups JSON file
        ratings_source (Standings, optional): A bigdance Standings object with team ratings
                                             If None, will create a new one
                                             
    Returns:
        list: List of Team objects ready for bigdance Bracket creation
    """
    # Get current team ratings if not provided
    if ratings_source is None:
        try:
            ratings_source = Standings()
            print(f"Successfully loaded {len(ratings_source.elo)} teams from Warren Nolan")
        except Exception as e:
            print(f"Warning: Could not load Standings: {e}")
            print("Will use approximate ratings based on seeds")
            ratings_source = None
    
    # Create mapping of region IDs to names
    region_names = {
        1: "South", 
        2: "West", 
        3: "East", 
        4: "Midwest"
    }
    
    # Default conference if not available
    default_conference = "Unknown"
    
    # List to store all teams
    teams = []
    
    # Process each game to extract teams
    for game in first_round:
        region_id = game.get("regionId")
        region_name = region_names.get(region_id, "Unknown")
        
        # Process first team
        team1_data = game.get("competitorOne", {})
        team1_name = team1_data.get("name")
        team1_seed = int(team1_data.get("seed", "16"))  # Default to 16 if not found
        
        # Process second team
        team2_data = game.get("competitorTwo", {})
        team2_name = team2_data.get("name")
        team2_seed = int(team2_data.get("seed", "16"))  # Default to 16 if not found
        
        # Lookup team ratings in Standings
        team1_rating = get_team_rating(ratings_source, team1_name, team1_seed)
        team2_rating = get_team_rating(ratings_source, team2_name, team2_seed)
        
        # Get conference info if available
        team1_conference = get_team_conference(ratings_source, team1_name, default_conference)
        team2_conference = get_team_conference(ratings_source, team2_name, default_conference)
        
        # Create Team objects
        team1 = Team(
            name=team1_name,
            seed=team1_seed,
            region=region_name,
            rating=team1_rating,
            conference=team1_conference
        )
        
        team2 = Team(
            name=team2_name,
            seed=team2_seed,
            region=region_name,
            rating=team2_rating,
            conference=team2_conference
        )
        
        # Add teams to list, checking for duplicates
        if not any(t.name == team1.name for t in teams):
            teams.append(team1)
        if not any(t.name == team2.name for t in teams):
            teams.append(team2)
    
    # Verify we have exactly 64 teams
    if len(teams) != 64:
        print(f"Warning: Expected 64 teams, but got {len(teams)}. Bracket may be incomplete.")
    
    return Bracket(teams)

def get_team_rating(ratings_source, team_name, seed):
    """Get a team's rating from the Standings object or estimate based on seed."""
    if ratings_source is not None:
        try:
            # Try to find exact match
            team_row = ratings_source.elo[ratings_source.elo['Team'] == team_name]
            if not team_row.empty:
                return float(team_row.iloc[0]['ELO'])
            
            # Try fuzzy matching
            for team in ratings_source.elo['Team']:
                if team.lower() in team_name.lower() or team_name.lower() in team.lower():
                    team_row = ratings_source.elo[ratings_source.elo['Team'] == team]
                    return float(team_row.iloc[0]['ELO'])
        except Exception as e:
            print(f"Warning: Error finding rating for {team_name}: {e}")
    
    # If we can't find the team or there's no ratings source, estimate based on seed
    # Higher seeds get higher ratings, with some randomness to make it interesting
    base_rating = 2000 - (seed * 50)
    random_adjustment = np.random.normal(0, 25)  # Small random component
    return base_rating + random_adjustment

def get_team_conference(ratings_source, team_name, default="Unknown"):
    """Get a team's conference from the Standings object."""
    if ratings_source is not None:
        try:
            # Try to find exact match
            team_row = ratings_source.elo[ratings_source.elo['Team'] == team_name]
            if not team_row.empty:
                return team_row.iloc[0]['Conference']
            
            # Try fuzzy matching
            for team in ratings_source.elo['Team']:
                if team.lower() in team_name.lower() or team_name.lower() in team.lower():
                    team_row = ratings_source.elo[ratings_source.elo['Team'] == team]
                    return team_row.iloc[0]['Conference']
        except Exception:
            pass
    
    return default

def main():
    # Get HTML content
    html_content = get_espn_bracket()
    
    if html_content is not None:
        # Save HTML for debugging
        with open("espn_bracket.html", "w", encoding="utf-8") as f:
            f.write(html_content)
        print("HTML saved to espn_bracket.html")
    else:
        print("Failed to extract bracket data")
        sys.exit(1)
    
    # Extract JSON data from HTML
    bracket_json = extract_json_data(html_content)
    
    if bracket_json:
        # Save JSON for debugging
        with open("bracket_data.json", "w", encoding="utf-8") as f:
            json.dump(bracket_json, f, indent=2)
        print("JSON data saved to bracket_data.json")
        
        # Extract teams and games from JSON
        first_round = extract_first_round_from_json(bracket_json)
        with open("first_round_matchups.json", "w", encoding="utf-8") as f:
            json.dump(first_round, f, indent=2)
        print("JSON data saved to first_round_matchups.json")
    else:
        print("Failed to extract JSON data, bailing...")
        sys.exit(1)
    
    actual_bracket = convert_espn_to_bigdance(first_round)

    # Apply a moderate upset factor to the actual tournament result
    # This ensures the actual tournament has a realistic amount of upsets
    for game in actual_bracket.games:
        game.upset_factor = 0.25  # Moderate upset factor for actual tournament

    results = actual_bracket.simulate_tournament()
    print(f"Simulated Final Four: {results["Elite 8"]}") # Technically the "results" of Elite 8
    print(f"Simulated Champion: {results["Champion"]}")

if __name__ == "__main__":
    main()
