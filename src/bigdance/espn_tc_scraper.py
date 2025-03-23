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
import optparse

def get_espn_bracket(women: bool = False, entry_id: str = ""):
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
        # Pulling a specific contestant's entry (or actual results if entry_id is empty)
        gender = "-women" if women else ""
        url = f"https://fantasy.espn.com/games/tournament-challenge-bracket{gender}-2025/bracket?id={entry_id}"
        driver.get(url)
        
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

def extract_entry_bracket(html_content, ratings_source=None, women: bool = False):
    """
    Extract the pick data embedded in the HTML.
    This contains the complete bracket information.
    """
    if ratings_source is None:
        try:
            ratings_source = Standings(women=women)
            print(f"Successfully loaded {len(ratings_source.elo)} teams from Warren Nolan")
        except Exception as e:
            print(f"Warning: Could not load Standings: {e}")
            print("Will use approximate ratings based on seeds")
            ratings_source = None

    # Soupify the raw html
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Find all picks, results, and seeding
    region_tags = soup.find_all('div',attrs={"class":"EtBYj UkSPS ZSuWB viYac NgsOb GpQCA NqeUA Mxk xTell"})
    pick_tags = soup.find_all('span', attrs={"class":"BracketPropositionHeaderDesktop-pickText"})
    team_tags = soup.find_all("label",attrs={"class":"BracketOutcome-label truncate"})
    team_id_tags = soup.find_all("img",attrs={"class":"Image BracketOutcome-image printHide"})
    seed_tags = soup.find_all("div", attrs={"class":"BracketOutcome-metadata"})

    # Extract actual bracket outcomes
    regions = [region.text.title() for region in region_tags]
    names = [team.text for team in team_tags][:64] # Focusing on first round for now
    ids = [int(team.attrs["src"].split("/")[-1].split(".")[0]) for team in team_id_tags][:64] # Focusing on first round for now
    seeds = [int(seed.text) for seed in seed_tags if len(seed.attrs["class"]) == 1][:64] # Focusing on first round for now

    # Create mapping between team names and ESPN id
    name_mapping = {ids[ind]: names[ind] for ind in range(len(ids))}

    # Creating list of team objects
    teams = []
    for ind in range(len(names)):
        teams.append(Team(names[ind], 
                          seeds[ind], 
                          regions[ind//16], 
                          get_team_rating(ratings_source, names[ind], seeds[ind]), 
                          get_team_conference(ratings_source, names[ind], "Unknown")))

    # Create an empty bracket with these teams
    bracket = Bracket(teams)
    
    # Extract picks made by user
    pick_ids = [int(pick.find("img").attrs["src"].split("/")[-1].split(".")[0]) for pick in pick_tags]
    picks = [name_mapping[id_val] for id_val in pick_ids]

    # Initialize results dictionary and round names list
    bracket.results = {}
    round_names = ["First Round","Second Round","Sweet 16","Elite 8","Final Four"]

    # Parse each round's picks
    for round_ind in range(5):
        bracket.results[round_names[round_ind]] = []
        for pick in picks[64 - 2**(6 - round_ind):64 - 2**(5 - round_ind)]: # Number of winners each round: 32, 16, 8, 4, 2
            pick = pick.replace("St.","St") # Not sure why ESPN has the St vs St. mismatch with "Saint" teams...
            winner = next((t for t in teams if pick == t.name), None) # Identifying winner's Team object
            if winner:
                bracket.results[round_names[round_ind]].append(winner) # Appending to bracket results
                if round_ind == 0:
                    for game in bracket.games: # Updating the first round games with a winner, used during log probability calculation
                        if game.team1.name == winner.name or game.team2.name == winner.name:
                            game.winner = winner
                            break
    
    # Extract champion pick
    champ_tag = soup.find("span", attrs={"class":"PrintChampionshipPickBody-outcomeName"})
    champion = champ_tag.text.replace("St.","St")
    if champion:
        winner = next((t for t in teams if champion.startswith(t.name)), None)
        if winner:
            bracket.results["Championship"] = [winner]
            bracket.results["Champion"] = winner
    
    # Calculate log probability and underdogs for reference
    bracket.log_probability = bracket.calculate_log_probability()
    bracket.identify_underdogs()
    # underdog_counts = bracket.count_underdogs_by_round()
    # underdog_counts["Total"] = bracket.total_underdogs()

    return bracket

def get_team_rating(ratings_source, team_name, seed):
    """Get a team's rating from the Standings object or estimate based on seed."""
    name_corrections = {"UConn":"Connecticut", 
                        "UNC Wilmington":"UNCW", 
                        "St John's":"Saint John's",
                        "Mount St Marys":"Mount Saint Mary's"}
    if team_name in name_corrections:
        team_name = name_corrections[team_name]
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
    print(f"Can't find {team_name}, using random seed-based rating...")
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
    parser = optparse.OptionParser()
    parser.add_option(
        "--women",
        action="store_true",
        dest="women",
        help="whether to pull stats for the NCAAW instead of NCAAM",
    )
    parser.add_option(
        "--entry_id",
        action="store",
        dest="entry_id",
        default="",
        help="ESPN entry ID of the bracket of interest",
    )
    options = parser.parse_args()[0]

    # Get HTML content
    html_content = get_espn_bracket(options.women, options.entry_id)
    
    if html_content is not None:
        # Save HTML for debugging
        with open("espn_bracket.html", "w", encoding="utf-8") as f:
            f.write(html_content)
        print("HTML saved to espn_bracket.html")
    else:
        print("Failed to extract bracket data")
        sys.exit(1)
    
    if options.entry_id == "": # Pulling most recent results and simulating
        # Apply a moderate upset factor to the actual tournament result
        # This ensures the actual tournament has a realistic amount of upsets
        for game in actual_bracket.games:
            game.upset_factor = 0.25  # Moderate upset factor for actual tournament
        results = actual_bracket.simulate_tournament()
        print(f"Simulated Final Four: {results["Elite 8"]}") # Technically the "results" of Elite 8
        print(f"Simulated Champion: {results["Champion"]}")
    else:
        # Extract bracket from raw HTML and pull elo ratings from Warren Nolan
        actual_bracket = extract_entry_bracket(html_content, women=options.women)
        print(f"Selected Final Four: {actual_bracket.results["Elite 8"]}") # Technically the "results" of Elite 8
        print(f"Selected Champion: {actual_bracket.results["Champion"]}")

if __name__ == "__main__":
    main()
