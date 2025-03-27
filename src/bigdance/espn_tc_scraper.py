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
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
from bigdance import Standings
from bigdance.cbb_brackets import Bracket, Team, Pool
import numpy as np
import optparse

# Set up logging
logger = logging.getLogger(__name__)

def get_espn_page(url: str = "", cache_dir: Optional[str] = None, cache_key: Optional[str] = None, 
                  check_pagination: bool = False, max_pages: int = 10):
    """
    Use Selenium to access ESPN pages and extract content, with support for pagination.
    
    Args:
        url: URL to retrieve
        cache_dir: Directory to store cached responses
        cache_key: Key to use for caching
        check_pagination: Whether to check for and navigate through paginated content
        max_pages: Maximum number of pages to attempt to retrieve if paginated
        
    Returns:
        HTML content of the page or dictionary of pages if paginated
    """
    # Check cache first for complete result if pagination is expected
    if check_pagination and cache_dir and cache_key:
        complete_cache_key = f"{cache_key}_complete"
        cached_content = _get_cached_response(cache_dir, complete_cache_key)
        if cached_content:
            logger.info(f"Using cached complete response for {url} with key {complete_cache_key}")
            return json.loads(cached_content)
    
    # Check cache for single page if not paginated
    if not check_pagination and cache_dir and cache_key:
        cached_content = _get_cached_response(cache_dir, cache_key)
        if cached_content:
            logger.info(f"Using cached response for {url} with key {cache_key}")
            return cached_content
    
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
        # Navigate to the URL
        driver.get(url)
        
        # Wait for the page to fully load
        time.sleep(5)
        
        if not check_pagination:
            # Single page mode - return the page source directly
            html_content = driver.page_source
            
            # Cache the content if cache parameters are provided
            if cache_dir and cache_key:
                _cache_response(cache_dir, cache_key, url, html_content)
            
            return html_content
        else:
            # Pagination mode - collect content from all pages
            all_pages_content = {}
            page_num = 1
            
            # Get the first page
            all_pages_content[page_num] = driver.page_source
            logger.info(f"Retrieved page {page_num}")
            
            # Check if pagination exists by looking for the pagination container
            try:
                # Use a more specific selector that might be more reliable
                pagination_selectors = [
                    ".Pagination",
                    "nav[aria-label='Pagination']",
                    ".pagination",
                    "div[role='navigation']",
                    "ul.pagination"
                ]
                
                pagination = None
                for selector in pagination_selectors:
                    try:
                        pagination = driver.find_element("css selector", selector)
                        logger.info(f"Pagination found with selector: {selector}")
                        break
                    except:
                        continue
                
                if pagination:
                    logger.info("Pagination controls found - checking for multiple pages")
                    
                    # Look for pagination elements
                    while page_num < max_pages:
                        try:
                            # Try different selectors for the next button
                            next_button_selectors = [
                                ".Pagination__Button--next",
                                "button[aria-label='Next']",
                                "button.pagination-next",
                                "a.pagination-next",
                                "button:contains('Next')",
                                "//button[contains(@class, 'next') or contains(text(), 'Next')]"
                            ]
                            
                            next_button = None
                            for selector in next_button_selectors:
                                try:
                                    if selector.startswith("//"):
                                        next_button = driver.find_element("xpath", selector)
                                    else:
                                        next_button = driver.find_element("css selector", selector)
                                    logger.info(f"Next button found with selector: {selector}")
                                    break
                                except:
                                    continue
                            
                            if not next_button:
                                logger.info("No next button found")
                                break
                            
                            # Check if the button is enabled/clickable (not disabled)
                            if "disabled" in next_button.get_attribute("class") or next_button.get_attribute("disabled"):
                                logger.info(f"Next button is disabled, reached last page ({page_num})")
                                break
                            
                            # Scroll the button into view
                            driver.execute_script("arguments[0].scrollIntoView(true);", next_button)
                            # Add a small delay after scrolling
                            time.sleep(1)
                            
                            # Try to use JavaScript to click the button (more reliable)
                            try:
                                logger.info("Clicking next button with JavaScript")
                                driver.execute_script("arguments[0].click();", next_button)
                            except:
                                # Fall back to regular click if JS click fails
                                logger.info("Falling back to regular click")
                                next_button.click()
                            
                            # Wait for the new page to load
                            time.sleep(3)
                            
                            # Increment page number and store the page content
                            page_num += 1
                            all_pages_content[page_num] = driver.page_source
                            logger.info(f"Retrieved page {page_num}")
                            
                        except Exception as e:
                            logger.info(f"No more pages or error navigating: {e}")
                            
                            # Take a screenshot for debugging if there's an error
                            if cache_dir:
                                screenshot_path = Path(cache_dir) / f"pagination_error_{cache_key}_{page_num}.png"
                                driver.save_screenshot(str(screenshot_path))
                                logger.info(f"Saved error screenshot to {screenshot_path}")
                            break
                else:
                    logger.info("No pagination controls found")
                        
            except Exception as e:
                logger.info(f"Error detecting pagination: {e}")
                # If we can't find pagination, we already have the single page content
                if cache_dir:
                    screenshot_path = Path(cache_dir) / f"pagination_detection_error_{cache_key}.png"
                    driver.save_screenshot(str(screenshot_path))
                    logger.info(f"Saved error screenshot to {screenshot_path}")
            
            # Cache the complete result
            if cache_dir and cache_key:
                complete_cache_key = f"{cache_key}_complete"
                _cache_response(cache_dir, complete_cache_key, url, json.dumps(all_pages_content))
            
            return all_pages_content
        
    except Exception as e:
        logger.error(f"An error occurred retrieving {url}: {e}")
        if cache_dir:
            screenshot_path = Path(cache_dir) / f"general_error_{cache_key}.png"
            try:
                driver.save_screenshot(str(screenshot_path))
                logger.info(f"Saved error screenshot to {screenshot_path}")
            except:
                pass
        return None
    
    finally:
        # Close the browser
        driver.quit()

def _get_cached_response(cache_dir: str, cache_key: str) -> Optional[str]:
    """
    Get a cached response if it exists and is not too old.
    
    Args:
        cache_dir: Directory where cache files are stored
        cache_key: Key to identify the cached item
        
    Returns:
        Cached content if available and fresh, None otherwise
    """
    try:
        # Create cache directory if it doesn't exist
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        
        # Check if cache file exists
        cache_file = cache_path / f"{cache_key}.json"
        if not cache_file.exists():
            return None
            
        # Load cache data
        cache_data = json.loads(cache_file.read_text())
        
        # Check if cache is fresh (less than 1 hour old for actual results, 1 day old for entries)
        time_since = (datetime.now() - datetime.fromisoformat(cache_data["timestamp"])).total_seconds()
        if (cache_key.endswith("_blank") and time_since > 3600) or time_since > 86400:
            logger.info(f"Cache expired for key {cache_key}")
            return None
        else:
            return cache_data["content"]
    
    except Exception as e:
        logger.warning(f"Error reading cache for {cache_key}: {e}")
        return None

def _cache_response(cache_dir: str, cache_key: str, url: str, content: str):
    """
    Cache a response for future use.
    
    Args:
        cache_dir: Directory where cache files should be stored
        cache_key: Key to identify the cached item
        url: Original URL that was requested
        content: Content to cache
    """
    try:
        # Create cache directory if it doesn't exist
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        
        # Create cache data structure
        cache_data = {
            "timestamp": datetime.now().isoformat(),
            "url": url,
            "content": content
        }
        
        # Write cache file
        cache_file = cache_path / f"{cache_key}.json"
        cache_file.write_text(json.dumps(cache_data))
        
        logger.info(f"Cached response to {cache_file}")
    
    except Exception as e:
        logger.warning(f"Error caching response for {cache_key}: {e}")

def get_espn_bracket(entry_id: str = "", women: bool = False, cache_dir: Optional[str] = None):
    """
    Pull a specific contestant's bracket entry (or actual results if entry_id is empty).
    
    Args:
        entry_id: ESPN entry ID to retrieve
        women: Whether to retrieve women's tournament data
        cache_dir: Directory to store cached responses
        
    Returns:
        HTML content of the bracket page
    """
    gender = "-women" if women else ""
    url = f"https://fantasy.espn.com/games/tournament-challenge-bracket{gender}-2025/bracket?id={entry_id}"
    
    # Create cache key
    cache_key = f"bracket_{'women' if women else 'men'}_{entry_id if entry_id else 'blank'}"
    
    # Get the page content with caching, explicitly not checking pagination
    return get_espn_page(url, cache_dir, cache_key, check_pagination=False)

def extract_entry_bracket(html_content, ratings_source=None, women: bool = False):
    """
    Extract the pick data embedded in the HTML.
    This contains the complete bracket information.
    
    Args:
        html_content: HTML content of the bracket page
        ratings_source: Optional Standings object for team ratings
        women: Whether to retrieve women's tournament data
        
    Returns:
        Bracket object representing the entry
    """
    if ratings_source is None:
        try:
            ratings_source = Standings(women=women)
            logger.info(f"Successfully loaded {len(ratings_source.elo)} teams from Warren Nolan")
        except Exception as e:
            logger.warning(f"Could not load Standings: {e}")
            logger.warning("Will use approximate ratings based on seeds")
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
    regions = [region.text for region in region_tags]
    names = [team.text for team in team_tags]
    ids = [team.attrs["src"].split("/")[-1].split(".")[0] for team in team_id_tags]
    seeds = [int(seed.text) for seed in seed_tags if len(seed.attrs["class"]) == 1]

    # Create mapping between team names and ESPN id
    name_mapping = {ids[ind]: names[ind] for ind in range(len(ids))}

    # Creating list of team objects
    teams = []
    for ind in range(64): # Focusing on first round here
        teams.append(Team(names[ind], 
                          seeds[ind], 
                          regions[ind//16], 
                          get_team_rating(ratings_source, names[ind], seeds[ind]), 
                          get_team_conference(ratings_source, names[ind], "Unknown")))

    # Create an empty bracket with these teams
    bracket = Bracket(teams)
    
    # Initialize results dictionary and round names list
    bracket.results = {}
    round_names = ["First Round","Second Round","Sweet 16","Elite 8","Final Four"]

    if len(pick_tags) == 0: # Reality
        picks = names[64:]
    else:
        # Extract picks made by user
        try:
            pick_ids = [pick.find("img").attrs["src"].split("/")[-1].split(".")[0] for pick in pick_tags]
            picks = [name_mapping[id_val] for id_val in pick_ids]
        except:
            logger.warning("Incomplete bracket, skipping...")
            return None

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
    if champ_tag:
        champion = champ_tag.text.replace("St.","St")
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
                        "Mount St Marys":"Mount Saint Mary's",
                        "NC State":"North Carolina State",
                        "UNC Greensboro":"UNCG",
                        "S Dakota St":"South Dakota State",
                        "SF Austin":"Stephen F. Austin",
                        "Fair Dickinson":"Fairleigh Dickinson"}
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
            logger.warning(f"Error finding rating for {team_name}: {e}")
    
    # If we can't find the team or there's no ratings source, estimate based on seed
    # Higher seeds get higher ratings, with some randomness to make it interesting
    logger.info(f"Can't find {team_name}, using random seed-based rating...")
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

def get_espn_pool(pool_id: str, women: bool = False, cache_dir: Optional[str] = None):
    """
    Pull list of entries for a specific league, supporting pagination for larger pools.
    
    Args:
        pool_id: ESPN pool ID to retrieve
        women: Whether to retrieve women's tournament data
        cache_dir: Directory to store cached responses
        
    Returns:
        Dictionary containing HTML content of all pages in the pool
    """
    gender = "-women" if women else ""
    url = f"https://fantasy.espn.com/games/tournament-challenge-bracket{gender}-2025/group?id={pool_id}"
    
    # Create cache key
    cache_key = f"pool_{'women' if women else 'men'}_{pool_id}"
    
    # Get the page content with caching, enabling pagination check
    return get_espn_page(url, cache_dir, cache_key, check_pagination=True)

def extract_entry_ids(html_content):
    """
    Extract entry IDs from pool HTML.
    
    Args:
        html_content: HTML content of the pool page,
                     or dictionary of HTML content from multiple pages
        
    Returns:
        Dictionary mapping entry names to entry IDs
    """
    entry_ids = {}
    
    # Handle both single page and paginated content
    if isinstance(html_content, dict):
        # Process each page
        for page_num, page_html in html_content.items():
            page_entries = _extract_entries_from_html(page_html)
            entry_ids.update(page_entries)
            logger.info(f"Found {len(page_entries)} entries on page {page_num}")
    else:
        # Single page as string
        entry_ids = _extract_entries_from_html(html_content)
    
    return entry_ids

def _extract_entries_from_html(html):
    """Helper function to extract entries from a single HTML page"""
    # Soupify the raw html
    soup = BeautifulSoup(html, 'html.parser')
    
    # Find all entries
    entry_tags = soup.find_all('td', attrs={"class":"BracketEntryTable-column--entryName Table__TD"})

    # Extract the entry ID's from the HTML
    return {entry.find("a").text: entry.find("a").attrs["href"].split("bracket?id=")[-1] 
            for entry in entry_tags if entry.find("a")}

def main():
    parser = optparse.OptionParser()
    parser.add_option(
        "--women",
        action="store_true",
        dest="women",
        help="whether to pull stats for the NCAAW instead of NCAAM",
    )
    parser.add_option(
        "--pool_id",
        action="store",
        dest="pool_id",
        help="ESPN group ID of the bracket pool of interest",
    )
    parser.add_option(
        "--as_of",
        action="store",
        dest="as_of",
        help='name of the round to simulate from ("First Round", ' + \
            '"Second Round", "Sweet 16", "Elite 8", "Final Four", "Championship")',
    )
    parser.add_option(
        "--cache_dir",
        action="store",
        dest="cache_dir",
        default="espn_cache",
        help="location of html cache directory",
    )
    options = parser.parse_args()[0]

    # Get HTML content with caching
    html_content = get_espn_pool(options.pool_id, options.women, options.cache_dir)

    # Extract bracket ID's for each entry
    entry_ids = extract_entry_ids(html_content) # Do we want to pull the pool name too???

    # Pull standings initially to reduce number of pulls throughout
    ratings_source = Standings(women=options.women)

    # Pulling blank entry to get seedings
    bracket_html = get_espn_bracket(women=options.women, cache_dir=options.cache_dir)
    actual_bracket = extract_entry_bracket(bracket_html, ratings_source, options.women)

    # If specified, erasing results from "as_of" round and beyond
    round_names = ["First Round","Second Round","Sweet 16","Elite 8","Final Four","Championship"]
    if options.as_of and options.as_of not in round_names:
        logger.warning("Don't recognize the round name provided for as_of, simulating from current state...")
        if options.as_of in ["First", "Second", "Sweet", "Elite", "Final"]:
            logger.warning('Hot tip: make sure to put multi-word round names in quotes, i.e. `--as_of "Second Round"` (thanks bash)')
        options.as_of = None
    if options.as_of:
        for round_name in round_names[round_names.index(options.as_of):]:
            actual_bracket.results[round_name] = []
        if "Champion" in actual_bracket.results:
            del actual_bracket.results["Champion"]

    # IMPORTANT: Add moderate upset factor to actual tournament results
    for game in actual_bracket.games:
        game.upset_factor = 0.25  # More realistic tournament has upsets
    
    # Initializing simulation pool
    pool_sim = Pool(actual_bracket)

    # Pulling each of the brackets in the league
    for entry_name in entry_ids:
        logger.info(f"Processing entry: {entry_name}")
        entry_html = get_espn_bracket(entry_ids[entry_name], options.women, options.cache_dir)
        entry_bracket = extract_entry_bracket(entry_html, ratings_source, options.women)
        if entry_bracket:
            pool_sim.add_entry(entry_name, entry_bracket, False)

    # Simulating pool
    logger.info(f"Simulating pool with {len(pool_sim.entries)} entries")
    pool_results = pool_sim.simulate_pool(num_sims=1000, fixed_winners=actual_bracket.results)

    # Printing results
    top_entries = pool_results.sort_values("win_pct", ascending=False)
    top_entries.to_csv("PoolSimResults.csv",index=False)
    print(top_entries[["name", "avg_score", "std_score", "win_pct"]])

if __name__ == "__main__":
    # Set up basic logging configuration
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    main()
