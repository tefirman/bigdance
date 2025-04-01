#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   espn_tc_scraper.py
@Time    :   2025/03/17
@Author  :   Taylor Firman
@Version :   0.3.2
@Contact :   tefirman@gmail.com
@Desc    :   Class-based implementation for extracting bracket data from ESPN Tournament Challenge
"""

import time
import json
import logging
from datetime import datetime
from pathlib import Path
import sys
import copy
from typing import Dict, List, Optional, Union, Any

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import WebDriverException
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import optparse

from bigdance import Standings
from bigdance.cbb_brackets import Bracket, Team, Pool

# Set up logging
logger = logging.getLogger(__name__)

# Constants
DEFAULT_CACHE_EXPIRY = 3600  # 1 hour for actual results
DEFAULT_ENTRY_CACHE_EXPIRY = 86400  # 1 day for entries
DEFAULT_DELAY = 0.1  # Seconds to delay between requests
MAX_RETRIES = 3  # Maximum number of retries for requests

# Selectors for finding pagination elements
PAGINATION_SELECTORS = [
    ".Pagination",
    "nav[aria-label='Pagination']",
    ".pagination",
    "div[role='navigation']",
    "ul.pagination",
]

# Selectors for finding next buttons in pagination
NEXT_BUTTON_SELECTORS = [
    ".Pagination__Button--next",
    "button[aria-label='Next']",
    "button.pagination-next",
    "a.pagination-next",
    "button:contains('Next')",
    "//button[contains(@class, 'next') or contains(text(), 'Next')]",
]

# Team name corrections to match ESPN to Warren Nolan
NAME_CORRECTIONS = {
    "UConn": "Connecticut",
    "UNC Wilmington": "UNCW",
    "St John's": "Saint John's",
    "Mount St Marys": "Mount Saint Mary's",
    "NC State": "North Carolina State",
    "UNC Greensboro": "UNCG",
    "S Dakota St": "South Dakota State",
    "SF Austin": "Stephen F. Austin",
    "Fair Dickinson": "Fairleigh Dickinson",
}


class BaseScraper:
    """Base class for scraping with common caching functionality"""

    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize base scraper with caching support"""
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cached_response(
        self, cache_key: str, max_age: int = DEFAULT_CACHE_EXPIRY
    ) -> Optional[str]:
        """
        Get cached response if available and not expired

        Args:
            cache_key: Key to identify the cached item
            max_age: Maximum age in seconds (default: 1 hour)

        Returns:
            Cached content if available and fresh, None otherwise
        """
        if not self.cache_dir:
            return None

        try:
            cache_file = self.cache_dir / f"{cache_key}.json"
            if not cache_file.exists():
                return None

            cache_data = json.loads(cache_file.read_text())

            # Check if cache is fresh
            time_since = (
                datetime.now() - datetime.fromisoformat(cache_data["timestamp"])
            ).total_seconds()
            if time_since > max_age:
                logger.debug(
                    f"Cache expired for key {cache_key} (age: {time_since:.1f}s)"
                )
                return None

            logger.debug(f"Using cached response for {cache_key}")
            return cache_data["content"]

        except Exception as e:
            logger.warning(f"Error reading cache for {cache_key}: {e}")
            return None

    def _cache_response(self, cache_key: str, url: str, content: str) -> None:
        """
        Cache a response for future use

        Args:
            cache_key: Key to identify the cached item
            url: Original URL that was requested
            content: Content to cache
        """
        if not self.cache_dir:
            return

        try:
            cache_data = {
                "timestamp": datetime.now().isoformat(),
                "url": url,
                "content": content,
            }

            cache_file = self.cache_dir / f"{cache_key}.json"
            cache_file.write_text(json.dumps(cache_data))

            logger.debug(f"Cached response to {cache_file}")

        except Exception as e:
            logger.warning(f"Error caching response for {cache_key}: {e}")

    def clear_cache(self, older_than_days: int = 7) -> int:
        """
        Clear cached responses older than specified days

        Args:
            older_than_days: Age threshold in days

        Returns:
            Number of cache files removed
        """
        if not self.cache_dir:
            return 0

        removed_count = 0
        now = datetime.now()

        for cache_file in self.cache_dir.glob("*.json"):
            try:
                cache_data = json.loads(cache_file.read_text())
                file_date = datetime.fromisoformat(cache_data["timestamp"])
                age_days = (now - file_date).total_seconds() / 86400

                if age_days > older_than_days:
                    cache_file.unlink()
                    removed_count += 1
                    logger.debug(
                        f"Removed cache file {cache_file} ({age_days:.1f} days old)"
                    )

            except Exception as e:
                logger.warning(f"Error processing cache file {cache_file}: {e}")

        return removed_count


class ESPNScraper(BaseScraper):
    """Class for scraping data from ESPN Tournament Challenge"""

    def __init__(self, women: bool = False, cache_dir: Optional[str] = None):
        """
        Initialize ESPN scraper

        Args:
            women: Whether to scrape women's tournament data
            cache_dir: Directory to store cached responses
        """
        super().__init__(cache_dir)
        self.women = women
        self.gender_suffix = "-women" if women else ""
        self.base_url = f"https://fantasy.espn.com/games/tournament-challenge-bracket{self.gender_suffix}-2025"

        # Initialize cached Chrome options and service to avoid repeated setup
        self._chrome_options = None
        self._chrome_service = None

    def _get_chrome_options(self) -> Options:
        """
        Get Chrome options for Selenium with proper settings

        Returns:
            Configured Chrome options
        """
        if self._chrome_options is None:
            chrome_options = Options()
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--window-size=1920,1080")

            # Add realistic user agent
            chrome_options.add_argument(
                "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/123.0.0.0 Safari/537.36"
            )

            self._chrome_options = chrome_options

        return self._chrome_options

    def _get_chrome_service(self) -> Service:
        """
        Get Chrome service for Selenium

        Returns:
            Configured Chrome service
        """
        if self._chrome_service is None:
            self._chrome_service = Service(ChromeDriverManager().install())

        return self._chrome_service

    def get_page(
        self,
        url: str,
        cache_key: Optional[str] = None,
        check_pagination: bool = False,
        max_pages: int = 10,
    ) -> Optional[Union[str, Dict[int, str]]]:
        """
        Retrieve a page using Selenium with support for pagination and caching

        Args:
            url: URL to retrieve
            cache_key: Key for caching (optional)
            check_pagination: Whether to check for pagination
            max_pages: Maximum number of pages to retrieve

        Returns:
            HTML content string, or dictionary of pages if paginated
        """
        # Check if we should use complete cache for paginated content
        if check_pagination and cache_key:
            complete_cache_key = f"{cache_key}_complete"
            cached_content = self._get_cached_response(
                complete_cache_key, DEFAULT_ENTRY_CACHE_EXPIRY
            )
            if cached_content:
                return json.loads(cached_content)

        # Check regular cache for non-paginated content
        if not check_pagination and cache_key:
            cache_expiry = (
                DEFAULT_CACHE_EXPIRY
                if "blank" in cache_key
                else DEFAULT_ENTRY_CACHE_EXPIRY
            )
            cached_content = self._get_cached_response(cache_key, cache_expiry)
            if cached_content:
                return cached_content

        # Try retrieval with retries
        for attempt in range(MAX_RETRIES):
            try:
                return self._retrieve_page(url, cache_key, check_pagination, max_pages)
            except WebDriverException as e:
                if attempt < MAX_RETRIES - 1:
                    wait_time = 2**attempt  # Exponential backoff
                    logger.warning(
                        f"Retrieval attempt {attempt+1} failed: {e}. Retrying in {wait_time}s..."
                    )
                    time.sleep(wait_time)
                else:
                    logger.error(f"All retrieval attempts failed for {url}: {e}")
                    return None
            except Exception as e:
                logger.error(f"Error retrieving {url}: {e}")
                return None

    def _retrieve_page(
        self, url: str, cache_key: Optional[str], check_pagination: bool, max_pages: int
    ) -> Union[str, Dict[int, str]]:
        """
        Internal method to retrieve a page with Selenium

        Args:
            url: URL to retrieve
            cache_key: Cache key if caching is enabled
            check_pagination: Whether to check for pagination
            max_pages: Maximum number of pages to retrieve

        Returns:
            Page content or dictionary of pages
        """
        driver = webdriver.Chrome(
            service=self._get_chrome_service(), options=self._get_chrome_options()
        )

        try:
            # Navigate to URL
            driver.get(url)
            time.sleep(5)  # Wait for page to load

            if not check_pagination:
                # Single page mode
                html_content = driver.page_source

                # Cache result if needed
                if cache_key and self.cache_dir:
                    self._cache_response(cache_key, url, html_content)

                return html_content
            else:
                # Pagination mode
                all_pages_content = {}
                page_num = 1

                # Get first page
                all_pages_content[page_num] = driver.page_source
                logger.debug(f"Retrieved page {page_num}")

                # Check for pagination elements
                pagination = self._find_pagination_element(driver)

                if pagination:
                    # Process pagination
                    while page_num < max_pages:
                        next_button = self._find_next_button(driver)

                        if not next_button or self._is_button_disabled(next_button):
                            logger.debug(
                                f"No more pages or next button disabled (at page {page_num})"
                            )
                            break

                        # Try to click the next button
                        self._click_next_button(driver, next_button)
                        time.sleep(3)  # Wait for page to load

                        # Store the new page content
                        page_num += 1
                        all_pages_content[page_num] = driver.page_source
                        logger.debug(f"Retrieved page {page_num}")
                else:
                    logger.debug("No pagination controls found")

                # Cache the complete result
                if cache_key and self.cache_dir:
                    complete_cache_key = f"{cache_key}_complete"
                    self._cache_response(
                        complete_cache_key, url, json.dumps(all_pages_content)
                    )

                return all_pages_content

        finally:
            # Ensure driver is closed
            driver.quit()

    def _find_pagination_element(self, driver) -> Optional[Any]:
        """Find pagination element using different selectors"""
        for selector in PAGINATION_SELECTORS:
            try:
                element = driver.find_element("css selector", selector)
                logger.debug(f"Pagination found with selector: {selector}")
                return element
            except Exception:
                continue
        return None

    def _find_next_button(self, driver) -> Optional[Any]:
        """Find next button in pagination"""
        for selector in NEXT_BUTTON_SELECTORS:
            try:
                if selector.startswith("//"):
                    element = driver.find_element("xpath", selector)
                else:
                    element = driver.find_element("css selector", selector)
                logger.debug(f"Next button found with selector: {selector}")
                return element
            except Exception:
                continue
        return None

    def _is_button_disabled(self, button) -> bool:
        """Check if a button is disabled"""
        return "disabled" in button.get_attribute("class") or button.get_attribute(
            "disabled"
        )

    def _click_next_button(self, driver, button) -> None:
        """Click a button with fallback strategies"""
        # Scroll into view
        driver.execute_script("arguments[0].scrollIntoView(true);", button)
        time.sleep(1)

        # Try JavaScript click first (more reliable)
        try:
            logger.debug("Clicking next button with JavaScript")
            driver.execute_script("arguments[0].click();", button)
        except Exception:
            # Fall back to regular click
            logger.debug("Falling back to regular click")
            button.click()

    def get_bracket(self, entry_id: str = "") -> Optional[str]:
        """
        Get an ESPN Tournament Challenge bracket page

        Args:
            entry_id: ESPN entry ID or empty string for blank/actual bracket

        Returns:
            HTML content of the bracket page
        """
        url = f"{self.base_url}/bracket?id={entry_id}"
        cache_key = f"bracket_{'women' if self.women else 'men'}_{entry_id if entry_id else 'blank'}"

        return self.get_page(url, cache_key, check_pagination=False)

    def get_pool(self, pool_id: str) -> Optional[Dict[int, str]]:
        """
        Get an ESPN Tournament Challenge pool with support for pagination

        Args:
            pool_id: ESPN pool ID

        Returns:
            Dictionary mapping page numbers to HTML content
        """
        url = f"{self.base_url}/group?id={pool_id}"
        cache_key = f"pool_{'women' if self.women else 'men'}_{pool_id}"

        return self.get_page(url, cache_key, check_pagination=True)


class ESPNBracket:
    """Class for handling ESPN bracket data extraction and analysis"""

    def __init__(self, women: bool = False, cache_dir: Optional[str] = None):
        """
        Initialize bracket handler

        Args:
            women: Whether to use women's tournament data
            cache_dir: Directory for caching
        """
        self.women = women
        self.ratings_source = None
        self.scraper = ESPNScraper(women, cache_dir)

        # Try to load ratings from Warren Nolan
        try:
            self.ratings_source = Standings(women=women, cache_dir=cache_dir)
            logger.info(
                f"Successfully loaded {len(self.ratings_source.elo)} teams from Warren Nolan"
            )
        except Exception as e:
            logger.warning(f"Could not load Standings: {e}")
            logger.warning("Will use approximate ratings based on seeds")

    def get_bracket(self, entry_id: str = "") -> Optional[str]:
        """
        Get bracket HTML content

        Args:
            entry_id: ESPN entry ID or empty string for blank/actual bracket

        Returns:
            HTML content of the bracket page
        """
        return self.scraper.get_bracket(entry_id)

    def extract_bracket(self, html_content: str) -> Optional[Bracket]:
        """
        Extract bracket data from HTML content

        Args:
            html_content: HTML content of the bracket page

        Returns:
            Bracket object or None if extraction fails
        """
        if not html_content:
            logger.error("No HTML content provided")
            return None

        try:
            # Parse the HTML
            soup = BeautifulSoup(html_content, "html.parser")

            # Find all parts of the bracket
            region_tags = soup.find_all(
                "div",
                attrs={"class": "EtBYj UkSPS ZSuWB viYac NgsOb GpQCA NqeUA Mxk xTell"},
            )
            pick_tags = soup.find_all(
                "span", attrs={"class": "BracketPropositionHeaderDesktop-pickText"}
            )
            team_tags = soup.find_all(
                "label", attrs={"class": "BracketOutcome-label truncate"}
            )
            team_id_tags = soup.find_all(
                "img", attrs={"class": "Image BracketOutcome-image printHide"}
            )
            seed_tags = soup.find_all("div", attrs={"class": "BracketOutcome-metadata"})

            # Extract actual bracket outcomes
            regions = [region.text for region in region_tags]
            names = [team.text for team in team_tags]
            ids = [
                team.attrs["src"].split("/")[-1].split(".")[0] for team in team_id_tags
            ]
            seeds = [
                int(seed.text) for seed in seed_tags if len(seed.attrs["class"]) == 1
            ]

            # Create mapping between team names and ESPN id
            name_mapping = {ids[ind]: names[ind] for ind in range(len(ids))}

            # Create Team objects
            teams = []
            for ind in range(64):  # First round teams
                teams.append(
                    Team(
                        names[ind],
                        seeds[ind],
                        regions[ind // 16],
                        self._get_team_rating(names[ind], seeds[ind]),
                        self._get_team_conference(names[ind]),
                    )
                )

            # Create bracket with these teams
            bracket = Bracket(teams)

            # Initialize results dictionary
            bracket.results = {}
            round_names = [
                "First Round",
                "Second Round",
                "Sweet 16",
                "Elite 8",
                "Final Four",
            ]

            # Determine if we're looking at the actual results or picks
            if len(pick_tags) == 0:  # Reality
                picks = names[64:]
            else:
                # Extract picks made by user
                try:
                    pick_ids = [
                        pick.find("img").attrs["src"].split("/")[-1].split(".")[0]
                        for pick in pick_tags
                    ]
                    picks = [name_mapping[id_val] for id_val in pick_ids]
                except Exception as e:
                    logger.warning(f"Incomplete bracket, skipping: {e}")
                    return None

            # Parse each round's picks
            for round_ind in range(5):
                bracket.results[round_names[round_ind]] = []
                for pick in picks[
                    64 - 2 ** (6 - round_ind) : 64 - 2 ** (5 - round_ind)
                ]:
                    pick = pick.replace("St.", "St")  # Fix Saint name mismatch
                    winner = next((t for t in teams if pick == t.name), None)
                    if winner:
                        bracket.results[round_names[round_ind]].append(winner)
                        if round_ind == 0:
                            for game in bracket.games:
                                if (
                                    game.team1.name == winner.name
                                    or game.team2.name == winner.name
                                ):
                                    game.winner = winner
                                    break

            # Extract champion pick
            champ_tag = soup.find(
                "span", attrs={"class": "PrintChampionshipPickBody-outcomeName"}
            )
            if champ_tag:
                champion = champ_tag.text.replace("St.", "St")
                winner = next((t for t in teams if champion.startswith(t.name)), None)
                if winner:
                    bracket.results["Championship"] = [winner]
                    bracket.results["Champion"] = winner

            # Calculate bracket statistics
            bracket.log_probability = bracket.calculate_log_probability()
            bracket.identify_underdogs()

            return bracket

        except Exception as e:
            logger.error(f"Error extracting bracket data: {str(e)}")
            return None

    def _get_team_rating(self, team_name: str, seed: int) -> float:
        """
        Get team rating from Standings or estimate based on seed

        Args:
            team_name: Team name
            seed: Team seed

        Returns:
            Team rating
        """
        # Apply name corrections
        if team_name in NAME_CORRECTIONS:
            team_name = NAME_CORRECTIONS[team_name]

        if self.ratings_source is not None:
            try:
                # Try exact match
                team_row = self.ratings_source.elo[
                    self.ratings_source.elo["Team"] == team_name
                ]
                if not team_row.empty:
                    return float(team_row.iloc[0]["ELO"])

                # Try fuzzy matching
                for team in self.ratings_source.elo["Team"]:
                    if (
                        team.lower() in team_name.lower()
                        or team_name.lower() in team.lower()
                    ):
                        team_row = self.ratings_source.elo[
                            self.ratings_source.elo["Team"] == team
                        ]
                        return float(team_row.iloc[0]["ELO"])
            except Exception as e:
                logger.warning(f"Error finding rating for {team_name}: {e}")

        # Estimate rating based on seed if not found
        logger.info(f"Can't find {team_name}, using random seed-based rating...")
        base_rating = 2000 - (seed * 50)
        random_adjustment = np.random.normal(0, 25)
        return base_rating + random_adjustment

    def _get_team_conference(self, team_name: str, default: str = "Unknown") -> str:
        """
        Get team conference from Standings

        Args:
            team_name: Team name
            default: Default conference name if not found

        Returns:
            Conference name
        """
        # Apply name corrections
        if team_name in NAME_CORRECTIONS:
            team_name = NAME_CORRECTIONS[team_name]

        if self.ratings_source is not None:
            try:
                # Try exact match
                team_row = self.ratings_source.elo[
                    self.ratings_source.elo["Team"] == team_name
                ]
                if not team_row.empty:
                    return team_row.iloc[0]["Conference"]

                # Try fuzzy matching
                for team in self.ratings_source.elo["Team"]:
                    if (
                        team.lower() in team_name.lower()
                        or team_name.lower() in team.lower()
                    ):
                        team_row = self.ratings_source.elo[
                            self.ratings_source.elo["Team"] == team
                        ]
                        return team_row.iloc[0]["Conference"]
            except Exception:
                pass

        return default


class ESPNPool:
    """Class for managing ESPN Tournament Challenge pools"""

    def __init__(self, women: bool = False, cache_dir: Optional[str] = None):
        """
        Initialize pool manager

        Args:
            women: Whether to use women's tournament data
            cache_dir: Directory for caching
        """
        self.women = women
        self.cache_dir = cache_dir
        self.scraper = ESPNScraper(women, cache_dir)
        self.bracket_handler = ESPNBracket(women, cache_dir)

    def get_pool(self, pool_id: str) -> Optional[Dict[int, str]]:
        """
        Get pool pages

        Args:
            pool_id: ESPN pool ID

        Returns:
            Dictionary of page HTML content
        """
        return self.scraper.get_pool(pool_id)

    def extract_entry_ids(
        self, html_content: Union[str, Dict[int, str]]
    ) -> Dict[str, str]:
        """
        Extract entry IDs from pool HTML

        Args:
            html_content: HTML content of the pool page or dictionary of pages

        Returns:
            Dictionary mapping entry names to entry IDs
        """
        entry_ids = {}

        # Handle both single page and paginated content
        if isinstance(html_content, dict):
            # Process each page
            for page_num, page_html in html_content.items():
                page_entries = self._extract_entries_from_html(page_html)
                entry_ids.update(page_entries)
                logger.debug(f"Found {len(page_entries)} entries on page {page_num}")
        else:
            # Single page as string
            entry_ids = self._extract_entries_from_html(html_content)

        return entry_ids

    def _extract_entries_from_html(self, html: str) -> Dict[str, str]:
        """
        Extract entries from a single HTML page

        Args:
            html: HTML content of a pool page

        Returns:
            Dictionary mapping entry names to entry IDs
        """
        soup = BeautifulSoup(html, "html.parser")
        entry_tags = soup.find_all(
            "td", attrs={"class": "BracketEntryTable-column--entryName Table__TD"}
        )

        return {
            entry.find("a").text: entry.find("a").attrs["href"].split("bracket?id=")[-1]
            for entry in entry_tags
            if entry.find("a")
        }

    def load_pool_entries(self, pool_id: str) -> Dict[str, Bracket]:
        """
        Load all entries from a pool

        Args:
            pool_id: ESPN pool ID

        Returns:
            Dictionary mapping entry names to Bracket objects
        """
        logger.info(f"Loading pool {pool_id}")

        # Get pool HTML
        pool_html = self.get_pool(pool_id)
        if not pool_html:
            logger.error(f"Failed to load pool {pool_id}")
            return {}

        # Extract entry IDs
        entry_ids = self.extract_entry_ids(pool_html)
        logger.info(f"Found {len(entry_ids)} entries in pool")

        # Load each entry
        entries = {}
        for entry_name, entry_id in entry_ids.items():
            logger.info(f"Loading entry: {entry_name}")
            entry_html = self.bracket_handler.get_bracket(entry_id)
            if entry_html:
                bracket = self.bracket_handler.extract_bracket(entry_html)
                if bracket:
                    entries[entry_name] = bracket
                else:
                    logger.warning(f"Failed to extract bracket for {entry_name}")
            else:
                logger.warning(f"Failed to load HTML for {entry_name}")

        logger.info(f"Successfully loaded {len(entries)} entries")
        return entries

    def create_simulation_pool(self, pool_id: str) -> Optional[Pool]:
        """
        Create a simulation Pool from an ESPN pool

        Args:
            pool_id: ESPN pool ID

        Returns:
            Pool object ready for simulation
        """
        # Load actual bracket from blank entry
        blank_html = self.bracket_handler.get_bracket()
        if not blank_html:
            logger.error("Failed to load actual bracket template")
            return None

        actual_bracket = self.bracket_handler.extract_bracket(blank_html)
        if not actual_bracket:
            logger.error("Failed to extract actual bracket data")
            return None

        # Add upset factor to actual tournament for realistic simulation
        for game in actual_bracket.games:
            game.upset_factor = 0.25  # More realistic tournament has upsets

        # Initialize pool with actual bracket
        pool_sim = Pool(actual_bracket)

        # Load and add entries
        entries = self.load_pool_entries(pool_id)
        for entry_name, bracket in entries.items():
            pool_sim.add_entry(
                entry_name, bracket, False
            )  # Don't re-simulate fixed picks

        return pool_sim


class GameImportanceAnalyzer:
    """Class for analyzing the importance of each game in a tournament"""

    def __init__(self, pool: Pool):
        """
        Initialize analyzer with a Pool

        Args:
            pool: Pool object containing entries and actual bracket
        """
        self.pool = pool

    def analyze_win_importance(
        self, current_round: Optional[str] = None, num_sims: int = 1000
    ) -> List[Dict]:
        """
        Analyze the importance of each game in the current round

        Args:
            current_round: Optional name of current round, will be inferred if None
            num_sims: Number of simulations to run

        Returns:
            List of dictionaries with game importance metrics
        """
        # Deep copy actual bracket to avoid modifying original
        actual_bracket = copy.deepcopy(self.pool.actual_results)

        # Infer current round if not provided
        if current_round is None:
            current_round = actual_bracket.infer_current_round()
            logger.debug(f"Inferred current round: {current_round}")

        # Validate current round
        valid_rounds = [
            "First Round",
            "Second Round",
            "Sweet 16",
            "Elite 8",
            "Final Four",
            "Championship",
        ]
        if current_round not in valid_rounds:
            raise ValueError(
                f"Invalid round name: {current_round}. Must be one of {valid_rounds}"
            )

        # Get teams in the current round
        teams_in_round = self._get_teams_in_round(actual_bracket, current_round)
        logger.debug(f"Analyzing {len(teams_in_round)//2} games in {current_round}")

        # Simulate baseline results
        logger.debug("Simulating baseline...")
        fixed_winners = copy.deepcopy(actual_bracket.results)
        baseline = self.pool.simulate_pool(
            num_sims=num_sims, fixed_winners=fixed_winners
        )

        # Analyze each game
        game_importance = []
        for game_ind in range(len(teams_in_round) // 2):
            team1 = teams_in_round[game_ind * 2]
            team2 = teams_in_round[game_ind * 2 + 1]
            if (
                team1 in actual_bracket.results[current_round]
                or team2 in actual_bracket.results[current_round]
            ):
                continue

            # Analyze importance of this matchup
            game_analysis = self._analyze_matchup(
                team1, team2, current_round, actual_bracket, baseline, num_sims
            )
            game_importance.append(game_analysis)

            logger.debug(
                f"Matchup impact: {game_analysis['max_impact']:.4f} (max), "
                f"{game_analysis['avg_impact']:.4f} (avg)"
            )

        return game_importance

    def _get_teams_in_round(self, bracket: Bracket, round_name: str) -> List[Team]:
        """
        Get teams participating in a specific round

        Args:
            bracket: Bracket object
            round_name: Round name

        Returns:
            List of teams in the round
        """
        if round_name == "First Round":
            teams = []
            for game in bracket.games:
                teams.extend([game.team1, game.team2])
            return teams
        else:
            prev_round = {
                "Second Round": "First Round",
                "Sweet 16": "Second Round",
                "Elite 8": "Sweet 16",
                "Final Four": "Elite 8",
                "Championship": "Final Four",
            }[round_name]
            return bracket.results[prev_round]

    def _analyze_matchup(
        self,
        team1: Team,
        team2: Team,
        round_name: str,
        actual_bracket,
        baseline,
        num_sims: int,
    ) -> Dict:
        """
        Analyze the importance of a specific matchup

        Args:
            team1: First team in matchup
            team2: Second team in matchup
            round_name: Round name
            actual_bracket: Actual tournament bracket
            baseline: Baseline simulation results
            num_sims: Number of simulations

        Returns:
            Dictionary with matchup importance metrics
        """
        # Simulate with Team 1 winning
        logger.debug(f"Simulating with {team1.name} winning...")
        fixed_winners_team1 = copy.deepcopy(actual_bracket.results)
        fixed_winners_team1[round_name] = fixed_winners_team1.get(round_name, []) + [
            team1
        ]
        results_team1 = self.pool.simulate_pool(
            num_sims=num_sims, fixed_winners=fixed_winners_team1
        )

        # Simulate with Team 2 winning
        logger.debug(f"Simulating with {team2.name} winning...")
        fixed_winners_team2 = copy.deepcopy(actual_bracket.results)
        fixed_winners_team2[round_name] = fixed_winners_team2.get(round_name, []) + [
            team2
        ]
        results_team2 = self.pool.simulate_pool(
            num_sims=num_sims, fixed_winners=fixed_winners_team2
        )

        # Merge results to calculate impact
        merged_results = pd.merge(
            results_team1[["name", "win_pct"]].rename(
                columns={"win_pct": "win_pct_team1"}
            ),
            results_team2[["name", "win_pct"]].rename(
                columns={"win_pct": "win_pct_team2"}
            ),
            on="name",
            how="outer",
        )
        merged_results = pd.merge(
            merged_results,
            baseline[["name", "win_pct"]].rename(
                columns={"win_pct": "win_pct_baseline"}
            ),
            on="name",
            how="outer",
        )

        # Calculate impact metrics
        merged_results["impact"] = abs(
            merged_results["win_pct_team1"] - merged_results["win_pct_team2"]
        )
        max_impact = merged_results["impact"].max()
        avg_impact = merged_results["impact"].mean()
        max_impact_entry = merged_results.loc[merged_results["impact"].idxmax()]

        # Create result dictionary
        return {
            "matchup": f"{team1.name} vs {team2.name}",
            "region": team1.region,
            "team1": {"name": team1.name, "seed": team1.seed},
            "team2": {"name": team2.name, "seed": team2.seed},
            "max_impact": max_impact,
            "avg_impact": avg_impact,
            "max_impact_entry": max_impact_entry["name"],
            "entry_win_pct_diff": {
                max_impact_entry["name"]: {
                    "team1_wins": max_impact_entry["win_pct_team1"],
                    "team2_wins": max_impact_entry["win_pct_team2"],
                    "baseline": max_impact_entry["win_pct_baseline"],
                    "impact": max_impact_entry["impact"],
                }
            },
            "all_entries_impact": merged_results.to_dict(orient="records"),
        }

    def print_importance_summary(
        self, game_importance: List[Dict], entry_name: Optional[str] = None
    ) -> None:
        """
        Print a human-readable summary of game importance analysis

        Args:
            game_importance: Game importance data from analyze_win_importance
            entry_name: Optional name of entry to focus on
        """
        if not game_importance:
            print("No games analyzed.")
            return

        # Check if the specified entry exists in the data
        if entry_name:
            entry_exists = False
            for details in game_importance:
                for entry_record in details["all_entries_impact"]:
                    if entry_record["name"] == entry_name:
                        entry_exists = True
                        break
                if entry_exists:
                    break

            if not entry_exists:
                print(f"Warning: Entry '{entry_name}' not found in the analysis data.")
                print("Defaulting to maximum impact entries for each game.")
                entry_name = None

        print("\n=== GAME IMPORTANCE SUMMARY ===\n")
        if entry_name:
            print(f"Focusing on entry: {entry_name}")

        for i, details in enumerate(game_importance):
            print(f"GAME #{i+1}: {details['matchup']} (Region: {details['region']})")
            print(
                f"  Max Impact: {details['max_impact']:.4f} | Avg Impact: {details['avg_impact']:.4f}"
            )

            if entry_name:
                # Find the specified entry's impact for this game
                entry_impact = None
                for entry_record in details["all_entries_impact"]:
                    if entry_record["name"] == entry_name:
                        entry_impact = {
                            "team1_wins": entry_record["win_pct_team1"],
                            "team2_wins": entry_record["win_pct_team2"],
                            "baseline": entry_record["win_pct_baseline"],
                            "impact": entry_record["impact"],
                        }
                        break

                if not entry_impact:
                    print(
                        f"  Note: Could not find impact data for {entry_name} on this game"
                    )
                    continue

                print(f"  Impact for {entry_name}: {entry_impact['impact']:.4f}")
            else:
                # Show the most affected entry
                print(f"  Most affected entry: {details['max_impact_entry']}")
                entry_impact = details["entry_win_pct_diff"][
                    details["max_impact_entry"]
                ]

            # Calculate percentages
            team1_pct = entry_impact["team1_wins"] * 100
            team2_pct = entry_impact["team2_wins"] * 100
            baseline_pct = entry_impact["baseline"] * 100

            # Determine which team benefits this entry
            if team1_pct > team2_pct:
                better_team = details["team1"]["name"]
                better_pct = team1_pct
                worse_team = details["team2"]["name"]
                worse_pct = team2_pct
            else:
                better_team = details["team2"]["name"]
                better_pct = team2_pct
                worse_team = details["team1"]["name"]
                worse_pct = team1_pct

            print(
                f"    Win chances: {better_pct:.1f}% if {better_team} wins vs {worse_pct:.1f}% if {worse_team} wins"
            )
            print(f"    Currently at: {baseline_pct:.1f}% baseline win probability")
            print(f"    Difference: {abs(team1_pct - team2_pct):.1f}%")
            print()

        print("=== END OF SUMMARY ===")


def main():
    """Command line interface for the module"""
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
        help='name of the round to simulate from ("First Round", '
        + '"Second Round", "Sweet 16", "Elite 8", "Final Four", "Championship")',
    )
    parser.add_option(
        "--importance",
        action="store_true",
        dest="importance",
        help="whether to assess the importance of each team winning in the current round",
    )
    parser.add_option(
        "--my_bracket",
        action="store",
        dest="my_bracket",
        help="name of the specific bracket to focus on in importance analysis",
    )
    parser.add_option(
        "--cache_dir",
        action="store",
        dest="cache_dir",
        default=".cache",
        help="location of html cache directory",
    )
    parser.add_option(
        "--verbose",
        action="store_true",
        dest="verbose",
        help="show all debugging messages",
    )
    options = parser.parse_args()[0]

    # Set up logging
    logging.basicConfig(
        level=logging.DEBUG if options.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Create pool manager
    pool_manager = ESPNPool(women=options.women, cache_dir=options.cache_dir)

    # Load actual bracket and pool entries
    pool_sim = pool_manager.create_simulation_pool(options.pool_id)
    if not pool_sim:
        logging.error("Failed to create simulation pool")
        return 1

    # Creating copy of pool simulator if importance calculation is requested
    if options.importance:
        importance_sim = copy.deepcopy(pool_sim)

    # If specified, erasing results from "as_of" round and beyond
    if options.as_of:
        round_names = [
            "First Round",
            "Second Round",
            "Sweet 16",
            "Elite 8",
            "Final Four",
            "Championship",
        ]
        if options.as_of not in round_names:
            logger.warning(
                "Don't recognize the round name provided for as_of, simulating from current state..."
            )
            if options.as_of in ["First", "Second", "Sweet", "Elite", "Final"]:
                logger.warning(
                    "Hot tip: make sure to put multi-word round names in quotes, "
                    'i.e. `--as_of "Second Round"` (thanks bash)'
                )
        else:
            for round_name in round_names[round_names.index(options.as_of) :]:
                pool_sim.actual_results.results[round_name] = []
            if "Champion" in pool_sim.actual_results.results:
                del pool_sim.actual_results.results["Champion"]

    # Simulating pool
    logger.info(f"Simulating pool with {len(pool_sim.entries)} entries")
    pool_results = pool_sim.simulate_pool(
        num_sims=1000, fixed_winners=pool_sim.actual_results.results
    )

    # Printing results
    top_entries = pool_results.sort_values("win_pct", ascending=False)
    top_entries.to_csv("PoolSimResults.csv", index=False)
    print()
    print(
        top_entries[["name", "avg_score", "std_score", "win_prob"]].to_string(
            index=False
        )
    )
    print()

    # Analyze game importance if requested
    if options.importance:
        analyzer = GameImportanceAnalyzer(importance_sim)
        importance = analyzer.analyze_win_importance(options.as_of, 1000)
        analyzer.print_importance_summary(importance, options.my_bracket)

    return 0


if __name__ == "__main__":
    sys.exit(main())
