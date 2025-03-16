#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   wn_cbb_elo.py
@Time    :   2024/02/22 10:58:06
@Author  :   Taylor Firman
@Version :   0.2.0
@Contact :   tefirman@gmail.com
@Desc    :   Elo ratings parser for the Warren Nolan college sports website (no affiliation)
"""

import json
import logging
import optparse
import time
from abc import ABC
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from io import StringIO
from pathlib import Path
from typing import Optional
from urllib.parse import quote

import pandas as pd
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class BaseScraper(ABC):
    """Base class for Warren Nolan scrapers with common functionality."""

    def __init__(self, cache_dir: Optional[str] = None):
        self.session = self._create_session()
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _create_session(self) -> requests.Session:
        """Create session with retry logic and connection pooling."""
        session = requests.Session()
        retries = Retry(
            total=5, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504]
        )
        adapter = HTTPAdapter(max_retries=retries, pool_connections=10, pool_maxsize=10)
        session.mount("https://", adapter)
        return session

    def _get_cached_response(self, url: str, cache_key: str) -> Optional[str]:
        """Get cached response if available."""
        if not self.cache_dir:
            return None

        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            cache_data = json.loads(cache_file.read_text())
            if (
                datetime.now() - datetime.fromisoformat(cache_data["timestamp"])
            ).total_seconds() < 3600:
                return cache_data["content"]
        return None

    def _cache_response(self, url: str, cache_key: str, content: str):
        """Cache response content."""
        if not self.cache_dir:
            return

        cache_data = {
            "timestamp": datetime.now().isoformat(),
            "url": url,
            "content": content,
        }
        cache_file = self.cache_dir / f"{cache_key}.json"
        cache_file.write_text(json.dumps(cache_data))

    def courteous_get(
        self, url: str, cache_key: Optional[str] = None, delay: float = 0.1
    ) -> str:
        """Enhanced GET request with caching and error handling."""
        if cache_key:
            cached = self._get_cached_response(url, cache_key)
            if cached:
                return cached

        try:
            response = self.session.get(url)
            response.raise_for_status()
            content = response.text

            if cache_key:
                self._cache_response(url, cache_key, content)

            time.sleep(delay)
            return content
        except requests.exceptions.RequestException as e:
            logging.error(f"Error fetching {url}: {str(e)}")
            raise


class Schedule(BaseScraper):
    """Enhanced Schedule class with parallel processing and validation."""

    def __init__(
        self,
        start: str,
        stop: str,
        gameset: str = "All Games",
        women: bool = False,
        elos: bool = True,
        max_workers: int = 4,
        cache_dir: Optional[str] = None,
    ):
        """Initialize Schedule with parallel processing support."""
        super().__init__(cache_dir)
        self.start = pd.to_datetime(start)
        self.stop = pd.to_datetime(stop)
        self._validate_dates()

        self.gameset = gameset
        self.gender = "w" if women else ""

        self.max_workers = max_workers
        self.pull_games(elos)

    def _validate_dates(self):
        """Validate date range."""
        if self.start > self.stop:
            raise ValueError("Start date must be before stop date")
        if (self.stop - self.start).days > 365:
            raise ValueError("Date range cannot exceed one year")

    def pull_games(self, elos: bool = True):
        """Pull games in parallel."""
        dates = [
            self.start + timedelta(days=x)
            for x in range((self.stop - self.start).days + 1)
            if (self.start + timedelta(days=x)).month in [11, 12, 1, 2, 3]
        ]

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            self.games_per_day = list(
                executor.map(
                    lambda d: Matchups(d, self.gameset, self.gender == "w", elos), dates
                )
            )

        self.games_per_day = [g for g in self.games_per_day if g.matchups.shape[0] > 0]


class Standings(BaseScraper):
    """Standings class that gathers team ratings and records."""

    def __init__(
        self,
        season: int = (datetime.now() + timedelta(days=90)).year,
        conference: str = None,
        women: bool = False,
        cache_dir: Optional[str] = None,
    ):
        """Initialize Standings with caching support."""
        super().__init__(cache_dir)  # Initialize the parent BaseScraper

        self.season = season
        self.gender = "w" if women else ""
        self.base_url = (
            f"https://www.warrennolan.com/basketball{self.gender}/{self.season}"
        )
        self.pull_elo_html()
        self.parse_elo_table()
        self.pull_conferences_html()
        self.parse_conference_list()
        self.add_conferences()
        self.add_ranks()
        if conference in self.conferences:
            self.elo = self.elo.loc[self.elo.Conference == conference].reset_index(
                drop=True
            )
        elif conference == "Top 25":
            self.elo = self.elo.loc[~self.elo["AP Rank"].isnull()].reset_index(
                drop=True
            )
        elif conference is not None and conference != "All Games":
            print("Invalid value for conference, including all teams...")

    def pull_elo_html(self):
        """Pull elo HTML using cached request."""
        cache_key = f"elo_{self.season}"
        self.elo_response = self.courteous_get(
            f"{self.base_url}/elo", cache_key=cache_key
        )
        self.elo_soup = BeautifulSoup(self.elo_response, "html.parser")

    def parse_elo_table(self):
        """
        Parses the elo table from the raw html and converts it to a pandas dataframe.
        """
        self.elo_tables = self.elo_soup.find_all(
            "table", attrs={"class": "normal-grid alternating-rows stats-table"}
        )
        self.elo = pd.concat(
            pd.read_html(StringIO(str(self.elo_tables))), ignore_index=True
        )
        self.elo = self.elo.rename(columns={"Rank": "ELO Rank"})

    def pull_conferences_html(self):
        """
        Pulls the raw html text for the list of conferences involved in the season of interest
        and converts it to a BeautifulSoup object for easier parsing.
        """
        cache_key = f"confs_{self.season}"
        self.confs_response = self.courteous_get(
            f"{self.base_url}/conferences", cache_key=cache_key
        )
        self.confs_soup = BeautifulSoup(self.confs_response, "html.parser")

    def parse_conference_list(self):
        """
        Parses the list of conferences from the raw html.
        """
        self.confs_div = self.confs_soup.find_all(
            "div", attrs={"class": "name-subcontainer"}
        )
        self.conferences = [team.text for team in self.confs_div]

    def pull_conference_teams(self, conference: str) -> list:
        """
        Pulls down a list of teams in a particular conference.

        Args:
            conference (str): name of the conference of interest.

        Returns:
            list: list of teams in the conference of interest for the season of interest.
        """
        conference = conference.replace(" ", "-")
        cache_key = f"{conference}_{self.season}"
        conf_response = self.courteous_get(
            f"{self.base_url}/conference/{conference}", cache_key=cache_key
        )
        conf_soup = BeautifulSoup(conf_response, "html.parser")
        conf_teams = conf_soup.find_all("div", attrs={"class": "name-subcontainer"})
        conf_teams = [team.text for team in conf_teams]
        return conf_teams

    def add_conferences(self):
        """
        Adds the respective conferences for each team in the elo dataframe.
        """
        for conf in self.conferences:
            teams = self.pull_conference_teams(conf)
            self.elo.loc[self.elo.Team.isin(teams), "Conference"] = conf

    def pull_ranks_html(self, poll: str = "AP"):
        """
        Pulls the raw html text for the poll rankings of the season of interest
        and converts it to a BeautifulSoup object for easier parsing.

        Args:
            poll (str, optional): which poll to use ("AP" or "Coaches"), defaults to "AP".
        """
        cache_key = f"rank{poll}_{self.season}"
        self.rank_response = self.courteous_get(
            f"{self.base_url}/polls-expanded/{poll.lower()}", cache_key=cache_key
        )
        self.rank_soup = BeautifulSoup(self.rank_response, "html.parser")

    def parse_ranks_table(self):
        """
        Parses the rankings table from the raw html and converts it to a pandas dataframe.
        """
        self.rank_table = self.rank_soup.find_all(
            "table", attrs={"class": "normal-grid alternating-rows stats-table"}
        )
        self.ranks = pd.read_html(StringIO(str(self.rank_table)))[0]

        # Modify this line to handle different DataFrame structures
        if isinstance(self.ranks.columns[0], tuple):
            # Handle multi-index columns
            self.ranks.columns = self.ranks.columns.get_level_values(-1)

        # Select first three columns and clean up team names
        self.ranks = self.ranks.iloc[:, :3]
        self.ranks = self.ranks.loc[~self.ranks.Team.isnull()].reset_index(drop=True)
        self.ranks.Team = self.ranks.Team.str.split(r"  \(").str[0]

    def add_ranks(self):
        """
        Adds the respective poll ranks for each team in the elo dataframe.
        """
        for poll in ["Coaches", "AP"]:
            self.pull_ranks_html(poll)
            self.parse_ranks_table()
            self.elo = pd.merge(
                left=self.elo,
                right=self.ranks.rename(columns={"Rank": f"{poll} Rank"}),
                how="left",
                on=["Team", "Record"],
            )


class Matchups(BaseScraper):
    """Matchups class that gathers the matchups projections and results."""

    def __init__(
        self,
        date: str = datetime.now(),
        gameset: str = "All Games",
        women: bool = False,
        elos: bool = True,
        cache_dir: Optional[str] = None,
    ):
        """Initialize Matchups with caching support."""
        super().__init__(cache_dir)  # Initialize the parent BaseScraper

        self.date = pd.to_datetime(date)
        self.gameset = gameset
        self.gender = "w" if women else ""
        self.season = (self.date + timedelta(days=90)).year
        self.base_url = (
            f"https://www.warrennolan.com/basketball{self.gender}/{self.season}"
        )
        self.pull_matchups_html()
        self.parse_matchups_table()
        if elos:
            self.add_elos()
        self.matchups = self.matchups[
            [col for col in self.matchups.columns if "1" in col]
            + [col for col in self.matchups.columns if "2" in col]
            + [
                col
                for col in self.matchups.columns
                if "1" not in col and "2" not in col
            ]
        ]

    def pull_matchups_html(self):
        """Pull matchups HTML using cached request."""
        gamestr = self.gameset.replace(" ", "%20")

        # Format date parts separately and encode properly
        weekday = self.date.strftime("%A")
        month = self.date.strftime("%B")
        day = self.date.strftime("%-d")  # %-d removes leading zeros
        date1 = quote(f"{weekday}, {month} {day}")
        date2 = self.date.strftime("%Y-%m-%d")
        url = f"{self.base_url}/predict-winners?type1={date1}&type2={gamestr}&date={date2}"
        logging.debug(f"Requesting URL: {url}")

        cache_key = (
            f"matchups_{self.date.strftime('%Y%m%d')}_{self.gameset.replace(' ','')}"
        )
        self.response = self.courteous_get(url, cache_key=cache_key)
        logging.debug(f"Response length: {len(self.response)}")

        self.soup = BeautifulSoup(self.response, "html.parser")

    def parse_matchups_table(self):
        """
        Parses the matchups table from the raw html and converts it to a pandas dataframe.
        """
        # Find all game boxes with class 'pbox'
        game_boxes = self.soup.find_all("div", class_="pbox")

        all_games = []
        for box in game_boxes:
            game_data = {}

            # Get team rows
            team1_row = box.find("tr", class_="pbox__info-team1-row")
            team2_row = box.find("tr", class_="pbox__info-team2-row")

            if team1_row and team2_row:
                # Extract team names
                game_data["team1"] = (
                    team1_row.find("div", class_="name-subcontainer")
                    .find("a")
                    .text.strip()
                )
                game_data["team2"] = (
                    team2_row.find("div", class_="name-subcontainer")
                    .find("a")
                    .text.strip()
                )

                # Extract scores (actual scores from live/final games)
                game_data["score1"] = int(
                    team1_row.find("td", class_="score center-line").text.strip() or 0
                )
                game_data["score2"] = int(
                    team2_row.find("td", class_="score center-line").text.strip() or 0
                )

                # Extract projected scores
                proj_score1 = team1_row.find_all("td", class_="score")[1].text.strip()
                proj_score2 = team2_row.find_all("td", class_="score")[1].text.strip()
                game_data["proj_score1"] = int(proj_score1)
                game_data["proj_score2"] = int(proj_score2)

                # Extract win probabilities
                prob1 = (
                    team1_row.find_all("td", class_="value")[1]
                    .text.strip()
                    .replace("%", "")
                )
                prob2 = (
                    team2_row.find_all("td", class_="value")[1]
                    .text.strip()
                    .replace("%", "")
                )
                game_data["rp_prob1"] = float(prob1) / 100.0
                game_data["rp_prob2"] = float(prob2) / 100.0

                # Extract over/under and spread
                over_under = team1_row.find_all("td", class_="value")[0].text.strip()
                spread = team2_row.find_all("td", class_="value")[0].text.strip()
                game_data["over_under"] = int(over_under)
                game_data["spread"] = int(spread.replace("+", ""))

                all_games.append(game_data)

        # Convert to DataFrame
        if all_games:
            self.matchups = pd.DataFrame(all_games)
        else:
            self.matchups = pd.DataFrame(
                columns=[
                    "team1",
                    "score1",
                    "proj_score1",
                    "rp_prob1",
                    "team2",
                    "score2",
                    "proj_score2",
                    "rp_prob2",
                    "over_under",
                    "spread",
                ]
            )

        # Log parsing results
        logging.debug(f"Parsed {len(all_games)} games from matchups table")

    def add_elos(
        self,
        s: Standings = None,
        base_elo: float = 1500.0,
        scale: float = 1.0,
        homefield: float = 100.0,
        verbose: bool = True,
    ):
        """
        Merges in the relevant elo ratings to the Matchups dataframe and
        calculates the corresponding win probabilities based on those elo ratings.

        Args:
            s (Standings, optional): Standings object containing the relevant elo ratings, defaults to latest rankings for the season in question.
            base_elo (float, optional): default elo rating for missing teams, defaults to 1500.0.
            scale (float, optional): relative scale of elo points compared to baseline, defaults to 1.0.
            homefield (float, optional): elo point bonus for homefield advantage, defaults to 100.0.
        """
        if self.date < datetime.now() - timedelta(days=1) and verbose:
            print(
                "Heads up: elo projections for past matchups should be taken with a grain of salt..."
            )
        if s is None:
            s = Standings(self.season, women=self.gender == "w")
        self.matchups = pd.merge(
            left=self.matchups,
            right=s.elo[["Team", "ELO"]].rename(
                columns={"Team": "team1", "ELO": "elo1"}
            ),
            how="left",
            on="team1",
        )
        self.matchups = pd.merge(
            left=self.matchups,
            right=s.elo[["Team", "ELO"]].rename(
                columns={"Team": "team2", "ELO": "elo2"}
            ),
            how="left",
            on="team2",
        )
        missing1 = self.matchups.elo1.isnull()
        missing2 = self.matchups.elo2.isnull()
        if missing1.any() or missing2.any():
            missing = (
                self.matchups.loc[missing1, "team1"].tolist()
                + self.matchups.loc[missing2, "team2"].tolist()
            )
            print(
                "Can't find the following team names in the current elo rankings: "
                + ", ".join(missing)
            )
            print(
                f"Assuming the average rating of {base_elo} for now, but might want to fix that..."
            )
            self.matchups.elo1 = self.matchups.elo1.fillna(base_elo)
            self.matchups.elo2 = self.matchups.elo2.fillna(base_elo)
        self.matchups["elo_prob1"] = self.matchups.apply(
            lambda x: elo_prob(x["elo1"], x["elo2"], scale, homefield), axis=1
        )
        self.matchups["elo_prob2"] = 1 - self.matchups["elo_prob1"]


def elo_prob(
    elo1: float, elo2: float, scale: float = 1.0, homefield: float = 100.0
) -> float:
    """
    Calculates the respective win probabilities of a matchup given the two corresponding elo ratings.

    Args:
        elo1 (float): elo rating of the away team.
        elo2 (float): elo rating of the home team.
        scale (float, optional): relative scale of elo points compared to baseline, defaults to 1.0.
        homefield (float, optional): elo point bonus for homefield advantage, defaults to 100.0.

    Returns:
        float: win probability for the away team (home probability = 1 - away probability)
    """
    # https://en.wikipedia.org/wiki/Elo_rating_system
    prob1 = 1 / (1 + 10 ** (scale * (elo2 + homefield - elo1) / 400))
    return prob1


def main():
    # Initializing command line inputs
    parser = optparse.OptionParser()
    parser.add_option(
        "--date",
        action="store",
        type="str",
        dest="date",
        default=datetime.now(),
        help="date of interest",
    )
    parser.add_option(
        "--conference",
        action="store",
        type="str",
        dest="conference",
        default="All Games",
        help="conference of interest",
    )
    parser.add_option(
        "--women",
        action="store_true",
        dest="women",
        help="whether to pull stats for the NCAAW instead of NCAAM",
    )
    parser.add_option(
        "--output",
        action="store",
        type="str",
        dest="output",
        help="where to save the Standings and Matchups data in the form of csv's",
    )
    parser.add_option(
        "--debug", action="store_true", dest="debug", help="prints debugging statements"
    )
    parser.add_option(
        "--cache_dir",
        action="store",
        type="str",
        dest="cache_dir",
        help="location of html cache",
    )
    options = parser.parse_args()[0]
    if options.debug:
        logging.basicConfig(level=logging.DEBUG)
    options.date = pd.to_datetime(options.date)
    season = (options.date + timedelta(days=90)).year
    if season < 2021:
        print(
            "Sadly, elo ratings were not stored on Warren Nolan before 2021. Try again with a more recent date."
        )
    else:
        # Pulling requested Standings and printing results
        s = Standings(season, options.conference, options.women, options.cache_dir)
        print(s.elo.to_string(index=False, na_rep=""))

        # Pulling requested Matchups for today and printing results
        m = Matchups(
            options.date, options.conference, options.women, False, options.cache_dir
        )
        if m.matchups.shape[0] > 0:
            print(m.matchups.to_string(index=False, na_rep=""))
        else:
            print(
                "No games were played on {}.".format(options.date.strftime("%B %d, %Y"))
            )

        # Saving as csv's if requested
        if options.output is not None:
            s.to_csv(
                "{}Standings_{}.csv".format(
                    options.output, datetime.now().strftime("%m%d%y")
                )
            )
            m.to_csv(
                "{}Matchups_{}.csv".format(
                    options.output, datetime.now().strftime("%m%d%y")
                )
            )


if __name__ == "__main__":
    main()
