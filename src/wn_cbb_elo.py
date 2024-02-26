#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   wn_cbb_elo.py
@Time    :   2024/02/22 10:58:06
@Author  :   Taylor Firman
@Version :   1.0
@Contact :   tefirman@gmail.com
@Desc    :   Elo ratings parser for the Warren Nolan college sports website (no affiliation)
'''

from io import StringIO
import pandas as pd
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import optparse
import time

class Standings:
    """
    Standings class that gathers the win-loss records and elo ratings
    for all college basketball teams for the season and gender provided.

    Attributes:  
        season: integer specifying the season of interest  
        gender: string specifiying the gender of interest  
        base_url: string specifying the exact Warren Nolan endpoint to pull from  
        elo: dataframe containing detailed information about each team  
        conferences: list of strings representing all conferences involved in the season of interest  
    """
    def __init__(self, season: int = (datetime.now() + timedelta(days=90)).year, conference: str = None, women: bool = False):
        """
        Initializes a Standings object using the parameters provided and class functions defined below.

        Args:  
            season (int, optional): season of interest, defaults to the current season.  
            women (bool, optional): whether to look at the women's game, defaults to False.  
        """
        self.season = season
        self.gender = "w" if women else ""
        self.base_url = f"https://www.warrennolan.com/basketball{self.gender}/{self.season}"
        self.pull_elo_html()
        self.parse_elo_table()
        self.pull_conferences_html()
        self.parse_conference_list()
        self.add_conferences()
        self.add_ranks()
        if conference in self.conferences:
            self.elo = self.elo.loc[self.elo.Conference == conference].reset_index(drop=True)
        elif conference == "Top 25":
            self.elo = self.elo.loc[~self.elo['AP Rank'].isnull()].reset_index(drop=True)
        elif conference is not None and conference != "All Games":
            print("Invalid value for conference, including all teams...")

    def pull_elo_html(self):
        """
        Pulls the raw html text of the elo rankings for the season of interest
        and converts it to a BeautifulSoup object for easier parsing.
        """
        self.elo_response = courteous_get(f"{self.base_url}/elo").text
        self.elo_soup = BeautifulSoup(self.elo_response, "html.parser")

    def parse_elo_table(self):
        """
        Parses the elo table from the raw html and converts it to a pandas dataframe.
        """
        self.elo_tables = self.elo_soup.find_all("table", attrs={"class": "normal-grid alternating-rows stats-table"})
        self.elo = pd.concat(pd.read_html(StringIO(str(self.elo_tables))), ignore_index=True)
        self.elo = self.elo.rename(columns={"Rank":"ELO Rank"})
    
    def pull_conferences_html(self):
        """
        Pulls the raw html text for the list of conferences involved in the season of interest
        and converts it to a BeautifulSoup object for easier parsing.
        """
        self.confs_response = courteous_get(f"{self.base_url}/conferences").text
        self.confs_soup = BeautifulSoup(self.confs_response, "html.parser")
    
    def parse_conference_list(self):
        """
        Parses the list of conferences from the raw html.
        """
        self.confs_div = self.confs_soup.find_all("div", attrs={"class": "name-subcontainer"})
        self.conferences = [team.text for team in self.confs_div]
    
    def pull_conference_teams(self, conference: str) -> list:
        """
        Pulls down a list of teams in a particular conference.

        Args:
            conference (str): name of the conference of interest.

        Returns:
            list: list of teams in the conference of interest for the season of interest.
        """
        conference = conference.replace(" ","-")
        conf_response = courteous_get(f"{self.base_url}/conference/{conference}").text
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
            self.elo.loc[self.elo.Team.isin(teams),"Conference"] = conf
    
    def pull_ranks_html(self, poll: str = "AP"):
        """
        Pulls the raw html text for the poll rankings of the season of interest
        and converts it to a BeautifulSoup object for easier parsing.

        Args:
            poll (str, optional): which poll to use ("AP" or "Coaches"), defaults to "AP".
        """
        self.rank_response = courteous_get(f"{self.base_url}/polls-expanded/{poll.lower()}").text
        self.rank_soup = BeautifulSoup(self.rank_response, "html.parser")
    
    def parse_ranks_table(self):
        """
        Parses the rankings table from the raw html and converts it to a pandas dataframe.
        """
        self.rank_table = self.rank_soup.find_all("table", attrs={"class": "normal-grid alternating-rows stats-table"})
        self.ranks = pd.read_html(StringIO(str(self.rank_table)))[0]
        self.ranks = self.ranks[self.ranks.columns[0][0]].iloc[:,:3]
        self.ranks = self.ranks.loc[~self.ranks.Team.isnull()].reset_index(drop=True)
        self.ranks.Team = self.ranks.Team.str.split("  \(").str[0]

    def add_ranks(self):
        """
        Adds the respective poll ranks for each team in the elo dataframe.
        """
        for poll in ["Coaches","AP"]:
            self.pull_ranks_html(poll)
            self.parse_ranks_table()
            self.elo = pd.merge(left=self.elo,right=self.ranks.rename(columns={"Rank":f"{poll} Rank"}),how="left",on=["Team","Record"])

class Matchups:
    """
    Matchups class that gathers the matchups projections and results for
    the season, gender, and gameset provided.

    Attributes:  
        date: integer specifying the season of interest  
        gameset: string specifiying the conference/rankings of interest  
        gender: string specifiying the gender of interest  
        base_url: string specifying the exact Warren Nolan endpoint to pull from  
        matchups: dataframe containing detailed information about each matchup on the date in question  
    """
    def __init__(self, date: str = datetime.now(), gameset: str = "All Games", women: bool = False, elos: bool = True):
        """
        Initializes a Matchups object using the parameters provided and class functions defined below.

        Args:
            date (str, optional): date of interest, defaults to today.  
            gameset (str, optional): conference/rankings of interest 
            ("All Games", "Top 25", or the name of a conference), defaults to "All Games".  
            women (bool, optional): whether to look at the women's game, defaults to False.  
            elos (bool, optional): whether to pull elo ratings/projections, defaults to True.  
        """
        self.date = pd.to_datetime(date)
        self.gameset = gameset
        self.gender = "w" if women else ""
        self.season = (self.date + timedelta(days=90)).year
        self.base_url = f"https://www.warrennolan.com/basketball{self.gender}/{self.season}"
        self.pull_matchups_html()
        self.parse_matchups_table()
        self.fix_matchups_dtype()
        if elos:
            self.add_elos()
        self.matchups = self.matchups[[col for col in self.matchups.columns if "1" in col] + \
        [col for col in self.matchups.columns if "2" in col] + \
        [col for col in self.matchups.columns if "1" not in col and "2" not in col]]

    def pull_matchups_html(self):
        """
        Pulls the raw html text of the matchup for the date of interest
        and converts it to a BeautifulSoup object for easier parsing.
        """
        gamestr = self.gameset.replace(" ","%20")
        date1 = self.date.strftime('%A, %B %d').replace(' ','%20')
        date2 = self.date.strftime('%Y-%m-%d')
        url = f"{self.base_url}/predict-winners?type1={date1}&type2={gamestr}&date={date2}"
        response = courteous_get(url).text
        self.soup = BeautifulSoup(response, "html.parser")
    
    def parse_matchups_table(self):
        """
        Parses the matchups table from the raw html and converts it to a pandas dataframe.
        """
        self.tables = self.soup.find_all("table", attrs={"class": "normal-grid white-rows no-border"})
        if len(self.tables) > 0:
            matchups = pd.concat(pd.read_html(StringIO(str(self.tables))), ignore_index=True)
            away = matchups.iloc[2::4,:5].reset_index(drop=True).rename(columns={0:"team1",\
            1:"score1",2:"proj_score1",3:"over_under",4:"rp_prob1"})
            home = matchups.iloc[3::4,:5].reset_index(drop=True).rename(columns={0:"team2",\
            1:"score2",2:"proj_score2",3:"spread",4:"rp_prob2"})
            self.matchups = pd.concat([away,home],axis=1)
        else:
            self.matchups = pd.DataFrame(columns=["team1","score1","proj_score1","rp_prob1",\
            "team2","score2","proj_score2","rp_prob2","over_under","spread"])
    
    def fix_matchups_dtype(self):
        """
        Performs basic datatype conversion for each of the columns in the matchups dataframe.
        """
        self.matchups.over_under = self.matchups.over_under.astype(int)
        self.matchups.spread = self.matchups.spread.astype(int)
        for i in ["1","2"]:
            self.matchups["team" + i] = self.matchups["team" + i].str.split(" \(").str[:-1].apply(" (".join)
            ranked = self.matchups["team" + i].str.startswith("#")
            self.matchups.loc[ranked,"team" + i] = self.matchups.loc[ranked,"team" + i].str.split(" ").str[1:].apply(" ".join)
            self.matchups["score" + i] = self.matchups["score" + i].astype(int)
            self.matchups["proj_score" + i] = self.matchups["proj_score" + i].astype(int)
            self.matchups["rp_prob" + i] = self.matchups["rp_prob" + i].str.replace("%","").astype(float)/100.0
    
    def add_elos(self, s: Standings = None, base_elo: float = 1500.0, scale: float = 1.0, homefield: float = 100.0, verbose: bool = True):
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
            print("Heads up: elo projections for past matchups should be taken with a grain of salt...")
        if s is None:
            s = Standings(self.season, women=self.gender == "w")
        self.matchups = pd.merge(left=self.matchups, right=s.elo[["Team","ELO"]].rename(columns={"Team":"team1","ELO":"elo1"}), how="left", on="team1")
        self.matchups = pd.merge(left=self.matchups, right=s.elo[["Team","ELO"]].rename(columns={"Team":"team2","ELO":"elo2"}), how="left", on="team2")
        missing1 = self.matchups.elo1.isnull()
        missing2 = self.matchups.elo2.isnull()
        if missing1.any() or missing2.any():
            missing = self.matchups.loc[missing1,"team1"].tolist() + self.matchups.loc[missing2,"team2"].tolist()
            print("Can't find the following team names in the current elo rankings: " + ", ".join(missing))
            print(f"Assuming the average rating of {base_elo} for now, but might want to fix that...")
            self.matchups.elo1 = self.matchups.elo1.fillna(base_elo)
            self.matchups.elo2 = self.matchups.elo2.fillna(base_elo)
        self.matchups['elo_prob1'] = self.matchups.apply(lambda x: elo_prob(x['elo1'], x['elo2'], scale, homefield),axis=1)
        self.matchups['elo_prob2'] = 1 - self.matchups['elo_prob1']

class Schedule:
    """
    Schedule class that collects a series of Matchups objects over the dates specified 
    and condenses the projections/results into a single dataframe.

    Attributes:  
        start: datetime specifying the first day of interest  
        stop: datetime specifying the last day of interest  
        gameset: string specifiying the conference/rankings of interest  
        gender: string specifiying the gender of interest  
        games_per_day: list of relevant Matchups objects over the specified timeframe  
        schedule: dataframe containing detailed information about each matchup during the timeframe in question  
    """
    def __init__(self, start: str, stop: str, gameset: str = "All Games", women: bool = False, elos: bool = True):
        """
        Initializes a Schedule object using the parameters provided and class functions defined below.

        Args:
            start (str): string specifying the first day of interest.  
            stop (str): string specifying the last day of interest.  
            gameset (str, optional): conference/rankings of interest 
            ("All Games", "Top 25", or the name of a conference), defaults to "All Games".  
            women (bool, optional): whether to look at the women's game, defaults to False.  
            elos (bool, optional): whether to pull elo ratings/projections, defaults to True.  
        """
        self.start = pd.to_datetime(start)
        self.stop = pd.to_datetime(stop)
        self.gameset = gameset
        self.gender = "w" if women else ""
        self.pull_games(elos=False)
        if elos:
            self.add_elos()
        self.flatten_schedule()

    def pull_games(self):
        """
        Pulls the list of Matchups objects for the timeframe of interest.
        """
        self.games_per_day = []
        for day in range((self.stop - self.start).days + 1):
            if (self.start + timedelta(days=day)).month in [11,12,1,2,3]:
                m = Matchups(self.start + timedelta(days=day),self.gameset,self.gender == "w")
                if m.matchups.shape[0] > 0:
                    self.games_per_day.append(m)
    
    def add_elos(self, base_elo: float = 1500.0, scale: float = 1.0, homefield: float = 100.0):
        """
        Merges in the relevant elo ratings to each Matchups dataframe and 
        calculates the corresponding win probabilities based on those elo ratings.

        Args:
            base_elo (float, optional): default elo rating for missing teams, defaults to 1500.0.  
            scale (float, optional): relative scale of elo points compared to baseline, defaults to 1.0.  
            homefield (float, optional): elo point bonus for homefield advantage, defaults to 100.0.  
        """
        seasons = [m.season for m in self.games_per_day]
        for season in range(min(seasons),max(seasons) + 1):
            s = Standings(season, women=self.gender == "w")
            for m in self.games_per_day:
                if m.season == season:
                    m.add_elos(s, base_elo, scale, homefield, False)

    def flatten_schedule(self):
        """
        Amalgamates the collection of matchup dataframes into a single dataframe.
        """
        self.schedule = pd.concat([m.matchups for m in self.games_per_day],ignore_index=True)


def elo_prob(elo1: float, elo2: float, scale: float = 1.0, homefield: float = 100.0) -> float:
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
    prob1 = 1/(1+10**(scale*(elo2 + homefield - elo1)/400))
    return prob1

def courteous_get(url: str, delay: float = 0.1):
    """
    Performs a GET request, but pauses for a short delay so as not to overwhelm the corresponding server.

    Args:
        url (str): endpoint on which to perform the GET.  
        delay (float, optional): number of seconds to pause after the GET, defaults to 0.1.  

    Returns:
        requests.models.Response: response object received from the GET request.
    """
    response = requests.get(url)
    time.sleep(delay)
    return response


def main():
    # Initializing command line inputs
    parser = optparse.OptionParser()
    parser.add_option(
        "--date",
        action="store",
        type="str",
        dest="date",
        default=datetime.now(),
        help="date of interest"
    )
    parser.add_option(
        "--conference",
        action="store",
        type="str",
        dest="conference",
        default="All Games",
        help="conference of interest"
    )
    parser.add_option(
        "--women",
        action="store_true",
        dest="women",
        help="whether to pull stats for the NCAAW instead of NCAAM"
    )
    parser.add_option(
        "--output",
        action="store",
        type="str",
        dest="output",
        help="where to save the Standings and Matchups data in the form of csv's"
    )
    options = parser.parse_args()[0]
    options.date = pd.to_datetime(options.date)
    season = (options.date + timedelta(days=90)).year
    if season < 2021:
        print("Sadly, elo ratings were not stored on Warren Nolan before 2021. Try again with a more recent date.")
    else:
        # Pulling requested Standings and printing results
        s = Standings(season, options.conference, options.women) # ADD CONFERENCE OPTION TO STANDINGS OBJECT!!!
        print(s.elo.to_string(index=False, na_rep=""))

        # Pulling requested Matchups for today and printing results
        m = Matchups(options.date, options.conference, options.women)
        if m.matchups.shape[0] > 0:
            print(m.matchups.to_string(index=False, na_rep=""))
        else:
            print("No games were played on {}.".format(options.date.strftime("%B %d, %Y")))

        # Saving as csv's if requested
        if options.output is not None:
            s.to_csv("{}Standings_{}.csv".format(options.output,datetime.now().strftime("%m%d%y")))
            m.to_csv("{}Matchups_{}.csv".format(options.output,datetime.now().strftime("%m%d%y")))


if __name__ == "__main__":
    main()

