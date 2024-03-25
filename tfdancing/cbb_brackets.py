#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   cbb_brackets.py
@Time    :   2024/02/23 12:30:35
@Author  :   Taylor Firman
@Version :   1.0
@Contact :   tefirman@gmail.com
@Desc    :   March Madness Bracket Pool Simulator
'''

import wn_cbb_elo as wn
import pandas as pd
import numpy as np
from datetime import datetime
from selenium import webdriver
import copy
from bs4 import BeautifulSoup
import json
import optparse
import os
import shutil
import time

corrections = pd.read_csv("name_corrections.csv")
round_names = ["Round of 32", "Sweet 16", "Elite 8", "Final 4", "Championship", "Winner"]

def sel_get(url: str, delay: float = 3.0):
    """
    Pulls the full html of a page using Selenium.

    Args:
        url (str): URL of the page in question.  
        delay (float, optional): amount of delay to allow the page to load before pulling, defaults to 3.0.  

    Returns:
        str: raw html behind the page in question.  
    """
    browser = webdriver.Chrome()
    browser.get(url)
    time.sleep(delay)
    html = browser.page_source
    browser.close()
    return html

class Picks:
    """
    Picks class that represents the predictions of a contestant.
    """
    def __init__(self, roundof32: list = [], sweet16: list = [], \
    elite8: list = [], final4: list = [], championship: list = [], winner: list = []):
        """
        Initializes a Picks object using the parameters provided and class functions defined below.

        Args:
            roundof32 (list, optional): list of teams picked to make the Round of 32, defaults to [].
            sweet16 (list, optional): list of teams picked to make the Sweet 16, defaults to [].
            elite8 (list, optional): list of teams picked to make the Elite 8, defaults to [].
            final4 (list, optional): list of teams picked to make the Final 4, defaults to [].
            championship (list, optional): list of teams picked to make the Championship, defaults to [].
            winner (list, optional): team picked to win the entire tournament, defaults to [].
        """
        self.roundof32 = roundof32
        self.sweet16 = sweet16
        self.elite8 = elite8
        self.final4 = final4
        self.championship = championship
        self.winner = winner
    
    def flatten(self):
        """
        Flattens a list object into a list of lists of picks for each round.  

        Returns:
            list: list of lists containing picks for each round
        """
        picks = []
        for ind in range(len(round_names)):
            picks.append([])
            for round_name in round_names[ind:]:
                picks[ind].append(getattr(self,round_name.replace(" ","").lower()))
        return picks

class Bracket:
    """
    Bracket class that represents the matchup projections/results of either a contestant's entry or reality.

    Attributes:  
        gender: string indicating which gender tournament is being analyzed  
        base_url: string indicating the base URL to be used when pulling bracket details  
        dancing: dataframe specifying the seeding and elo ratings of all teams in the tournament  
        rounds: list of dataframes specifying the matchup projections and outcomes for each round  
        points: float specifying the number of points earned by a bracket entry  
        earnings: float specifying the amount of money earned by a bracket entry  
    """
    def __init__(self, s: wn.Standings, bracket_id: str = None, randomize: bool = False, picks: Picks = Picks()):
        """
        Initializes a Bracket object using the parameters provided and class functions defined below.

        Args:
            s (wn.Standings): Standings objects specifying the elo ratings of all teams in college basketball.  
            bracket_id (str, optional): ESPN Tournament Challenge bracket ID to analyze as your own entry.  
            randomize (bool, optional): whether to use the picks in the bracket provided.  
            picks (Picks, optional): teams picked to make each round, defaults to no preferred picks.  
        """
        self.gender = "-women" if s.gender == "w" else ""
        self.base_url = f"https://fantasy.espn.com/games/tournament-challenge-bracket{self.gender}-{s.season}"
        if bracket_id is not None:
            self.pull_bracket(bracket_id)
            self.extract_seeding(s)
            self.extract_picks()
        else:
            self.dummy_seeding(s)
        if randomize or not hasattr(self, "picks"):
            self.picks = picks.flatten()
        self.rounds = []
        for ind in range(len(self.picks)):
            self.advance_round(self.picks[ind])

    def dummy_seeding(self, s: wn.Standings):
        conf_winners = s.elo.loc[s.elo.Conference != "Independent"].drop_duplicates(subset="Conference",keep="first")
        at_large = s.elo.loc[~s.elo.Team.isin(conf_winners.Team.tolist())].iloc[:64 - conf_winners.shape[0]]
        self.dancing = pd.concat([conf_winners,at_large],ignore_index=True).sort_values(by="ELO",ascending=False,ignore_index=True)
        self.dancing['tourney_rank'] = np.arange(1,self.dancing.shape[0] + 1)
        self.dancing['tourney_seed'] = (self.dancing.tourney_rank - 1)//4 + 1

    def pull_bracket(self, bracket_id: str, cache_loc: str = ".brackets"):
        if not os.path.exists(cache_loc):
            os.mkdir(cache_loc)
        if os.path.exists(cache_loc + "/" + bracket_id):
            tempData = open(cache_loc + "/" + bracket_id,"r")
            self.bracket_html = tempData.read()
            tempData.close()
        else:
            self.bracket_html = sel_get(f"{self.base_url}/bracket?id={bracket_id}")
            tempData = open(cache_loc + "/" + bracket_id,"w")
            tempData.write(self.bracket_html)
            tempData.close()
        self.bracket_soup = BeautifulSoup(self.bracket_html,"html.parser")

    def extract_seeding(self, s: wn.Standings):
        scriptTag = self.bracket_soup.find_all("script")
        scriptTag = [tag for tag in scriptTag if ";window['__project-chui__']=" in tag.text][0]
        bracket = json.loads(scriptTag.text.split(";window['__project-chui__']=")[-1][:-1])
        games = bracket['page']["content"]["challenge"]["propositions"]
        self.rankings = pd.DataFrame()
        for game in games:
            self.rankings = pd.concat([self.rankings,pd.DataFrame(game['possibleOutcomes'])],ignore_index=True)
        self.rankings["tourney_rank"] = self.rankings.mappings.apply(lambda x: \
        [val["value"] for val in x if val['type'] == "RANKING"][0]).astype(int)
        self.rankings['tourney_seed'] = (self.rankings.tourney_rank - 1)//4 + 1
        self.rankings = pd.merge(left=self.rankings,right=corrections,how="left",on="name")
        self.rankings.Team = self.rankings.Team.fillna(self.rankings.name)
        self.rankings.loc[self.rankings.Team.str.endswith(" St"),"Team"] += "ate"
        self.dancing = pd.merge(left=self.rankings[["Team","tourney_rank","tourney_seed"]],right=s.elo,how="left",on="Team")
        missing = self.dancing.ELO.isnull()
        if missing.any():
            print("MISSING SOME TEAM NAMES: " + ", ".join(self.dancing.loc[missing,"Team"].tolist()))

    def extract_picks(self):
        pickTag = self.bracket_soup.find_all("div", attrs={"class": "PrintBracketOutcome-teamName CLASS_TRUNCATE"})
        picks = pd.DataFrame({"name":[pick.text for pick in pickTag]})
        num_picks = picks.groupby("name").size().to_frame("freq").reset_index()
        num_picks = pd.merge(left=num_picks,right=corrections,how="left",on="name")
        num_picks.Team = num_picks.Team.fillna(num_picks.name)
        num_picks.loc[num_picks.Team.str.endswith(" St"),"Team"] += "ate"
        self.picks = []
        for ind in range(2,7):
            self.picks.append(num_picks.loc[num_picks.freq >= ind,'Team'].tolist())
        champ = self.bracket_soup.find_all("span", attrs={"class": "PrintChampionshipPickBody-outcomeName"})
        self.picks.append(self.rankings.loc[self.rankings.description == champ[0].text,"Team"].tolist())

    def infer_matchups(self):
        """
        Infers the next round's matchups based on the teams remaining and their respective seeds.
        """
        self.matchups = pd.concat([self.dancing.iloc[:self.dancing.shape[0]//2]\
        .rename(columns={col:col + "1" for col in self.dancing.columns}),\
        self.dancing.iloc[self.dancing.shape[0]//2:].iloc[::-1].reset_index(drop=True)\
        .rename(columns={col:col + "2" for col in self.dancing.columns})],axis=1)
        self.matchups["elo_prob1"] = self.matchups.apply(lambda x: wn.elo_prob(x['ELO1'], x['ELO2'], 1.0, 0.0),axis=1)
        self.matchups["elo_prob2"] = 1 - self.matchups["elo_prob1"]
    
    def make_picks(self, fixed: list = []):
        """
        Randomly selects a contestant's picks (or reality's winners) for the latest round.

        Args:
            fixed (list, optional): list of teams to advance by default, defaults to [].
        """
        self.matchups["pick_rand"] = np.random.rand(self.matchups.shape[0])
        self.matchups.loc[self.matchups.Team1.isin(fixed),"pick_rand"] = 0.0
        self.matchups.loc[self.matchups.Team2.isin(fixed),"pick_rand"] = 1.0
        self.upset = self.matchups.pick_rand > self.matchups.elo_prob1
        self.matchups.loc[~self.upset,"pick"] = self.matchups.loc[~self.upset,"Team1"]
        self.matchups.loc[self.upset,"pick"] = self.matchups.loc[self.upset,"Team2"]
        self.rounds.append(self.matchups.pick.tolist())
    
    def update_standings(self):
        """
        Updates the standings based on the results from the latest round.
        """
        for col in self.dancing.columns:
            self.matchups.loc[self.upset,col + "1"] = self.matchups.loc[self.upset,col + "2"]
        self.matchups = self.matchups[[col for col in self.matchups.columns if col.endswith("1")]]
        self.matchups = self.matchups.rename(columns={col:col[:-1] for col in self.matchups.columns})
        self.dancing = self.matchups.copy()
        del self.matchups, self.upset

    def advance_round(self, fixed: list = []):
        """
        Updates all of the relevant Bracket settings to advance the entry by a round.

        Args:
            fixed (list, optional): list of teams to advance by default, defaults to [].  
        """
        self.infer_matchups()
        self.make_picks(fixed)
        self.update_standings()

class Pool:
    """
    Pool class that consists of a collection of brackets and a simulation of reality to represent a March Madness Pool.

    Attributes:  
        entries: list of Bracket objects representing each contestant's bracket entry  
        reality: Bracket object representing what actually happened during the tournament  
        payouts: list of payouts for the top placing entries  
    """
    def __init__(self, s: wn.Standings, group_id: str = None, num_entries: int = 10, payouts: list = [100.0]):
        """
        Initializes a Pool object using the parameters provided and class functions defined below.

        Args:
            s (wn.Standings): Standings objects specifying the elo ratings of all teams in college basketball.  
            bracket_id (str, optional): ESPN Tournament Challenge group ID to analyze as your own entry.  
            num_entries (int, optional): number of contestants in the pool, defaults to 10.  
            payouts (list, optional): list of payouts for the top placing entries, defaults to $100 winner-take-all.  
        """
        self.gender = "-women" if s.gender == "w" else ""
        self.base_url = f"https://fantasy.espn.com/games/tournament-challenge-bracket{self.gender}-{s.season}"
        self.group_id = group_id
        self.num_entries = num_entries
        self.payouts = payouts
        self.pull_group(s)
        self.add_reality(s)
        self.score_entries()
        self.pay_out_earnings()
        self.pull_standings()
    
    def pull_group(self, s: wn.Standings):
        if self.group_id is not None:
            self.pull_group_details()
            self.parse_group_entries(s)
        else:
            self.entries = []
            for entry in range(self.num_entries):
                self.entries.append(Bracket(s, None, True))
                self.entries[-1].Name = f"Entry #{entry + 1}"

    def pull_group_details(self, cache_loc: str = ".groups"):
        if not os.path.exists(cache_loc):
            os.mkdir(cache_loc)
        if os.path.exists(cache_loc + "/" + self.group_id):
            tempData = open(cache_loc + "/" + self.group_id,"r")
            self.group_html = tempData.read()
            tempData.close()
        else:
            self.group_html = sel_get(f"{self.base_url}/group?id={self.group_id}")
            tempData = open(cache_loc + "/" + self.group_id,"w")
            tempData.write(self.group_html)
            tempData.close()
        self.group_soup = BeautifulSoup(self.group_html, features="lxml")
    
    def parse_group_entries(self, s: wn.Standings):
        self.group_name = self.group_soup.find_all("div", attrs={"class": "GroupCard-titleContainer"})[0].text
        entry_deets = self.group_soup.find_all("div", attrs={"class":"EntryLink-nameContainer EntryLink-nameContainer--vertical"})
        self.entries = []
        for entry in entry_deets:
            if entry.text == "a42many42's Picks 1":
                continue
            entry_id = entry.find_all("a")[0].attrs["href"].split("/bracket?id=")[-1]
            self.entries.append(Bracket(s, entry_id, False))
            self.entries[-1].Name = entry.find_all("a")[0].text
            self.entries[-1].User = entry.find_all("span")[0].text
            self.entries[-1].bracket_id = entry_id
        del self.group_html, self.group_soup
    
    def add_reality(self, s: wn.Standings):
        if self.group_id is None:
            self.reality = Bracket(s, None, True)
        else:
            outcomeTag = self.entries[0].bracket_soup.find_all("label", attrs={"class": "BracketOutcome-label truncate"})
            outcomes = pd.DataFrame({"name":[outcome.text for outcome in outcomeTag]})
            num_games = outcomes.groupby("name").size().to_frame("freq").reset_index()
            num_games = pd.merge(left=num_games,right=corrections,how="left",on="name")
            num_games.Team = num_games.Team.fillna(num_games.name)
            num_games.loc[num_games.Team.str.endswith(" St"),"Team"] += "ate"
            self.reality_picks = Picks()
            for ind in range(2,8):
                setattr(self.reality_picks, round_names[ind - 2].replace(" ","").lower(), \
                num_games.loc[num_games.freq >= ind,'Team'].tolist())
            self.reality = Bracket(s, self.entries[0].bracket_id, True, self.reality_picks)
            for ind in range(len(self.entries)):
                del self.entries[ind].bracket_html, self.entries[ind].bracket_soup, self.entries[ind].rankings
            del self.reality.bracket_html, self.reality.bracket_soup, self.reality.rankings
            # champ = self.bracket_soup.find_all("span", attrs={"class": "PrintChampionshipPickBody-outcomeName"})
            # setattr(self.reality_picks, "winner", self.rankings.loc[self.rankings.description == champ[0].text,"Team"].tolist())

    def score_entries(self, base_pts: float = 10.0):
        """
        Calculates the points earned by each entry by comparing it to reality.

        Args:
            base_pts (float, optional): number of points awarded for each win in round 1, defaults to 10.0.
        """
        for entry in range(len(self.entries)):
            self.entries[entry].points = 0.0
            for round in range(len(self.reality.rounds)):
                correct = np.intersect1d(self.reality.rounds[round],self.entries[entry].rounds[round])
                self.entries[entry].points += len(correct)*base_pts*(2**round)
        self.entries.sort(key=lambda x: x.points + np.random.rand(), reverse=True) # Random jitter for tiebreaker
    
    def pay_out_earnings(self):
        """
        Assigns corresponding earnings to each entry in the Pool.
        """
        for ind in range(len(self.entries)):
            self.entries[ind].earnings = self.payouts[ind] if ind < len(self.payouts) else 0.0
    
    def pull_standings(self):
        self.standings = pd.DataFrame()
        for entry in self.entries:
            self.standings = pd.concat([self.standings,pd.DataFrame({"Name":[entry.Name],"User":[entry.User],\
            "Points":[entry.points],"Earnings":[entry.earnings]})],ignore_index=True)
        self.standings = self.standings.sort_values(by="Earnings",ascending=False,ignore_index=True)

class Simulation:
    """
    Simulation class that consists of a collection of March Madness Pools to analyze trends among winning entries.

    Attributes:  
        pools: list of Pool objects representing each simulated contest  
        winners: list of Bracket objects representing the winners of each simulated contest  
        picks: dataframe of picks made be the winning entries  
        pick_probs: dataframe containing probabilities of each team being picked in each round in a winning entry  
    """
    def __init__(self, s: wn.Standings, group_id: str = None, num_entries: int = 10, payouts: list = [100.0], num_sims: int = 1000):
        """
        Initializes a Simulation object using the parameters provided and class functions defined below.

        Args:
            s (wn.Standings): Standings objects specifying the elo ratings of all teams in college basketball.  
            group_id (str, optional): ESPN Tournament Challenge group ID to analyze.
            num_entries (int, optional): number of contestants in the pool, defaults to 10.
            payouts (list, optional): list of payouts for the top placing entries, defaults to $100 winner-take-all.  
            num_sims (int, optional): number of simulations to run, defaults to 1000.
        """
        self.group_id = group_id
        self.num_entries = num_entries
        self.payouts = payouts
        self.num_sims = num_sims
        self.create_pools(s)
        self.avg_standings()
        self.amass_winner_picks()
        self.analyze_winner_picks()
        self.pick_probs = pd.merge(left=s.elo[["Team","Record","ELO","AP Rank"]],\
        right=self.pick_probs.reset_index().rename(columns={"Pick":"Team"}),how='inner',on="Team")

    def create_pools(self, s: wn.Standings):
        self.pools = []
        self.winners = []
        for pool in range(self.num_sims):
            if (pool + 1)%100 == 0:
                print(f"Simulation {pool + 1} out of {self.num_sims}, {datetime.now()}")
            if pool == 0 or self.group_id is None:
                self.pools.append(Pool(s, self.group_id, self.num_entries, self.payouts))
            else:
                self.pools.append(copy.deepcopy(self.pools[0]))
                self.pools[-1].reality = Bracket(s, self.pools[-1].entries[0].bracket_id, True, self.pools[-1].reality_picks)
                self.pools[-1].score_entries()
                self.pools[-1].pay_out_earnings()
                self.pools[-1].pull_standings()
            self.winners.append(self.pools[-1].entries[0])
            if pool > 0:
                del self.pools[-1].entries, self.pools[-1].reality

    def avg_standings(self):
        self.standings = pd.concat([p.standings for p in self.pools],ignore_index=True)
        self.standings = self.standings.groupby(["Name","User"]).mean().reset_index()
        self.standings = self.standings.sort_values(by=["Earnings","Points"],ascending=False,ignore_index=True)

    def amass_winner_picks(self):
        """
        Collects the picks made by the winning entries into a single dataframe for easier analysis.
        """
        self.picks = pd.DataFrame()
        for round in range(len(self.winners[0].rounds)):
            picks = []
            for winner in self.winners:
                picks.extend(winner.rounds[round])
            self.picks = pd.concat([self.picks,pd.DataFrame({"Round":round,"Pick":picks})],ignore_index=True)
    
    def analyze_winner_picks(self):
        """
        Analyzes the probability of each team being picked in each round in a winning entry.
        """
        self.pick_probs = self.picks.groupby(["Pick","Round"]).size().unstack().fillna(0.0)
        self.pick_probs = self.pick_probs.rename(columns={ind:round_names[ind] for ind in range(len(round_names))})
        for col in self.pick_probs.columns:
            self.pick_probs[col] /= len(self.pools)
        self.pick_probs = self.pick_probs.sort_values(by=round_names[::-1],ascending=False)

def main():
    # Initializing command line inputs
    parser = optparse.OptionParser()
    parser.add_option(
        "--season",
        action="store",
        type="int",
        dest="season",
        default=datetime.now().year,
        help="season of interest"
    )
    parser.add_option(
        "--women",
        action="store_true",
        dest="women",
        help="whether to pull stats for the NCAAW instead of NCAAM"
    )
    parser.add_option(
        "--entries",
        action="store",
        type="int",
        dest="entries",
        default=10,
        help="number of bracket entries in each simulated pool"
    )
    parser.add_option(
        "--sims",
        action="store",
        type="int",
        dest="sims",
        default=1000,
        help="number of bracket pools to simulate"
    )
    parser.add_option(
        "--payouts",
        action="store",
        type="str",
        dest="payouts",
        default="100",
        help="comma separated list of payouts for the top entries"
    )
    parser.add_option(
        "--groupid",
        action="store",
        type="str",
        dest="group_id",
        help="ESPN Tournament Challenge group ID to analyze"
    )
    parser.add_option(
        "--breakdown",
        action="store_true",
        dest="breakdown",
        help="whether to display pick breakdown among winners"
    )
    parser.add_option(
        "--resetcache",
        action="store_true",
        dest="reset_cache",
        help="whether to reset your group and bracket caches"
    )
    parser.add_option(
        "--output",
        action="store",
        type="str",
        dest="output",
        help="where to save the pick percentage data in the form of a csv"
    )
    options = parser.parse_args()[0]
    try:
        options.payouts = [float(val) for val in options.payouts.split(",")]
    except:
        print("Can't parse the provided payouts, assuming a $100 winner-take-all...")
        options.payouts = [100.0]
    if options.reset_cache:
        if os.path.isdir(".brackets"):
            shutil.rmtree(".brackets")
        if os.path.isdir(".groups"):
            shutil.rmtree(".groups")

    # Pulling current elo standings
    s = wn.Standings(options.season, women=options.women)

    # Simulating bracket pools as requested and printing the results
    sim = Simulation(s, options.group_id, options.entries, options.payouts, options.sims)
    print(sim.standings.to_string(index=False, na_rep=""))
    if options.output is not None:
        sim.standings.to_excel("{}ProjectedStandings_{}_{}.xlsx".format(options.output,\
        sim.pools[0].group_name.replace(" ",""),datetime.now().strftime("%m%d%y")),index=False)

    if options.breakdown:
        print("Overall Winner Pick Breakdown:")
        print(sim.pick_probs.to_string(index=False, na_rep=""))
        # Saving as csv if requested
        if options.output is not None:
            sim.pick_probs.to_excel("{}WinningPickPcts_{}.xlsx".format(options.output,datetime.now().strftime("%m%d%y")),index=False)

if __name__ == "__main__":
    main()

