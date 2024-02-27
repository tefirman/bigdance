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

import wn_cbb_elo as cbb
import pandas as pd
import numpy as np
from datetime import datetime
import optparse

class Bracket:
    """
    Bracket class that represents the matchup projections/results of either a contestant's entry or reality.

    Attributes:  
        dancing: dataframe specifying the seeding and elo ratings of all teams in the tournament  
        rounds: list of dataframes specifying the matchup projections and outcomes for each round  
        points: float specifying the number of points earned by a bracket entry  
        earnings: float specifying the amount of money earned by a bracket entry  
    """
    def __init__(self, dancing: pd.DataFrame, winner: str = None, ship: list = [], \
    final4: list = [], elite8: list = [], sweet16: list = [], roundof32: list = []):
        """
        Initializes a Bracket object using the parameters provided and class functions defined below.

        Args:
            dancing (pd.DataFrame): dataframe specifying the seeding and elo ratings of all teams in the tournament.  
            winner (str, optional): team picked to win the entire tournament, defaults to None.  
            ship (list, optional): list of teams picked to make the championship, defaults to [].  
            final4 (list, optional): list of teams picked to make the Final 4, defaults to [].  
            elite8 (list, optional): list of teams picked to make the Elite 8, defaults to [].  
            sweet16 (list, optional): list of teams picked to make the Sweet 16, defaults to [].  
            roundof32 (list, optional): list of teams picked to make the Round of 32, defaults to [].  
        """
        self.dancing = dancing
        self.rounds = []
        while self.dancing.shape[0] > 1:
            fixed = [winner] + (ship if len(self.rounds) < 5 else []) \
                + (final4 if len(self.rounds) < 4 else []) \
                + (elite8 if len(self.rounds) < 3 else []) \
                + (sweet16 if len(self.rounds) < 2 else []) \
                + (roundof32 if len(self.rounds) < 1 else [])
            self.advance_round(fixed)

    def infer_matchups(self):
        """
        Infers the next round's matchups based on the teams remaining and their respective seeds.
        """
        self.matchups = pd.concat([self.dancing.iloc[:self.dancing.shape[0]//2]\
        .rename(columns={col:col + "1" for col in self.dancing.columns}),\
        self.dancing.iloc[self.dancing.shape[0]//2:].iloc[::-1].reset_index(drop=True)\
        .rename(columns={col:col + "2" for col in self.dancing.columns})],axis=1)
        self.matchups["elo_prob1"] = self.matchups.apply(lambda x: cbb.elo_prob(x['ELO1'], x['ELO2'], 1.0, 0.0),axis=1)
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
        self.rounds.append(self.matchups.copy())
    
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

def dummy_dancing(season: int = datetime.now().year, women: bool = False):
    """
    Generates an example of bracket seeding based on current elo ratings from Warren Nolan.

    Args:
        season (int, optional): season of interest, defaults to the current year.  
        women (bool, optional): whether to look at the women's game, defaults to False.  

    Returns:
        pd.DataFrame: dataframe specifying the seeding and elo ratings of all teams in the tournament.  
    """
    s = cbb.Standings(season, women=women)
    conf_winners = s.elo.loc[s.elo.Conference != "Independent"].drop_duplicates(subset="Conference",keep="first")
    at_large = s.elo.loc[~s.elo.Team.isin(conf_winners.Team.tolist())].iloc[:64 - conf_winners.shape[0]]
    dancing = pd.concat([conf_winners,at_large],ignore_index=True).sort_values(by="ELO",ascending=False,ignore_index=True)
    dancing['tourney_rank'] = np.arange(1,dancing.shape[0] + 1)
    dancing['tourney_seed'] = (dancing.tourney_rank - 1)//4 + 1
    return dancing

class Pool:
    """
    Pool class that consists of a collection of brackets and a simulation of reality to represent a March Madness Pool.

    Attributes:  
        entries: list of Bracket objects representing each contestant's bracket entry  
        reality: Bracket object representing what actually happened during the tournament  
        payouts: list of payouts for the top placing entries  
    """
    def __init__(self, dancing: pd.DataFrame, num_entries: int = 10, payouts: list = [100.0]):
        """
        Initializes a Pool object using the parameters provided and class functions defined below.

        Args:
            dancing (pd.DataFrame): dataframe specifying the seeding and elo ratings of all teams in the tournament.  
            num_entries (int, optional): number of contestants in the pool, defaults to 10.
            payouts (list, optional): list of payouts for the top placing entries, defaults to $100 winner-take-all.  
        """
        self.payouts = payouts
        self.reality = Bracket(dancing)
        self.entries = []
        for entry in range(num_entries):
            self.entries.append(Bracket(dancing))
            self.entries[-1].Name = f"Entry #{entry + 1}"
        self.score_entries()
        self.pay_out_earnings()
    
    def score_entries(self, base_pts: float = 10.0):
        """
        Calculates the points earned by each entry by comparing it to reality.

        Args:
            base_pts (float, optional): number of points awarded for each win in round 1, defaults to 10.0.
        """
        for entry in range(len(self.entries)):
            self.entries[entry].points = 0.0
            for round in range(len(self.reality.rounds)):
                correct = np.intersect1d(self.reality.rounds[round].pick.tolist(),self.entries[entry].rounds[round].pick.tolist())
                self.entries[entry].points += len(correct)*base_pts*(2**round)
        self.entries.sort(key=lambda x: x.points + np.random.rand(), reverse=True) # Random jitter for tiebreaker
    
    def pay_out_earnings(self):
        """
        Assigns corresponding earnings to each entry in the Pool.
        """
        for ind in range(len(self.entries)):
            self.entries[ind].earnings = self.payouts[ind] if ind < len(self.payouts) else 0.0

class Simulation:
    """
    Simulation class that consists of a collection of March Madness Pools to analyze trends among winning entries.

    Attributes:  
        pools: list of Pool objects representing each simulated contest  
        winners: list of Bracket objects representing the winners of each simulated contest  
        picks: dataframe of picks made be the winning entries  
        pick_probs: dataframe containing probabilities of each team being picked in each round in a winning entry  
    """
    def __init__(self, dancing: pd.DataFrame, num_entries: int = 10, payouts: list = [100.0], num_sims: int = 1000):
        """
        Initializes a Simulation object using the parameters provided and class functions defined below.

        Args:
            dancing (pd.DataFrame): dataframe specifying the seeding and elo ratings of all teams in the tournament.  
            num_entries (int, optional): number of contestants in the pool, defaults to 10.
            payouts (list, optional): list of payouts for the top placing entries, defaults to $100 winner-take-all.  
            num_sims (int, optional): number of simulations to run, defaults to 1000.
        """
        self.pools = []
        for pool in range(num_sims):
            if (pool + 1)%100 == 0:
                print(f"Simulation {pool + 1} out of {num_sims}, {datetime.now()}")
            self.pools.append(Pool(dancing, num_entries, payouts))
        self.winners = [pool.entries[0] for pool in self.pools]
        self.amass_winner_picks()
        self.analyze_winner_picks()
        self.pick_probs = pd.merge(left=dancing[["Team","Record","ELO","AP Rank","tourney_seed"]],\
        right=self.pick_probs.reset_index().rename(columns={"Pick":"Team"}),how='inner',on="Team")

    def amass_winner_picks(self):
        """
        Collects the picks made by the winning entries into a single dataframe for easier analysis.
        """
        self.picks = pd.DataFrame()
        for round in range(len(self.winners[0].rounds)):
            picks = []
            for winner in self.winners:
                picks.extend(winner.rounds[round].pick.tolist())
            self.picks = pd.concat([self.picks,pd.DataFrame({"Round":round,"Pick":picks})],ignore_index=True)
    
    def analyze_winner_picks(self):
        """
        Analyzes the probability of each team being picked in each round in a winning entry.
        """
        self.pick_probs = self.picks.groupby(["Pick","Round"]).size().unstack().fillna(0.0)
        round_names = ["Round of 32", "Sweet 16", "Elite 8", "Final 4", "Ship", "Winner"]
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
        dest="entires",
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
        dest="sims",
        default="100",
        help="comma separated list of payouts for the top entries"
    )
    parser.add_option(
        "--output",
        action="store",
        type="str",
        dest="output",
        help="where to save the Standings and Matchups data in the form of csv's"
    )
    options = parser.parse_args()[0]
    try:
        options.payouts = [float(val) for val in options.payouts.split(",")]
    except:
        print("Can't parse the provided payouts, assuming a $100 winner-take-all...")
        options.payouts = [100.0]

    # Pulling dummy seeding for the requested season
    dancing = dummy_dancing(options.season, options.women)
    # Simulating bracket pools as requested and printing the results
    sim = Simulation(dancing, options.entries, options.payouts, options.num_sims)
    print(sim.pick_probs.to_string(index=False, na_rep=""))
    # Saving as csv's if requested
    if options.output is not None:
        sim.to_csv("{}WinningPickPcts_{}.csv".format(options.output,datetime.now().strftime("%m%d%y")))

if __name__ == "__main__":
    main()

