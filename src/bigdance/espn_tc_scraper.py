#!/usr/bin/env python
"""
@File    :   espn_tc_scraper.py
@Time    :   2025/03/17
@Author  :   Taylor Firman
@Version :   0.8.2
@Contact :   tefirman@gmail.com
@Desc    :   ESPN Tournament Challenge integration via the Gambit JSON API
"""

import argparse
import copy
import logging
import sys
from typing import Optional

import numpy as np
import pandas as pd
import requests

from bigdance.cbb_brackets import Bracket, Pool, Team
from bigdance.wn_cbb_scraper import Standings

# Set up logging
logger = logging.getLogger(__name__)

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
    "CA Baptist": "California Baptist",
    "N Dakota St": "North Dakota State",
    "Hawai'i": "Hawaii",
    "FDU": "Fairleigh Dickinson",
    "Miami OH": "Miami (OH)",
}


class ESPNApi:
    """ESPN Gambit API client — fetches bracket/pool data as JSON."""

    GAMBIT_API_BASE = "https://gambit-api.fantasy.espn.com/apis/v1/challenges"

    ROUND_NAMES = ["First Round", "Second Round", "Sweet 16", "Elite 8", "Final Four"]

    def __init__(self, women: bool = False, cache_dir: Optional[str] = None):
        """
        Initialize ESPN API client.

        Args:
            women: Whether to use women's tournament data
            cache_dir: Directory for caching (used for Warren Nolan ratings)
        """
        self.women = women
        gender_suffix = "-women" if women else ""
        self.challenge_slug = f"tournament-challenge-bracket{gender_suffix}-2026"
        self.ratings_source: Optional[Standings] = None

        try:
            self.ratings_source = Standings(women=women, cache_dir=cache_dir)
        except Exception as e:
            logger.warning(f"Could not load Standings: {e}")

    def fetch_challenge(self) -> dict:
        """Fetch the master challenge data (propositions, scoring periods, etc.).

        ESPN only returns propositions for the current scoring period.  To
        build a complete outcome map (needed for resolving entry picks that
        reference earlier rounds), we fetch each prior period's propositions
        and merge them into the response.
        """
        url = f"{self.GAMBIT_API_BASE}/{self.challenge_slug}"
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = dict(resp.json())

        # Collect proposition IDs already present (current period)
        current_prop_ids = {p["id"] for p in data.get("propositions", [])}
        current_period = data.get("currentScoringPeriod", {}).get("id", 1)

        # Fetch propositions from prior scoring periods
        for period in data.get("scoringPeriods", []):
            pid = period.get("id", 0)
            if pid >= current_period:
                continue
            try:
                prior_resp = requests.get(url, params={"scoringPeriodId": pid}, timeout=30)
                prior_resp.raise_for_status()
                prior_data = prior_resp.json()
                for prop in prior_data.get("propositions", []):
                    if prop["id"] not in current_prop_ids:
                        data["propositions"].append(prop)
                        current_prop_ids.add(prop["id"])
            except Exception as e:
                logger.warning(f"Could not fetch scoring period {pid}: {e}")

        return data

    def fetch_group(self, group_id: str) -> dict:
        """Fetch group (pool) data including all entries."""
        url = f"{self.GAMBIT_API_BASE}/{self.challenge_slug}/groups/{group_id}"
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        return dict(resp.json())

    def fetch_entry(self, entry_id: str) -> dict:
        """Fetch a single entry's picks."""
        url = f"{self.GAMBIT_API_BASE}/{self.challenge_slug}/entries/{entry_id}"
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        return dict(resp.json())

    def _build_outcome_map(self, challenge: dict) -> dict[str, dict]:
        """Build a lookup from outcomeId to team info."""
        outcome_map: dict[str, dict] = {}
        for prop in challenge["propositions"]:
            for outcome in prop["possibleOutcomes"]:
                mappings = {m["type"]: m["value"] for m in outcome.get("mappings", [])}
                outcome_map[outcome["id"]] = {
                    "name": outcome["name"],
                    "seed": outcome.get("regionSeed", 1),
                    "region_id": outcome.get("regionId", 1),
                    "espn_id": mappings.get("COMPETITOR_ID"),
                    "proposition_id": prop["id"],
                    "scoring_period": prop["scoringPeriodId"],
                }
        return outcome_map

    def _build_prop_map(self, challenge: dict) -> dict[str, dict]:
        """Build a lookup from propositionId to proposition info."""
        prop_map: dict[str, dict] = {}
        for prop in challenge["propositions"]:
            correct = prop.get("correctOutcomes", [])
            prop_map[prop["id"]] = {
                "name": prop["name"],
                "status": prop["status"],
                "scoring_period": prop["scoringPeriodId"],
                "correct_outcome_ids": correct,
                "winner_id": correct[0] if correct else None,
                "actual_outcome_ids": prop.get("actualOutcomeIds", []),
            }
        return prop_map

    def _get_team_rating(self, team_name: str, seed: int) -> float:
        """Get team rating from Standings or estimate from seed."""
        if team_name in NAME_CORRECTIONS:
            team_name = NAME_CORRECTIONS[team_name]
        if self.ratings_source is not None:
            try:
                row = self.ratings_source.elo[self.ratings_source.elo["Team"] == team_name]
                if not row.empty:
                    return float(row.iloc[0]["ELO"])
                for team in self.ratings_source.elo["Team"]:
                    if team.lower() in team_name.lower() or team_name.lower() in team.lower():
                        row = self.ratings_source.elo[self.ratings_source.elo["Team"] == team]
                        return float(row.iloc[0]["ELO"])
            except Exception as e:
                logger.warning(f"Error finding rating for {team_name}: {e}")
        logger.info(f"Can't find {team_name}, using random seed-based rating...")
        return 2000 - (seed * 50) + np.random.normal(0, 25)

    def _get_team_conference(self, team_name: str, default: str = "Unknown") -> str:
        """Get team conference from Standings."""
        if team_name in NAME_CORRECTIONS:
            team_name = NAME_CORRECTIONS[team_name]
        if self.ratings_source is not None:
            try:
                row = self.ratings_source.elo[self.ratings_source.elo["Team"] == team_name]
                if not row.empty:
                    return str(row.iloc[0]["Conference"])
                for team in self.ratings_source.elo["Team"]:
                    if team.lower() in team_name.lower() or team_name.lower() in team.lower():
                        row = self.ratings_source.elo[self.ratings_source.elo["Team"] == team]
                        return str(row.iloc[0]["Conference"])
            except Exception:
                pass
        return default

    def build_teams(self, challenge: dict) -> list[Team]:
        """Build 64 Team objects from challenge propositions."""
        region_names: dict[int, str] = {}
        for prop in challenge["propositions"]:
            for outcome in prop["possibleOutcomes"]:
                rid = outcome.get("regionId")
                if rid and rid not in region_names:
                    region_names[rid] = f"Region {rid}"

        teams: list[Team] = []
        seen: set[str] = set()
        sorted_props = sorted(challenge["propositions"], key=lambda p: p.get("displayOrder", 0))
        for prop in sorted_props:
            for outcome in prop["possibleOutcomes"]:
                key = f"{outcome.get('regionId')}-{outcome.get('regionSeed')}"
                if key in seen:
                    continue
                seen.add(key)
                name = outcome["name"]
                seed = outcome.get("regionSeed", 1)
                region = region_names.get(outcome.get("regionId", 1), "Unknown")
                teams.append(
                    Team(
                        name,
                        seed,
                        region,
                        self._get_team_rating(name, seed),
                        self._get_team_conference(name),
                    )
                )
        return teams

    def build_actual_bracket(self, challenge: dict) -> Bracket:
        """Build the actual tournament bracket with real results from completed games."""
        teams = self.build_teams(challenge)
        bracket = Bracket(teams)
        prop_map = self._build_prop_map(challenge)
        outcome_map = self._build_outcome_map(challenge)
        team_by_name = {t.name: t for t in teams}

        bracket.results = {rn: [] for rn in self.ROUND_NAMES}

        # Determine which scoring periods are available in the API response.
        # ESPN only returns propositions for the current scoring period, so
        # prior rounds must be inferred from actualOutcomeIds on later props.
        min_period = min((p["scoring_period"] for p in prop_map.values()), default=1)

        # Infer winners for prior rounds from actualOutcomeIds.
        # Each prop's actualOutcomeIds lists the teams actually playing in
        # that game, which means they won their prior round matchups.
        if min_period > 1:
            prior_winner_names: set[str] = set()
            for prop_info in prop_map.values():
                if prop_info["scoring_period"] == min_period:
                    for oid in prop_info["actual_outcome_ids"]:
                        info = outcome_map.get(oid)
                        if info:
                            prior_winner_names.add(info["name"])

            # Set winners on First Round games. Teams in prior_winner_names
            # definitely won their First Round game. For games where neither
            # team is in prior_winner_names (both were eliminated before the
            # current round), assign the higher seed as winner — their runs
            # are over so the choice doesn't affect current-round analysis,
            # but we need all games resolved to advance the bracket tree.
            for game in bracket.games:
                if game.team1.name in prior_winner_names:
                    game.winner = game.team1
                elif game.team2.name in prior_winner_names:
                    game.winner = game.team2
                elif not game.winner:
                    game.winner = game.team1 if game.team1.seed <= game.team2.seed else game.team2

            bracket.results["First Round"] = [game.winner for game in bracket.games if game.winner]

            # Advance through intermediate completed rounds if min_period > 2.
            # The teams in prior_winner_names won ALL rounds before min_period,
            # so they are the winners in every intermediate round as well.
            current_games = bracket.games.copy()
            for round_ind in range(1, min_period - 1):
                if all(g.winner for g in current_games):
                    current_games = bracket.advance_round(current_games)
                    for game in current_games:
                        if game.team1.name in prior_winner_names:
                            game.winner = game.team1
                            bracket.results[self.ROUND_NAMES[round_ind]].append(game.team1)
                        elif game.team2.name in prior_winner_names:
                            game.winner = game.team2
                            bracket.results[self.ROUND_NAMES[round_ind]].append(game.team2)
                        elif not game.winner:
                            game.winner = (
                                game.team1 if game.team1.seed <= game.team2.seed else game.team2
                            )
                            bracket.results[self.ROUND_NAMES[round_ind]].append(game.winner)
        else:
            # First Round props are available — use correctOutcomes directly
            for _prop_id, prop_info in prop_map.items():
                if prop_info["scoring_period"] == 1 and prop_info["winner_id"]:
                    winner_info = outcome_map.get(prop_info["winner_id"])
                    if winner_info:
                        winner_team = team_by_name.get(winner_info["name"])
                        if winner_team:
                            for game in bracket.games:
                                if (
                                    game.team1.name == winner_team.name
                                    or game.team2.name == winner_team.name
                                ):
                                    game.winner = winner_team
                                    break

            bracket.results["First Round"] = [game.winner for game in bracket.games if game.winner]

        # Build current and subsequent rounds from game tree.
        # Start from where prior-round reconstruction left off to avoid
        # duplicating work and to ensure we use the same game objects.
        start_round = max(min_period - 1, 1)
        current_games = bracket.games.copy()
        for step in range(start_round):
            # Set winners from already-populated results so advance_round()
            # uses actual winners instead of randomly simulating.
            round_winners = {t.name for t in bracket.results.get(self.ROUND_NAMES[step], [])}
            for game in current_games:
                if not game.winner and game.team1.name in round_winners:
                    game.winner = game.team1
                elif not game.winner and game.team2.name in round_winners:
                    game.winner = game.team2
            # Advance all but the last completed round — the main loop
            # expects current_games to be the last fully-completed round
            # so it can call advance_round() to enter the current round.
            if step < start_round - 1 and all(g.winner for g in current_games):
                current_games = bracket.advance_round(current_games)
        for round_ind in range(start_round, 5):
            if all(g.winner for g in current_games):
                current_games = bracket.advance_round(current_games)
                for _prop_id, prop_info in prop_map.items():
                    if prop_info["scoring_period"] == round_ind + 1 and prop_info["winner_id"]:
                        winner_info = outcome_map.get(prop_info["winner_id"])
                        if winner_info:
                            for game in current_games:
                                if game.team1.name == winner_info["name"]:
                                    game.winner = game.team1
                                    bracket.results[self.ROUND_NAMES[round_ind]].append(game.team1)
                                    break
                                elif game.team2.name == winner_info["name"]:
                                    game.winner = game.team2
                                    bracket.results[self.ROUND_NAMES[round_ind]].append(game.team2)
                                    break
            else:
                if not bracket.results[self.ROUND_NAMES[round_ind]]:
                    break

        return bracket

    def build_entry_bracket(
        self, entry_data: dict, challenge: dict, teams: list[Team]
    ) -> Optional[Bracket]:
        """
        Build a Bracket from an entry's picks.

        Args:
            entry_data: Entry JSON from fetch_entry()
            challenge: Challenge JSON from fetch_challenge()
            teams: List of Team objects (from build_teams)

        Returns:
            Bracket with picks populated, or None if incomplete
        """
        outcome_map = self._build_outcome_map(challenge)
        team_by_name = {t.name: t for t in teams}

        bracket = Bracket(list(teams))
        bracket.results = {rn: [] for rn in self.ROUND_NAMES}

        picks = entry_data.get("picks", [])
        if len(picks) != 63:
            logger.warning(f"Expected 63 picks, got {len(picks)} for {entry_data.get('name')}")
            return None

        # periodReached = the furthest scoring period this team reaches.
        # A team with periodReached=2 wins R1 (reaches R2), so rounds_won = periodReached - 1.
        team_max_period: dict[str, int] = {}
        for pick in picks:
            outcome_id = pick["outcomesPicked"][0]["outcomeId"]
            period_reached = pick.get("periodReached", 1)
            info = outcome_map.get(outcome_id)
            if info:
                name = info["name"]
                team_max_period[name] = max(team_max_period.get(name, 0), period_reached)

        for team_name, max_period in team_max_period.items():
            team = team_by_name.get(team_name)
            if not team:
                continue
            rounds_won = max_period - 1
            for round_idx in range(min(rounds_won, 5)):
                if team not in bracket.results[self.ROUND_NAMES[round_idx]]:
                    bracket.results[self.ROUND_NAMES[round_idx]].append(team)
                if round_idx == 0:
                    for game in bracket.games:
                        if game.team1.name == team_name or game.team2.name == team_name:
                            game.winner = team
                            break

            if max_period >= 6:
                bracket.results["Championship"] = [team]
                bracket.results["Champion"] = team  # type: ignore[assignment]

        bracket.log_probability = bracket.calculate_log_probability()
        bracket.identify_underdogs()
        return bracket

    def create_simulation_pool(self, pool_id: str) -> Optional[Pool]:
        """
        Create a simulation Pool from an ESPN pool using the JSON API.

        Args:
            pool_id: ESPN group/pool ID

        Returns:
            Pool object ready for simulation, or None on failure
        """
        logger.info("Fetching challenge data...")
        challenge = self.fetch_challenge()

        logger.info("Building actual bracket...")
        actual_bracket = self.build_actual_bracket(challenge)
        for game in actual_bracket.games:
            game.upset_factor = 0.25

        pool_sim = Pool(actual_bracket)
        teams = actual_bracket.teams

        logger.info(f"Fetching pool {pool_id}...")
        group = self.fetch_group(pool_id)
        entries = group.get("entries", [])
        logger.info(f"Found {len(entries)} entries")

        for entry_info in entries:
            entry_name = entry_info.get("name", "Unknown")
            logger.info(f"Loading entry: {entry_name}")
            entry_data = self.fetch_entry(entry_info["id"])
            bracket = self.build_entry_bracket(entry_data, challenge, teams)
            if bracket:
                pool_sim.add_entry(entry_name, bracket, False)
            else:
                logger.warning(f"Failed to build bracket for {entry_name}")

        logger.info(f"Successfully loaded {len(pool_sim.entries)} entries")
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
    ) -> list[dict]:
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
            raise ValueError(f"Invalid round name: {current_round}. Must be one of {valid_rounds}")

        # Get teams in the current round
        teams_in_round = self._get_teams_in_round(actual_bracket, current_round)
        logger.debug(f"Analyzing {len(teams_in_round) // 2} games in {current_round}")

        # Simulate baseline results
        logger.debug("Simulating baseline...")
        fixed_winners = copy.deepcopy(actual_bracket.results)
        baseline = self.pool.simulate_pool(num_sims=num_sims, fixed_winners=fixed_winners)

        # Analyze each game
        game_importance = []
        current_round_results = actual_bracket.results.get(current_round, [])
        for game_ind in range(len(teams_in_round) // 2):
            team1 = teams_in_round[game_ind * 2]
            team2 = teams_in_round[game_ind * 2 + 1]
            if team1 in current_round_results or team2 in current_round_results:
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

    def _get_teams_in_round(self, bracket: Bracket, round_name: str) -> list[Team]:
        """
        Get teams participating in a specific round by walking the game tree.

        Uses advance_round to build correct matchup pairings from the bracket
        structure rather than relying on the flat results list ordering.

        Args:
            bracket: Bracket object
            round_name: Round name

        Returns:
            List of teams in the round (ordered for sequential pairing)
        """
        round_order = [
            "First Round",
            "Second Round",
            "Sweet 16",
            "Elite 8",
            "Final Four",
            "Championship",
        ]
        target_idx = round_order.index(round_name)

        if target_idx == 0:
            teams = []
            for game in bracket.games:
                teams.extend([game.team1, game.team2])
            return teams

        # Walk the game tree forward using actual results to set winners
        # on intermediate rounds so advance_round pairs teams correctly.
        current_games = copy.deepcopy(bracket.games)
        for step in range(target_idx):
            completed_round = round_order[step]
            winners_set = {t.name for t in bracket.results.get(completed_round, [])}
            for game in current_games:
                if not game.winner and game.team1.name in winners_set:
                    game.winner = game.team1
                elif not game.winner and game.team2.name in winners_set:
                    game.winner = game.team2
            current_games = bracket.advance_round(current_games)

        teams = []
        for game in current_games:
            teams.extend([game.team1, game.team2])
        return teams

    def _analyze_matchup(
        self,
        team1: Team,
        team2: Team,
        round_name: str,
        actual_bracket: Bracket,
        baseline: pd.DataFrame,
        num_sims: int,
    ) -> dict:
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
        fixed_winners_team1[round_name] = fixed_winners_team1.get(round_name, []) + [team1]
        results_team1 = self.pool.simulate_pool(
            num_sims=num_sims, fixed_winners=fixed_winners_team1
        )

        # Simulate with Team 2 winning
        logger.debug(f"Simulating with {team2.name} winning...")
        fixed_winners_team2 = copy.deepcopy(actual_bracket.results)
        fixed_winners_team2[round_name] = fixed_winners_team2.get(round_name, []) + [team2]
        results_team2 = self.pool.simulate_pool(
            num_sims=num_sims, fixed_winners=fixed_winners_team2
        )

        # Merge results to calculate impact
        merged_results = pd.merge(
            results_team1[["name", "win_pct"]].rename(columns={"win_pct": "win_pct_team1"}),
            results_team2[["name", "win_pct"]].rename(columns={"win_pct": "win_pct_team2"}),
            on="name",
            how="outer",
        )
        merged_results = pd.merge(
            merged_results,
            baseline[["name", "win_pct"]].rename(columns={"win_pct": "win_pct_baseline"}),
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
        self, game_importance: list[dict], entry_name: Optional[str] = None
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
            print(f"GAME #{i + 1}: {details['matchup']} (Region: {details['region']})")
            print(
                f"  Max Impact: {details['max_impact']:.4f} | "
                f"Avg Impact: {details['avg_impact']:.4f}"
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
                    print(f"  Note: Could not find impact data for {entry_name} on this game")
                    continue

                print(f"  Impact for {entry_name}: {entry_impact['impact']:.4f}")
            else:
                # Show the most affected entry
                print(f"  Most affected entry: {details['max_impact_entry']}")
                entry_impact = details["entry_win_pct_diff"][details["max_impact_entry"]]

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
                f"    Win chances: {better_pct:.1f}% if {better_team} wins "
                f"vs {worse_pct:.1f}% if {worse_team} wins"
            )
            print(f"    Currently at: {baseline_pct:.1f}% baseline win probability")
            print(f"    Difference: {abs(team1_pct - team2_pct):.1f}%")
            print()

        print("=== END OF SUMMARY ===")


def main(argv=None):
    """Command line interface for the module"""
    parser = argparse.ArgumentParser(description="Simulate ESPN Tournament Challenge bracket pool")
    parser.add_argument(
        "--gender",
        choices=["men", "women"],
        default="men",
        help="Which tournament to use (default: men)",
    )
    parser.add_argument(
        "--pool_id",
        type=str,
        default=None,
        help="ESPN group ID of the bracket pool of interest",
    )
    parser.add_argument(
        "--as_of",
        type=str,
        default=None,
        help='name of the round to simulate from ("First Round", '
        + '"Second Round", "Sweet 16", "Elite 8", "Final Four", "Championship")',
    )
    parser.add_argument(
        "--importance",
        action="store_true",
        help="whether to assess the importance of each team winning in the current round",
    )
    parser.add_argument(
        "--my_bracket",
        type=str,
        default=None,
        help="name of the specific bracket to focus on in importance analysis",
    )
    parser.add_argument(
        "--team_probs",
        action="store_true",
        help="show each team's probability of reaching each round instead of pool standings",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="show all debugging messages",
    )
    args = parser.parse_args(argv)
    women = args.gender == "women"

    # Set up logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Create API client and load pool
    api = ESPNApi(women=women)
    pool_sim = api.create_simulation_pool(args.pool_id)
    if not pool_sim:
        logging.error("Failed to create simulation pool")
        return 1

    # Show team round probabilities if requested (skips pool simulation)
    if args.team_probs:
        from bigdance.bigdance_integration import simulate_round_probabilities

        df = simulate_round_probabilities(bracket=pool_sim.actual_results, num_sims=1000)
        print()
        print(df.to_string(index=False))
        print()
        return 0

    # Creating copy of pool simulator if importance calculation is requested
    if args.importance:
        importance_sim = copy.deepcopy(pool_sim)

    # If specified, erasing results from "as_of" round and beyond
    if args.as_of:
        round_names = [
            "First Round",
            "Second Round",
            "Sweet 16",
            "Elite 8",
            "Final Four",
            "Championship",
        ]
        if args.as_of not in round_names:
            logger.warning(
                "Don't recognize the round name provided for as_of, "
                "simulating from current state..."
            )
            if args.as_of in ["First", "Second", "Sweet", "Elite", "Final"]:
                logger.warning(
                    "Hot tip: make sure to put multi-word round names in quotes, "
                    'i.e. `--as_of "Second Round"` (thanks bash)'
                )
        else:
            for round_name in round_names[round_names.index(args.as_of) :]:
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
    print(top_entries[["name", "avg_score", "std_score", "win_prob"]].to_string(index=False))
    print()

    # Analyze game importance if requested
    if args.importance:
        analyzer = GameImportanceAnalyzer(importance_sim)
        importance = analyzer.analyze_win_importance(args.as_of, 1000)
        analyzer.print_importance_summary(importance, args.my_bracket)

    return 0


if __name__ == "__main__":
    sys.exit(main())
