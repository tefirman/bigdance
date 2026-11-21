#!/usr/bin/env python
"""
@File    :   betting.py
@Author  :   Taylor Firman
@Version :   0.8.5
@Contact :   tefirman@gmail.com
@Desc    :   In-season college basketball betting tools: compare Warren Nolan Elo win
             probabilities against sportsbook moneylines, size bets, and track closing line value.

Nothing here is betting advice. A public Elo model will rarely beat sharp closing lines on its
own, which is why `find_edges` shrinks the model toward the market by default and why `BetLog`
tracks closing line value (CLV), a much faster signal of real edge than win/loss results.
"""

import argparse
import html
import logging
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import requests

ODDS_API_BASE = "https://api.the-odds-api.com/v4"
SPORT_KEYS = {False: "basketball_ncaab", True: "basketball_wncaab"}

# Manual {odds-api name: Warren Nolan name} fixes for names the automatic matcher can't resolve.
NAME_OVERRIDES: dict[str, str] = {
    "LIU Sharks": "Long Island",
    "Saint Mary's Gaels": "Saint Mary's College",
    "Seattle Redhawks": "Seattle University",
    "Seattle U Redhawks": "Seattle University",
    "San José St Spartans": "San Jose State",
    "N Colorado Bears": "Northern Colorado",
    "Boston Univ. Terriers": "Boston University",
    "IUPUI Jaguars": "IU Indianapolis",
    "Fort Wayne Mastodons": "Purdue Fort Wayne",
    "Loyola (Chi) Ramblers": "Loyola-Chicago",
    "UNC Wilmington Seahawks": "UNCW",
    "UNC Greensboro Spartans": "UNCG",
    "Presbyterian Blue Hose": "Presbyterian College",
    "Hawai'i Rainbow Warriors": "Hawaii",
    "UConn Huskies": "Connecticut",
    "Arkansas-Little Rock Trojans": "Little Rock",
    "CSU Bakersfield Roadrunners": "Cal State Bakersfield",
    "CSU Northridge Matadors": "Cal State Northridge",
    "GW Revolutionaries": "George Washington",
    "Mt. St. Mary's Mountaineers": "Mount Saint Mary's",
    "NC State Wolfpack": "North Carolina State",
    "SE Louisiana Lions": "Southeastern Louisiana",
    "SE Missouri St Redhawks": "Southeast Missouri",
    "Tenn-Martin Skyhawks": "Tennessee-Martin",
    "UT-Arlington Mavericks": "UTA",
    "Cal Baptist Lancers": "California Baptist",
    "Massachusetts Minutemen": "UMass",
    "New Orleans Privateers": "LSU New Orleans",
    "Pennsylvania Quakers": "Penn",
    "UT Rio Grande Valley Vaqueros": "UTRGV",
    "Texas A&M-CC Islanders": "Texas A&M-Corpus Christi",
}


# --------------------------------------------------------------------------------------
# Odds math
# --------------------------------------------------------------------------------------
def american_to_prob(odds: float) -> float:
    """Implied probability (including vig) of American odds."""
    if odds < 0:
        return -odds / (-odds + 100.0)
    return 100.0 / (odds + 100.0)


def american_to_decimal(odds: float) -> float:
    """Decimal odds (total payout per 1 staked) from American odds."""
    if odds < 0:
        return 1.0 + 100.0 / -odds
    return 1.0 + odds / 100.0


def prob_to_american(p: float) -> float:
    """American odds corresponding to a fair probability."""
    if not 0 < p < 1:
        raise ValueError("Probability must be strictly between 0 and 1")
    if p >= 0.5:
        return -100.0 * p / (1 - p)
    return 100.0 * (1 - p) / p


def devig_two_way(p1: float, p2: float) -> tuple[float, float]:
    """Remove the vig from a two-way market by proportional normalization."""
    total = p1 + p2
    return p1 / total, p2 / total


def expected_value(p: float, decimal_odds: float) -> float:
    """Expected profit per 1 staked given win probability p."""
    return p * decimal_odds - 1.0


def kelly_stake(
    p: float, decimal_odds: float, fraction: float = 0.25, max_fraction: float = 0.05
) -> float:
    """
    Fractional Kelly stake as a share of bankroll, capped at max_fraction.
    Returns 0 for non-positive-EV bets.
    """
    b = decimal_odds - 1.0
    if b <= 0:
        return 0.0
    full_kelly = (p * decimal_odds - 1.0) / b
    return float(min(max(full_kelly * fraction, 0.0), max_fraction))


# --------------------------------------------------------------------------------------
# Team name matching (odds feeds use "Duke Blue Devils", Warren Nolan uses "Duke")
# --------------------------------------------------------------------------------------
def normalize_name(name: str) -> str:
    """Lowercase, strip punctuation, and expand leading/trailing 'St.' for comparison."""
    s = re.sub(r"[^\w\s]", " ", name.lower().replace("&", "and"))
    tokens = s.split()
    out = []
    for i, tok in enumerate(tokens):
        if tok == "st":
            tok = "saint" if i == 0 else "state"
        out.append(tok)
    return " ".join(out)


def map_team_names(
    odds_names: list[str],
    wn_names: list[str],
    overrides: Optional[dict[str, str]] = None,
) -> tuple[dict[str, str], list[str]]:
    """
    Map odds-feed team names to Warren Nolan names.

    A Warren Nolan name matches when the odds name equals it or starts with it followed by a
    mascot (longest match wins, so "Michigan State Spartans" resolves to "Michigan St." rather
    than "Michigan").

    Returns:
        (mapping, unmatched): dict of odds name -> WN name, and the odds names left unresolved.
    """
    overrides = {**NAME_OVERRIDES, **(overrides or {})}
    wn_norm = sorted(((normalize_name(n), n) for n in wn_names), key=lambda t: -len(t[0]))
    mapping: dict[str, str] = {}
    unmatched: list[str] = []
    for name in dict.fromkeys(odds_names):
        if name in overrides:
            mapping[name] = overrides[name]
            continue
        norm = normalize_name(name)
        for wn_n, wn_orig in wn_norm:
            if norm == wn_n or norm.startswith(wn_n + " "):
                mapping[name] = wn_orig
                break
        else:
            unmatched.append(name)
    return mapping, unmatched


# --------------------------------------------------------------------------------------
# Odds retrieval (The Odds API: https://the-odds-api.com)
# --------------------------------------------------------------------------------------
ODDS_COLUMNS = [
    "game_id",
    "commence_time",
    "home_team",
    "away_team",
    "book",
    "home_price",
    "away_price",
]


def parse_odds_response(events: list[dict]) -> pd.DataFrame:
    """Flatten The Odds API h2h response into one row per (game, book)."""
    rows = []
    for ev in events:
        home, away = ev.get("home_team"), ev.get("away_team")
        for bk in ev.get("bookmakers", []):
            for mkt in bk.get("markets", []):
                if mkt.get("key") != "h2h":
                    continue
                prices = {o["name"]: o["price"] for o in mkt.get("outcomes", [])}
                if home in prices and away in prices:
                    rows.append(
                        {
                            "game_id": ev["id"],
                            "commence_time": pd.to_datetime(ev["commence_time"], utc=True),
                            "home_team": home,
                            "away_team": away,
                            "book": bk["key"],
                            "home_price": float(prices[home]),
                            "away_price": float(prices[away]),
                        }
                    )
    return pd.DataFrame(rows, columns=ODDS_COLUMNS)


class OddsApiError(RuntimeError):
    """Odds API failure. Messages never include the request URL, which contains the API key."""


class OddsApi:
    """Minimal The Odds API client. Reads the key from ODDS_API_KEY if not given."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout: float = 15.0,
        retries: int = 4,
        backoff: float = 5.0,
    ):
        self.api_key = api_key or os.environ.get("ODDS_API_KEY")
        if not self.api_key:
            raise ValueError("No Odds API key: pass api_key or set the ODDS_API_KEY env variable")
        self.timeout = timeout
        self.retries = retries
        self.backoff = backoff
        self.requests_remaining: Optional[str] = None

    def fetch_h2h(
        self,
        women: bool = False,
        regions: str = "us",
        bookmakers: Optional[str] = None,
    ) -> pd.DataFrame:
        """Fetch current moneylines (American odds) for upcoming games."""
        params = {
            "apiKey": self.api_key,
            "markets": "h2h",
            "oddsFormat": "american",
            "dateFormat": "iso",
        }
        if bookmakers:
            params["bookmakers"] = bookmakers
        else:
            params["regions"] = regions
        url = f"{ODDS_API_BASE}/sports/{SPORT_KEYS[women]}/odds"
        for attempt in range(self.retries):
            try:
                resp = requests.get(url, params=params, timeout=self.timeout)
            except requests.RequestException as e:
                # Network trouble (e.g. a Mac waking from sleep): retry with backoff. Exception
                # text embeds the URL and thus the API key, so only the type is surfaced.
                failure = f"network error ({type(e).__name__})"
            else:
                if resp.status_code < 400:
                    break
                failure = f"HTTP {resp.status_code}"
                if resp.status_code < 500 and resp.status_code != 429:
                    raise OddsApiError(f"Odds API request failed: {failure}") from None
            if attempt == self.retries - 1:
                raise OddsApiError(
                    f"Odds API request failed after {self.retries} attempts: {failure}"
                ) from None
            time.sleep(self.backoff * 3**attempt)
        self.requests_remaining = resp.headers.get("x-requests-remaining")
        return parse_odds_response(resp.json())


# --------------------------------------------------------------------------------------
# Edge finding
# --------------------------------------------------------------------------------------
def consensus_home_prob(odds: pd.DataFrame) -> pd.Series:
    """No-vig home win probability per game_id, averaged across books."""
    fair_home = odds.apply(
        lambda r: devig_two_way(
            american_to_prob(r["home_price"]), american_to_prob(r["away_price"])
        )[0],
        axis=1,
    )
    return fair_home.groupby(odds["game_id"]).mean()


EDGE_COLUMNS = [
    "game_id",
    "commence_time",
    "side",
    "odds_team",
    "wn_team",
    "opponent",
    "book",
    "price",
    "model_prob",
    "market_prob",
    "blended_prob",
    "ev",
    "stake_frac",
]


def find_edges(
    matchups: pd.DataFrame,
    odds: pd.DataFrame,
    model_weight: float = 0.5,
    min_ev: float = 0.0,
    kelly_fraction: float = 0.25,
    max_stake_fraction: float = 0.05,
    name_overrides: Optional[dict[str, str]] = None,
    game_date: Optional[str] = None,
    max_divergence: float = 0.10,
    max_ev: float = 0.25,
) -> pd.DataFrame:
    """
    Compare Elo win probabilities to sportsbook moneylines and return +EV bets.

    Args:
        matchups: `Matchups.matchups` with elo columns (team1 = away, team2 = home).
        odds: Output of `OddsApi.fetch_h2h` (one row per game and book).
        model_weight: Weight on the Elo probability when blending with the no-vig market
            consensus (1.0 = trust Elo fully, 0.0 = trust the market fully). Elo alone is far
            noisier than a sharp closing line, so the default shrinks hard toward the market.
        min_ev: Minimum expected profit per unit staked to report a bet.
        kelly_fraction: Fraction of full Kelly to stake.
        max_stake_fraction: Cap on any single stake as a share of bankroll.
        name_overrides: Extra {odds name: Warren Nolan name} fixes.
        game_date: Only consider games on this US/Eastern calendar date (YYYY-MM-DD).
        max_divergence: Skip bets where the Elo and no-vig market probabilities differ by more
            than this. Gaps that large (e.g. Elo 25% vs a 5% market) are almost always Elo error
            rather than real edge, since early-season Elo is poorly calibrated for mismatches.
        max_ev: Skip bets with expected profit per unit above this. A +25% edge on a college
            basketball moneyline is almost never real, and this catches longshots where a small
            absolute probability gap is still a huge relative one.

    Returns:
        DataFrame of bets sorted by EV, one row per bet, priced at the best book available.
    """
    if not 0.0 <= model_weight <= 1.0:
        raise ValueError("model_weight must be between 0 and 1")
    if odds.empty or matchups.empty:
        return pd.DataFrame(columns=EDGE_COLUMNS)
    if "elo_prob2" not in matchups.columns:
        raise ValueError("matchups needs Elo probabilities (call Matchups.add_elos())")

    odds = odds.dropna(subset=["home_price", "away_price"]).copy()
    if game_date is not None:
        local = odds["commence_time"].dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d")
        odds = odds[local == pd.to_datetime(game_date).strftime("%Y-%m-%d")]
    if odds.empty:
        return pd.DataFrame(columns=EDGE_COLUMNS)

    wn_names = sorted(set(matchups["team1"]) | set(matchups["team2"]))
    mapping, unmatched = map_team_names(
        list(set(odds["home_team"]) | set(odds["away_team"])), wn_names, name_overrides
    )
    if unmatched:
        logging.warning("Unmatched team names (add to name_overrides): %s", ", ".join(unmatched))

    market_home = consensus_home_prob(odds)
    rows = []
    for gid, g in odds.groupby("game_id"):
        home, away = g["home_team"].iloc[0], g["away_team"].iloc[0]
        home_wn, away_wn = mapping.get(home), mapping.get(away)
        if home_wn is None or away_wn is None:
            continue
        m = matchups[(matchups["team1"] == away_wn) & (matchups["team2"] == home_wn)]
        if m.empty:
            continue
        sides = {
            "home": (home, home_wn, away, float(m.iloc[0]["elo_prob2"]), market_home[gid]),
            "away": (away, away_wn, home, float(m.iloc[0]["elo_prob1"]), 1 - market_home[gid]),
        }
        for side, (team, wn_team, opp, model_p, market_p) in sides.items():
            decimals = g[f"{side}_price"].map(american_to_decimal)
            best = decimals.idxmax()
            if abs(model_p - market_p) > max_divergence:
                continue
            blended = model_weight * model_p + (1 - model_weight) * market_p
            dec = float(decimals[best])
            ev = expected_value(blended, dec)
            if ev < min_ev or ev <= 0 or ev > max_ev:
                continue
            rows.append(
                {
                    "game_id": gid,
                    "commence_time": g["commence_time"].iloc[0],
                    "side": side,
                    "odds_team": team,
                    "wn_team": wn_team,
                    "opponent": opp,
                    "book": g.loc[best, "book"],
                    "price": float(g.loc[best, f"{side}_price"]),
                    "model_prob": model_p,
                    "market_prob": market_p,
                    "blended_prob": blended,
                    "ev": ev,
                    "stake_frac": kelly_stake(blended, dec, kelly_fraction, max_stake_fraction),
                }
            )
    out = pd.DataFrame(rows, columns=EDGE_COLUMNS)
    return out.sort_values("ev", ascending=False).reset_index(drop=True)


SNAPSHOT_COLUMNS = [
    "snapshot_at",
    "game_id",
    "commence_time",
    "home_team",
    "away_team",
    "model_prob_home",
    "market_prob_home",
    "closing_prob_home",
    "closing_at",
    "home_won",
]


def game_snapshot(
    matchups: pd.DataFrame,
    odds: pd.DataFrame,
    game_date: Optional[str] = None,
    name_overrides: Optional[dict[str, str]] = None,
) -> pd.DataFrame:
    """
    Elo vs no-vig market home-win probability for EVERY matched game, not just flagged bets.
    Accumulated over a season, this is the dataset for calibrating the Elo scale against the
    market (and for seeing how often Elo disagrees with it).
    """
    odds = odds.dropna(subset=["home_price", "away_price"])
    if game_date is not None and not odds.empty:
        local = odds["commence_time"].dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d")
        odds = odds[local == pd.to_datetime(game_date).strftime("%Y-%m-%d")]
    if odds.empty or matchups.empty or "elo_prob2" not in matchups.columns:
        return pd.DataFrame(columns=SNAPSHOT_COLUMNS)
    wn_names = sorted(set(matchups["team1"]) | set(matchups["team2"]))
    mapping, _ = map_team_names(
        list(set(odds["home_team"]) | set(odds["away_team"])), wn_names, name_overrides
    )
    market_home = consensus_home_prob(odds)
    now = datetime.now(timezone.utc).isoformat()
    rows = []
    for gid, g in odds.groupby("game_id"):
        home, away = g["home_team"].iloc[0], g["away_team"].iloc[0]
        if home not in mapping or away not in mapping:
            continue
        m = matchups[(matchups["team1"] == mapping[away]) & (matchups["team2"] == mapping[home])]
        if m.empty:
            continue
        rows.append(
            {
                "snapshot_at": now,
                "game_id": gid,
                "commence_time": g["commence_time"].iloc[0].isoformat(),
                "home_team": home,
                "away_team": away,
                "model_prob_home": float(m.iloc[0]["elo_prob2"]),
                "market_prob_home": float(market_home[gid]),
            }
        )
    return pd.DataFrame(rows, columns=SNAPSHOT_COLUMNS)


def _read_snapshots(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=SNAPSHOT_COLUMNS)
    df = pd.read_csv(path).reindex(columns=SNAPSHOT_COLUMNS)
    df["closing_at"] = df["closing_at"].astype(object)
    return df


def append_snapshot(path: str, snapshot: pd.DataFrame) -> int:
    """
    Add new games to a snapshot CSV, one row per game. Re-running never clobbers rows that
    already have closing lines or results. Returns the number of games in the file.
    """
    if snapshot.empty:
        return 0
    old = _read_snapshots(path)
    locked = old[old["closing_prob_home"].notna() | old["home_won"].notna()]
    frames = [
        old[~old["game_id"].isin(snapshot["game_id"])],
        locked,
        snapshot[~snapshot["game_id"].isin(locked["game_id"])],
    ]
    combined = pd.concat([f for f in frames if not f.empty]).drop_duplicates("game_id")
    combined.reindex(columns=SNAPSHOT_COLUMNS).to_csv(path, index=False)
    return len(combined)


def _tipping_soon(df: pd.DataFrame, now_ts: pd.Timestamp, window_minutes: float) -> pd.Series:
    tip = pd.to_datetime(df["commence_time"], utc=True)
    return (tip > now_ts) & (tip <= now_ts + pd.Timedelta(minutes=window_minutes))


def snapshot_games_due(
    path: str, window_minutes: float = 35, now: Optional[datetime] = None
) -> int:
    """Number of snapshot games tipping off within the window (cheap, local, no API call)."""
    df = _read_snapshots(path)
    if df.empty:
        return 0
    return int(_tipping_soon(df, BetLog._utc(now), window_minutes).sum())


def update_snapshot_closing(
    path: str,
    odds: pd.DataFrame,
    now: Optional[datetime] = None,
    window_minutes: float = 35,
) -> int:
    """
    Record the latest pre-game no-vig market probability for every snapshot game tipping within
    the window. Later runs overwrite earlier ones, so the final pre-tip value is the closing line.
    """
    df = _read_snapshots(path)
    odds = odds.dropna(subset=["home_price", "away_price"])
    if df.empty or odds.empty:
        return 0
    now_ts = BetLog._utc(now)
    market_home = consensus_home_prob(odds)
    tips = odds.drop_duplicates("game_id").set_index("game_id")["commence_time"]
    updated = 0
    for i, row in df[_tipping_soon(df, now_ts, window_minutes)].iterrows():
        gid = row["game_id"]
        if gid not in market_home.index or now_ts >= tips[gid]:
            continue
        df.loc[i, "closing_prob_home"] = market_home[gid]
        df.loc[i, "closing_at"] = now_ts.isoformat()
        updated += 1
    if updated:
        df.to_csv(path, index=False)
    return updated


def _default_matchups_fetcher(women: bool):
    from bigdance.wn_cbb_scraper import Matchups

    def fetch(date: str) -> pd.DataFrame:
        return Matchups(date, women=women, elos=False).matchups

    return fetch


def _team_won(
    matchups: pd.DataFrame,
    team: str,
    opponent: str,
    overrides: Optional[dict[str, str]] = None,
) -> Optional[bool]:
    """
    Did `team` beat `opponent` per Warren Nolan final scores? None if the game can't be found
    or has no score yet (postponed, or not updated).
    """
    wn_names = sorted(set(matchups["team1"]) | set(matchups["team2"]))
    mapping, _ = map_team_names([team, opponent], wn_names, overrides)
    mine, theirs = mapping.get(team), mapping.get(opponent)
    if mine is None or theirs is None:
        return None
    row = matchups[
        ((matchups["team1"] == mine) & (matchups["team2"] == theirs))
        | ((matchups["team1"] == theirs) & (matchups["team2"] == mine))
    ]
    if row.empty:
        return None
    r = row.iloc[0]
    if r["score1"] == 0 and r["score2"] == 0:
        return None
    my_score = r["score1"] if r["team1"] == mine else r["score2"]
    their_score = r["score2"] if r["team1"] == mine else r["score1"]
    return bool(my_score > their_score)


def settle_snapshots(
    path: str,
    now: Optional[datetime] = None,
    min_hours_after_tip: float = 4.0,
    matchups_for_date=None,
    women: bool = False,
    name_overrides: Optional[dict[str, str]] = None,
) -> int:
    """Fill in `home_won` for snapshot games that have finished. Returns games settled."""
    df = _read_snapshots(path)
    if df.empty:
        return 0
    now_ts = BetLog._utc(now)
    tip = pd.to_datetime(df["commence_time"], utc=True)
    due = df["home_won"].isna() & (tip <= now_ts - pd.Timedelta(hours=min_hours_after_tip))
    if not due.any():
        return 0
    fetch = matchups_for_date or _default_matchups_fetcher(women)
    dates = tip[due].dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d")
    settled = 0
    for date, idx in dates.groupby(dates).groups.items():
        m = fetch(date)
        if m.empty:
            continue
        for i in idx:
            won = _team_won(m, df.loc[i, "home_team"], df.loc[i, "away_team"], name_overrides)
            if won is not None:
                df.loc[i, "home_won"] = float(won)
                settled += 1
    if settled:
        df.to_csv(path, index=False)
    return settled


def read_quota(path: str) -> Optional[int]:
    """Last known Odds API credits remaining (stored beside the snapshot file)."""
    try:
        with open(path + ".quota") as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return None


def write_quota(path: str, remaining) -> None:
    try:
        with open(path + ".quota", "w") as f:
            f.write(str(int(remaining)))
    except (OSError, TypeError, ValueError):
        pass


# --------------------------------------------------------------------------------------
# Bet log and closing line value
# --------------------------------------------------------------------------------------
class BetLog:
    """
    CSV-backed record of bets, used to track closing line value (CLV).

    CLV here is the expected profit of the price you took, measured against the no-vig closing
    probability: `decimal_taken * closing_fair_prob - 1`. Consistently positive CLV is the best
    evidence that a model has a real edge, long before win/loss results become meaningful.
    """

    COLUMNS = [
        "bet_id",
        "logged_at",
        "game_id",
        "commence_time",
        "odds_team",
        "opponent",
        "book",
        "price",
        "model_prob",
        "blended_prob",
        "market_prob",
        "stake",
        "closing_fair_prob",
        "clv",
        "result",
        "profit",
    ]

    def __init__(self, path: str):
        self.path = path
        if os.path.exists(path):
            self.df = pd.read_csv(path)
        else:
            self.df = pd.DataFrame(columns=self.COLUMNS)
        self.df["result"] = self.df["result"].astype(object)

    def save(self):
        self.df.to_csv(self.path, index=False)

    def add_edges(self, edges: pd.DataFrame, bankroll: float) -> int:
        """Log recommended bets (skipping ones already logged). Returns the number added."""
        existing = set(zip(self.df["game_id"], self.df["odds_team"]))
        new = []
        for _, e in edges.iterrows():
            if (e["game_id"], e["odds_team"]) in existing:
                continue
            new.append(
                {
                    "bet_id": uuid.uuid4().hex[:8],
                    "logged_at": datetime.now(timezone.utc).isoformat(),
                    "game_id": e["game_id"],
                    "commence_time": pd.to_datetime(e["commence_time"], utc=True).isoformat(),
                    "odds_team": e["odds_team"],
                    "opponent": e["opponent"],
                    "book": e["book"],
                    "price": e["price"],
                    "model_prob": e["model_prob"],
                    "blended_prob": e["blended_prob"],
                    "market_prob": e["market_prob"],
                    "stake": round(e["stake_frac"] * bankroll, 2),
                }
            )
        if new:
            added = pd.DataFrame(new).reindex(columns=self.COLUMNS).astype({"result": object})
            self.df = added if self.df.empty else pd.concat([self.df, added], ignore_index=True)
            self.save()
        return len(new)

    @staticmethod
    def _utc(now: Optional[datetime]) -> pd.Timestamp:
        ts = pd.Timestamp(now or datetime.now(timezone.utc))
        return ts.tz_localize("UTC") if ts.tzinfo is None else ts

    def _tipoff_in_window(self, now_ts: pd.Timestamp, window_minutes: float) -> pd.Series:
        """Open bets whose game starts within the next window_minutes."""
        if self.df.empty:
            return pd.Series(dtype=bool)
        tip = pd.to_datetime(self.df["commence_time"], utc=True)
        return (
            self.df["result"].isna()
            & (tip > now_ts)
            & (tip <= now_ts + pd.Timedelta(minutes=window_minutes))
        )

    def due_for_closing(self, window_minutes: float = 20, now: Optional[datetime] = None) -> int:
        """
        Number of open bets tipping off within window_minutes. Cheap and local, so a scheduler
        can call it every few minutes and only spend an API request when it returns > 0.
        """
        return int(self._tipoff_in_window(self._utc(now), window_minutes).sum())

    def update_closing(
        self,
        odds: pd.DataFrame,
        now: Optional[datetime] = None,
        window_minutes: Optional[float] = None,
    ) -> int:
        """
        Refresh closing no-vig probabilities from a fresh odds snapshot. Only games that haven't
        started are updated (and, if window_minutes is given, only those tipping within it), so
        the last pre-game snapshot becomes the closing line. Returns the number of bets updated.
        """
        if self.df.empty or odds.empty:
            return 0
        now_ts = self._utc(now)
        market_home = consensus_home_prob(odds.dropna(subset=["home_price", "away_price"]))
        games = odds.drop_duplicates("game_id").set_index("game_id")
        eligible = (
            self._tipoff_in_window(now_ts, window_minutes)
            if window_minutes is not None
            else self.df["result"].isna()
        )
        updated = 0
        for i, bet in self.df[eligible].iterrows():
            gid = bet["game_id"]
            if gid not in market_home.index or now_ts >= games.loc[gid, "commence_time"]:
                continue
            home_p = market_home[gid]
            fair = home_p if bet["odds_team"] == games.loc[gid, "home_team"] else 1 - home_p
            self.df.loc[i, "closing_fair_prob"] = fair
            self.df.loc[i, "clv"] = expected_value(fair, american_to_decimal(bet["price"]))
            updated += 1
        if updated:
            self.save()
        return updated

    def settle(
        self,
        now: Optional[datetime] = None,
        min_hours_after_tip: float = 4.0,
        matchups_for_date=None,
        women: bool = False,
        name_overrides: Optional[dict[str, str]] = None,
    ) -> int:
        """
        Grade open bets from Warren Nolan final scores. Only bets whose game tipped off at least
        min_hours_after_tip ago are considered (Warren Nolan has no final-vs-live flag), and
        games with no recorded score (postponed, or not yet updated) are left open.
        Returns the number of bets settled.
        """
        if self.df.empty:
            return 0
        now_ts = self._utc(now)
        tip = pd.to_datetime(self.df["commence_time"], utc=True)
        due = self.df["result"].isna() & (tip <= now_ts - pd.Timedelta(hours=min_hours_after_tip))
        if not due.any():
            return 0
        fetch = matchups_for_date or _default_matchups_fetcher(women)
        dates = tip[due].dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d")
        settled = 0
        for date, idx in dates.groupby(dates).groups.items():
            m = fetch(date)
            if m.empty:
                continue
            for i in idx:
                bet = self.df.loc[i]
                won = _team_won(m, bet["odds_team"], bet["opponent"], name_overrides)
                if won is not None:
                    self._settle_row(i, won=won)
                    settled += 1
        if settled:
            self.save()
        return settled

    def _settle_row(self, i, won: bool):
        bet = self.df.loc[i]
        self.df.loc[i, "result"] = "win" if won else "loss"
        self.df.loc[i, "profit"] = (
            bet["stake"] * (american_to_decimal(bet["price"]) - 1) if won else -bet["stake"]
        )

    def record_result(self, bet_id: str, won: bool):
        """Settle a bet as a win or loss and compute its profit."""
        mask = self.df["bet_id"] == bet_id
        if not mask.any():
            raise KeyError(f"No bet with id {bet_id}")
        self._settle_row(self.df.index[mask][0], won)
        self.save()

    def summary(self) -> dict:
        """Aggregate CLV and results."""
        df = self.df
        settled = df[df["result"].notna()]
        clv = pd.to_numeric(df["clv"], errors="coerce").dropna()
        staked = float(settled["stake"].sum())
        return {
            "bets": len(df),
            "bets_with_clv": len(clv),
            "avg_clv": float(clv.mean()) if len(clv) else float("nan"),
            "pct_positive_clv": float((clv > 0).mean()) if len(clv) else float("nan"),
            "settled": len(settled),
            "profit": float(settled["profit"].sum()),
            "roi": float(settled["profit"].sum() / staked) if staked else float("nan"),
        }


# --------------------------------------------------------------------------------------
# Shortlist: a phone-friendly page of upcoming games with prices, fair prices and Elo's view
# --------------------------------------------------------------------------------------
SHORTLIST_COLUMNS = [
    "day",
    "game_id",
    "tip",
    "home",
    "away",
    "home_best_price",
    "home_best_book",
    "away_best_price",
    "away_best_book",
    "home_fair_price",
    "away_fair_price",
    "market_home",
    "elo_home",
    "elo_away",
    "venue_flag",
    "lean_side",
    "lean_price",
    "lean_book",
    "lean_min_price",
    "lean_ev",
]


def build_shortlist(
    matchups: pd.DataFrame,
    odds: pd.DataFrame,
    game_date: Optional[str] = None,
    model_weight: float = 0.5,
    min_ev: float = 0.02,
    max_divergence: float = 0.10,
    max_ev: float = 0.25,
    neutral: Optional[set[str]] = None,
    name_overrides: Optional[dict[str, str]] = None,
    now: Optional[datetime] = None,
) -> pd.DataFrame:
    """
    One row per upcoming game: best price per side, no-vig fair price, Elo's probability, and an
    optional "lean" (same blending and sanity caps as `find_edges`).

    The odds feed has no venue information, but Elo gives the listed home team a home bonus. Games
    involving a team named in `neutral` (Warren Nolan names) are marked neutral-site and use Elo
    without the bonus.
    """
    cols = SHORTLIST_COLUMNS
    odds = odds.dropna(subset=["home_price", "away_price"])
    if game_date is not None and not odds.empty:
        local = odds["commence_time"].dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d")
        odds = odds[local == pd.to_datetime(game_date).strftime("%Y-%m-%d")]
    odds = odds[odds["commence_time"] > BetLog._utc(now)]
    if odds.empty or matchups.empty or "elo1" not in matchups.columns:
        return pd.DataFrame(columns=cols)

    from bigdance.wn_cbb_scraper import elo_prob

    neutral = neutral or set()
    wn_names = sorted(set(matchups["team1"]) | set(matchups["team2"]))
    mapping, unmatched = map_team_names(
        list(set(odds["home_team"]) | set(odds["away_team"])), wn_names, name_overrides
    )
    if unmatched:
        logging.warning("Unmatched team names (add to name_overrides): %s", ", ".join(unmatched))
    market_home = consensus_home_prob(odds)

    rows = []
    for gid, g in odds.groupby("game_id"):
        home_wn, away_wn = mapping.get(g["home_team"].iloc[0]), mapping.get(g["away_team"].iloc[0])
        if home_wn is None or away_wn is None:
            continue
        m = matchups[(matchups["team1"] == away_wn) & (matchups["team2"] == home_wn)]
        if m.empty:
            continue
        r = m.iloc[0]
        mh = float(market_home[gid])
        elo_home = float(r["elo_prob2"])
        elo_neutral = 1 - elo_prob(r["elo1"], r["elo2"], 1.0, 0.0)
        is_neutral = home_wn in neutral or away_wn in neutral
        model_home = elo_neutral if is_neutral else elo_home
        flag = "neutral" if is_neutral else None

        best, fair, side_prob = {}, {}, {"home": model_home, "away": 1 - model_home}
        for side, mkt in (("home", mh), ("away", 1 - mh)):
            decimals = g[f"{side}_price"].map(american_to_decimal)
            idx = decimals.idxmax()
            best[side] = (
                float(g.loc[idx, f"{side}_price"]),
                g.loc[idx, "book"],
                float(decimals[idx]),
            )
            fair[side] = prob_to_american(mkt)

        lean = None
        for side, mkt in (("home", mh), ("away", 1 - mh)):
            blended = model_weight * side_prob[side] + (1 - model_weight) * mkt
            ev = expected_value(blended, best[side][2])
            if (
                min_ev <= ev <= max_ev
                and abs(side_prob[side] - mkt) <= max_divergence
                and (lean is None or ev > lean[1])
            ):
                lean = (side, ev, prob_to_american(blended))
        rows.append(
            {
                "day": game_date or g["commence_time"].iloc[0].strftime("%Y-%m-%d"),
                "game_id": gid,
                "tip": g["commence_time"].iloc[0],
                "home": home_wn,
                "away": away_wn,
                "home_best_price": best["home"][0],
                "home_best_book": best["home"][1],
                "away_best_price": best["away"][0],
                "away_best_book": best["away"][1],
                "home_fair_price": fair["home"],
                "away_fair_price": fair["away"],
                "market_home": mh,
                "elo_home": side_prob["home"],
                "elo_away": side_prob["away"],
                "venue_flag": flag,
                "lean_side": lean[0] if lean else None,
                "lean_price": best[lean[0]][0] if lean else None,
                "lean_book": best[lean[0]][1] if lean else None,
                "lean_min_price": lean[2] if lean else None,
                "lean_ev": lean[1] if lean else None,
            }
        )
    out = pd.DataFrame(rows, columns=cols)
    return out.sort_values(["day", "tip"]).reset_index(drop=True)


def _price(x: float) -> str:
    return f"{x:+.0f}"


def _book(name: str) -> str:
    pretty = {
        "draftkings": "DraftKings",
        "fanduel": "FanDuel",
        "betmgm": "BetMGM",
        "betrivers": "BetRivers",
        "bovada": "Bovada",
        "betonlineag": "BetOnline",
        "lowvig": "LowVig",
        "betus": "BetUS",
        "mybookieag": "MyBookie",
        "williamhill_us": "Caesars",
        "espnbet": "ESPN BET",
        "fanatics": "Fanatics",
        "hardrockbet": "Hard Rock",
    }
    return pretty.get(name, name)


SHORTLIST_CSS = """
:root{--bg:#f4f6f8;--card:#fff;--fg:#14202b;--muted:#5b6b7a;--line:#dde3e9;--accent:#0b6bcb;
--accent-bg:#e7f1fc;--warn:#a85a00;--warn-bg:#fdf0dc;--font:'Inter',system-ui,sans-serif;
--display:'Barlow Condensed','Arial Narrow',sans-serif}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#0e151c;--card:#16202a;
--fg:#e7edf3;--muted:#93a3b3;--line:#263341;--accent:#6cb0ff;--accent-bg:#14304d;--warn:#f0b35c;
--warn-bg:#3a2a10;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#0e151c;--card:#16202a;--fg:#e7edf3;--muted:#93a3b3;--line:#263341;
--accent:#6cb0ff;--accent-bg:#14304d;--warn:#f0b35c;--warn-bg:#3a2a10;color-scheme:dark}
body{background:var(--bg);color:var(--fg);font:15px/1.45 var(--font);
padding-block:16px 32px;padding-left:max(16px,env(safe-area-inset-left,0px));
padding-right:max(16px,env(safe-area-inset-right,0px))}
main,header.top,footer{max-width:640px;margin-inline:auto}
h1{font:700 2rem/1 var(--display);letter-spacing:.01em;margin:0;text-wrap:balance}
.sub{color:var(--muted);font-size:.85rem;margin:.35rem 0 0}
nav.days{display:flex;flex-wrap:wrap;gap:8px;margin:14px 0 4px}
nav.days a{padding:6px 12px;border:1px solid var(--line);border-radius:999px;color:var(--fg);
text-decoration:none;font-weight:600;font-size:.85rem;background:var(--card)}
.filter{display:flex;align-items:center;gap:8px;margin:12px auto;font-size:.9rem;max-width:640px}
.filter input{width:18px;height:18px}
details{max-width:640px;margin:10px auto;font-size:.85rem;color:var(--muted)}
details summary{cursor:pointer;font-weight:600;color:var(--fg)}
details p{margin:.5rem 0}
section.day h2{font:700 1.5rem/1.1 var(--display);margin:22px 0 10px;letter-spacing:.01em}
.game{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px;
margin-bottom:10px;break-inside:avoid}
.game.has-lean{background:var(--accent-bg);border-color:var(--accent)}
.meta{display:flex;align-items:center;gap:8px;flex-wrap:wrap}
.tip{font:700 1.15rem/1 var(--display);letter-spacing:.02em}
.tag{font-size:.68rem;font-weight:700;letter-spacing:.06em;text-transform:uppercase;
padding:3px 7px;border-radius:5px}
.tag.lean{background:var(--accent);color:var(--card)}
.tag.warn{background:var(--warn-bg);color:var(--warn)}
.tag.info{background:var(--line);color:var(--muted)}
.game h3{font:600 1rem/1.25 var(--font);margin:8px 0}
.game h3 span{color:var(--muted);font-weight:400}
.tbl{overflow-x:auto}
table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}
th{font-size:.68rem;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);
text-align:right;font-weight:600;padding:0 0 4px 8px}
th:first-child,td:first-child{text-align:left;padding-left:0}
td{padding:5px 0 5px 8px;text-align:right;border-top:1px solid var(--line);white-space:nowrap}
td.team{white-space:normal;font-weight:600;min-width:0}
td small{display:block;color:var(--muted);font-size:.72rem}
tr.pick td{font-weight:700}
.leanline{margin:8px 0 0;font-size:.9rem}
.leanline b{font-weight:700}
.empty{color:var(--muted);padding:20px 0}
.banner{max-width:640px;margin:10px auto;padding:10px 12px;border-radius:8px;font-size:.85rem;
background:var(--warn-bg);color:var(--warn)}
footer{color:var(--muted);font-size:.78rem;margin-top:26px}
.filter:has(input:checked) ~ main .game:not(.has-lean){display:none}
@page{size:4.6in 8.6in;margin:.3in}
@media print{body{background:#fff;color:#000;padding:0;font-size:13px}.filter,nav.days,details{display:none}
.game{border-color:#999;background:#fff}.game.has-lean{background:#eef5fd}}
"""


def render_shortlist(
    df: pd.DataFrame,
    generated_at: Optional[datetime] = None,
    books: Optional[list[str]] = None,
    tz: str = "America/Los_Angeles",
) -> str:
    """Render the shortlist as an HTML fragment (title, style, content; no html/head/body)."""
    esc = html.escape
    now = pd.Timestamp(generated_at or datetime.now(timezone.utc)).tz_convert(tz)
    n_leans = int(df["lean_side"].notna().sum()) if not df.empty else 0
    days = list(dict.fromkeys(df["day"])) if not df.empty else []
    label = lambda d: pd.to_datetime(d).strftime("%a %b %-d")  # noqa: E731

    parts = [
        "<title>College Hoops Shortlist</title>",
        '<meta name="claude:viewport-fit-sides" content="cover">',
        '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:'
        'wght@600;700&family=Inter:wght@400;600;700&display=swap">',
        f"<style>{SHORTLIST_CSS}</style>",
        '<header class="top"><h1>College Hoops Shortlist</h1>',
        f'<p class="sub">Prices as of {esc(now.strftime("%a %-I:%M %p"))} PT &middot; '
        f"{len(df)} games &middot; {n_leans} leans</p>",
    ]
    if days:
        parts.append(
            '<nav class="days">'
            + "".join(f'<a href="#day-{esc(d)}">{esc(label(d))}</a>' for d in days)
            + "</nav>"
        )
    parts += [
        "</header>",
        '<label class="filter"><input type="checkbox" id="leans-only">Leans only</label>',
        "<details><summary>How to read this</summary>"
        "<p><b>Best price</b> is the highest payout across books right now. <b>Fair price</b> is the "
        "average of the books with the vig removed, so it is what the odds would be with no house "
        "cut. Compare it to the board: the further your price is below fair, the more you pay.</p>"
        "<p><b>Elo</b> is Warren Nolan's rating-based win chance. A <b>lean</b> means a 50/50 mix of Elo "
        "and the market, at the best price, beats the odds by at least 2%. The line shows the "
        "worst price you would still take.</p>"
        "<p>Elo gives the listed home team a home-court bonus. At <b>neutral-site</b> games that "
        "overstates the home team, so those are marked when known.</p></details>",
    ]
    if any(pd.to_datetime(d).month == 11 and pd.to_datetime(d).day >= 22 for d in days):
        parts.append(
            '<p class="banner">Thanksgiving week: many games are tournament games at neutral sites. '
            "Elo adds a home bonus to the listed home team, so treat any lean with extra suspicion "
            "unless the game is marked Neutral site.</p>"
        )
    parts.append("<main>")
    if df.empty:
        parts.append('<p class="empty">No upcoming games with prices yet. Check back later.</p>')
    for d in days:
        sub = df[df["day"] == d]
        parts.append(
            f'<section class="day" id="day-{esc(d)}"><h2>{esc(label(d))} '
            f'<span style="color:var(--muted)">&middot; {len(sub)} games</span></h2>'
        )
        for _, r in sub.iterrows():
            tip = r["tip"].tz_convert(tz).strftime("%-I:%M %p")
            lean = r["lean_side"]
            tags = ""
            if lean:
                tags += '<span class="tag lean">Lean</span>'
            if r["venue_flag"] == "neutral":
                tags += '<span class="tag info">Neutral site</span>'
            rows = ""
            for side, team in (("away", r["away"]), ("home", r["home"])):
                pick = ' class="pick"' if lean == side else ""
                rows += (
                    f'<tr{pick}><td class="team">{esc(team)}</td>'
                    f"<td>{_price(r[f'{side}_best_price'])}"
                    f"<small>{esc(_book(r[f'{side}_best_book']))}</small></td>"
                    f"<td>{_price(r[f'{side}_fair_price'])}</td>"
                    f"<td>{r[f'elo_{side}'] * 100:.0f}%</td></tr>"
                )
            leanline = ""
            if lean:
                team = r[lean]
                leanline = (
                    f'<p class="leanline">Lean: <b>{esc(team)} {_price(r["lean_price"])}</b> at '
                    f"{esc(_book(r['lean_book']))}. Pass if it is worse than "
                    f"<b>{_price(r['lean_min_price'])}</b>.</p>"
                )
            parts.append(
                f'<article class="game{" has-lean" if lean else ""}">'
                f'<div class="meta"><span class="tip">{esc(tip)}</span>{tags}</div>'
                f"<h3>{esc(r['away'])} <span>at</span> {esc(r['home'])}</h3>"
                '<div class="tbl"><table><thead><tr><th>Team</th><th>Best</th><th>Fair</th>'
                f"<th>Elo</th></tr></thead><tbody>{rows}</tbody></table></div>{leanline}</article>"
            )
        parts.append("</section>")
    book_txt = ", ".join(_book(b) for b in sorted(books or []))
    parts.append(
        "</main><footer><p>Prices: The Odds API"
        + (f" ({esc(book_txt)})" if book_txt else "")
        + ". Ratings: Warren Nolan Elo. Lines move and books differ, so check the board before you "
        "bet. In our tracking so far, Elo has not beaten the market, so treat leans as "
        "entertainment, not advice. Bet only what you would spend on a night out.</p></footer>"
    )
    return "\n".join(parts)


def standalone_html(fragment: str) -> str:
    """Wrap the fragment as a complete document (for opening locally or printing to PDF)."""
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">'
        f"</head><body>{fragment}</body></html>"
    )


def html_to_pdf(html_path: str, pdf_path: str) -> bool:
    """Print an HTML file to PDF with headless Chrome. Returns False if Chrome isn't found."""
    chrome = shutil.which("google-chrome") or shutil.which("chromium")
    mac = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    chrome = chrome or (mac if os.path.exists(mac) else None)
    if not chrome:
        return False
    subprocess.run(
        [
            chrome,
            "--headless=new",
            "--disable-gpu",
            "--no-pdf-header-footer",
            "--virtual-time-budget=4000",
            f"--print-to-pdf={pdf_path}",
            "file://" + os.path.abspath(html_path),
        ],
        check=True,
        capture_output=True,
        timeout=90,
    )
    return os.path.exists(pdf_path)


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="bigdance edges",
        description="Compare Warren Nolan Elo probabilities to sportsbook moneylines",
    )
    parser.add_argument("--date", help="Game date YYYY-MM-DD (default: today)")
    parser.add_argument("--women", action="store_true", help="Women's basketball")
    parser.add_argument("--api-key", help="The Odds API key (default: $ODDS_API_KEY)")
    parser.add_argument("--regions", default="us", help="Odds API regions (default: us)")
    parser.add_argument("--bookmakers", help="Comma-separated bookmaker keys (overrides regions)")
    parser.add_argument(
        "--model-weight",
        type=float,
        default=0.5,
        help="Weight on Elo vs the no-vig market, 0-1 (default: 0.5)",
    )
    parser.add_argument("--min-ev", type=float, default=0.02, help="Minimum EV (default: 0.02)")
    parser.add_argument(
        "--max-divergence",
        type=float,
        default=0.10,
        help="Skip bets where Elo and the market differ by more than this (default: 0.10)",
    )
    parser.add_argument(
        "--max-ev", type=float, default=0.25, help="Skip bets above this EV (default: 0.25)"
    )
    parser.add_argument("--bankroll", type=float, default=1000.0, help="Bankroll for sizing")
    parser.add_argument("--kelly", type=float, default=0.25, help="Kelly fraction (default: 0.25)")
    parser.add_argument("--log", metavar="CSV", help="Append the printed bets to this bet log")
    parser.add_argument(
        "--update-closing",
        metavar="CSV",
        help="Refresh closing lines for bets in this log (skips the API call unless a logged "
        "bet tips off within --closing-window minutes)",
    )
    parser.add_argument(
        "--closing-window",
        type=float,
        default=20.0,
        help="Minutes before tip-off to capture closing lines; keep above the scheduler "
        "interval (default: 20)",
    )
    parser.add_argument(
        "--credit-reserve",
        type=int,
        default=100,
        help="With --snapshot, capture closing lines for ALL games only while at least this many "
        "Odds API credits remain; below it, only logged bets are tracked (default: 100)",
    )
    parser.add_argument(
        "--shortlist",
        metavar="DIR",
        help="Write a phone-friendly shortlist of upcoming games (shortlist.html, "
        "shortlist_artifact.html, and shortlist.pdf with --pdf) into DIR and exit",
    )
    parser.add_argument(
        "--dates", help="Comma-separated game dates for --shortlist (default: --date or today)"
    )
    parser.add_argument(
        "--neutral",
        help="Comma-separated Warren Nolan team names playing at neutral sites (no home bonus)",
    )
    parser.add_argument(
        "--pdf", action="store_true", help="Also write shortlist.pdf (needs Chrome)"
    )
    parser.add_argument("--settle", metavar="CSV", help="Grade finished bets from final scores")
    parser.add_argument(
        "--snapshot",
        metavar="CSV",
        help="Track every game, not just bets: record Elo vs market probabilities (with the "
        "picks run), closing lines (--update-closing) and results (--settle)",
    )
    parser.add_argument("--summary", metavar="CSV", help="Print CLV/results summary and exit")
    args = parser.parse_args(argv)

    if args.summary:
        for k, v in BetLog(args.summary).summary().items():
            print(f"{k:>18}: {v:.4f}" if isinstance(v, float) else f"{k:>18}: {v}")
        return 0

    if args.settle:
        msg = f"Settled {BetLog(args.settle).settle(women=args.women)} bets"
        if args.snapshot:
            msg += f", {settle_snapshots(args.snapshot, women=args.women)} snapshot games"
        print(msg)
        return 0

    closing_log = BetLog(args.update_closing) if args.update_closing else None
    track_games = False
    if closing_log is not None:
        due = closing_log.due_for_closing(args.closing_window)
        if args.snapshot:
            games_due = snapshot_games_due(args.snapshot, args.closing_window)
            remaining = read_quota(args.snapshot)
            track_games = games_due > 0 and (remaining is None or remaining >= args.credit_reserve)
            if games_due and not track_games:
                print(
                    f"Credit reserve reached ({remaining} < {args.credit_reserve}): "
                    "tracking logged bets only"
                )
            due += games_due if track_games else 0
        if not due:
            print(f"No games tip off within {args.closing_window:g} min; skipping API call")
            return 0

    api = OddsApi(args.api_key)
    try:
        odds = api.fetch_h2h(women=args.women, regions=args.regions, bookmakers=args.bookmakers)
    except OddsApiError as e:
        print(e, file=sys.stderr)
        return 1
    if args.snapshot and api.requests_remaining is not None:
        write_quota(args.snapshot, api.requests_remaining)

    if closing_log is not None:
        n = closing_log.update_closing(odds, window_minutes=args.closing_window)
        msg = f"Updated closing lines for {n} bets"
        if track_games:
            msg += f" and {update_snapshot_closing(args.snapshot, odds, window_minutes=args.closing_window)} games"
        print(f"{msg} (API requests left: {api.requests_remaining})")
        return 0

    from bigdance.wn_cbb_scraper import Matchups

    if args.shortlist:
        dates = [d.strip() for d in (args.dates or args.date or "").split(",") if d.strip()]
        dates = dates or [datetime.now().strftime("%Y-%m-%d")]
        neutral = {n.strip() for n in (args.neutral or "").split(",") if n.strip()}
        frames = [
            build_shortlist(
                Matchups(d, women=args.women).matchups,
                odds,
                game_date=d,
                model_weight=args.model_weight,
                min_ev=args.min_ev,
                max_divergence=args.max_divergence,
                max_ev=args.max_ev,
                neutral=neutral,
            )
            for d in dates
        ]
        frames = [f for f in frames if not f.empty]
        df = (
            pd.concat(frames, ignore_index=True)
            if frames
            else pd.DataFrame(columns=SHORTLIST_COLUMNS)
        )
        fragment = render_shortlist(df, books=sorted(odds["book"].unique()))
        os.makedirs(args.shortlist, exist_ok=True)
        page = os.path.join(args.shortlist, "shortlist.html")
        with open(page, "w") as f:
            f.write(standalone_html(fragment))
        with open(os.path.join(args.shortlist, "shortlist_artifact.html"), "w") as f:
            f.write(fragment)
        print(f"{len(df)} games, {int(df['lean_side'].notna().sum())} leans -> {page}")
        if args.pdf:
            pdf = os.path.join(args.shortlist, "shortlist.pdf")
            print(f"PDF -> {pdf}" if html_to_pdf(page, pdf) else "No Chrome found; skipped PDF")
        return 0

    date = args.date or datetime.now().strftime("%Y-%m-%d")
    matchups = Matchups(date, women=args.women).matchups
    if args.snapshot:
        n = append_snapshot(args.snapshot, game_snapshot(matchups, odds, game_date=date))
        print(f"Snapshot file now holds {n} games ({args.snapshot})")
    edges = find_edges(
        matchups,
        odds,
        model_weight=args.model_weight,
        min_ev=args.min_ev,
        kelly_fraction=args.kelly,
        game_date=date,
        max_divergence=args.max_divergence,
        max_ev=args.max_ev,
    )
    if edges.empty:
        print("No bets meet the EV threshold.")
        return 0

    show = edges.assign(stake=(edges["stake_frac"] * args.bankroll).round(2))
    print(
        show[
            ["odds_team", "opponent", "book", "price", "model_prob", "market_prob", "ev", "stake"]
        ].to_string(index=False, float_format=lambda x: f"{x:.3f}")
    )
    if args.log:
        print(f"Logged {BetLog(args.log).add_edges(edges, args.bankroll)} new bets to {args.log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
