"""
March Madness Pool Tracker — live pool standings and game importance analysis.

Uses the ESPN Gambit API (JSON) to fetch pool and bracket data, then runs
Monte Carlo simulations via the bigdance package to project win probabilities
and analyze which upcoming games matter most for each entry.

Usage:
    streamlit run app/tracker.py
"""

import copy
import logging

import pandas as pd
import streamlit as st

from bigdance.cbb_brackets import Pool
from bigdance.espn_tc_scraper import ESPNApi, GameImportanceAnalyzer

logger = logging.getLogger(__name__)

st.set_page_config(page_title="bigdance pool tracker", layout="wide")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PERIOD_LABELS = {
    1: "Round of 64",
    2: "Round of 32",
    3: "Sweet 16",
    4: "Elite 8",
    5: "Final Four",
    6: "National Championship",
}

# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------


@st.cache_resource
def load_api(women: bool = False) -> ESPNApi:
    """Create a cached ESPNApi instance."""
    return ESPNApi(women=women)


@st.cache_data(ttl=300)
def load_pool_data(group_id: str, women: bool = False) -> tuple[dict, dict, list[dict]]:
    """Fetch all pool data from the ESPN API."""
    api = load_api(women)
    challenge = api.fetch_challenge()
    group = api.fetch_group(group_id)
    entry_data_list = []
    for entry in group.get("entries", []):
        entry_data_list.append(api.fetch_entry(entry["id"]))
    return challenge, group, entry_data_list


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


def run_pool_simulation(
    challenge: dict,
    entry_data_list: list[dict],
    women: bool,
    num_sims: int,
) -> tuple[Pool, pd.DataFrame]:
    """Build Pool from API data and run simulation."""
    api = load_api(women)

    actual_bracket = api.build_actual_bracket(challenge)
    for game in actual_bracket.games:
        game.upset_factor = 0.25

    teams = actual_bracket.teams
    pool = Pool(actual_bracket)

    for entry_data in entry_data_list:
        entry_name = entry_data.get("name", "Unknown")
        bracket = api.build_entry_bracket(entry_data, challenge, teams)
        if bracket:
            pool.add_entry(entry_name, bracket, False)

    # Deep copy the pool before simulation since simulate_pool mutates
    # the actual bracket's results (fills in all rounds). We need the
    # original state for importance analysis.
    pool_for_importance = copy.deepcopy(pool)

    fixed_winners = copy.deepcopy(actual_bracket.results)
    results_df = pool.simulate_pool(num_sims=num_sims, fixed_winners=fixed_winners)

    return pool_for_importance, results_df


def run_importance_analysis(pool: Pool, num_sims: int, current_round: str) -> list[dict]:
    """Run game importance analysis on the pool."""
    analyzer = GameImportanceAnalyzer(pool)
    return analyzer.analyze_win_importance(current_round=current_round, num_sims=num_sims)


# ---------------------------------------------------------------------------
# UI Components
# ---------------------------------------------------------------------------


def render_standings(results_df: pd.DataFrame, my_bracket: str = ""):
    """Render the pool standings table."""
    display_df = results_df[["name", "avg_score", "std_score", "win_prob"]].copy()
    display_df.columns = ["Entry", "Avg Score", "Std Score", "Win Prob"]
    display_df["Avg Score"] = display_df["Avg Score"].apply(lambda x: f"{x:.1f}")
    display_df["Std Score"] = display_df["Std Score"].apply(lambda x: f"{x:.1f}")

    if my_bracket:

        def highlight_row(row):
            if row["Entry"] == my_bracket:
                return ["background-color: #e6f3ff; font-weight: bold"] * len(row)
            return [""] * len(row)

        styled = display_df.style.apply(highlight_row, axis=1)
        st.dataframe(styled, width="stretch", hide_index=True)
    else:
        st.dataframe(display_df, width="stretch", hide_index=True)


def render_importance(importance_data: list[dict], my_bracket: str = ""):
    """Render the game importance analysis."""
    if not importance_data:
        st.info("No upcoming games to analyze in the current round.")
        return

    sorted_games = sorted(importance_data, key=lambda g: g["max_impact"], reverse=True)

    for i, game in enumerate(sorted_games):
        team1 = game["team1"]
        team2 = game["team2"]
        matchup_label = f"({team1['seed']}) {team1['name']}  vs  ({team2['seed']}) {team2['name']}"

        with st.expander(
            f"**Game {i + 1}: {matchup_label}** — {game['region']}",
            expanded=(i < 3),
        ):
            col1, col2 = st.columns(2)
            col1.metric("Max Impact", f"{game['max_impact'] * 100:.1f}%")
            col2.metric("Avg Impact", f"{game['avg_impact'] * 100:.1f}%")

            impact_records = game.get("all_entries_impact", [])
            if impact_records:
                impact_df = pd.DataFrame(impact_records)
                impact_df = impact_df.rename(
                    columns={
                        "name": "Entry",
                        "win_pct_team1": f"Win% if {team1['name']}",
                        "win_pct_team2": f"Win% if {team2['name']}",
                        "win_pct_baseline": "Baseline Win%",
                        "impact": "Impact",
                    }
                )

                impact_df = impact_df.sort_values("Impact", ascending=False)

                pct_cols = [c for c in impact_df.columns if "Win%" in c or c == "Impact"]
                col_config = {}
                for col in pct_cols:
                    impact_df[col] = (impact_df[col] * 100).round(1)
                    col_config[col] = st.column_config.NumberColumn(format="%.1f%%")

                if my_bracket and my_bracket in impact_df["Entry"].values:

                    def highlight_entry(row):
                        if row["Entry"] == my_bracket:
                            return ["background-color: #e6f3ff; font-weight: bold"] * len(row)
                        return [""] * len(row)

                    styled = impact_df.style.apply(highlight_entry, axis=1)
                    st.dataframe(styled, column_config=col_config, width="stretch", hide_index=True)
                else:
                    st.dataframe(
                        impact_df, column_config=col_config, width="stretch", hide_index=True
                    )


def render_entry_details(
    entry_data_list: list[dict],
    challenge: dict,
    importance_data: list[dict],
    selected_entry: str,
    women: bool = False,
):
    """Render detailed view for a specific entry."""
    api = load_api(women)
    outcome_map = api._build_outcome_map(challenge)

    entry = next((e for e in entry_data_list if e.get("name") == selected_entry), None)
    if not entry:
        st.warning(f"Entry '{selected_entry}' not found.")
        return

    # Score summary
    score = entry.get("score", {})
    record = score.get("record", {})
    col1, col2, col3 = st.columns(3)
    col1.metric("Score", score.get("overallScore", 0))
    col2.metric("Record", f"{record.get('wins', 0)}W - {record.get('losses', 0)}L")
    col3.metric("Possible Points", score.get("possiblePointsRemaining", "?"))

    st.subheader("Picks by Round")

    picks = entry.get("picks", [])
    picks_by_round: dict[int, list[dict]] = {}
    for pick in picks:
        outcome_id = pick["outcomesPicked"][0]["outcomeId"]
        result = pick["outcomesPicked"][0].get("result", "UNDECIDED")
        period_reached = pick.get("periodReached", 1)
        info = outcome_map.get(outcome_id)
        if not info:
            continue  # Skip later-round picks with unknown outcome IDs

        prop_info = next(
            (p for p in challenge["propositions"] if p["id"] == pick["propositionId"]),
            None,
        )
        base_period = prop_info["scoringPeriodId"] if prop_info else 1

        for period in range(base_period, period_reached + 1):
            if period not in picks_by_round:
                picks_by_round[period] = []
            picks_by_round[period].append(
                {
                    "Team": info.get("name", "Unknown"),
                    "Seed": info.get("seed", 0),
                    "Result": result if period == base_period else "UNDECIDED",
                }
            )

    for period in sorted(picks_by_round.keys()):
        label = PERIOD_LABELS.get(period, f"Round {period}")
        picks_df = pd.DataFrame(picks_by_round[period])

        def color_result(val):
            if val == "CORRECT":
                return "color: green; font-weight: bold"
            elif val == "INCORRECT":
                return "color: red"
            return ""

        st.write(f"**{label}**")
        styled = picks_df.style.map(color_result, subset=["Result"])
        st.dataframe(styled, width="stretch", hide_index=True)

    # Game impact for this entry
    if importance_data:
        st.subheader("How Upcoming Games Affect You")
        for game in importance_data:
            entry_impact = next(
                (e for e in game.get("all_entries_impact", []) if e["name"] == selected_entry),
                None,
            )
            if entry_impact and entry_impact["impact"] > 0.001:
                team1 = game["team1"]
                team2 = game["team2"]
                t1_pct = entry_impact["win_pct_team1"] * 100
                t2_pct = entry_impact["win_pct_team2"] * 100

                if t1_pct > t2_pct:
                    better, worse = team1["name"], team2["name"]
                    better_pct, worse_pct = t1_pct, t2_pct
                else:
                    better, worse = team2["name"], team1["name"]
                    better_pct, worse_pct = t2_pct, t1_pct

                st.write(
                    f"**{team1['name']} vs {team2['name']}**: "
                    f"You want **{better}** ({better_pct:.1f}% win chance) "
                    f"over {worse} ({worse_pct:.1f}%)"
                )


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

st.title("March Madness Pool Tracker")

with st.sidebar:
    st.header("Pool Settings")
    gender = st.radio("Tournament", options=["Men's", "Women's"], horizontal=True)
    women = gender == "Women's"

    pool_id = st.text_input(
        "ESPN Pool ID",
        placeholder="e.g. 3e3a8cfd-b6ef-460c-b096-3044bc76ae15",
        help="Found in the URL of your ESPN Tournament Challenge group page",
    )

    num_sims = st.selectbox("Simulations", options=[500, 1000, 2500], index=1)

    my_bracket = st.text_input(
        "Your bracket name (optional)",
        placeholder="e.g. Firman's BigDance",
        help="Highlights your entry in the results",
    )

    analyze_btn = st.button("Analyze Pool", type="primary", width="stretch")

    st.divider()
    st.caption("Built by [Taylor Firman](https://github.com/tefirman)")

# Main area
if not pool_id:
    st.info(
        "Enter your ESPN pool ID in the sidebar to get started. "
        "You can find it in the URL of your ESPN Tournament Challenge group page: "
        "`/group?id=YOUR_POOL_ID`"
    )
    st.stop()

if analyze_btn:
    with st.spinner("Fetching pool data from ESPN..."):
        try:
            challenge, group, entry_data_list = load_pool_data(pool_id, women)
            st.session_state["challenge"] = challenge
            st.session_state["group"] = group
            st.session_state["entry_data_list"] = entry_data_list
        except Exception as e:
            st.error(f"Failed to fetch pool data: {e}")
            st.stop()

    with st.spinner(f"Running {num_sims} simulations..."):
        pool, results_df = run_pool_simulation(challenge, entry_data_list, women, num_sims)
        st.session_state["pool"] = pool
        st.session_state["results_df"] = results_df

    current_round = pool.actual_results.infer_current_round()
    st.session_state["current_round"] = current_round

    if current_round:
        with st.spinner(f"Analyzing game importance ({current_round})..."):
            importance = run_importance_analysis(pool, min(num_sims, 500), current_round)
            st.session_state["importance"] = importance
    else:
        st.session_state["importance"] = []

# Display results
if "results_df" not in st.session_state:
    st.stop()

results_df = st.session_state["results_df"]
pool = st.session_state["pool"]
challenge = st.session_state["challenge"]
entry_data_list = st.session_state["entry_data_list"]
importance = st.session_state.get("importance", [])
current_round = st.session_state.get("current_round")

entry_names = [e.get("name", "?") for e in entry_data_list]
current_period = challenge.get("currentScoringPeriod", {})

col1, col2, col3 = st.columns(3)
col1.metric("Entries", len(entry_names))
col2.metric("Current Round", current_period.get("label", current_round or "Unknown"))
col3.metric("Games to Analyze", len(importance))

tab_standings, tab_importance, tab_details = st.tabs(
    ["Pool Standings", "Game Importance", "Entry Details"]
)

with tab_standings:
    render_standings(results_df, my_bracket)

with tab_importance:
    render_importance(importance, my_bracket)

with tab_details:
    selected = st.selectbox("Select entry", options=sorted(entry_names))
    if selected:
        render_entry_details(entry_data_list, challenge, importance, selected, women)
