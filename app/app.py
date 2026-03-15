import streamlit as st
from collections import defaultdict
from pathlib import Path
import pandas as pd
from bigdance import Standings, create_teams_from_standings, simulate_round_probabilities
from bigdance.cbb_brackets import Bracket, Team, Game, Pool
from bigdance.wn_cbb_scraper import elo_prob

st.set_page_config(page_title="bigdance bracket pool", layout="wide")

# ---------------------------------------------------------------------------
# Password gate
# ---------------------------------------------------------------------------

def check_password() -> bool:
    if st.session_state.get("authenticated"):
        return True
    _, col, _ = st.columns([1, 1, 1])
    with col:
        st.title("March Madness Bracket Pool")
        pwd = st.text_input("Password", type="password", key="pwd_input")
        if pwd:
            if pwd == st.secrets.get("password", ""):
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Incorrect password.")
    return False

if not check_password():
    st.stop()

st.title("March Madness Bracket Pool Simulator")

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

@st.cache_resource
def load_standings(women: bool = False) -> Standings:
    return Standings(women=women)


@st.cache_resource
def load_bracket(women: bool = False) -> Bracket:
    return create_teams_from_standings(load_standings(women=women))


@st.cache_resource
def load_round_probabilities(women: bool = False):
    return simulate_round_probabilities(standings=load_standings(women=women), num_sims=1000)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Pool Settings")
    gender = st.radio("Tournament", options=["Men's", "Women's"], horizontal=True)
    women = gender == "Women's"
    gender_key = "women" if women else "men"
    pool_size = st.selectbox("Pool size (# of entries)", options=[5, 10, 15, 20, 30, 50], index=1)
    num_sims = st.selectbox("Simulations", options=[500, 1000, 2500, 5000], index=1)
    simulate_btn = st.button("Simulate", type="primary", width="stretch")
    st.divider()
    if st.button("Reset bracket", width="stretch"):
        st.session_state.picks = {r: {} for r in range(6)}
        st.session_state.reset_count += 1
        st.rerun()

bracket_key = f"bracket_{gender}"
if bracket_key not in st.session_state:
    with st.spinner(f"Loading {gender} bracket from Warren Nolan rankings..."):
        st.session_state[bracket_key] = load_bracket(women=women)

bracket: Bracket = st.session_state[bracket_key]

# picks[round_idx][game_idx] = Team
# round 0 = Round of 64 (32 games), ..., round 5 = Championship (1 game)
if "picks" not in st.session_state:
    st.session_state.picks = {r: {} for r in range(6)}

if "reset_count" not in st.session_state:
    st.session_state.reset_count = 0

picks: dict[int, dict[int, Team]] = st.session_state.picks


def better_seed(t1: Team, t2: Team) -> Team:
    return t1 if t1.seed <= t2.seed else t2


def seed_defaults() -> None:
    """Pre-populate all missing picks with the better-seeded team so every
    widget has a value on the very first page load, before any radio fires."""
    _bracket: Bracket = bracket
    _region_r0: dict = defaultdict(list)
    for _i, _g in enumerate(_bracket.games):
        _region_r0[_g.region].append((_i, _g))

    REGIONS_LOCAL = ["East", "West", "South", "Midwest"]
    BLOCK = {r: i for i, r in enumerate(REGIONS_LOCAL)}

    def _rgidx(region: str, rnd: int, loc: int) -> int:
        return BLOCK[region] * (8 // (2 ** rnd)) + loc

    # Round 0 — seeded from bracket.games directly
    for region in REGIONS_LOCAL:
        for local_idx, (bracket_idx, game) in enumerate(_region_r0[region]):
            if bracket_idx not in picks[0]:
                picks[0][bracket_idx] = better_seed(game.team1, game.team2)

    # Rounds 1–3 — seeded from previous round's picks
    for rnd in range(1, 4):
        block_size = 8 // (2 ** rnd)
        for region in REGIONS_LOCAL:
            for loc in range(block_size):
                gidx = _rgidx(region, rnd, loc)
                if gidx not in picks[rnd]:
                    prev_loc1, prev_loc2 = loc * 2, loc * 2 + 1
                    if rnd == 1:
                        prev_idx1 = _region_r0[region][prev_loc1][0]
                        prev_idx2 = _region_r0[region][prev_loc2][0]
                    else:
                        prev_idx1 = _rgidx(region, rnd - 1, prev_loc1)
                        prev_idx2 = _rgidx(region, rnd - 1, prev_loc2)
                    t1 = picks[rnd - 1].get(prev_idx1)
                    t2 = picks[rnd - 1].get(prev_idx2)
                    if t1 and t2:
                        picks[rnd][gidx] = better_seed(t1, t2)

    # Round 4 — Final Four: South vs Midwest (0), East vs West (1)
    ff_pairs = [
        (0, _rgidx("South", 3, 0), _rgidx("Midwest", 3, 0)),
        (1, _rgidx("East", 3, 0), _rgidx("West", 3, 0)),
    ]
    for ff_idx, idx1, idx2 in ff_pairs:
        if ff_idx not in picks[4]:
            t1, t2 = picks[3].get(idx1), picks[3].get(idx2)
            if t1 and t2:
                picks[4][ff_idx] = better_seed(t1, t2)

    # Round 5 — Championship
    if 0 not in picks[5]:
        t1, t2 = picks[4].get(0), picks[4].get(1)
        if t1 and t2:
            picks[5][0] = better_seed(t1, t2)


seed_defaults()

# ---------------------------------------------------------------------------
# Game index helpers
#
# First-round games in bracket.games are grouped by region.
# We build a lookup once so the rest of the code is region-name driven.
#
# For rounds 1–3 (within-region games) we assign a flat index per round:
#   region_game_idx(region, round, local) → global index for that round
#   where local is 0..block_size-1 within the region.
#
# Round 4 (Final Four): indices 0 (South vs Midwest) and 1 (East vs West)
# Round 5 (Championship): index 0
# ---------------------------------------------------------------------------

REGIONS = ["East", "West", "South", "Midwest"]
REGION_BLOCK = {r: i for i, r in enumerate(REGIONS)}  # East=0, West=1, South=2, Midwest=3

# region_r0_games[region] = list of (global_bracket_index, Game)
region_r0_games: dict[str, list[tuple[int, Game]]] = defaultdict(list)
for _i, _g in enumerate(bracket.games):
    region_r0_games[_g.region].append((_i, _g))


def region_game_idx(region: str, round_idx: int, local_idx: int) -> int:
    """Return the flat game index for rounds 1–3 (not used for round 0)."""
    block_size = 8 // (2 ** round_idx)
    return REGION_BLOCK[region] * block_size + local_idx


def get_pick(round_idx: int, game_idx: int) -> Team | None:
    return picks[round_idx].get(game_idx)


def set_pick(round_idx: int, game_idx: int, team: Team) -> None:
    if picks[round_idx].get(game_idx) is team:
        return
    picks[round_idx][game_idx] = team
    # Invalidate all downstream picks for rounds 1–3
    if round_idx < 3:
        child = game_idx // 2
        for r in range(round_idx + 1, 4):
            picks[r].pop(child, None)
            child = child // 2
    # Any change in rounds 0–3 clears Final Four and Championship
    if round_idx <= 3:
        picks[4].clear()
        picks[5].clear()
    elif round_idx == 4:
        picks[5].clear()


def render_matchup(round_idx: int, game_idx: int, team1: Team | None, team2: Team | None) -> None:
    if team1 is None or team2 is None:
        st.markdown("_Waiting..._")
        return

    label1 = f"({team1.seed}) {team1.name}"
    label2 = f"({team2.seed}) {team2.name}"
    current = picks[round_idx].get(game_idx)
    if current is not None:
        current_label = f"({current.seed}) {current.name}"
        if current_label not in (label1, label2):
            current_label = None  # stale pick after upstream change
    else:
        current_label = None

    if current_label is None:
        # Default to the better seed (lower seed number)
        current_label = label1 if team1.seed <= team2.seed else label2

    rc = st.session_state.reset_count
    key = f"pick_r{round_idx}_g{game_idx}_v{rc}"
    choice = st.radio(
        label=key,
        options=[label1, label2],
        index=0 if current_label == label1 else 1,
        key=key,
        label_visibility="collapsed",
    )
    set_pick(round_idx, game_idx, team1 if choice == label1 else team2)

    p1 = elo_prob(team1.rating, team2.rating, homefield=0.0)
    st.caption(f"{team1.name} {p1:.0%} · {team2.name} {1 - p1:.0%}")


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    .stTabs [data-baseweb="tab"] {
        font-size: 1.1rem;
        font-weight: 600;
        padding: 0.5rem 1.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

tab_bracket, tab_results, tab_strategy, tab_probs = st.tabs(["🏀  Pick Your Bracket", "📊  Sim Results", "🎯  Upset Strategy", "📈  Team Probabilities"])

# ---------------------------------------------------------------------------
# Tab 1: bracket picker
# ---------------------------------------------------------------------------

with tab_bracket:
    for region in REGIONS:
        st.markdown(f"### {region} Region")
        cols = st.columns(4)
        round_labels = ["Round of 64", "Round of 32", "Sweet 16", "Elite 8"]

        for col, round_label, round_idx in zip(cols, round_labels, range(4)):
            with col:
                st.markdown(f"**{round_label}**")
                block_size = 8 // (2 ** round_idx)
                # Spacers to vertically center games relative to their feeder games.
                # Before the first game: 2^round_idx - 1 blank lines.
                # Between games: 2^(round_idx+1) - 1 blank lines.
                leading = 6 * (2 ** round_idx - 1)
                between = 6 * (2 ** (round_idx + 1) - 1)

                for local_idx in range(block_size):
                    for _ in range(leading if local_idx == 0 else between):
                        st.write("")
                    if round_idx == 0:
                        bracket_idx, game = region_r0_games[region][local_idx]
                        render_matchup(0, bracket_idx, game.team1, game.team2)
                    else:
                        gidx = region_game_idx(region, round_idx, local_idx)
                        prev_local_1 = local_idx * 2
                        prev_local_2 = local_idx * 2 + 1
                        if round_idx == 1:
                            prev_idx1, _ = region_r0_games[region][prev_local_1]
                            prev_idx2, _ = region_r0_games[region][prev_local_2]
                        else:
                            prev_idx1 = region_game_idx(region, round_idx - 1, prev_local_1)
                            prev_idx2 = region_game_idx(region, round_idx - 1, prev_local_2)
                        team1 = picks[round_idx - 1].get(prev_idx1)
                        team2 = picks[round_idx - 1].get(prev_idx2)
                        render_matchup(round_idx, gidx, team1, team2)

        st.divider()

    st.markdown("### Final Four")
    ff_cols = st.columns(2)

    south_winner = picks[3].get(region_game_idx("South", 3, 0))
    midwest_winner = picks[3].get(region_game_idx("Midwest", 3, 0))
    east_winner = picks[3].get(region_game_idx("East", 3, 0))
    west_winner = picks[3].get(region_game_idx("West", 3, 0))

    with ff_cols[0]:
        st.markdown("**South vs Midwest**")
        render_matchup(4, 0, south_winner, midwest_winner)

    with ff_cols[1]:
        st.markdown("**East vs West**")
        render_matchup(4, 1, east_winner, west_winner)

    st.divider()

    st.markdown("### Championship")
    _, champ_col, _ = st.columns([1, 2, 1])
    with champ_col:
        render_matchup(5, 0, picks[4].get(0), picks[4].get(1))

    champion = picks[5].get(0)
    if champion:
        st.success(f"Your champion: ({champion.seed}) **{champion.name}**")

# ---------------------------------------------------------------------------
# Simulation — runs from sidebar button, stores results in session state
# ---------------------------------------------------------------------------

champion = picks[5].get(0)


def build_fixed_winners() -> dict[str, list[Team]]:
    """Convert session picks into the fixed_winners format expected by Bracket.simulate_tournament."""
    round_name_map = {
        0: "First Round",
        1: "Second Round",
        2: "Sweet 16",
        3: "Elite 8",
        4: "Final Four",
        5: "Championship",
    }
    fixed: dict[str, list[Team]] = {}
    for round_idx, round_picks in picks.items():
        if round_picks:
            fixed[round_name_map[round_idx]] = list(round_picks.values())
    return fixed


if simulate_btn:
    if champion is None:
        with tab_bracket:
            st.warning("Please complete your bracket (pick a champion) before simulating.")
    else:
        with st.spinner(f"Running {num_sims:,} simulations with {pool_size} entries..."):
            fixed_winners = build_fixed_winners()

            standings = load_standings(women=women)
            actual_bracket = create_teams_from_standings(standings)
            pool = Pool(actual_results=actual_bracket)

            user_bracket = create_teams_from_standings(standings)
            user_bracket.simulate_tournament(fixed_winners=fixed_winners)
            pool.add_entry("You", user_bracket, simulate=False)

            for i in range(pool_size - 1):
                opp_bracket = create_teams_from_standings(standings)
                pool.add_entry(f"Opponent {i + 1}", opp_bracket, simulate=True)

            st.session_state.sim_results = pool.simulate_pool(num_sims=num_sims)

# ---------------------------------------------------------------------------
# Tab 2: results
# ---------------------------------------------------------------------------

with tab_results:
    if "sim_results" not in st.session_state:
        st.info("Configure your pool in the sidebar and click **Simulate** to see results.")
    else:
        results = st.session_state.sim_results
        user_row = results[results["name"] == "You"].iloc[0]

        col1, col2 = st.columns(2)
        col1.metric("Win probability", f"{user_row['win_pct']:.1%}")
        col2.metric("Avg score", f"{user_row['avg_score']:.1f}")

        st.caption("_2nd and 3rd place probabilities are a planned future enhancement._")

        st.subheader("Full Pool Standings")
        display = (
            results[["name", "avg_score", "win_pct"]]
            .rename(columns={"name": "Entry", "avg_score": "Avg Score", "win_pct": "Win %"})
            .sort_values("Win %", ascending=False)
            .reset_index(drop=True)
        )
        display["Win %"] = display["Win %"].map("{:.1%}".format)
        display["Avg Score"] = display["Avg Score"].map("{:.1f}".format)
        st.dataframe(display, width="stretch")

# ---------------------------------------------------------------------------
# Tab 3: upset strategy
# ---------------------------------------------------------------------------

ROUND_ORDER = ["First Round", "Second Round", "Sweet 16", "Elite 8", "Final Four", "Championship"]


def count_upsets_from_picks() -> dict[str, int]:
    """Count underdogs per round using the same seed-threshold definition as BracketAnalysis.

    Mirrors cbb_brackets.Bracket.is_underdog():
        First Round  — seed > 8  (9–16 are underdogs)
        Second Round — seed > 4  (5–16 are underdogs)
        Sweet 16     — seed > 2  (3–16 are underdogs)
        Elite 8+     — seed > 1  (2–16 are underdogs)
    """
    seed_thresholds = {
        "First Round": 8,
        "Second Round": 4,
        "Sweet 16": 2,
        "Elite 8": 1,
        "Final Four": 1,
        "Championship": 1,
    }
    round_name_map = {
        0: "First Round",
        1: "Second Round",
        2: "Sweet 16",
        3: "Elite 8",
        4: "Final Four",
        5: "Championship",
    }
    upset_counts: dict[str, int] = {r: 0 for r in ROUND_ORDER}

    for rnd in range(6):
        round_name = round_name_map[rnd]
        threshold = seed_thresholds[round_name]
        for winner in picks[rnd].values():
            if winner is not None and winner.seed > threshold:
                upset_counts[round_name] += 1

    return upset_counts


def load_upset_strategy(pool_size: int, gender: str = "men") -> pd.DataFrame | None:
    strategy_path = Path(__file__).parent / "data" / gender / f"pool_{pool_size}" / "optimal_upset_strategy.csv"
    if not strategy_path.exists():
        return None
    return pd.read_csv(strategy_path)


def load_log_prob_strategy(pool_size: int, gender: str = "men") -> pd.DataFrame | None:
    path = Path(__file__).parent / "data" / gender / f"pool_{pool_size}" / "log_prob_strategy.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def load_common_underdogs(pool_size: int, gender: str = "men") -> pd.DataFrame | None:
    path = Path(__file__).parent / "data" / gender / f"pool_{pool_size}" / "common_underdogs.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df.empty:
        return df
    # Pivot from long (one row per team×round) to wide (one row per team, one col per round)
    rounds = ["First Round", "Second Round", "Sweet 16", "Elite 8", "Final Four"]
    pivot = df.pivot_table(index=["seed", "team"], columns="advanced_through", values="frequency", aggfunc="first")
    pivot = pivot.reindex(columns=[r for r in rounds if r in pivot.columns])
    pivot.columns = [c for c in pivot.columns]
    pivot = pivot.reset_index()
    # Leave frequencies as floats (NaN for missing); format via column_config in the display layer
    pivot = pivot.rename(columns={"seed": "Seed", "team": "Team"})
    # Sort by First Round frequency desc (most common underdogs first), then by seed
    sort_col = next((r for r in rounds if r in pivot.columns), None)
    if sort_col:
        pivot = pivot.sort_values(sort_col, ascending=False, na_position="last").reset_index(drop=True)
    return pivot


def compute_madness_scores(women: bool = False) -> dict[str, float]:
    """Compute Madness Score (negative log probability) per round from current picks.

    Builds a temporary bracket from current picks and delegates to
    Bracket.calculate_log_probability() from the bigdance package.
    Higher score = more surprising/upset-heavy picks in that round.
    """
    tmp = create_teams_from_standings(load_standings(women=women))
    tmp.simulate_tournament(fixed_winners=build_fixed_winners())
    tmp.calculate_log_probability()
    return tmp.log_probability_by_round


with tab_strategy:
    strategy_df = load_upset_strategy(pool_size, gender=gender_key)
    if strategy_df is None:
        st.info(
            f"No upset strategy data found for pool size {pool_size}. "
            "Run `python app/generate_upset_analysis.py` to generate it."
        )
    else:
        num_pools_used = int(strategy_df["num_pools"].iloc[0]) if "num_pools" in strategy_df.columns else "?"
        st.caption(
            f"Optimal upset counts per round for a {pool_size}-person pool, "
            f"based on {num_pools_used} simulated pools. Pick counts update live as you fill in your bracket."
        )

        def _z_to_color(z: float) -> str:
            """Interpolate light-green→light-yellow→light-red based on z-score distance from mean.
            z=0 → light green, z=0.5 → light yellow, z>=1 → light red.
            """
            z = min(abs(z), 1.0)
            if z <= 0.5:
                # #90ee90 (light green) → #ffff99 (light yellow)
                t = z / 0.5
                r = int(144 + t * (255 - 144))
                g = int(238 + t * (255 - 238))
                b = int(144 - t * 144)
            else:
                # #ffff99 (light yellow) → #ff9999 (light red)
                t = (z - 0.5) / 0.5
                r = 255
                g = int(255 - t * (255 - 153))
                b = int(153 - t * 153)
            return f"background-color: rgb({r},{g},{b}); color: #222"

        user_upsets = count_upsets_from_picks()

        rows = []
        z_scores_upsets = []
        for _, row in strategy_df.iterrows():
            rnd = row["round"]
            if rnd == "Total Upsets":
                continue
            mean_val = row["mean_upsets"] if pd.notna(row.get("mean_upsets")) else None
            std_val = row["std_upsets"] if pd.notna(row.get("std_upsets")) else None
            mean_str = f"{mean_val:.1f}" if mean_val is not None else "—"
            std_str = f"{std_val:.1f}" if std_val is not None else "—"
            your_count = user_upsets.get(rnd, 0)
            if mean_val is not None and std_val is not None and std_val > 0:
                z = (your_count - mean_val) / std_val
                direction = "—" if abs(z) < 0.1 else ("↑ too bold" if z > 0 else "↓ too chalk")
            else:
                z = 0.0
                direction = "—"
            z_scores_upsets.append(z)
            rows.append({
                "Round": rnd,
                "Your Upsets": your_count,
                "Winners Avg": mean_str,
                "Winners Std": std_str,
                "Direction": direction,
            })

        display_df = pd.DataFrame(rows)

        def _color_col(z_scores: list[float], col: str):
            def _styler(s):
                return [_z_to_color(z) for z in z_scores]
            return _styler

        styled = display_df.style.apply(_color_col(z_scores_upsets, "Your Upsets"), subset=["Your Upsets"])
        st.dataframe(styled, width="stretch", hide_index=True)

        st.markdown("#### Madness Score by Round")
        st.caption(
            "Negative log probability of your picks — higher = more surprising. "
            "Compare your score per round to what winners typically look like."
        )
        log_prob_df = load_log_prob_strategy(pool_size, gender=gender_key)
        if log_prob_df is None:
            st.info("Run `python app/generate_upset_analysis.py` to generate Madness Score data.")
        else:
            madness_scores = compute_madness_scores(women=women)
            madness_rows = []
            z_scores_madness = []
            for _, row in log_prob_df.iterrows():
                rnd = row["round"]
                mean_val = row["mean_madness"] if pd.notna(row.get("mean_madness")) else None
                std_val = row["std_madness"] if pd.notna(row.get("std_madness")) else None
                your_score = madness_scores.get(rnd, 0.0)
                mean_str = f"{mean_val:.2f}" if mean_val is not None else "—"
                std_str = f"{std_val:.2f}" if std_val is not None else "—"
                if mean_val is not None and std_val is not None and std_val > 0:
                    z = (your_score - mean_val) / std_val
                    direction = "—" if abs(z) < 0.1 else ("↑ too bold" if z > 0 else "↓ too chalk")
                else:
                    z = 0.0
                    direction = "—"
                z_scores_madness.append(z)
                madness_rows.append({
                    "Round": rnd,
                    "Your Madness": f"{your_score:.2f}",
                    "Winners Avg": mean_str,
                    "Winners Std": std_str,
                    "Direction": direction,
                })

            madness_display = pd.DataFrame(madness_rows)
            styled_madness = madness_display.style.apply(
                _color_col(z_scores_madness, "Your Madness"), subset=["Your Madness"]
            )
            st.dataframe(styled_madness, width="stretch", hide_index=True)

        st.markdown("#### Common Underdogs in Winning Brackets")
        st.caption("How often each underdog team appears in winning pool brackets, by the round they upset through.")
        underdogs_df = load_common_underdogs(pool_size, gender=gender_key)
        if underdogs_df is None:
            st.info("Run `python app/generate_upset_analysis.py` to generate common underdogs data.")
        elif underdogs_df.empty:
            st.info("No common underdog data available for this pool size.")
        else:
            round_cols = ["First Round", "Second Round", "Sweet 16", "Elite 8", "Final Four"]
            display_underdogs = underdogs_df.copy()
            for rnd in round_cols:
                if rnd in display_underdogs.columns:
                    display_underdogs[rnd] = display_underdogs[rnd] * 100
            col_config = {
                rnd: st.column_config.NumberColumn(rnd, format="%.0f%%", min_value=0, max_value=100)
                for rnd in round_cols
                if rnd in display_underdogs.columns
            }
            st.dataframe(display_underdogs, column_config=col_config, width="stretch", hide_index=True)

# ---------------------------------------------------------------------------
# Tab 4: team round probabilities
# ---------------------------------------------------------------------------

def american_odds(p: float) -> str:
    """Convert a win probability to American-style odds string."""
    if p <= 0 or p >= 1:
        return "N/A"
    if p >= 0.5:
        return f"-{round(p / (1 - p) * 100)}"
    else:
        return f"+{round((1 / p - 1) * 100)}"


with tab_probs:
    st.caption("Probability of each team reaching each round, based on 1,000 simulated tournaments using current Warren Nolan Elo ratings.")
    with st.spinner("Computing team probabilities..."):
        probs_df = load_round_probabilities(women=women).copy()
    champ_pct = probs_df["Championship"].str.rstrip("%").astype(float) / 100
    probs_df["Championship Odds"] = champ_pct.apply(american_odds)
    st.dataframe(probs_df, width="stretch", hide_index=True)
