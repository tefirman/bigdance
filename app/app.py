import streamlit as st
from collections import defaultdict
from bigdance import Standings, create_teams_from_standings, simulate_round_probabilities
from bigdance.cbb_brackets import Bracket, Team, Game, Pool
from bigdance.wn_cbb_scraper import elo_prob

st.set_page_config(page_title="bigdance bracket pool", layout="wide")
st.title("March Madness Bracket Pool Simulator")

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

@st.cache_resource
def load_standings() -> Standings:
    return Standings()


@st.cache_resource
def load_bracket() -> Bracket:
    return create_teams_from_standings(load_standings())


@st.cache_resource
def load_round_probabilities():
    return simulate_round_probabilities(standings=load_standings(), num_sims=1000)


if "bracket" not in st.session_state:
    with st.spinner("Loading bracket from Warren Nolan rankings..."):
        st.session_state.bracket = load_bracket()

bracket: Bracket = st.session_state.bracket

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
    _bracket: Bracket = st.session_state.bracket
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
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Pool Settings")
    pool_size = st.selectbox("Pool size (# of entries)", options=list(range(2, 51)), index=8)
    num_sims = st.selectbox("Simulations", options=[500, 1000, 2500, 5000], index=1)
    simulate_btn = st.button("Simulate", type="primary", width="stretch")
    st.divider()
    if st.button("Reset bracket", width="stretch"):
        st.session_state.picks = {r: {} for r in range(6)}
        st.session_state.reset_count += 1
        st.rerun()

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

tab_bracket, tab_results, tab_probs = st.tabs(["🏀  Pick Your Bracket", "📊  Results", "📈  Team Probabilities"])

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

                for local_idx in range(block_size):
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

            standings = load_standings()
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
# Tab 3: team round probabilities
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
        probs_df = load_round_probabilities().copy()
    champ_pct = probs_df["Championship"].str.rstrip("%").astype(float) / 100
    probs_df["Championship Odds"] = champ_pct.apply(american_odds)
    st.dataframe(probs_df, width="stretch", hide_index=True)
