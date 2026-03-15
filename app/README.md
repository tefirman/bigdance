# bigdance bracket app

A Streamlit web app for picking your March Madness bracket and estimating your win probability in a bracket pool.

## Overview

The app lets you fill out a complete NCAA tournament bracket using real or simulated team data, specify your pool size, and run Monte Carlo simulations to estimate how likely your bracket is to win. It supports both the men's and women's NCAA tournaments. It is designed for personal and small-group use — no local server setup required for friends, just share the Streamlit Community Cloud URL.

## User Flow

### Sidebar: Configure Your Pool

- **Tournament:** Toggle between Men's and Women's NCAA tournament
- **Pool size:** Number of entries in your bracket pool (options: 5, 10, 15, 20, 30, 50)
- **Number of simulations:** How many Monte Carlo simulations to run (options: 500, 1000, 2500, 5000)
- **Simulate button:** Runs the pool simulation with your current bracket picks
- **Reset bracket:** Clears all picks back to the better-seed defaults

### Tab 1: Pick Your Bracket

The bracket picker displays the 64-team field in a round-by-round cascading layout. For each game:

- Two radio buttons show the current opponents (team name + seed)
- Each matchup shows the Elo-based win probability for each team as a caption (e.g., `Auburn 78% · Florida 22%`)
- Selecting a team advances them to the next round as a radio button option
- Defaults to the better seed throughout; all rounds populate automatically on page load
- This continues through all six rounds: Round of 64, Round of 32, Sweet 16, Elite 8, Final Four, Championship

The bracket is displayed by region (East, West, South, Midwest), with all four regions visible simultaneously. The national semifinals and championship game populate automatically as regional winners are selected.

**Before Selection Sunday:** The bracket field is generated from current Warren Nolan Elo ratings via `create_teams_from_standings()`, which simulates the likely 64-team field and seedings based on in-season rankings. This allows full testing and exploration of the app before the real bracket is announced.

**After Selection Sunday:** The real bracket field is pulled automatically from ESPN via `ESPNBracket`. This ensures accuracy (play-in results, late seed changes) and keeps the app reusable in future seasons without manual updates.

### Tab 2: Sim Results

Displays after clicking **Simulate**:

- **Win probability:** Your estimated chance of finishing first in the pool
- **Avg score:** Your average simulated score across all tournament outcomes
- **Full pool standings:** Win % and avg score for all entries, sorted by win probability

### Tab 3: Upset Strategy

Pre-computed analysis of winning bracket patterns for your pool size, across three sections:

- **Upset count by round:** How many upsets winning brackets typically pick per round, color-coded by how your current picks compare (green = on target, yellow = slightly off, red = too chalk or too bold)
- **Madness Score by round:** Negative log probability of your picks per round — higher = more surprising. Shows how your picks compare to the typical winning bracket's surprise level
- **Common underdogs in winning brackets:** Which underdog teams appear most often in winning brackets, broken down by how far they upset through

All data is pre-computed for the selected pool size and tournament (men's/women's) via `generate_upset_analysis.py` and stored in `app/data/{gender}/pool_{n}/`.

### Tab 4: Team Probabilities

A reference table showing each tournament team's probability of reaching each round, based on 1,000 simulated tournaments using current Warren Nolan Elo ratings. Columns: Team, Seed, Region, First Round through Championship, and **Championship Odds** in American betting format (e.g., `+500`, `-150`).

Computed once per session and cached — instant on subsequent visits.

## Tech Stack

- **Framework:** [Streamlit](https://streamlit.io/)
- **Simulation engine:** [`bigdance`](https://github.com/tefirman/bigdance) (this repo)
- **Deployment:** [Streamlit Community Cloud](https://bigdance-bracket.streamlit.app) (shareable URL, no local setup required)

## Running Locally

```bash
# From the repo root
pip install -e ".[dev]"
pip install streamlit
streamlit run app/app.py
```

## Pre-computing Upset Strategy Data

The Upset Strategy tab requires pre-baked data files. Generate them before deploying:

```bash
# Men's tournament (Warren Nolan hypothetical bracket, pre-Selection Sunday)
python app/generate_upset_analysis.py --gender men

# Women's tournament
python app/generate_upset_analysis.py --gender women

# After Selection Sunday, use the real ESPN bracket
python app/generate_upset_analysis.py --gender men --use_espn
python app/generate_upset_analysis.py --gender women --use_espn
```

Output is saved to `app/data/{gender}/pool_{n}/`. This can also be triggered via the **Generate Upset Analysis Data** GitHub Action, which parallelizes across all pool sizes.

## Deployment

The app is deployed at [bigdance-bracket.streamlit.app](https://bigdance-bracket.streamlit.app) via Streamlit Community Cloud.

To redeploy after changes: push to `main` and Streamlit Community Cloud will pick up the update automatically.

## File Structure

```
app/
├── README.md                    # this file
├── app.py                       # main Streamlit app
├── generate_upset_analysis.py   # pre-bake script for upset strategy data
└── data/
    ├── men/
    │   ├── pool_5/
    │   ├── pool_10/
    │   └── ...
    └── women/
        ├── pool_5/
        ├── pool_10/
        └── ...
```

## Future Enhancements

- **2nd and 3rd place probability:** Finish position distribution beyond just win %, requires tracking full rank distribution in pool simulation
- **Pick improvement suggestions:** Identify which picks are hurting your odds, based on log-likelihood analysis — picks that diverge from winning brackets in simulation are flagged with an estimated improvement if switched
- **Round-by-round survival probability:** % chance your bracket is still in contention after each round resolves
- **Head-to-head comparison:** Compare your bracket against a specific opponent's bracket to see where the key differentiators are
- **Session persistence:** Persist a user's picks between sessions via URL params or Streamlit session state
- **Mobile layout:** Radio buttons in a cascading bracket may be cramped on small screens — worth a responsive design pass before sharing widely
- **Human-bias opponent modeling:** Sample simulated opponent brackets from a realistic human-pick distribution (e.g., bias toward popular teams) rather than purely Elo-random
