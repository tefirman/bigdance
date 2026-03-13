# bigdance bracket app

A Streamlit web app for picking your March Madness bracket and estimating your win probability in a bracket pool.

## Overview

The app lets you fill out a complete NCAA tournament bracket using real or simulated team data, specify your pool size, and run Monte Carlo simulations to estimate how likely your bracket is to win. It is designed for personal and small-group use — no local server setup required for friends, just share the Streamlit Community Cloud URL.

## User Flow

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

### Sidebar: Configure Your Pool

Two dropdowns plus a Simulate button:

- **Pool size:** Number of entries in your bracket pool (options: 2–50)
- **Number of simulations:** How many Monte Carlo simulations to run (options: 500, 1000, 2500, 5000)
- **Reset bracket:** Clears all picks back to the better-seed defaults

The opponent brackets in the pool are generated using the same Elo-based simulation used to build the bracket field. Each simulated opponent picks independently via Elo win probabilities, producing a plausible distribution of competing brackets.

### Tab 2: Results

Displays after clicking **Simulate**:

- **Win probability:** Your estimated chance of finishing first in the pool
- **Avg score:** Your average simulated score across all tournament outcomes
- **Full pool standings:** Win % and avg score for all entries, sorted by win probability

### Tab 3: Team Probabilities

A reference table showing each tournament team's probability of reaching each round, based on 1,000 simulated tournaments using current Warren Nolan Elo ratings. Columns: Team, Seed, Region, First Round through Championship, and **Championship Odds** in American betting format (e.g., `+500`, `-150`).

Computed once per session and cached — instant on subsequent visits.

## Tech Stack

- **Framework:** [Streamlit](https://streamlit.io/)
- **Simulation engine:** [`bigdance`](https://github.com/tefirman/bigdance) (this repo)
- **Deployment:** Streamlit Community Cloud (shareable URL, no local setup required)

## Running Locally

```bash
# From the repo root
pip install -e ".[dev]"
pip install streamlit
streamlit run app/app.py
```

## Deployment

The app is deployed via Streamlit Community Cloud. [TBD — add link once deployed]

To redeploy after changes: push to `main` and Streamlit Community Cloud will pick up the update automatically.

## File Structure

```
app/
├── README.md       # this file
├── app.py          # main Streamlit app
└── requirements.txt  # app-specific dependencies (if needed beyond the package)
```

## Future Enhancements

- **2nd and 3rd place probability:** Finish position distribution beyond just win %, requires tracking full rank distribution in pool simulation
- **Pick improvement suggestions:** Identify which picks are hurting your odds, based on log-likelihood analysis — picks that diverge from winning brackets in simulation are flagged with an estimated improvement if switched
- **Optimal upset count by round:** Given your pool size, show how many upsets per round tends to maximize win probability
- **Round-by-round survival probability:** % chance your bracket is still in contention after each round resolves
- **Head-to-head comparison:** Compare your bracket against a specific opponent's bracket to see where the key differentiators are
- **Session persistence:** Persist a user's picks between sessions via URL params or Streamlit session state
- **Mobile layout:** Radio buttons in a cascading bracket may be cramped on small screens — worth a responsive design pass before sharing widely
- **Human-bias opponent modeling:** Sample simulated opponent brackets from a realistic human-pick distribution (e.g., bias toward popular teams) rather than purely Elo-random
