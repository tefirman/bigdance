<table>
<tr>
  <td><img src="https://github.com/tefirman/bigdance/blob/main/assets/DancingHex.png?raw=true" width="400" alt="bigdance logo"></td>
  <td>
    <h1>bigdance</h1>
    <p>A Python package for NCAA March Madness bracket simulation combining real-time ratings with customizable tournament simulations.</p>
  </td>
</tr>
</table>

[![PyPI version](https://badge.fury.io/py/bigdance.svg)](https://badge.fury.io/py/bigdance)
[![Run Tests](https://github.com/tefirman/bigdance/actions/workflows/test.yml/badge.svg)](https://github.com/tefirman/bigdance/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

`bigdance` is a comprehensive Python package for simulating NCAA basketball tournament brackets. It provides tools for:

- Pulling real-time college basketball team ratings and matchups from [Warren Nolan](https://www.warrennolan.com/)
- Creating and simulating hypothetical tournament brackets with in-season rankings
- Extracting and simulating real bracket pools from [ESPN's Tournament Challenge](https://fantasy.espn.com/games/tournament-challenge-bracket-2025/)
- Simulating tournament outcomes with adjustable "upset factors"
- Analyzing bracket pools to determine winning strategies
- Visualizing results and generating insights on optimal bracket selection
- Analyzing the importance of specific games in a tournament

Whether you're a fan looking to improve your bracket picks, a data scientist analyzing tournament patterns, or a researcher studying sports predictions, `bigdance` offers powerful, customizable tools to help you simulate and analyze the Big Dance of March Madness.

## Installation

```bash
pip install bigdance
```

## Quick Start

From the command line:
```bash
# Analyze a bracket pool from ESPN, pool ID found in the URL after "bracket?id="
python -m bigdance.espn_tc_scraper --pool_id 77268ce6-7989-4e01-97dc-6681c63c6890
```

Example output:
```
                     name  avg_score  std_score  win_pct
Taylor's Educated Guesses  89.963768  28.193593 0.265083
   Crazylegs329's Picks 1  79.592105  23.647902 0.146500
     Marclemore's Picks 1  75.286765  26.658584 0.123583
      trev_wood's Picks 1  80.643411  32.009227 0.112583
                    Tyler  82.548077  29.134399 0.097500
        ddehart's Picks 1  82.314815  29.403620 0.096083
     KyleStokes's Picks 1  81.021978  27.432737 0.088000
        dsutt06's Picks 1  84.612500  28.695474 0.070667
```

You can also run detailed game importance analysis:
```bash
python -m bigdance.espn_tc_scraper --pool_id 77268ce6-7989-4e01-97dc-6681c63c6890 --importance
```

Example output:
```
=== GAME IMPORTANCE SUMMARY ===

GAME #1: Auburn vs Florida (Region: SOUTH)
  Max Impact: 0.5330 | Avg Impact: 0.2665
  Most affected entry: Tyler
    Win chances: 72.7% if Auburn wins vs 19.4% if Florida wins
    Currently at: 43.0% baseline win probability
    Difference: 53.3%

GAME #2: Duke vs Houston (Region: EAST)
  Max Impact: 0.4750 | Avg Impact: 0.2375
  Most affected entry: Tyler
    Win chances: 69.6% if Duke wins vs 22.1% if Houston wins
    Currently at: 43.0% baseline win probability
    Difference: 47.5%

=== END OF SUMMARY ===
```

## Key Features

### Real-time Basketball Data

Pull current team ratings, rankings, and matchup predictions:

```python
from bigdance import Standings, Matchups

# Get current team standings (with Elo ratings)
standings = Standings()

# Get predictions for today's games
today_games = Matchups()

# Get women's basketball ratings instead
womens_standings = Standings(women=True)

# Filter by conference
acc_teams = Standings(conference="ACC")

# Print top teams by Elo rating
print(standings.elo.sort_values("ELO", ascending=False).head(10))
```

### Hypothetical Tournament Simulation Before Bracket Release

Create a bracket based on current Warren Nolan rankings and accounting for automatic conference bids:

```python
from bigdance import create_teams_from_standings, Standings

# Get current standings
standings = Standings()

# Create bracket with automatic conference bids and seeding
bracket = create_teams_from_standings(standings)

# Simulate tournament once
results = bracket.simulate_tournament()

# Get the champion
champion = results["Champion"]
print(f"Simulated champion: {champion.name} (Seed {champion.seed})")

# Print all Final Four teams
for team in results["Final Four"]:
    print(f"{team.name} (Seed {team.seed}, {team.region} Region)")
```

### Customizing Upset Likelihood

Control how often upsets occur in your simulations:

```python
from bigdance import create_teams_from_standings, Standings

# Get current standings
standings = Standings()

# Create bracket 
bracket = create_teams_from_standings(standings)

# Adjust upset factor for all games
# Range from -1.0 (chalk/favorites always win) to 1.0 (coin flip/50-50)
for game in bracket.games:
    # Values around 0.3 tend to match historical upset rates
    game.upset_factor = 0.3  

# Simulate tournament with adjusted upset factor
results = bracket.simulate_tournament()
```

### ESPN Tournament Challenge Bracket Pool Simulation

Pull brackets directly from ESPN Tournament Challenge:

```python
from bigdance.espn_tc_scraper import ESPNBracket, ESPNPool

# Create a bracket handler for men's tournament (use women=True for women's tournament)
bracket_handler = ESPNBracket()

# Get the current tournament bracket
bracket_html = bracket_handler.get_bracket()
actual_bracket = bracket_handler.extract_bracket(bracket_html)

# Load a pool and all its entries
pool_manager = ESPNPool()
pool_id = "1234567"  # ESPN pool ID found in the URL after "bracket?id="
entries = pool_manager.load_pool_entries(pool_id)

# Create a simulation pool from ESPN entries
pool_sim = pool_manager.create_simulation_pool(pool_id)

# Simulate and display top entries
results = pool_sim.simulate_pool(num_sims=1000)
print(results.head(10))
```

### Game Importance Analysis

Analyze which games have the most impact on a pool's outcome:

```python
from bigdance.espn_tc_scraper import ESPNPool, GameImportanceAnalyzer

# Load a pool from ESPN
pool_manager = ESPNPool()
pool_sim = pool_manager.create_simulation_pool("1234567") # ESPN pool ID

# Create analyzer
analyzer = GameImportanceAnalyzer(pool_sim)

# Analyze the importance of each remaining game
importance = analyzer.analyze_win_importance()

# Print human-readable summary
analyzer.print_importance_summary(importance)

# Focus on impact for a specific entry
analyzer.print_importance_summary(importance, entry_name="My Bracket")
```

### Advanced Analysis

Analyze winning strategies and optimal upset selections using a hypothetical bracket based on current Warren Nolan rankings:

```python
from bigdance import Standings
from bigdance.bracket_analysis import BracketAnalysis

# Get current standings
standings = Standings()

# Create analyzer
analyzer = BracketAnalysis(standings, num_pools=100)

# Run simulations
analyzer.simulate_pools(entries_per_pool=10)

# Generate comparative visualizations
analyzer.plot_comparative_upset_distributions()

# Find optimal upset strategy
strategy = analyzer.identify_optimal_upset_strategy()
print(strategy)

# Find common underdog picks in winning brackets
underdogs = analyzer.find_common_underdogs()
print(underdogs)

# Save comprehensive analysis
analyzer.save_all_comparative_data()
```

Or after the bracket is released, you can integrate with ESPN Tournament Challenge to work with the real tournament bracket and analyze winning strategies:

```python
from bigdance.bracket_analysis import BracketAnalysis

# Use ESPN data instead of Warren Nolan
analyzer = BracketAnalysis(use_espn=True, women=False, num_pools=100)

# For Second Chance brackets (starting from Sweet 16)
analyzer = BracketAnalysis(use_espn=True, second_chance=True, num_pools=100)

# Run simulations with ESPN data as the reference bracket
analyzer.simulate_pools(entries_per_pool=10)
```

## Historical Scheduling and Results

Access game schedules and results:

```python
from bigdance import Schedule
from datetime import datetime, timedelta

# Get last week's games
last_week = datetime.now() - timedelta(days=7)
today = datetime.now()
schedule = Schedule(
    start=last_week.strftime("%Y-%m-%d"),
    stop=today.strftime("%Y-%m-%d")
)

# View games from each day
for day_games in schedule.games_per_day:
    print(f"Games on {day_games.date.strftime('%Y-%m-%d')}:")
    print(day_games.matchups)
```

## Command Line Tools

The package includes command-line tools for accessing functionality:

```bash
# Analyze a bracket pool from ESPN
python -m bigdance.espn_tc_scraper --pool_id 1234567

# Find most important remaining games
python -m bigdance.espn_tc_scraper --pool_id 1234567 --importance

# Run bracket analysis with ESPN data
python -m bigdance.bracket_analysis --use_espn --num_pools 100
```

## Development

To install the package for development:

```bash
git clone https://github.com/tefirman/bigdance
cd bigdance
pip install -e ".[dev]"
```

Run tests:

```bash
pytest
```

## Documentation

For detailed documentation on all functions and classes, use Python's built-in help:

```python
import bigdance
help(bigdance)
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Warren Nolan website for providing college basketball data (no affiliation)
- ESPN Tournament Challenge for tournament brackets (no affiliation)
- Andrew Sundberg for [historical tournament data](https://www.kaggle.com/datasets/andrewsundberg/college-basketball-dataset/) used in testing

## Author

- Taylor Firman ([@tefirman](https://github.com/tefirman))
