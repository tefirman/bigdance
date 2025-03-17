<table>
<tr>
  <td><img src="https://github.com/tefirman/bigdance/blob/main/assets/DancingHex.png?raw=true" width="400" alt="bigdance logo"></td>
  <td>
    <h1>bigdance</h1>
    <p>A Python package for NCAA March Madness bracket simulation combining real-time ratings from Warren Nolan with customizable tournament simulations.</p>
  </td>
</tr>
</table>

[![PyPI version](https://badge.fury.io/py/bigdance.svg)](https://badge.fury.io/py/bigdance)
[![Run Tests](https://github.com/tefirman/bigdance/actions/workflows/test.yml/badge.svg)](https://github.com/tefirman/bigdance/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

`bigdance` is a comprehensive Python package for simulating NCAA basketball tournament brackets. It provides tools for:

- Pulling real-time college basketball team ratings and statistics from Warren Nolan
- Creating realistic tournament brackets with automatic bids and seeding
- Simulating tournament outcomes with adjustable "upset factors"
- Analyzing bracket pools to determine winning strategies
- Visualizing results and generating insights on optimal bracket selection

Whether you're a fan looking to improve your bracket picks, a data scientist analyzing tournament patterns, or a researcher studying sports predictions, `bigdance` offers powerful, customizable tools to help you simulate and analyze March Madness.

## Installation

```bash
pip install bigdance
```

## Quick Start

```python
from bigdance import Standings, simulate_bracket_pool

# Get current team ratings
standings = Standings()

# Simulate a bracket pool
results = simulate_bracket_pool(standings, num_entries=100)
print(results)
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

### Tournament Bracket Creation and Simulation

Create and simulate entire tournament brackets:

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

### Bracket Pool Simulation

Simulate entire bracket pools with multiple entries:

```python
from bigdance import simulate_bracket_pool, Standings

# Get current standings
standings = Standings()

# Simulate pool with 100 entries using varying upset factors
results = simulate_bracket_pool(
    standings,
    num_entries=100,
    # Optional: provide specific upset factors 
    # upset_factors=[0.1, 0.2, 0.3, ...],
)

# Print winning entries
print(results.head(10))
```

### Advanced Analysis

Analyze winning strategies and optimal upset selections:

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

### Historical Scheduling and Results

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

## Advanced Usage Examples

### Creating a Custom Bracket with Specific Picks

```python
from bigdance import Standings, create_teams_from_standings
from bigdance.bigdance_integration import create_bracket_with_picks

# Get team data
standings = Standings()
teams = create_teams_from_standings(standings).teams

# Define your picks (team names by round)
picks = {
    "First Round": ["Duke", "North Carolina", "Kansas", ...],
    "Second Round": ["Duke", "Kansas", ...],
    "Sweet 16": ["Duke", "Purdue", ...],
    "Elite 8": ["Duke", "UConn"],
    "Final Four": ["Duke"],
    "Championship": ["Duke"]
}

# Create bracket with your picks
my_bracket = create_bracket_with_picks(teams, picks)

# Check bracket details
print(f"Champion: {my_bracket.results['Champion'].name}")
```

### Finding the Optimal Upset Factor

```python
from bigdance import Standings, create_teams_from_standings, Pool
import numpy as np

# Get team data
standings = Standings()
actual_bracket = create_teams_from_standings(standings)

# Create pool
pool = Pool(actual_bracket)

# Try different upset factors
upset_factors = np.arange(-0.8, 0.81, 0.1)

for factor in upset_factors:
    entry = create_teams_from_standings(standings)
    for game in entry.games:
        game.upset_factor = factor
    pool.add_entry(f"Factor_{factor:.1f}", entry)

# Simulate and find best factor
results = pool.simulate_pool(num_sims=1000)
print(results[["name", "win_pct", "avg_score"]].sort_values("win_pct", ascending=False))
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
- Andrew Sundberg for historical tournament data used in testing

## Author

- Taylor Firman ([@tefirman](https://github.com/tefirman))
