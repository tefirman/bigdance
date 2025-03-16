"""
bigdance - A March Madness bracket simulation package.

This package provides tools for NCAA basketball bracket simulation by combining:
- Real-time stats and ratings from Warren Nolan (no affiliation)
- Bracket simulation with customizable parameters
- Integration tools for tournament pool analysis

Main components:
- Standings: Get current NCAA basketball team ratings and rankings
- Matchups: Get game predictions and results
- Bracket: Create and simulate tournament brackets
- Pool: Simulate tournament pools with multiple entries
"""

from .bigdance_integration import create_teams_from_standings, simulate_bracket_pool
from .cbb_brackets import Bracket, Game, Pool, Team
from .wn_cbb_scraper import Matchups, Schedule, Standings, elo_prob

__version__ = "0.1.0"
__author__ = "Taylor Firman"
__email__ = "tefirman@gmail.com"

# List of public objects that should be available when using "from dancing import *"
__all__ = [
    # Core classes
    "Standings",
    "Matchups",
    "Schedule",
    "Team",
    "Game",
    "Bracket",
    "Pool",
    # Helper functions
    "elo_prob",
    "create_teams_from_standings",
    "simulate_bracket_pool",
]
