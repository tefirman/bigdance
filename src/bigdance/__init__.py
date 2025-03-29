"""
bigdance - A March Madness bracket simulation package.

This package provides tools for NCAA basketball bracket simulation by combining:
- Real-time stats and ratings from Warren Nolan (no affiliation)
- Bracket simulation with customizable parameters
- Integration with ESPN Tournament Challenge (no affiliation)
- Tools for tournament pool analysis and simulation

Main components:
- Standings: Get current NCAA basketball team ratings and rankings
- Matchups: Get game predictions and results
- Bracket: Create and simulate tournament brackets
- Pool: Simulate tournament pools with multiple entries
- ESPNBracket: Extract bracket data from ESPN Tournament Challenge
- ESPNPool: Analyze ESPN Tournament Challenge bracket pools
- GameImportanceAnalyzer: Analyze the importance of specific games
- BracketAnalysis: Analyze winning strategies across multiple pools
"""

from .bigdance_integration import create_teams_from_standings, simulate_hypothetical_bracket_pool
from .cbb_brackets import Bracket, Game, Pool, Team
from .wn_cbb_scraper import Matchups, Schedule, Standings, elo_prob
from .espn_tc_scraper import ESPNScraper, ESPNBracket, ESPNPool, GameImportanceAnalyzer
from .bracket_analysis import BracketAnalysis

__version__ = "0.3.0"
__author__ = "Taylor Firman"
__email__ = "tefirman@gmail.com"

# List of public objects that should be available when using "from bigdance import *"
__all__ = [
    # Core classes
    "Standings",
    "Matchups",
    "Schedule",
    "Team",
    "Game",
    "Bracket",
    "Pool",
    # ESPN integration
    "ESPNScraper",
    "ESPNBracket",
    "ESPNPool",
    "GameImportanceAnalyzer",
    # Analysis tools
    "BracketAnalysis",
    # Helper functions
    "elo_prob",
    "create_teams_from_standings",
    "simulate_hypothetical_bracket_pool",
]