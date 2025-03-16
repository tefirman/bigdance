from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from bigdance.bigdance_integration import (
    create_teams_from_standings,
    simulate_bracket_pool,
)
from bigdance.wn_cbb_scraper import Standings


@pytest.fixture
def mock_standings():
    """Create mock standings data for testing"""
    standings = MagicMock()

    # Create sample standings DataFrame with all major conferences
    teams_data = {
        "ACC": [
            ("Duke", 1800),
            ("North Carolina", 1780),
            ("Virginia", 1760),
            ("Miami FL", 1740),
            ("Clemson", 1720),
            ("Pittsburgh", 1700),
            ("Wake Forest", 1680),
            ("NC State", 1660),
        ],
        "Big 12": [
            ("Kansas", 1795),
            ("Houston", 1775),
            ("Texas", 1755),
            ("Baylor", 1735),
            ("Iowa State", 1715),
            ("TCU", 1695),
            ("Oklahoma", 1675),
            ("Kansas State", 1655),
        ],
        "Big East": [
            ("UConn", 1790),
            ("Marquette", 1770),
            ("Creighton", 1750),
            ("Villanova", 1730),
            ("Providence", 1710),
            ("Butler", 1690),
            ("St. John's", 1670),
            ("Seton Hall", 1650),
        ],
        "Big Ten": [
            ("Purdue", 1785),
            ("Illinois", 1765),
            ("Michigan State", 1745),
            ("Wisconsin", 1725),
            ("Indiana", 1705),
            ("Ohio State", 1685),
            ("Michigan", 1665),
            ("Iowa", 1645),
        ],
        "Pac-12": [
            ("Arizona", 1780),
            ("UCLA", 1760),
            ("USC", 1740),
            ("Utah", 1720),
            ("Colorado", 1700),
            ("Oregon", 1680),
            ("Washington", 1660),
            ("Stanford", 1640),
        ],
        "SEC": [
            ("Tennessee", 1775),
            ("Kentucky", 1755),
            ("Alabama", 1735),
            ("Auburn", 1715),
            ("Florida", 1695),
            ("Arkansas", 1675),
            ("Texas A&M", 1655),
            ("South Carolina", 1635),
        ],
        "American": [
            ("Memphis", 1720),
            ("FAU", 1700),
            ("SMU", 1680),
            ("South Florida", 1660),
            ("UAB", 1640),
            ("Charlotte", 1620),
        ],
        "Mountain West": [
            ("San Diego State", 1710),
            ("Utah State", 1690),
            ("Nevada", 1670),
            ("Boise State", 1650),
            ("New Mexico", 1630),
            ("Colorado State", 1610),
        ],
        "WCC": [
            ("Gonzaga", 1750),
            ("Saint Mary's", 1730),
            ("San Francisco", 1650),
            ("Santa Clara", 1630),
            ("Loyola Marymount", 1610),
        ],
        "Atlantic 10": [
            ("Dayton", 1700),
            ("VCU", 1680),
            ("Richmond", 1640),
            ("Saint Louis", 1620),
            ("Davidson", 1600),
        ],
    }

    # Convert to DataFrame format
    teams = []
    for conference, team_list in teams_data.items():
        for team, elo in team_list:
            teams.append({"Team": team, "ELO": elo, "Conference": conference})

    standings.elo = pd.DataFrame(teams)
    return standings


def test_create_teams_basic(mock_standings):
    """Test basic team creation functionality"""
    bracket = create_teams_from_standings(mock_standings)

    # Check we got the right number of teams
    assert len(bracket.teams) == 64

    # Check all teams have valid seeds (1-16)
    seeds = [team.seed for team in bracket.teams]
    assert min(seeds) >= 1
    assert max(seeds) <= 16

    # Check all teams have valid regions
    regions = set(team.region for team in bracket.teams)
    assert regions == {"East", "West", "South", "Midwest"}

    # Check highest rated team from each conference got in
    conference_champs = {
        "ACC": "Duke",
        "Big 12": "Kansas",
        "Big East": "UConn",
        "Big Ten": "Purdue",
        "Pac-12": "Arizona",
        "SEC": "Tennessee",
        "American": "Memphis",
        "Mountain West": "San Diego State",
        "WCC": "Gonzaga",
        "Atlantic 10": "Dayton",
    }

    tournament_teams = [team.name for team in bracket.teams]
    for conference, expected_champ in conference_champs.items():
        assert expected_champ in tournament_teams


def test_create_teams_seeding(mock_standings):
    """Test that teams are seeded appropriately based on ratings"""
    bracket = create_teams_from_standings(mock_standings)

    # Top 4 overall seeds should be 1 seeds
    one_seeds = [team for team in bracket.teams if team.seed == 1]
    assert len(one_seeds) == 4
    assert all(team.name in ["Duke", "Kansas", "UConn", "Purdue"] for team in one_seeds)

    # Check that each region has one team of each seed
    for region in ["East", "West", "South", "Midwest"]:
        region_teams = [team for team in bracket.teams if team.region == region]
        region_seeds = [team.seed for team in region_teams]
        assert sorted(region_seeds) == list(range(1, 17))


def test_create_teams_with_regions(mock_standings):
    """Test team creation with predefined regions"""
    regions = {"Duke": "East", "Kansas": "West", "UConn": "South", "Purdue": "Midwest"}

    bracket = create_teams_from_standings(mock_standings, regions=regions)

    # Verify specified teams went to correct regions
    for team in bracket.teams:
        if team.name in regions:
            assert team.region == regions[team.name]


def test_create_teams_validation(mock_standings):
    """Test bracket structure validation"""
    # Create invalid regions mapping that would put too many teams in one region
    invalid_regions = {team: "East" for team in mock_standings.elo["Team"]}

    with pytest.raises(ValueError):
        create_teams_from_standings(mock_standings, regions=invalid_regions)


def test_simulate_bracket_pool(mock_standings):
    """Test bracket pool simulation"""
    num_entries = 5
    results = simulate_bracket_pool(mock_standings, num_entries=num_entries)

    # Check basic structure of results
    assert len(results) == num_entries
    assert all(
        col in results.columns
        for col in ["name", "avg_score", "std_score", "wins", "win_pct"]
    )

    # Check win percentages sum to approximately 1
    assert abs(results["win_pct"].sum() - 1.0) < 0.01


def test_simulate_bracket_pool_with_upset_factors(mock_standings):
    """Test bracket pool simulation with custom upset factors"""
    num_entries = 3
    upset_factors = [0.1, 0.2, 0.3]

    # Just verify that the simulation runs with custom upset factors
    results = simulate_bracket_pool(
        mock_standings, num_entries=num_entries, upset_factors=upset_factors
    )

    # Check basic structure and requirements
    assert len(results) == num_entries
    assert all(
        col in results.columns
        for col in ["name", "avg_score", "std_score", "wins", "win_pct"]
    )
    assert abs(results["win_pct"].sum() - 1.0) < 0.01  # Win percentages should sum to 1

    # Verify all entries have some non-zero standard deviation
    assert all(results["std_score"] > 0)


def test_integration_with_real_standings(mock_standings):
    """Test full integration with actual Warren Nolan data"""
    with patch("bigdance.bigdance_integration.Standings") as mock_standings_class:
        # Configure mock to return our fixture data
        mock_standings_class.return_value = mock_standings

        # Try running the main simulation
        results = simulate_bracket_pool(Standings(), num_entries=5)

        assert len(results) == 5
        assert all(
            col in results.columns
            for col in ["name", "avg_score", "std_score", "wins", "win_pct"]
        )


def test_simulate_bracket_pool_with_full_upset_range(mock_standings):
    """Test bracket pool simulation with the full range of upset factors (-1.0 to 1.0)"""
    num_entries = 5
    # Include a full range from extreme chalk to coin flip
    upset_factors = [-0.8, -0.4, 0.0, 0.4, 0.8]

    # Just verify that the simulation runs with the new upset factor range
    results = simulate_bracket_pool(
        mock_standings, num_entries=num_entries, upset_factors=upset_factors
    )

    # Check basic structure and requirements
    assert len(results) == num_entries
    assert all(
        col in results.columns
        for col in ["name", "avg_score", "std_score", "wins", "win_pct"]
    )
    assert abs(results["win_pct"].sum() - 1.0) < 0.01  # Win percentages should sum to 1

    # Verify all entries have some non-zero standard deviation
    assert all(results["std_score"] > 0)

    # Verify the simulation runs with out-of-bounds values by clipping them
    # This tests the robustness of the implementation
    extreme_factors = [-2.0, -0.5, 0.5, 2.0, 0.0]
    results_extreme = simulate_bracket_pool(
        mock_standings, num_entries=num_entries, upset_factors=extreme_factors
    )
    assert len(results_extreme) == num_entries
