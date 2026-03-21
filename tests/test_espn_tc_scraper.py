from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from bigdance.cbb_brackets import Bracket, Team
from bigdance.espn_tc_scraper import (
    ESPNApi,
    GameImportanceAnalyzer,
)


# Fixtures
@pytest.fixture
def mock_bracket_with_teams():
    """Create a valid bracket with 64 teams across 4 regions"""
    teams = []
    for region in ["Region 1", "Region 2", "Region 3", "Region 4"]:
        for seed in range(1, 17):
            teams.append(
                Team(
                    f"Team{seed}_{region}",
                    seed,
                    region,
                    1800 - (seed * 30),
                    "Conference",
                )
            )
    return Bracket(teams)


@pytest.fixture
def sample_challenge():
    """Create a minimal challenge response for testing."""
    propositions = []
    seed_matchups = [(1, 16), (8, 9), (5, 12), (4, 13), (6, 11), (3, 14), (7, 10), (2, 15)]

    display_order = 0
    for region_id in range(1, 5):
        for seed1, seed2 in seed_matchups:
            prop_id = f"prop-r{region_id}-{seed1}v{seed2}"
            propositions.append(
                {
                    "id": prop_id,
                    "name": f"Team{seed1} vs Team{seed2}",
                    "status": "LOCKED",
                    "scoringPeriodId": 1,
                    "displayOrder": display_order,
                    "correctOutcomes": [],
                    "possibleOutcomes": [
                        {
                            "id": f"outcome-r{region_id}-s{seed1}",
                            "name": f"Team{seed1}_Region {region_id}",
                            "regionId": region_id,
                            "regionSeed": seed1,
                            "mappings": [
                                {"type": "COMPETITOR_ID", "value": str(seed1 + region_id * 100)},
                            ],
                        },
                        {
                            "id": f"outcome-r{region_id}-s{seed2}",
                            "name": f"Team{seed2}_Region {region_id}",
                            "regionId": region_id,
                            "regionSeed": seed2,
                            "mappings": [
                                {"type": "COMPETITOR_ID", "value": str(seed2 + region_id * 100)},
                            ],
                        },
                    ],
                }
            )
            display_order += 1

    return {
        "propositions": propositions,
        "currentScoringPeriod": {"id": 1, "label": "Round of 64"},
        "scoringPeriods": [
            {"id": i, "label": label}
            for i, label in enumerate(
                ["Round of 64", "Round of 32", "Sweet 16", "Elite 8", "Final Four", "Championship"],
                start=1,
            )
        ],
    }


@pytest.fixture
def sample_entry(sample_challenge):
    """Create a sample entry with 63 picks."""
    picks = []
    # For each first-round game, pick the higher seed (lower number)
    for prop in sample_challenge["propositions"]:
        outcome = prop["possibleOutcomes"][0]  # higher seed
        picks.append(
            {
                "propositionId": prop["id"],
                "outcomesPicked": [{"outcomeId": outcome["id"], "result": "UNDECIDED"}],
                "periodReached": 2,  # wins first round
            }
        )

    # Add later-round picks (31 more for rounds 2-6)
    # Pick teams from region 1 to go far
    region1_outcomes = [
        p["possibleOutcomes"][0]
        for p in sample_challenge["propositions"]
        if p["possibleOutcomes"][0]["regionId"] == 1
    ]
    for i, outcome in enumerate(region1_outcomes[:8]):
        # Sweet 16 picks (8)
        picks.append(
            {
                "propositionId": f"prop-r2-{i}",
                "outcomesPicked": [{"outcomeId": outcome["id"], "result": "UNDECIDED"}],
                "periodReached": 3,
            }
        )
    for i, outcome in enumerate(region1_outcomes[:4]):
        # Elite 8 picks (4)
        picks.append(
            {
                "propositionId": f"prop-r3-{i}",
                "outcomesPicked": [{"outcomeId": outcome["id"], "result": "UNDECIDED"}],
                "periodReached": 4,
            }
        )
    for i, outcome in enumerate(region1_outcomes[:2]):
        # Final Four picks (2)
        picks.append(
            {
                "propositionId": f"prop-r4-{i}",
                "outcomesPicked": [{"outcomeId": outcome["id"], "result": "UNDECIDED"}],
                "periodReached": 5,
            }
        )
    # Championship pick (1)
    picks.append(
        {
            "propositionId": "prop-r5-0",
            "outcomesPicked": [{"outcomeId": region1_outcomes[0]["id"], "result": "UNDECIDED"}],
            "periodReached": 6,
        }
    )
    # Pad remaining picks to reach 63
    while len(picks) < 63:
        picks.append(
            {
                "propositionId": f"prop-pad-{len(picks)}",
                "outcomesPicked": [{"outcomeId": region1_outcomes[0]["id"], "result": "UNDECIDED"}],
                "periodReached": 2,
            }
        )

    return {"id": "test-entry-id", "name": "Test Entry", "picks": picks[:63]}


# Tests for ESPNApi
class TestESPNApi:
    def test_initialization(self):
        """Test ESPNApi initialization"""
        with patch("bigdance.wn_cbb_scraper.Standings"):
            api = ESPNApi()
            assert api.women is False
            assert "tournament-challenge-bracket-2026" in api.challenge_slug

            api_women = ESPNApi(women=True)
            assert api_women.women is True
            assert "-women-" in api_women.challenge_slug

    def test_build_outcome_map(self, sample_challenge):
        """Test building outcome map from challenge data"""
        with patch("bigdance.wn_cbb_scraper.Standings"):
            api = ESPNApi()
            outcome_map = api._build_outcome_map(sample_challenge)

            # Should have 64 outcomes (2 per proposition × 32 propositions)
            assert len(outcome_map) == 64

            # Check structure of an outcome
            first_key = list(outcome_map.keys())[0]
            entry = outcome_map[first_key]
            assert "name" in entry
            assert "seed" in entry
            assert "region_id" in entry

    def test_build_prop_map(self, sample_challenge):
        """Test building proposition map from challenge data"""
        with patch("bigdance.wn_cbb_scraper.Standings"):
            api = ESPNApi()
            prop_map = api._build_prop_map(sample_challenge)

            assert len(prop_map) == 32
            first_key = list(prop_map.keys())[0]
            entry = prop_map[first_key]
            assert "status" in entry
            assert "scoring_period" in entry
            assert "winner_id" in entry

    def test_build_teams(self, sample_challenge):
        """Test building teams from challenge data"""
        with patch("bigdance.wn_cbb_scraper.Standings"):
            api = ESPNApi()
            api.ratings_source = None  # Force seed-based ratings
            teams = api.build_teams(sample_challenge)

            assert len(teams) == 64
            regions = set(t.region for t in teams)
            assert len(regions) == 4
            for region in regions:
                region_teams = [t for t in teams if t.region == region]
                assert len(region_teams) == 16
                seeds = sorted(t.seed for t in region_teams)
                assert seeds == list(range(1, 17))

    def test_build_actual_bracket_no_results(self, sample_challenge):
        """Test building actual bracket when no games are complete"""
        with patch("bigdance.wn_cbb_scraper.Standings"):
            api = ESPNApi()
            api.ratings_source = None
            bracket = api.build_actual_bracket(sample_challenge)

            assert len(bracket.teams) == 64
            assert len(bracket.games) == 32
            assert len(bracket.results["First Round"]) == 0

    def test_build_actual_bracket_with_results(self, sample_challenge):
        """Test building actual bracket with some completed games"""
        # Mark first proposition as complete with team1 winning
        prop = sample_challenge["propositions"][0]
        prop["status"] = "COMPLETE"
        prop["correctOutcomes"] = [prop["possibleOutcomes"][0]["id"]]

        with patch("bigdance.wn_cbb_scraper.Standings"):
            api = ESPNApi()
            api.ratings_source = None
            bracket = api.build_actual_bracket(sample_challenge)

            assert len(bracket.results["First Round"]) == 1

    def test_build_entry_bracket(self, sample_challenge, sample_entry):
        """Test building an entry bracket from picks"""
        with patch("bigdance.wn_cbb_scraper.Standings"):
            api = ESPNApi()
            api.ratings_source = None
            teams = api.build_teams(sample_challenge)
            bracket = api.build_entry_bracket(sample_entry, sample_challenge, teams)

            assert bracket is not None
            assert len(bracket.results["First Round"]) == 32
            # Fixture picks 8 teams from region 1 to advance past R1
            assert len(bracket.results["Second Round"]) == 8

    def test_build_entry_bracket_incomplete(self, sample_challenge):
        """Test that incomplete entries return None"""
        with patch("bigdance.wn_cbb_scraper.Standings"):
            api = ESPNApi()
            api.ratings_source = None
            teams = api.build_teams(sample_challenge)

            incomplete_entry = {"name": "Bad Entry", "picks": [{"fake": "data"}] * 10}
            result = api.build_entry_bracket(incomplete_entry, sample_challenge, teams)
            assert result is None

    @patch("bigdance.espn_tc_scraper.requests.get")
    def test_fetch_challenge(self, mock_get):
        """Test fetching challenge data"""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"propositions": [], "scoringPeriods": []}
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        with patch("bigdance.wn_cbb_scraper.Standings"):
            api = ESPNApi()
            result = api.fetch_challenge()

            assert "propositions" in result
            mock_get.assert_called_once()
            assert "gambit-api" in mock_get.call_args[0][0]

    @patch("bigdance.espn_tc_scraper.requests.get")
    def test_fetch_group(self, mock_get):
        """Test fetching group data"""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"entries": [{"id": "e1", "name": "Entry 1"}]}
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        with patch("bigdance.wn_cbb_scraper.Standings"):
            api = ESPNApi()
            result = api.fetch_group("test-group-id")

            assert "entries" in result
            assert len(result["entries"]) == 1

    @patch("bigdance.espn_tc_scraper.requests.get")
    def test_fetch_entry(self, mock_get):
        """Test fetching entry data"""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"name": "Test Entry", "picks": []}
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        with patch("bigdance.wn_cbb_scraper.Standings"):
            api = ESPNApi()
            result = api.fetch_entry("test-entry-id")

            assert result["name"] == "Test Entry"


# Tests for GameImportanceAnalyzer
class TestGameImportanceAnalyzer:
    def test_initialization(self):
        """Test GameImportanceAnalyzer initialization"""
        mock_pool = MagicMock()
        analyzer = GameImportanceAnalyzer(mock_pool)
        assert analyzer.pool is mock_pool

    def test_analyze_win_importance(self):
        """Test analyzing the importance of games"""
        mock_pool = MagicMock()
        mock_bracket = MagicMock()
        mock_bracket.infer_current_round.return_value = "Sweet 16"

        team1 = Team("Kansas", 1, "Midwest", 1800, "Big 12")
        team2 = Team("Duke", 1, "East", 1750, "ACC")

        mock_bracket.results = {
            "First Round": [team1, team2],
            "Second Round": [team1, team2],
            "Sweet 16": [],
            "Elite 8": [],
            "Final Four": [],
            "Championship": [],
        }

        mock_pool.actual_results = mock_bracket
        mock_pool.simulate_pool.return_value = pd.DataFrame(
            {"name": ["Entry1", "Entry2"], "win_pct": [0.6, 0.4]}
        )

        analyzer = GameImportanceAnalyzer(mock_pool)

        with patch.object(analyzer, "_get_teams_in_round") as mock_get_teams:
            mock_get_teams.return_value = [team1, team2]

            with patch.object(analyzer, "_analyze_matchup") as mock_analyze:
                mock_analyze.return_value = {
                    "matchup": "Kansas vs Duke",
                    "region": "Midwest",
                    "max_impact": 0.3,
                    "avg_impact": 0.2,
                    "team1": {"name": "Kansas", "seed": 1},
                    "team2": {"name": "Duke", "seed": 1},
                    "max_impact_entry": "Entry1",
                }

                results = analyzer.analyze_win_importance()

                assert len(results) == 1
                assert mock_analyze.call_count == 1

    def test_print_importance_summary(self):
        """Test printing a human-readable summary of game importance analysis"""
        analyzer = GameImportanceAnalyzer(MagicMock())

        game_importance = [
            {
                "matchup": "Kansas vs Duke",
                "region": "Midwest",
                "team1": {"name": "Kansas", "seed": 1},
                "team2": {"name": "Duke", "seed": 1},
                "max_impact": 0.25,
                "avg_impact": 0.15,
                "max_impact_entry": "Entry1",
                "entry_win_pct_diff": {
                    "Entry1": {
                        "team1_wins": 0.7,
                        "team2_wins": 0.45,
                        "baseline": 0.6,
                        "impact": 0.25,
                    }
                },
                "all_entries_impact": [
                    {
                        "name": "Entry1",
                        "win_pct_team1": 0.7,
                        "win_pct_team2": 0.45,
                        "win_pct_baseline": 0.6,
                        "impact": 0.25,
                    },
                    {
                        "name": "Entry2",
                        "win_pct_team1": 0.3,
                        "win_pct_team2": 0.4,
                        "win_pct_baseline": 0.35,
                        "impact": 0.1,
                    },
                ],
            }
        ]

        with patch("builtins.print") as mock_print:
            analyzer.print_importance_summary(game_importance)
            assert mock_print.call_count > 5

        with patch("builtins.print") as mock_print:
            analyzer.print_importance_summary(game_importance, entry_name="Entry1")
            assert mock_print.call_count > 5

        with patch("builtins.print") as mock_print:
            analyzer.print_importance_summary([])
            mock_print.assert_called_with("No games analyzed.")

        with patch("builtins.print") as mock_print:
            analyzer.print_importance_summary(game_importance, entry_name="NonExistentEntry")
            mock_print.assert_any_call(
                "Warning: Entry 'NonExistentEntry' not found in the analysis data."
            )


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
