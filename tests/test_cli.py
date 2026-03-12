from unittest.mock import patch

import pandas as pd
import pytest

from bigdance.cli import main


@pytest.fixture
def mock_simulate_df():
    """Minimal DataFrame that simulate_round_probabilities would return."""
    return pd.DataFrame(
        {
            "Team": ["TeamA", "TeamB"],
            "Seed": [1, 2],
            "Region": ["East", "West"],
            "First Round": ["100.0%", "100.0%"],
            "Second Round": ["80.0%", "60.0%"],
            "Sweet 16": ["50.0%", "30.0%"],
            "Elite 8": ["30.0%", "15.0%"],
            "Final Four": ["20.0%", "5.0%"],
            "Championship": ["10.0%", "2.0%"],
        }
    )


@pytest.fixture
def mock_pool_df():
    """Minimal DataFrame that simulate_pool would return."""
    return pd.DataFrame(
        {
            "name": ["Entry1", "Entry2"],
            "avg_score": [120.0, 110.0],
            "std_score": [20.0, 18.0],
            "wins": [600, 400],
            "win_pct": [0.6, 0.4],
            "win_prob": ["60.0%", "40.0%"],
        }
    )


def test_no_command_prints_help(capsys):
    """Running bigdance with no subcommand prints help and returns 0."""
    result = main([])
    assert result == 0
    captured = capsys.readouterr()
    assert "usage" in captured.out.lower() or "commands" in captured.out.lower()


def test_simulate_subcommand(mock_simulate_df, capsys):
    """bigdance simulate calls simulate_round_probabilities and prints results."""
    with (
        patch("bigdance.bigdance_integration.Standings"),
        patch(
            "bigdance.bigdance_integration.simulate_round_probabilities",
            return_value=mock_simulate_df,
        ) as mock_fn,
    ):
        result = main(["simulate", "--num_sims", "10"])

    assert result == 0
    mock_fn.assert_called_once()
    captured = capsys.readouterr()
    assert "TeamA" in captured.out


def test_simulate_top_flag(mock_simulate_df, capsys):
    """--top N limits output to N rows."""
    with (
        patch("bigdance.bigdance_integration.Standings"),
        patch(
            "bigdance.bigdance_integration.simulate_round_probabilities",
            return_value=mock_simulate_df,
        ),
    ):
        main(["simulate", "--num_sims", "10", "--top", "1"])

    captured = capsys.readouterr()
    assert "TeamA" in captured.out
    assert "TeamB" not in captured.out


def test_simulate_women_flag():
    """--women flag is forwarded to Standings and simulate_round_probabilities."""
    with (
        patch("bigdance.bigdance_integration.Standings") as mock_standings_cls,
        patch(
            "bigdance.bigdance_integration.simulate_round_probabilities",
            return_value=pd.DataFrame(
                columns=[
                    "Team",
                    "Seed",
                    "Region",
                    "First Round",
                    "Second Round",
                    "Sweet 16",
                    "Elite 8",
                    "Final Four",
                    "Championship",
                ]
            ),
        ) as mock_fn,
    ):
        main(["simulate", "--num_sims", "5", "--women"])

    mock_standings_cls.assert_called_once_with(conference=None, women=True)
    _, kwargs = mock_fn.call_args
    assert kwargs.get("women") is True


def test_standings_subcommand():
    """bigdance standings delegates to wn_cbb_scraper.main."""
    with patch("bigdance.wn_cbb_scraper.main") as mock_main:
        result = main(["standings"])
    assert result == 0
    mock_main.assert_called_once_with([])


def test_espn_subcommand_pool(mock_pool_df):
    """bigdance espn --pool_id delegates to espn_tc_scraper.main."""
    with patch("bigdance.espn_tc_scraper.main", return_value=0) as mock_main:
        result = main(["espn", "--pool_id", "abc123"])
    assert result == 0
    mock_main.assert_called_once_with(["--pool_id", "abc123"])


def test_espn_subcommand_team_probs(mock_simulate_df, capsys):
    """bigdance espn --team_probs prints round probabilities and returns early."""
    with patch("bigdance.espn_tc_scraper.main", return_value=0) as mock_main:
        result = main(["espn", "--pool_id", "abc123", "--team_probs"])
    assert result == 0
    mock_main.assert_called_once_with(["--pool_id", "abc123", "--team_probs"])


def test_analyze_subcommand():
    """bigdance analyze delegates to bracket_analysis.main."""
    with patch("bigdance.bracket_analysis.main", return_value=0) as mock_main:
        result = main(["analyze"])
    assert result == 0
    mock_main.assert_called_once_with([])
