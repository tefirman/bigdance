from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from bigdance.wn_cbb_scraper import BaseScraper, Matchups, Schedule, Standings

SAMPLE_ELO_HTML = """
<table class="normal-grid alternating-rows stats-table">
    <thead>
        <tr>
            <th class="top-header">Team</th>
            <th class="top-header">Record</th>
            <th class="top-header"><abbr title="ELO Chess Rating">ELO</abbr></th>
            <th class="top-header cell-right-black"><abbr title="ELO Chess Ranking">Rank</abbr></th>
            <th class="top-header"><abbr title="ELO Chess Ranking Delta">ELO Delta</abbr></th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td class="data-cell data-medium">
                <div class="logo-name-container">
                    <div class="logo-subcontainer"><ul class="team-logo"><li><a class="Duke" href="/basketball/2025/schedule/Duke"></a></li></ul></div>
                    <div class="name-subcontainer"><a class="blue-black" href="/basketball/2025/schedule/Duke">Duke</a></div>
                </div>
            </td>
            <td class="data-cell data-center data-medium">15-2</td>
            <td class="data-cell data-center data-medium">1750.50</td>
            <td class="data-cell data-center data-medium cell-right-black">1</td>
            <td class="data-cell data-center data-medium data-bold" style="background-color: #99FF00; color: black;">+2</td>
        </tr>
        <tr>
            <td class="data-cell data-medium">
                <div class="logo-name-container">
                    <div class="logo-subcontainer"><ul class="team-logo"><li><a class="North-Carolina" href="/basketball/2025/schedule/North-Carolina"></a></li></ul></div>
                    <div class="name-subcontainer"><a class="blue-black" href="/basketball/2025/schedule/North-Carolina">North Carolina</a></div>
                </div>
            </td>
            <td class="data-cell data-center data-medium">14-3</td>
            <td class="data-cell data-center data-medium">1725.30</td>
            <td class="data-cell data-center data-medium cell-right-black">2</td>
            <td class="data-cell data-center data-medium data-bold" style="background-color: #FF0000; color: white;">-1</td>
        </tr>
    </tbody>
</table>
"""

SAMPLE_RANKS_HTML = """
<table class="normal-grid alternating-rows stats-table">
    <thead>
        <tr>
            <th class="top-header-dark">Rank</th>
            <th class="top-header-dark">Team</th>
            <th class="top-header-dark">Record</th>
            <th class="top-header-dark">Pts</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td class="data-cell data-center data-medium">1</td>
            <td class="data-cell data-medium">
                <div class="logo-name-container">
                    <div class="logo-subcontainer"><ul class="team-logo"><li><a class="Duke" href="/basketball/2025/schedule/Duke"></a></li></ul></div>
                    <div class="name-subcontainer"><a class="blue-black" href="/basketball/2025/schedule/Duke">Duke</a></div>
                </div>
            </td>
            <td class="data-cell data-center data-medium">15-2</td>
            <td class="data-cell data-center data-medium cell-right-black">1550</td>
        </tr>
        <tr>
            <td class="data-cell data-center data-medium">2</td>
            <td class="data-cell data-medium">
                <div class="logo-name-container">
                    <div class="logo-subcontainer"><ul class="team-logo"><li><a class="North-Carolina" href="/basketball/2025/schedule/North-Carolina"></a></li></ul></div>
                    <div class="name-subcontainer"><a class="blue-black" href="/basketball/2025/schedule/North-Carolina">North Carolina</a></div>
                </div>
            </td>
            <td class="data-cell data-center data-medium">14-3</td>
            <td class="data-cell data-center data-medium cell-right-black">1484</td>
        </tr>
    </tbody>
</table>
"""

SAMPLE_MATCHUPS_HTML = """
<div id="20635" class="pbox">
    <div class="pbox__header"></div>
    <div class="pbox__info">
        <table>
            <tr class="pbox__info-top-row">
                <th class="live-game-cell" rowspan="2"><img class="live-game" src="/images/live.gif" /></th>
                <th class="time-clock center-line" colspan="2" rowspan="2">End of 2nd Half</th>
                <th class="predict-heading" colspan="5">RP Prediction</th>
            </tr>
            <tr>
                <td class="heading2">Score</td>
                <td class="heading2">O/U-Spread</td>
                <td class="heading2">Win Prob.</td>
                <td class="heading2">Confidence</td>
                <td class="heading2"></td>
            </tr>
            <tr class="pbox__info-team1-row">
                <td class="winner"></td>
                <td class="team-info">
                    <div class="logo-name-container">
                        <div class="logo-subcontainer"><ul class="team-logo"><li>
                            <a class="Central-Arkansas" href="/basketball/2025/schedule/Central-Arkansas"></a>
                        </li></ul></div>
                        <div class="name-subcontainer"><span class="team-rank"></span>
                            <a class="blue-black" href="/basketball/2025/schedule/Central-Arkansas">Central Arkansas</a><br>
                            (4-16, Road 0-12)
                        </div>
                    </div>
                </td>
                <td class="score center-line">65</td>
                <td class="score">64</td>
                <td class="value">144</td>
                <td class="value">10%</td>
                <td class="value">L</td>
                <td class="value"></td>
            </tr>
            <tr class="pbox__info-team2-row">
                <td class="winner"></td>
                <td class="team-info">
                    <div class="logo-name-container">
                        <div class="logo-subcontainer"><ul class="team-logo"><li>
                            <a class="North-Alabama" href="/basketball/2025/schedule/North-Alabama"></a>
                        </li></ul></div>
                        <div class="name-subcontainer"><span class="team-rank"></span>
                            <a class="blue-black" href="/basketball/2025/schedule/North-Alabama">North Alabama</a><br>
                            (12-8, Home 7-1)
                        </div>
                    </div>
                </td>
                <td class="score center-line">94</td>
                <td class="score">80</td>
                <td class="value">-16</td>
                <td class="value">90%</td>
                <td class="value">L</td>
                <td class="value"></td>
            </tr>
        </table>
    </div>
    <div class="pbox__footer"></div>
</div>
"""

SAMPLE_CONFERENCES_HTML = """
<div class="conferences-list__container conf-height--33-teams">
    <div class="conferences-list__info-block">
        <div class="logo-name-container">
            <div class="logo-subcontainer"><ul class="conf-logo"><li><a class="ACC" href="/basketball/2025/conference/ACC"></a></li></ul></div>
            <div class="name-subcontainer"><a class="blue-black" href="/basketball/2025/conference/ACC">ACC</a></div>
        </div>
        <div class="conferences-list__links">
            <a class="lt-blue-black" href="/basketball/2025/conference/ACC">Standings</a> |
            <a class="lt-blue-black" href="/basketball/2025/conf-schedule?conference=ACC">Schedule</a> |
            <a class="lt-blue-black" href="/basketball/2025/conf-prediction?conference=ACC">Prediction</a>
        </div>
    </div>

    <div class="conferences-list__info-block">
        <div class="logo-name-container">
            <div class="logo-subcontainer"><ul class="conf-logo"><li><a class="Big-12" href="/basketball/2025/conference/Big-12"></a></li></ul></div>
            <div class="name-subcontainer"><a class="blue-black" href="/basketball/2025/conference/Big-12">Big 12</a></div>
        </div>
        <div class="conferences-list__links">
            <a class="lt-blue-black" href="/basketball/2025/conference/Big-12">Standings</a> |
            <a class="lt-blue-black" href="/basketball/2025/conf-schedule?conference=Big-12">Schedule</a> |
            <a class="lt-blue-black" href="/basketball/2025/conf-prediction?conference=Big-12">Prediction</a>
        </div>
    </div>
</div>
"""

SAMPLE_ACC_HTML = """
<div class="main-body-row-flex-scroll">
    <div class="full-width-box-x">
        <table class="normal-grid alternating-rows stats-table">
            <tr>
                <td class="data-cell data-medium">
                    <div class="logo-name-container">
                        <div class="logo-subcontainer"><ul class="team-logo"><li><a class="Duke" href="/basketball/2025/schedule/Duke"></a></li></ul></div>
                        <div class="name-subcontainer"><a class="blue-black" href="/basketball/2025/schedule/Duke">Duke</a></div>
                    </div>
                </td>
            </tr>
            <tr>
                <td class="data-cell data-medium">
                    <div class="logo-name-container">
                        <div class="logo-subcontainer"><ul class="team-logo"><li><a class="North-Carolina" href="/basketball/2025/schedule/North-Carolina"></a></li></ul></div>
                        <div class="name-subcontainer"><a class="blue-black" href="/basketball/2025/schedule/North-Carolina">North Carolina</a></div>
                    </div>
                </td>
            </tr>
            <tr>
                <td class="data-cell data-medium">
                    <div class="logo-name-container">
                        <div class="logo-subcontainer"><ul class="team-logo"><li><a class="Louisville" href="/basketball/2025/schedule/Louisville"></a></li></ul></div>
                        <div class="name-subcontainer"><a class="blue-black" href="/basketball/2025/schedule/Louisville">Louisville</a></div>
                    </div>
                </td>
            </tr>
        </table>
    </div>
</div>
"""

SAMPLE_BIG12_HTML = """
<div class="main-body-row-flex-scroll">
    <div class="full-width-box-x">
        <table class="normal-grid alternating-rows stats-table">
            <tr>
                <td class="data-cell data-medium">
                    <div class="logo-name-container">
                        <div class="logo-subcontainer"><ul class="team-logo"><li><a class="Kansas" href="/basketball/2025/schedule/Kansas"></a></li></ul></div>
                        <div class="name-subcontainer"><a class="blue-black" href="/basketball/2025/schedule/Kansas">Kansas</a></div>
                    </div>
                </td>
            </tr>
            <tr>
                <td class="data-cell data-medium">
                    <div class="logo-name-container">
                        <div class="logo-subcontainer"><ul class="team-logo"><li><a class="Houston" href="/basketball/2025/schedule/Houston"></a></li></ul></div>
                        <div class="name-subcontainer"><a class="blue-black" href="/basketball/2025/schedule/Houston">Houston</a></div>
                    </div>
                </td>
            </tr>
            <tr>
                <td class="data-cell data-medium">
                    <div class="logo-name-container">
                        <div class="logo-subcontainer"><ul class="team-logo"><li><a class="Texas" href="/basketball/2025/schedule/Texas"></a></li></ul></div>
                        <div class="name-subcontainer"><a class="blue-black" href="/basketball/2025/schedule/Texas">Texas</a></div>
                    </div>
                </td>
            </tr>
        </table>
    </div>
</div>
"""


@pytest.fixture
def mock_session():
    """Fixture providing a mock requests session"""
    with patch("requests.Session") as mock:
        session = mock.return_value
        response = MagicMock()
        response.text = SAMPLE_MATCHUPS_HTML
        response.raise_for_status = MagicMock()
        session.get.return_value = response
        yield session


@pytest.fixture
def mock_elo_response():
    """Fixture providing mock ELO rankings page"""
    with patch("requests.Session") as mock:
        session = mock.return_value
        response = MagicMock()
        response.text = SAMPLE_ELO_HTML
        session.get.return_value = response
        yield session


@pytest.fixture
def mock_rank_response():
    """Fixture providing mock rankings page"""
    with patch("requests.Session") as mock:
        session = mock.return_value
        session.get.return_value.text = SAMPLE_RANKS_HTML
        session.get.return_value.raise_for_status = MagicMock()
        yield session


@pytest.fixture
def mock_matchups_response():
    """Fixture providing mock matchups page"""
    with patch("requests.Session") as mock:
        session = mock.return_value
        response = MagicMock()
        response.text = SAMPLE_MATCHUPS_HTML
        session.get.return_value = response
        yield session


@pytest.fixture
def mock_standings_responses():
    """Fixture providing mock responses for all Standings HTML requests"""
    mock = MagicMock()

    def mock_get(url, **kwargs):
        print(f"Mock received request for URL: {url}")  # Debug URL

        response = MagicMock()
        response.raise_for_status = MagicMock()

        if "elo" in url:
            response.text = SAMPLE_ELO_HTML
        elif "/conferences" in url:
            response.text = SAMPLE_CONFERENCES_HTML
        elif "/conference/ACC" in url:
            response.text = SAMPLE_ACC_HTML
        elif "/conference/Big-12" in url:
            response.text = SAMPLE_BIG12_HTML
        elif "/polls-expanded" in url:
            response.text = SAMPLE_RANKS_HTML
        else:
            response.text = ""
            print(f"WARNING: No mock response for URL: {url}")

        return response

    mock.get = MagicMock(side_effect=mock_get)
    mock.mount = MagicMock()
    mock.adapters = {}

    return mock


@pytest.fixture
def clean_cache():
    """Fixture to provide a clean cache directory for each test"""
    import os
    import shutil

    cache_dir = "test_cache"
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
    os.makedirs(cache_dir)
    yield cache_dir
    # Clean up after test
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)


class TestBaseScraper:
    def test_initialization(self, mock_session):
        """Test basic scraper initialization"""
        scraper = BaseScraper(cache_dir="test_cache")
        assert scraper.cache_dir == Path("test_cache")
        assert scraper.cache_dir.exists()

    def test_courteous_get(self, mock_session):
        """Test courteous GET request functionality"""
        with patch(
            "bigdance.wn_cbb_scraper.BaseScraper._create_session",
            return_value=mock_session,
        ):
            # Set up the mock response
            mock_response = MagicMock()
            mock_response.text = "<html>Mock HTML</html>"
            mock_session.get.return_value = mock_response

            scraper = BaseScraper(cache_dir="test_cache")

            # Ensure cache miss by mocking get_cached_response
            with patch.object(scraper, "_get_cached_response", return_value=None):
                response = scraper.courteous_get("https://test.com", "test_key")

                # Verify response
                assert response == "<html>Mock HTML</html>"

                # Verify get was called
                mock_session.get.assert_called_once_with("https://test.com")

    def test_cache_handling(self, clean_cache):
        """Test cache read/write operations"""
        scraper = BaseScraper(cache_dir=clean_cache)

        # Test cache miss
        cached = scraper._get_cached_response("https://test.com", "test_key")
        assert cached is None

        # Test cache write
        scraper._cache_response("https://test.com", "test_key", "<html>Cached</html>")

        # Test cache hit
        cached = scraper._get_cached_response("https://test.com", "test_key")
        assert cached == "<html>Cached</html>"


class TestStandings:
    def test_initialization(self, mock_elo_response, mock_rank_response):
        """Test standings initialization"""
        with patch("bigdance.wn_cbb_scraper.Standings._create_session") as mock_session:
            # Configure mock to return different responses for ELO vs ranks
            def mock_get(url):
                if "elo" in url:
                    return MagicMock(text=SAMPLE_ELO_HTML)
                else:
                    return MagicMock(text=SAMPLE_RANKS_HTML)

            mock_session.return_value.get = MagicMock(side_effect=mock_get)

            standings = Standings(season=2024)
            assert standings.season == 2024
            assert not standings.gender
            assert standings.base_url == "https://www.warrennolan.com/basketball/2024"

    def test_elo_parsing(self, mock_elo_response):
        """Test parsing of ELO rankings"""
        with patch("bigdance.wn_cbb_scraper.Standings._create_session") as mock_session:
            mock_session.return_value.get.return_value = MagicMock(text=SAMPLE_ELO_HTML)
            standings = Standings(season=2024)
            assert len(standings.elo) == 2
            assert standings.elo.iloc[0]["Team"] == "Duke"
            assert float(standings.elo.iloc[0]["ELO"]) == 1750.50

    def test_full_conference_list(self, mock_standings_responses):
        """Test parsing of full conference list"""
        with patch("requests.Session", autospec=True) as mock_session_class:
            # Configure the mock session to return our mock_standings_responses
            mock_session_class.return_value = mock_standings_responses

            # Add debug logging
            print("Mock session configured")

            standings = Standings(season=2024)

            # Add more debug logging
            print(f"Mock get calls: {mock_standings_responses.get.call_args_list}")

            # Test that both conferences are found (from our sample HTML)
            assert (
                len(standings.conferences) == 2
            ), f"Found conferences: {standings.conferences}"
            assert "ACC" in standings.conferences
            assert "Big 12" in standings.conferences

    def test_conference_teams_parsing(self, mock_standings_responses):
        """Test parsing teams within a conference"""
        with patch("requests.Session", autospec=True) as mock_session_class:
            # Configure the mock session to return our mock_standings_responses
            mock_session_class.return_value = mock_standings_responses

            standings = Standings(season=2024)
            teams = standings.pull_conference_teams("ACC")

            # Debug logging
            print(f"Mock get calls: {mock_standings_responses.get.call_args_list}")
            print(f"Found teams: {teams}")

            # Verify expected teams are found
            assert len(teams) == 3, f"Found teams: {teams}"
            assert "Duke" in teams
            assert "North Carolina" in teams
            assert "Louisville" in teams

    def test_conference_filtering(self, mock_standings_responses):
        """Test filtering standings by conference"""
        with patch("requests.Session", autospec=True) as mock_session_class:
            # Configure the mock session to return our mock_standings_responses
            mock_session_class.return_value = mock_standings_responses

            # Add debug print to see initial data
            standings = Standings(season=2024, conference="ACC")
            print("\nInitial ELO data:")
            print("Number of teams:", len(standings.elo))
            print("First few rows:")
            print(standings.elo)

            # Should only show the teams from our SAMPLE_ELO_HTML
            assert (
                len(standings.elo) == 2
            ), "Should only have Duke and UNC from sample data"
            # Verify only ACC teams are included
            for _, row in standings.elo.iterrows():
                assert row["Conference"] == "ACC"

    def test_women_parameter(self, mock_elo_response):
        """Test women's basketball parameter"""
        with patch(
            "bigdance.wn_cbb_scraper.Standings._create_session",
            return_value=mock_elo_response,
        ):
            standings = Standings(season=2024, women=True)
            assert standings.gender == "w"
            assert "basketballw" in standings.base_url


class TestMatchups:
    def test_initialization(self, mock_matchups_response):
        """Test matchups initialization"""
        with patch(
            "bigdance.wn_cbb_scraper.Matchups._create_session",
            return_value=mock_matchups_response,
        ):
            matchups = Matchups(date=datetime.now(), elos=False)  # Disable ELO addition
            assert isinstance(matchups.matchups, pd.DataFrame)

    def test_matchups_parsing(self, mock_matchups_response):
        """Test parsing of matchups data"""
        with patch(
            "bigdance.wn_cbb_scraper.Matchups._create_session",
            return_value=mock_matchups_response,
        ):
            # Initialize without ELO calculations
            matchups = Matchups(date=datetime.now(), elos=False)
            assert len(matchups.matchups) == 1
            assert (
                matchups.matchups.iloc[0]["team1"] == "Central Arkansas"
            )  # Update expected team name
            assert matchups.matchups.iloc[0]["score1"] == 65  # Update expected score

    def test_elo_addition(self, mock_matchups_response):
        """Test adding ELO ratings to matchups"""
        with patch(
            "bigdance.wn_cbb_scraper.Matchups._create_session",
            return_value=mock_matchups_response,
        ):
            # Create matchups without initial ELO calculation
            matchups = Matchups(date=datetime.now(), elos=False)

            # Create mock standings with matching team names
            mock_standings = MagicMock()
            mock_standings.elo = pd.DataFrame(
                {
                    "Team": ["Central Arkansas", "North Alabama"],  # Update team names
                    "ELO": [1600, 1550],
                }
            )

            matchups.add_elos(mock_standings)
            assert "elo1" in matchups.matchups.columns
            assert "elo2" in matchups.matchups.columns
            assert matchups.matchups.iloc[0]["elo1"] == 1600


class TestSchedule:
    def test_initialization(self, mock_session):
        """Test schedule initialization"""
        with patch(
            "bigdance.wn_cbb_scraper.Schedule._create_session",
            return_value=mock_session,
        ):
            start = datetime.now()
            stop = start + timedelta(days=7)
            schedule = Schedule(
                start=start.strftime("%Y-%m-%d"),
                stop=stop.strftime("%Y-%m-%d"),
                elos=False,  # Disable ELO calculations during testing
            )
            assert schedule.start == pd.to_datetime(start.strftime("%Y-%m-%d"))
            assert schedule.stop == pd.to_datetime(stop.strftime("%Y-%m-%d"))

    def test_date_validation(self):
        """Test date range validation"""
        with pytest.raises(ValueError):
            start = datetime.now()
            stop = start - timedelta(days=1)
            Schedule(start=start.strftime("%Y-%m-%d"), stop=stop.strftime("%Y-%m-%d"))

    def test_pull_games(self, mock_session):
        """Test game pulling functionality"""
        with patch(
            "bigdance.wn_cbb_scraper.Schedule._create_session",
            return_value=mock_session,
        ):
            start = datetime.now()
            stop = start + timedelta(days=7)
            schedule = Schedule(
                start=start.strftime("%Y-%m-%d"),
                stop=stop.strftime("%Y-%m-%d"),
                elos=False,  # Disable ELO calculations
            )
            assert hasattr(schedule, "games_per_day")

    def test_women_parameter(self, mock_session):
        """Test women's basketball parameter"""
        with patch(
            "bigdance.wn_cbb_scraper.Schedule._create_session",
            return_value=mock_session,
        ):
            start = datetime.now()
            stop = start + timedelta(days=7)
            schedule = Schedule(
                start=start.strftime("%Y-%m-%d"),
                stop=stop.strftime("%Y-%m-%d"),
                women=True,
                elos=False,  # Disable ELO calculations
            )
            assert schedule.gender == "w"
