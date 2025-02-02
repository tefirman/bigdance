import pytest
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
from pathlib import Path
from src.dancing.wn_cbb_scraper import BaseScraper, Standings, Matchups, Schedule

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
                    <div class="logo-subcontainer"><ul class="team-logo"><li><a class="Team-A" href="/basketball/2025/schedule/Team-A"></a></li></ul></div>
                    <div class="name-subcontainer"><a class="blue-black" href="/basketball/2025/schedule/Team-A">Team A</a></div>
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
                    <div class="logo-subcontainer"><ul class="team-logo"><li><a class="Team-B" href="/basketball/2025/schedule/Team-B"></a></li></ul></div>
                    <div class="name-subcontainer"><a class="blue-black" href="/basketball/2025/schedule/Team-B">Team B</a></div>
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
                    <div class="logo-subcontainer"><ul class="team-logo"><li><a class="Team-A" href="/basketball/2025/schedule/Team-A"></a></li></ul></div>
                    <div class="name-subcontainer"><a class="blue-black" href="/basketball/2025/schedule/Team-A">Team A</a>&nbsp;&nbsp;(62)</div>
                </div>
            </td>
            <td class="data-cell data-center data-medium">15-2</td>
            <td class="data-cell data-center data-medium cell-right-black">1550</td>
        </tr>
        <tr>
            <td class="data-cell data-center data-medium">2</td>
            <td class="data-cell data-medium">
                <div class="logo-name-container">
                    <div class="logo-subcontainer"><ul class="team-logo"><li><a class="Team-B" href="/basketball/2025/schedule/Team-B"></a></li></ul></div>
                    <div class="name-subcontainer"><a class="blue-black" href="/basketball/2025/schedule/Team-B">Team B</a></div>
                </div>
            </td>
            <td class="data-cell data-center data-medium">14-3</td>
            <td class="data-cell data-center data-medium cell-right-black">1484</td>
        </tr>
    </tbody>
</table>
"""

SAMPLE_MATCHUPS_HTML = '''
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
'''

@pytest.fixture
def mock_session():
    """Fixture providing a mock requests session"""
    with patch('requests.Session') as mock:
        session = mock.return_value
        response = MagicMock()
        response.text = SAMPLE_MATCHUPS_HTML
        response.raise_for_status = MagicMock()
        session.get.return_value = response
        yield session

@pytest.fixture
def mock_elo_response():
    """Fixture providing mock ELO rankings page"""
    with patch('requests.Session') as mock:
        session = mock.return_value
        response = MagicMock()
        response.text = SAMPLE_ELO_HTML
        session.get.return_value = response
        yield session

@pytest.fixture
def mock_rank_response():
    """Fixture providing mock rankings page"""
    with patch('requests.Session') as mock:
        session = mock.return_value
        session.get.return_value.text = SAMPLE_RANKS_HTML
        session.get.return_value.raise_for_status = MagicMock()
        yield session

@pytest.fixture 
def mock_matchups_response():
    """Fixture providing mock matchups page"""
    with patch('requests.Session') as mock:
        session = mock.return_value
        response = MagicMock()
        response.text = SAMPLE_MATCHUPS_HTML 
        session.get.return_value = response
        yield session

@pytest.fixture
def clean_cache():
    """Fixture to provide a clean cache directory for each test"""
    import shutil
    import os
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
        with patch('dancing.wn_cbb_scraper.BaseScraper._create_session', return_value=mock_session):
            # Set up the mock response
            mock_response = MagicMock()
            mock_response.text = "<html>Mock HTML</html>"
            mock_session.get.return_value = mock_response
            
            scraper = BaseScraper(cache_dir="test_cache")
            
            # Ensure cache miss by mocking get_cached_response
            with patch.object(scraper, '_get_cached_response', return_value=None):
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
        with patch('dancing.wn_cbb_scraper.Standings._create_session') as mock_session:
            # Configure mock to return different responses for ELO vs ranks
            def mock_get(url):
                if 'elo' in url:
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
        with patch('dancing.wn_cbb_scraper.Standings._create_session') as mock_session:
            mock_session.return_value.get.return_value = MagicMock(text=SAMPLE_ELO_HTML)
            standings = Standings(season=2024)
            assert len(standings.elo) == 2
            assert standings.elo.iloc[0]['Team'] == 'Team A'
            assert float(standings.elo.iloc[0]['ELO']) == 1750.50

    def test_conference_filtering(self, mock_elo_response):
        """Test conference-specific standings"""
        with patch('dancing.wn_cbb_scraper.Standings._create_session', return_value=mock_elo_response):
            standings = Standings(season=2024, conference="Conf 1")
            standing_teams = set(standings.elo['Team'])
            assert 'Team A' in standing_teams or 'Team B' in standing_teams

    def test_women_parameter(self, mock_elo_response):
        """Test women's basketball parameter"""
        with patch('dancing.wn_cbb_scraper.Standings._create_session', return_value=mock_elo_response):
            standings = Standings(season=2024, women=True)
            assert standings.gender == "w"
            assert "basketballw" in standings.base_url

class TestMatchups:
    def test_initialization(self, mock_matchups_response):
        """Test matchups initialization"""
        with patch('dancing.wn_cbb_scraper.Matchups._create_session', return_value=mock_matchups_response):
            matchups = Matchups(date=datetime.now(), elos=False)  # Disable ELO addition
            assert isinstance(matchups.matchups, pd.DataFrame)

    def test_matchups_parsing(self, mock_matchups_response):
        """Test parsing of matchups data"""
        with patch('dancing.wn_cbb_scraper.Matchups._create_session', return_value=mock_matchups_response):
            matchups = Matchups(date=datetime.now())
            assert len(matchups.matchups) == 1
            assert matchups.matchups.iloc[0]['team1'] == 'Team A'
            assert matchups.matchups.iloc[0]['score1'] == 75

    def test_elo_addition(self, mock_matchups_response):
        """Test adding ELO ratings to matchups"""
        with patch('dancing.wn_cbb_scraper.Matchups._create_session', return_value=mock_matchups_response):
            matchups = Matchups(date=datetime.now())
            
            mock_standings = MagicMock()
            mock_standings.elo = pd.DataFrame({
                'Team': ['Team A', 'Team B'],
                'ELO': [1600, 1550]
            })
            
            matchups.add_elos(mock_standings)
            assert 'elo1' in matchups.matchups.columns
            assert 'elo2' in matchups.matchups.columns
            assert matchups.matchups.iloc[0]['elo1'] == 1600

class TestSchedule:
    def test_initialization(self, mock_session):
        """Test schedule initialization"""
        with patch('dancing.wn_cbb_scraper.Schedule._create_session', return_value=mock_session):
            start = datetime.now()
            stop = start + timedelta(days=7)
            schedule = Schedule(
                start=start.strftime('%Y-%m-%d'),
                stop=stop.strftime('%Y-%m-%d'),
                elos=False  # Disable ELO calculations during testing
            )
            assert schedule.start == pd.to_datetime(start.strftime('%Y-%m-%d'))
            assert schedule.stop == pd.to_datetime(stop.strftime('%Y-%m-%d'))

    def test_date_validation(self):
        """Test date range validation"""
        with pytest.raises(ValueError):
            start = datetime.now()
            stop = start - timedelta(days=1)
            Schedule(start=start.strftime('%Y-%m-%d'), 
                    stop=stop.strftime('%Y-%m-%d'))

    def test_pull_games(self, mock_session):
        """Test game pulling functionality"""
        with patch('dancing.wn_cbb_scraper.Schedule._create_session', return_value=mock_session):
            start = datetime.now()
            stop = start + timedelta(days=7)
            schedule = Schedule(start=start.strftime('%Y-%m-%d'), 
                             stop=stop.strftime('%Y-%m-%d'))
            assert hasattr(schedule, 'games_per_day')

    def test_women_parameter(self, mock_session):
        """Test women's basketball parameter"""
        with patch('dancing.wn_cbb_scraper.Schedule._create_session', return_value=mock_session):
            start = datetime.now()
            stop = start + timedelta(days=7)
            schedule = Schedule(start=start.strftime('%Y-%m-%d'), 
                             stop=stop.strftime('%Y-%m-%d'),
                             women=True)
            assert schedule.gender == "w"
