import pytest
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
from pathlib import Path
from src.dancing.wn_cbb_scraper import BaseScraper, Standings, Matchups, Schedule

SAMPLE_ELO_HTML = """
<table class="normal-grid alternating-rows stats-table">
    <tr>
        <th>Rank</th>
        <th>Team</th>
        <th>ELO</th>
        <th>Record</th>
    </tr>
    <tr>
        <td>1</td>
        <td>Team A</td>
        <td>1600</td>
        <td>20-5</td>
    </tr>
    <tr>
        <td>2</td>
        <td>Team B</td>
        <td>1550</td>
        <td>18-7</td>
    </tr>
</table>
"""

SAMPLE_MATCHUPS_HTML = """
<div class="pbox">
    <tr class="pbox__info-team1-row">
        <div class="name-subcontainer"><a>Team A</a></div>
        <td class="score center-line">75</td>
        <td class="score">78</td>
        <td class="value">60%</td>
    </tr>
    <tr class="pbox__info-team2-row">
        <div class="name-subcontainer"><a>Team B</a></div>
        <td class="score center-line">70</td>
        <td class="score">72</td>
        <td class="value">40%</td>
    </tr>
</div>
"""

@pytest.fixture
def mock_session():
    """Fixture providing a mock requests session"""
    with patch('requests.Session') as mock:
        session = mock.return_value
        session.get.return_value.text = "<html>Mock HTML</html>"
        session.get.return_value.raise_for_status = MagicMock()
        yield session

@pytest.fixture
def mock_elo_response():
    """Fixture providing mock ELO rankings page"""
    with patch('requests.Session') as mock:
        session = mock.return_value
        session.get.return_value.text = SAMPLE_ELO_HTML
        session.get.return_value.raise_for_status = MagicMock()
        yield session

@pytest.fixture
def mock_matchups_response():
    """Fixture providing mock matchups page"""
    with patch('requests.Session') as mock:
        session = mock.return_value
        session.get.return_value.text = SAMPLE_MATCHUPS_HTML
        session.get.return_value.raise_for_status = MagicMock()
        yield session

class TestBaseScraper:
    def test_initialization(self, mock_session):
        """Test basic scraper initialization"""
        scraper = BaseScraper(cache_dir="test_cache")
        assert scraper.cache_dir == Path("test_cache")
        assert scraper.cache_dir.exists()

    def test_courteous_get(self, mock_session):
        """Test courteous GET request functionality"""
        scraper = BaseScraper(cache_dir="test_cache")
        response = scraper.courteous_get("https://test.com", "test_key")
        assert response == "<html>Mock HTML</html>"
        mock_session.get.assert_called_once()

    def test_cache_handling(self, mock_session):
        """Test cache read/write operations"""
        scraper = BaseScraper(cache_dir="test_cache")
        
        # Test cache miss
        cached = scraper._get_cached_response("https://test.com", "test_key")
        assert cached is None
        
        # Test cache write
        scraper._cache_response("https://test.com", "test_key", "<html>Cached</html>")
        
        # Test cache hit
        cached = scraper._get_cached_response("https://test.com", "test_key")
        assert cached == "<html>Cached</html>"

class TestStandings:
    def test_initialization(self, mock_elo_response):
        """Test standings initialization"""
        with patch('dancing.wn_cbb_scraper.Standings._create_session', return_value=mock_elo_response):
            standings = Standings(season=2024)
            assert standings.season == 2024
            assert not standings.gender
            assert standings.base_url == "https://www.warrennolan.com/basketball/2024"

    def test_elo_parsing(self, mock_elo_response):
        """Test parsing of ELO rankings"""
        with patch('dancing.wn_cbb_scraper.Standings._create_session', return_value=mock_elo_response):
            standings = Standings(season=2024)
            assert len(standings.elo) == 2
            assert standings.elo.iloc[0]['Team'] == 'Team A'
            assert standings.elo.iloc[0]['ELO'] == 1600

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
            matchups = Matchups(date=datetime.now())
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
            schedule = Schedule(start=start.strftime('%Y-%m-%d'), 
                             stop=stop.strftime('%Y-%m-%d'))
            assert schedule.start == pd.to_datetime(start)
            assert schedule.stop == pd.to_datetime(stop)

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
