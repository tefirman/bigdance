#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   data.py
@Time    :   2024/02/20
@Author  :   Taylor Firman
@Version :   1.0
@Contact :   tefirman@gmail.com
@Desc    :   Tournament data and management for March Madness bracket app
'''

from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

# Initial tournament team data
teams = {
    "East": [
        {"Team": "UConn", "Seed": 1, "Rating": 2000},
        {"Team": "Grambling", "Seed": 16, "Rating": 1500},
        {"Team": "FAU", "Seed": 8, "Rating": 1800},
        {"Team": "Northwestern", "Seed": 9, "Rating": 1750},
        {"Team": "San Diego St", "Seed": 5, "Rating": 1850},
        {"Team": "UAB", "Seed": 12, "Rating": 1650},
        {"Team": "Auburn", "Seed": 4, "Rating": 1900},
        {"Team": "Yale", "Seed": 13, "Rating": 1600},
        {"Team": "BYU", "Seed": 6, "Rating": 1825},
        {"Team": "Duquesne", "Seed": 11, "Rating": 1675},
        {"Team": "Illinois", "Seed": 3, "Rating": 1925},
        {"Team": "Morehead St", "Seed": 14, "Rating": 1575},
        {"Team": "Washington St", "Seed": 7, "Rating": 1775},
        {"Team": "Drake", "Seed": 10, "Rating": 1700},
        {"Team": "Iowa St", "Seed": 2, "Rating": 1950},
        {"Team": "South Dakota St", "Seed": 15, "Rating": 1525},
    ],
    "West": [
        {"Team": "Purdue", "Seed": 1, "Rating": 1990},
        {"Team": "Montana St", "Seed": 16, "Rating": 1510},
        {"Team": "Utah St", "Seed": 8, "Rating": 1790},
        {"Team": "TCU", "Seed": 9, "Rating": 1760},
        {"Team": "Gonzaga", "Seed": 5, "Rating": 1840},
        {"Team": "McNeese", "Seed": 12, "Rating": 1660},
        {"Team": "Kansas", "Seed": 4, "Rating": 1890},
        {"Team": "Samford", "Seed": 13, "Rating": 1610},
        {"Team": "Texas Tech", "Seed": 6, "Rating": 1815},
        {"Team": "NC State", "Seed": 11, "Rating": 1685},
        {"Team": "Kentucky", "Seed": 3, "Rating": 1915},
        {"Team": "Oakland", "Seed": 14, "Rating": 1585},
        {"Team": "Florida", "Seed": 7, "Rating": 1765},
        {"Team": "Boise St", "Seed": 10, "Rating": 1710},
        {"Team": "Tennessee", "Seed": 2, "Rating": 1940},
        {"Team": "Western Ky", "Seed": 15, "Rating": 1535},
    ],
    "South": [
        {"Team": "Houston", "Seed": 1, "Rating": 1995},
        {"Team": "Longwood", "Seed": 16, "Rating": 1505},
        {"Team": "Nebraska", "Seed": 8, "Rating": 1795},
        {"Team": "Texas A&M", "Seed": 9, "Rating": 1755},
        {"Team": "Wisconsin", "Seed": 5, "Rating": 1845},
        {"Team": "James Madison", "Seed": 12, "Rating": 1655},
        {"Team": "Duke", "Seed": 4, "Rating": 1895},
        {"Team": "Vermont", "Seed": 13, "Rating": 1605},
        {"Team": "Texas", "Seed": 6, "Rating": 1820},
        {"Team": "Virginia", "Seed": 11, "Rating": 1680},
        {"Team": "Marquette", "Seed": 3, "Rating": 1920},
        {"Team": "Akron", "Seed": 14, "Rating": 1580},
        {"Team": "Nevada", "Seed": 7, "Rating": 1770},
        {"Team": "Colorado", "Seed": 10, "Rating": 1705},
        {"Team": "Arizona", "Seed": 2, "Rating": 1945},
        {"Team": "Long Beach St", "Seed": 15, "Rating": 1530},
    ],
    "Midwest": [
        {"Team": "North Carolina", "Seed": 1, "Rating": 1985},
        {"Team": "Howard", "Seed": 16, "Rating": 1515},
        {"Team": "Mississippi St", "Seed": 8, "Rating": 1785},
        {"Team": "Michigan St", "Seed": 9, "Rating": 1765},
        {"Team": "St. Mary's", "Seed": 5, "Rating": 1835},
        {"Team": "Grand Canyon", "Seed": 12, "Rating": 1665},
        {"Team": "Alabama", "Seed": 4, "Rating": 1885},
        {"Team": "Charleston", "Seed": 13, "Rating": 1615},
        {"Team": "Clemson", "Seed": 6, "Rating": 1810},
        {"Team": "New Mexico", "Seed": 11, "Rating": 1690},
        {"Team": "Baylor", "Seed": 3, "Rating": 1910},
        {"Team": "Colgate", "Seed": 14, "Rating": 1590},
        {"Team": "Dayton", "Seed": 7, "Rating": 1760},
        {"Team": "South Carolina", "Seed": 10, "Rating": 1715},
        {"Team": "Creighton", "Seed": 2, "Rating": 1935},
        {"Team": "Saint Peter's", "Seed": 15, "Rating": 1540},
    ]
}

# Map of conferences for each team
team_conferences = {
    "UConn": "Big East",
    "Illinois": "Big Ten",
    "Iowa St": "Big 12",
    "Purdue": "Big Ten",
    "Kansas": "Big 12",
    "Kentucky": "SEC",
    "Tennessee": "SEC",
    "Houston": "Big 12",
    "Duke": "ACC",
    "Marquette": "Big East",
    "Arizona": "Pac-12",
    "North Carolina": "ACC",
    "Alabama": "SEC",
    "Baylor": "Big 12",
    "Creighton": "Big East"
    # Add more as needed
}

def get_team_conference(team_name: str) -> str:
    """Get conference for a given team"""
    return team_conferences.get(team_name, "Other")

def filter_teams_by_conference(conference: str) -> Dict[str, List[Dict]]:
    """Filter teams by conference"""
    if conference == "All Games":
        return teams
        
    filtered_teams = {}
    for region, region_teams in teams.items():
        filtered_teams[region] = [
            team for team in region_teams 
            if get_team_conference(team["Team"]) == conference
        ]
    return filtered_teams

def initialize_tournament_data():
    """Initialize or update tournament data"""
    logger.info("Initializing tournament data...")
    # In the future, this could pull real data from an API or database
    return teams

def validate_tournament_data():
    """Validate tournament data structure"""
    try:
        # Check for required regions
        required_regions = {"East", "West", "South", "Midwest"}
        if set(teams.keys()) != required_regions:
            raise ValueError("Missing required tournament regions")
            
        # Check each region has 16 teams
        for region, region_teams in teams.items():
            if len(region_teams) != 16:
                raise ValueError(f"{region} region does not have exactly 16 teams")
                
            # Check team seeds are 1-16 with no duplicates
            seeds = [team["Seed"] for team in region_teams]
            if sorted(seeds) != list(range(1, 17)):
                raise ValueError(f"{region} region has invalid seed numbers")
                
            # Check each team has required fields
            required_fields = {"Team", "Seed", "Rating"}
            for team in region_teams:
                if not all(field in team for field in required_fields):
                    raise ValueError(f"Team {team.get('Team', 'Unknown')} missing required fields")
                    
        logger.info("Tournament data validation successful")
        return True
        
    except Exception as e:
        logger.error(f"Tournament data validation failed: {str(e)}")
        raise

if __name__ == "__main__":
    # Test data validation
    logging.basicConfig(level=logging.INFO)
    validate_tournament_data()
