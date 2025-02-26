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

import logging
from bigdance.wn_cbb_scraper import Standings
from bigdance.bigdance_integration import create_teams_from_standings

logger = logging.getLogger(__name__)

# Pulling tournament team data from Warren Nolan for now
standings = Standings()
actual_bracket = create_teams_from_standings(standings)
teams = {}
for region in ["East","West","South","Midwest"]:
    teams[region] = [{"Team": team.name, "Seed": team.seed, "Rating": team.rating} \
                     for team in actual_bracket.teams if team.region == region]

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
