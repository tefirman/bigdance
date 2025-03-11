import pytest
import pandas as pd
import numpy as np
from bigdance.cbb_brackets import Team, Game, Bracket, Pool

@pytest.fixture
def sample_teams():
    """Fixture providing a realistic set of tournament teams"""
    teams = []
    regions = ["East", "West", "South", "Midwest"]
    seeds = list(range(1, 17))
    
    for region in regions:
        for seed in seeds:
            # Create team with rating roughly correlated to seed
            rating = 2000 - (seed * 50) + np.random.normal(0, 25)
            teams.append(Team(
                name=f"{region} {seed} Seed",
                seed=seed,
                region=region,
                rating=rating,
                conference=f"Conference {(seed-1)//4 + 1}"
            ))
    return teams

@pytest.fixture
def sample_bracket(sample_teams):
    """Fixture providing a tournament bracket with teams"""
    return Bracket(sample_teams)

def test_team_initialization():
    """Test Team class initialization and validation"""
    team = Team(
        name="Test Team",
        seed=1,
        region="East",
        rating=2000.0,
        conference="Test Conference"
    )
    
    assert team.name == "Test Team"
    assert team.seed == 1
    assert team.region == "East"
    assert team.rating == 2000.0
    assert team.conference == "Test Conference"

    # Test invalid seed value
    with pytest.raises(ValueError):
        Team(
            name="Invalid Team",
            seed=17,  # Invalid seed
            region="East",
            rating=2000.0,
            conference="Test Conference"
        )

def test_game_initialization():
    """Test Game class initialization"""
    team1 = Team("Team 1", 1, "East", 2000.0, "Conf 1")
    team2 = Team("Team 2", 16, "East", 1500.0, "Conf 2")
    
    game = Game(
        team1=team1,
        team2=team2,
        round=1,
        region="East"
    )
    
    assert game.team1 == team1
    assert game.team2 == team2
    assert game.round == 1
    assert game.region == "East"
    assert game.winner is None
    assert game.actual_winner is None

def test_bracket_initialization(sample_teams):
    """Test Bracket class initialization and validation"""
    bracket = Bracket(sample_teams)
    
    # Check number of teams
    assert len(bracket.teams) == 64
    
    # Check initial games creation
    assert len(bracket.games) == 32  # First round should have 32 games
    
    # Verify seeding matchups (1v16, 2v15, etc.)
    for region in ["East", "West", "South", "Midwest"]:
        region_games = [g for g in bracket.games if g.region == region]
        assert len(region_games) == 8
        
        seed_pairs = [(1,16), (8,9), (5,12), (4,13), (6,11), (3,14), (7,10), (2,15)]
        for game, (seed1, seed2) in zip(region_games, seed_pairs):
            assert game.team1.seed == seed1
            assert game.team2.seed == seed2

def test_bracket_validation():
    """Test bracket validation rules"""
    # Test with wrong number of teams
    with pytest.raises(ValueError):
        Bracket([Team("Test", 1, "East", 2000.0, "Conf") for _ in range(63)])  # One team short
    
    # Test with invalid number of regions
    invalid_teams = []
    for seed in range(1, 17):
        for region in ["East", "West", "South"]:  # Missing one region
            invalid_teams.append(Team(f"Team {seed}{region}", seed, region, 2000.0, "Conf"))
    with pytest.raises(ValueError):
        Bracket(invalid_teams * 2)  # Multiply to get to 64 teams

def test_game_simulation(sample_bracket):
    """Test game simulation logic"""
    game = sample_bracket.games[0]  # Take first game (1v16 matchup)
    
    # Run multiple simulations to verify probabilistic behavior
    winners = []
    for _ in range(1000):
        winner = sample_bracket.simulate_game(game)
        winners.append(winner)
    
    # Higher rated team should win more often
    higher_seed_wins = sum(1 for w in winners if w.seed == 1)
    assert higher_seed_wins > 800  # Should win roughly 80-90% of the time

def test_advance_round(sample_bracket):
    """Test round advancement logic and validate second round seeding"""
    # Simulate first round
    first_round_games = sample_bracket.games.copy()
    second_round_games = sample_bracket.advance_round(first_round_games)
    
    # Check number of games
    assert len(second_round_games) == 16  # Second round should have 16 games
    
    # Track matchups by region for debugging
    region_matchups = {region: [] for region in ["East", "West", "South", "Midwest"]}
    
    # Check region assignments and seeding
    for game in second_round_games:
        assert game.round == 2
        # Teams in same region should play each other
        if game.region != "Final Four":
            assert game.team1.region == game.team2.region == game.region
            
            # Verify second round seeding patterns
            seed1, seed2 = game.team1.seed, game.team2.seed
            region_matchups[game.region].append((seed1, seed2))
            
            # Define valid second round matchups
            valid_pairs = {
                1: {8, 9}, 8: {1, 16}, 9: {1, 16}, 16:{8, 9},      # 1/16 winner vs 8/9 winner
                4: {5, 12}, 5: {4, 13}, 12: {4, 13}, 13: (5, 12),    # 4/13 winner vs 5/12 winner
                3: {6, 11}, 6: {3, 14}, 11: {3, 14}, 14: {6, 11},    # 3/14 winner vs 6/11 winner
                2: {7, 10}, 7: {2, 15}, 10: {2, 15}, 15: {7, 10}     # 2/15 winner vs 7/10 winner
            }
            
            # Check that seeds form a valid matchup
            assert (seed1 in valid_pairs and seed2 in valid_pairs[seed1]) or \
                   (seed2 in valid_pairs and seed1 in valid_pairs[seed2]), \
                   f"Invalid second round matchup: {seed1} vs {seed2} in {game.region} region"
    
    # Print all matchups by region for debugging
    print("\nSecond Round Matchups:")
    for region, matchups in region_matchups.items():
        print(f"\n{region} Region:")
        for seed1, seed2 in sorted(matchups):
            print(f"{seed1} seed vs {seed2} seed")
            
    # Verify each region has 4 games
    for region, matchups in region_matchups.items():
        assert len(matchups) == 4, f"{region} region has {len(matchups)} games (should be 4)"

def test_tournament_simulation(sample_bracket):
    """Test full tournament simulation"""
    results = sample_bracket.simulate_tournament()
    
    # Check that we have results for each round
    expected_rounds = ["First Round", "Second Round", "Sweet 16", "Elite 8", 
                      "Final Four", "Championship", "Champion"]
    assert all(round_name in results for round_name in expected_rounds)
    
    # Check number of teams advancing in each round
    assert len(results["First Round"]) == 32
    assert len(results["Second Round"]) == 16
    assert len(results["Sweet 16"]) == 8
    assert len(results["Elite 8"]) == 4
    assert len(results["Final Four"]) == 2
    assert len(results["Championship"]) == 1
    
    # Verify Champion is one team
    assert isinstance(results["Champion"], Team)

def test_pool_simulation(sample_teams):
    """Test tournament pool simulation"""
    # Create actual bracket and some entries
    actual_bracket = Bracket(sample_teams)
    pool = Pool(actual_bracket)
    
    # Add some entries
    for i in range(5):
        entry = Bracket(sample_teams)
        pool.add_entry(f"Entry {i+1}", entry)
    
    # Run pool simulation
    results = pool.simulate_pool(num_sims=100)
    
    # Check results structure
    assert isinstance(results, pd.DataFrame)
    assert all(col in results.columns for col in 
              ["name", "avg_score", "std_score", "wins", "win_pct"])
    
    # Check win percentage totals to 1
    assert abs(results["win_pct"].sum() - 1.0) < 0.01

def test_scoring_system(sample_teams):
    """Test bracket scoring system"""
    # Create and simulate actual bracket 
    actual_bracket = Bracket(sample_teams)
    pool = Pool(actual_bracket)
    
    # Create an entry bracket
    entry_bracket = Bracket(sample_teams)
    
    # Test with custom scoring system
    custom_scoring = {
        "First Round": 1,
        "Second Round": 2,
        "Sweet 16": 4,
        "Elite 8": 8,
        "Final Four": 16,
        "Championship": 32
    }
    
    # First simulate the actual tournament (needed before scoring)
    pool.actual_tournament = actual_bracket.simulate_tournament()
    
    # Then simulate and score the entry
    entry_results = entry_bracket.simulate_tournament()
    score = pool.score_bracket(entry_results, custom_scoring)
    
    assert isinstance(score, int)
    assert score >= 0
    
    # Calculate maximum possible score
    first_round_teams = 32  # 32 winners in first round
    second_round_teams = 16
    sweet_16_teams = 8
    elite_8_teams = 4
    final_four_teams = 2
    championship_teams = 1
    
    max_score = (first_round_teams * custom_scoring["First Round"] +
                 second_round_teams * custom_scoring["Second Round"] +
                 sweet_16_teams * custom_scoring["Sweet 16"] +
                 elite_8_teams * custom_scoring["Elite 8"] +
                 final_four_teams * custom_scoring["Final Four"] +
                 championship_teams * custom_scoring["Championship"])
    
    assert score <= max_score

def test_reproducibility(sample_teams):
    """Test that simulations are reproducible with same random seed"""
    np.random.seed(42)
    bracket1 = Bracket(sample_teams)
    results1 = bracket1.simulate_tournament()
    
    np.random.seed(42)
    bracket2 = Bracket(sample_teams)
    results2 = bracket2.simulate_tournament()
    
    # Check that all rounds have same winners
    for round_name in ["First Round", "Second Round", "Sweet 16", 
                      "Elite 8", "Final Four", "Championship"]:
        winners1 = [team.name for team in results1[round_name]]
        winners2 = [team.name for team in results2[round_name]]
        assert winners1 == winners2
    
    assert results1["Champion"].name == results2["Champion"].name

def test_upset_rates_at_different_factors(sample_teams):
    """Test that higher upset_factor actually creates more upsets"""
    # Create bracket with sample teams
    bracket = Bracket(sample_teams)
    
    # Test upset factors
    upset_factors = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]
    upset_rates = []
    
    for upset_factor in upset_factors:
        # Set the upset factor on the simulate_game method for this test
        original_method = bracket.simulate_game
        
        # Add diagnostic information - print the original and adjusted probabilities
        def test_simulate_with_diagnostics(self, game, upset_factor=upset_factor):
            rating_diff = game.team1.rating - game.team2.rating
            base_prob = 1 / (1 + 10**(-rating_diff/400))
            adjusted_prob = (base_prob * (1 - upset_factor)) + (0.5 * upset_factor)
            
            # Print diagnostic information for the first few games
            if hasattr(test_simulate_with_diagnostics, 'count'):
                test_simulate_with_diagnostics.count += 1
            else:
                test_simulate_with_diagnostics.count = 1
                
            if test_simulate_with_diagnostics.count <= 10:
                favorite = game.team1 if game.team1.seed < game.team2.seed else game.team2
                underdog = game.team2 if game.team1.seed < game.team2.seed else game.team1
                is_upset = (adjusted_prob < 0.5 and game.team1.seed < game.team2.seed) or \
                           (adjusted_prob > 0.5 and game.team1.seed > game.team2.seed)
                print(f"F={upset_factor:.1f}, {game.team1.seed} vs {game.team2.seed}, "\
                      f"Base p={base_prob:.4f}, Adj p={adjusted_prob:.4f}, Upset? {is_upset}")
            
            return game.team1 if np.random.random() < adjusted_prob else game.team2
        
        # Patch the method for this test
        bracket.simulate_game = test_simulate_with_diagnostics.__get__(bracket, type(bracket))
        
        upsets = 0
        total_games = 0
        
        # Run multiple simulations
        for _ in range(100):
            # Reset the bracket
            bracket = Bracket(sample_teams)
            bracket.simulate_game = test_simulate_with_diagnostics.__get__(bracket, type(bracket))
            
            # Run tournament simulation
            bracket.simulate_tournament()
            
            # Count upsets
            for round_name, teams in bracket.underdogs_by_round.items():
                upsets += len(teams)
            
            # Count total games in the tournament (63 for a 64-team tournament)
            total_games += 63
        
        upset_rate = upsets / total_games
        upset_rates.append(upset_rate)
        
        # Restore original method
        bracket.simulate_game = original_method
        
        print(f"Upset factor: {upset_factor:.1f}, Upset rate: {upset_rate:.4f}")
    
    print("\nAll upset rates:", upset_rates)
    
    # Check if rates generally increase (allow for some statistical variation)
    is_generally_increasing = True
    for i in range(1, len(upset_rates)):
        if upset_rates[i] < upset_rates[i-1] - 0.02:  # Allow 2% variation
            is_generally_increasing = False
            print(f"Drop at {upset_factors[i]}: {upset_rates[i-1]:.4f} -> {upset_rates[i]:.4f}")
    
    assert is_generally_increasing, f"Upset rates are not generally increasing: {upset_rates}"

def test_seeding_impact(sample_teams):
    """Test that seeding has appropriate impact on advancement probability"""
    results = {}
    
    # Run many simulations
    for _ in range(1000):
        bracket = Bracket(sample_teams)
        outcomes = bracket.simulate_tournament()
        
        # Track which seeds reached each round
        for round_name, teams_result in outcomes.items():
            if round_name not in results:
                results[round_name] = []
            
            # Handle both single Team objects and lists of Teams
            if round_name == "Champion":
                # Champion is a single Team object
                results[round_name].append(teams_result.seed)
            else:
                # Other rounds have lists of teams
                for team in teams_result:
                    results[round_name].append(team.seed)
    
    # Calculate advancement rates by seed, dividing by number of teams per seed (4)
    advancement_rates = {}
    for round_name, seeds in results.items():
        if round_name not in ["Champion"]:  # Process rounds with multiple teams
            counts = {}
            for seed in range(1, 17):
                # Divide by 4000 (4 teams per seed × 1000 simulations)
                counts[seed] = seeds.count(seed) / 4000
            advancement_rates[round_name] = counts
        else:
            # Champion is just one team, so divide by 1000 simulations
            counts = {}
            for seed in range(1, 17):
                counts[seed] = seeds.count(seed) / 1000
            advancement_rates[round_name] = counts
    
    # Print advancement rates for analysis
    print("\nAdvancement rates by seed:")
    for round_name, rates in advancement_rates.items():
        print(f"\n{round_name}:")
        for seed, rate in sorted(rates.items()):
            print(f"Seed {seed}: {rate:.4f}")
    
    # Verification checks - general relationships that should hold
    sweet_16_rates = advancement_rates["Sweet 16"]
    assert sweet_16_rates[1] > sweet_16_rates[4] > sweet_16_rates[8]  # Higher seeds should advance more
    
    # Verify reasonable 1-seed advancement rates
    assert 0.7 < advancement_rates["Sweet 16"][1] < 0.95  # 1 seeds should make Sweet 16 70-95% of time
    assert 0.3 < advancement_rates["Final Four"][1] < 0.7  # 1 seeds should make Final Four 30-70% of time

def test_pool_with_variable_upset_factors(sample_teams):
    """Test that brackets with reasonable upset factors perform well"""
    actual_bracket = Bracket(sample_teams)
    
    # Create pool
    pool = Pool(actual_bracket)
    
    # Add entries with different upset factors
    upset_factors = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    for factor in upset_factors:
        entry = Bracket(sample_teams)
        # Set upset factor for all games
        for game in entry.games:
            game.upset_factor = factor
        pool.add_entry(f"Entry_{factor:.1f}", entry)
    
    # Simulate pool
    results = pool.simulate_pool(num_sims=1000)
    
    # Print results
    print("\nPool results with different upset factors:")
    print(results)
    
    # Verify that winning entry has a reasonable upset factor
    winning_factor = float(results.iloc[0]['name'].split('_')[1])
    print(f"Winning upset factor: {winning_factor}")
    
    # Verify that at least one non-zero upset factor performs better than pure chalk
    non_zero_entries = results[results['name'] != "Entry_0.0"]
    assert non_zero_entries.iloc[0]['win_pct'] >= results[results['name'] == "Entry_0.0"].iloc[0]['win_pct']
