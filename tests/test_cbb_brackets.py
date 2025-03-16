import numpy as np
import pandas as pd
import pytest

from bigdance.cbb_brackets import Bracket, Game, Pool, Team


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
            teams.append(
                Team(
                    name=f"{region} {seed} Seed",
                    seed=seed,
                    region=region,
                    rating=rating,
                    conference=f"Conference {(seed-1)//4 + 1}",
                )
            )
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
        conference="Test Conference",
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
            conference="Test Conference",
        )


def test_game_initialization():
    """Test Game class initialization"""
    team1 = Team("Team 1", 1, "East", 2000.0, "Conf 1")
    team2 = Team("Team 2", 16, "East", 1500.0, "Conf 2")

    game = Game(team1=team1, team2=team2, round=1, region="East")

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

        seed_pairs = [
            (1, 16),
            (8, 9),
            (5, 12),
            (4, 13),
            (6, 11),
            (3, 14),
            (7, 10),
            (2, 15),
        ]
        for game, (seed1, seed2) in zip(region_games, seed_pairs):
            assert game.team1.seed == seed1
            assert game.team2.seed == seed2


def test_bracket_validation():
    """Test bracket validation rules"""
    # Test with wrong number of teams
    with pytest.raises(ValueError):
        Bracket(
            [Team("Test", 1, "East", 2000.0, "Conf") for _ in range(63)]
        )  # One team short

    # Test with invalid number of regions
    invalid_teams = []
    for seed in range(1, 17):
        for region in ["East", "West", "South"]:  # Missing one region
            invalid_teams.append(
                Team(f"Team {seed}{region}", seed, region, 2000.0, "Conf")
            )
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

    print(winners)

    higher_seed_wins = sum(1 for w in winners if w.seed == 1)

    print(str(higher_seed_wins))

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
                1: {8, 9},
                8: {1, 16},
                9: {1, 16},
                16: {8, 9},  # 1/16 winner vs 8/9 winner
                4: {5, 12},
                5: {4, 13},
                12: {4, 13},
                13: (5, 12),  # 4/13 winner vs 5/12 winner
                3: {6, 11},
                6: {3, 14},
                11: {3, 14},
                14: {6, 11},  # 3/14 winner vs 6/11 winner
                2: {7, 10},
                7: {2, 15},
                10: {2, 15},
                15: {7, 10},  # 2/15 winner vs 7/10 winner
            }

            # Check that seeds form a valid matchup
            assert (seed1 in valid_pairs and seed2 in valid_pairs[seed1]) or (
                seed2 in valid_pairs and seed1 in valid_pairs[seed2]
            ), f"Invalid second round matchup: {seed1} vs {seed2} in {game.region} region"

    # Print all matchups by region for debugging
    print("\nSecond Round Matchups:")
    for region, matchups in region_matchups.items():
        print(f"\n{region} Region:")
        for seed1, seed2 in sorted(matchups):
            print(f"{seed1} seed vs {seed2} seed")

    # Verify each region has 4 games
    for region, matchups in region_matchups.items():
        assert (
            len(matchups) == 4
        ), f"{region} region has {len(matchups)} games (should be 4)"


def test_tournament_simulation(sample_bracket):
    """Test full tournament simulation"""
    results = sample_bracket.simulate_tournament()

    # Check that we have results for each round
    expected_rounds = [
        "First Round",
        "Second Round",
        "Sweet 16",
        "Elite 8",
        "Final Four",
        "Championship",
        "Champion",
    ]
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
    assert all(
        col in results.columns
        for col in ["name", "avg_score", "std_score", "wins", "win_pct"]
    )

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
        "Championship": 32,
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

    max_score = (
        first_round_teams * custom_scoring["First Round"]
        + second_round_teams * custom_scoring["Second Round"]
        + sweet_16_teams * custom_scoring["Sweet 16"]
        + elite_8_teams * custom_scoring["Elite 8"]
        + final_four_teams * custom_scoring["Final Four"]
        + championship_teams * custom_scoring["Championship"]
    )

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
    for round_name in [
        "First Round",
        "Second Round",
        "Sweet 16",
        "Elite 8",
        "Final Four",
        "Championship",
    ]:
        winners1 = [team.name for team in results1[round_name]]
        winners2 = [team.name for team in results2[round_name]]
        assert winners1 == winners2

    assert results1["Champion"].name == results2["Champion"].name


def test_upset_rates_at_different_factors(sample_teams):
    """Test that the upset_factor scale works as expected across the range from -1.0 to 1.0"""
    # Test upset factors across the full spectrum
    # upset_factors = [-1.0, -0.5, 0.0, 0.5, 1.0]
    upset_factors = np.arange(-1.0, 1.01, 0.1)
    upset_rates = []

    num_brackets = 100  # Simulate 100 brackets for each upset factor to reduce noise

    for upset_factor in upset_factors:
        total_upsets = 0

        # Simulate multiple brackets with this upset factor
        for _ in range(num_brackets):
            # Create a new bracket for each simulation
            test_bracket = Bracket(sample_teams)

            # Set the upset factor on all games
            for game in test_bracket.games:
                game.upset_factor = upset_factor

            # Run tournament simulation
            test_bracket.simulate_tournament()

            # Count upsets (underdogs winning)
            total_upsets += test_bracket.total_underdogs()

        # Calculate average upset rate across all simulations
        avg_upset_rate = total_upsets / (
            63 * num_brackets
        )  # 63 total games per tournament
        upset_rates.append(avg_upset_rate)
        print(f"Upset factor: {upset_factor:.1f}, Avg upset rate: {avg_upset_rate:.4f}")

    print("\nAll upset rates:", upset_rates)

    # Check if rates generally increase with increasing upset factor
    # This should now be a monotonic relationship from low to high
    for i in range(1, len(upset_rates)):
        assert (
            upset_rates[i] > upset_rates[i - 1] - 0.01
        ), f"Upset rates should generally increase. Drop at {upset_factors[i]}: {upset_rates[i-1]:.4f} -> {upset_rates[i]:.4f}"


def test_negative_upset_factor_behavior(sample_teams):
    """Test that negative upset factors properly favor higher seeds beyond elo predictions"""
    # Create brackets with different upset factors
    chalk_bracket = Bracket(sample_teams)  # extreme chalk
    elo_bracket = Bracket(sample_teams)  # pure elo-based

    # Set different upset factors
    for game in chalk_bracket.games:
        game.upset_factor = -1.0  # Extreme chalk (100% favorite wins)

    for game in elo_bracket.games:
        game.upset_factor = 0.0  # Pure elo (no adjustment)

    # Run multiple simulations to get stable results
    chalk_upsets = 0
    elo_upsets = 0
    num_sims = 50

    for _ in range(num_sims):
        # Create fresh brackets to reset results
        chalk_bracket = Bracket(sample_teams)
        elo_bracket = Bracket(sample_teams)

        # Set upset factors
        for game in chalk_bracket.games:
            game.upset_factor = -1.0
        for game in elo_bracket.games:
            game.upset_factor = 0.0

        # Simulate tournaments
        chalk_bracket.simulate_tournament()
        elo_bracket.simulate_tournament()

        # Count upsets
        chalk_upsets += chalk_bracket.total_underdogs()
        elo_upsets += elo_bracket.total_underdogs()

    # Calculate average upsets per bracket
    avg_chalk_upsets = chalk_upsets / num_sims
    avg_elo_upsets = elo_upsets / num_sims

    print(f"Average upsets with extreme chalk (-1.0): {avg_chalk_upsets:.2f}")
    print(f"Average upsets with pure elo (0.0): {avg_elo_upsets:.2f}")

    # Extreme chalk should produce significantly fewer upsets than elo-based
    assert (
        avg_chalk_upsets < avg_elo_upsets * 0.7
    ), "Extreme chalk should generate far fewer upsets than elo-based picks"

    # For extreme chalk (-1.0), expect very few upsets overall
    assert (
        avg_chalk_upsets < 1.0
    ), f"Extreme chalk should produce very few upsets, got {avg_chalk_upsets:.2f}"


def test_seeding_impact(sample_teams):
    """Test that seeding has appropriate impact on advancement probability"""
    results = {}

    # Set a realistic upset factor to match historical upset rates
    realistic_upset_factor = 0.3

    # Run many simulations
    for _ in range(1000):
        bracket = Bracket(sample_teams)

        # Set a realistic upset factor for all games
        for game in bracket.games:
            game.upset_factor = realistic_upset_factor

        outcomes = bracket.simulate_tournament()

        # Track which seeds reached each round
        for round_name, teams_result in outcomes.items():
            if round_name not in results:
                results[round_name] = []

            # Handle both single Team objects and lists of Teams
            if round_name != "Champion":
                # Other rounds have lists of teams
                for team in teams_result:
                    results[round_name].append(team.seed)

    # Calculate advancement rates by seed, dividing by number of teams per seed (4)
    advancement_rates = {}
    for round_name, seeds in results.items():
        if round_name != "Champion":  # Process rounds with multiple teams
            counts = {}
            for seed in range(1, 17):
                # Divide by 4000 (4 teams per seed × 1000 simulations)
                counts[seed] = seeds.count(seed) / 4000
            advancement_rates[round_name] = counts

    # Loading and processing historical tournament data
    # Pulled from Andrew Sundberg's College Basketball Dataset on Kaggle
    # https://www.kaggle.com/datasets/andrewsundberg/college-basketball-dataset?resource=download
    teams = pd.read_csv("assets/cbb.csv")
    actual = teams.loc[~teams.SEED.isnull() & ~teams.POSTSEASON.isin(["R68"])]
    rounds = pd.DataFrame(
        {
            "POSTSEASON": ["R32", "S16", "E8", "F4", "2ND", "Champions"],
            "round_rank": [1, 2, 3, 4, 5, 6],
        }
    )
    actual = pd.merge(left=actual, right=rounds, how="inner", on=["POSTSEASON"])
    round_names = [
        "First Round",
        "Second Round",
        "Sweet 16",
        "Elite 8",
        "Final Four",
        "Championship",
        "Champion",
    ]
    historical_rates = {}
    for round_ind in range(rounds.shape[0]):
        historical_rates[round_names[round_ind]] = (
            actual.loc[actual.round_rank > round_ind].groupby("SEED").size()
            / (4 * actual.YEAR.nunique())
        ).to_dict()
        for seed in range(1, 17):
            if seed not in historical_rates[round_names[round_ind]]:
                historical_rates[round_names[round_ind]][seed] = 0.0

    # Print advancement rates for analysis
    print(
        f"\nAdvancement rates by seed (with upset_factor = {realistic_upset_factor}):"
    )
    rmsd_expectation = {
        "First Round": 0.12,
        "Second Round": 0.1,
        "Sweet 16": 0.08,
        "Elite 8": 0.05,
        "Final Four": 0.05,
        "Championship": 0.05,
    }
    for round_name, rates in advancement_rates.items():
        square_diff = []
        for seed, rate in sorted(rates.items()):
            square_diff.append((rate - historical_rates[round_name][seed]) ** 2.0)
        rmsd = (sum(square_diff) / len(square_diff)) ** 0.5
        print(f"{round_name} Root Mean Square Diff: {rmsd}")
        assert rmsd < rmsd_expectation[round_name]

    # # Optional: Create visualization comparing to historical data
    # import matplotlib.pyplot as plt
    # fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    # fig.suptitle(f'Tournament Advancement Rates: Simulation (upset_factor={realistic_upset_factor}) vs Historical',
    #             fontsize=16)
    # rounds_to_plot = ["First Round", "Second Round", "Sweet 16", "Elite 8"]
    # for i, round_name in enumerate(rounds_to_plot):
    #     row, col = divmod(i, 2)
    #     ax = axes[row, col]
    #     # Get data for this round
    #     sim_rates = [advancement_rates[round_name][seed] for seed in range(1, 17)]
    #     hist_rates = [historical_rates[round_name][seed] for seed in range(1, 17)]
    #     # Plot bar chart
    #     x = np.arange(1, 17)
    #     width = 0.35
    #     ax.bar(x - width/2, sim_rates, width, label='Simulation', color='skyblue')
    #     ax.bar(x + width/2, hist_rates, width, label='Historical', color='lightcoral')
    #     # Add labels and formatting
    #     ax.set_title(round_name)
    #     ax.set_xlabel('Seed')
    #     ax.set_ylabel('Advancement Rate')
    #     ax.set_xticks(x)
    #     ax.set_ylim(0, 1.0)
    #     ax.legend()
    #     ax.grid(True, linestyle='--', alpha=0.6)
    # plt.tight_layout(rect=[0, 0, 1, 0.95])
    # plt.savefig("seed_advancement_comparison.png", dpi=300, bbox_inches='tight')
    # print("\nVisualization saved as 'seed_advancement_comparison.png'")


def test_pool_with_variable_upset_factors(sample_teams):
    """Test that brackets with reasonable upset factors perform well"""
    actual_bracket = Bracket(sample_teams)

    # Create pool
    pool = Pool(actual_bracket)

    # Add entries with different upset factors
    # upset_factors = [-0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8]  # Include some negative values
    upset_factors = np.arange(-1.0, 1.01, 0.1)

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
    print(results[["name", "avg_score", "std_score", "wins", "win_pct"]])

    # results['upset_factor'] = results.name.str.split("_").str[-1].astype(float)
    # results.sort_values(by="upset_factor",inplace=True)
    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.plot(results.upset_factor,results.win_pct)
    # plt.grid(True)
    # plt.xlabel("Upset Factor")
    # plt.ylabel("Win Pct")
    # plt.savefig("UpsetFactorWinPcts.pdf")

    # Verify that winning entry has a reasonable upset factor
    winning_factor = float(results.iloc[0]["name"].split("_")[1])
    print(f"Winning upset factor: {winning_factor}")

    # Verify that at least one non-chalk upset factor performs better than pure chalk
    nonchalk_entries = results[results["name"] != "Entry_-1.0"]
    assert (
        nonchalk_entries.iloc[0]["win_pct"]
        >= results[results["name"] == "Entry_-1.0"].iloc[0]["win_pct"]
    )
