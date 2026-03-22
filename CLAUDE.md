# CLAUDE.md

## Project Overview

**bigdance** is a Python package for NCAA March Madness bracket simulation. It combines real-time Elo ratings (from Warren Nolan), tournament bracket simulation, ESPN Tournament Challenge integration (via Gambit JSON API), and pool strategy analysis. Published on PyPI. Current version: 0.8.2.

## Repository Structure

```
src/bigdance/           # Main package
  cbb_brackets.py       # Core classes: Team, Game, Bracket, Pool
  wn_cbb_scraper.py     # Warren Nolan scraper: Standings, Matchups, Schedule
  espn_tc_scraper.py    # ESPN API client: ESPNApi, GameImportanceAnalyzer
  bracket_analysis.py   # Strategy analysis: BracketAnalysis
  bigdance_integration.py  # Helpers: create_teams_from_standings(), etc.
  cli.py                # CLI dispatcher (bigdance standings|simulate|espn|analyze)

app/                    # Streamlit web apps
  app.py                # Main bracket picker + simulator app
  tracker.py            # Live pool tracker (Game Importance, Pool Standings, Entry Details)
  generate_upset_analysis.py  # Pre-compute strategy data for app
  requirements.txt      # App-specific deps (streamlit, bigdance>=0.8.2)
  data/                 # Pre-computed analysis by gender/pool size

tests/                  # pytest test suite
.github/workflows/      # CI: test.yml, publish.yml, generate_upset_analysis.yml
```

## Development Commands

```bash
# Install for development
pip install -e ".[dev]"

# Lint & format (what CI runs)
ruff check src/ tests/ app/
ruff format --check src/ tests/

# Auto-fix formatting
make format

# Run all tests
pytest --cov=bigdance tests/

# Full check (lint + typecheck + test)
make check
```

## Key Conventions

- **Line length**: 100 characters
- **Linter**: ruff (rules: E, W, F, I, UP, B, SIM). CI runs both `ruff check` and `ruff format --check` on `src/`, `tests/`, and `app/`
- **Python**: supports 3.9-3.13, target 3.9 for ruff/black
- **Type checking**: mypy (continue-on-error in CI, not blocking)
- **Tests**: pytest. ESPN scraper tests take ~3-5 minutes (they hit live APIs)
- **Pre-commit**: ruff check + format, trailing whitespace, YAML validation

## ESPN API Behavior

ESPN's Gambit API (`gambit-api.fantasy.espn.com`) only returns propositions for the **current scoring period**. Once a round ends, those props disappear. Prior round results must be reconstructed from `actualOutcomeIds` on current round props. This is handled in `build_actual_bracket()`.

The `scoringPeriodId` mapping: 1=Round of 64, 2=Round of 32, 3=Sweet 16, 4=Elite 8, 5=Final Four, 6=Championship.

Team names from ESPN may not match Warren Nolan names. The `NAME_CORRECTIONS` dict in `espn_tc_scraper.py` handles known mismatches.

## Version Bumping

When releasing a new version, update ALL of these files:
- `pyproject.toml` (version field)
- `src/bigdance/__init__.py` (__version__)
- `src/bigdance/espn_tc_scraper.py` (@Version header)
- `src/bigdance/bracket_analysis.py` (@Version header)
- `src/bigdance/wn_cbb_scraper.py` (@Version header)
- `src/bigdance/bigdance_integration.py` (@Version header)
- `app/requirements.txt` (bigdance>=X.Y.Z)

Release notes go in `notes/release_notes_vX.Y.Z.md` (git-ignored, used for GitHub releases).

## CI/CD Pipeline

- **test.yml**: On PR/push to main. Lint job (Python 3.12) must pass before test matrix (3.9-3.13). Coverage uploaded to Codecov on 3.12.
- **publish.yml**: On GitHub release. Builds and publishes to PyPI.
- **generate_upset_analysis.yml**: Manual dispatch. Pre-computes strategy data across pool sizes.

## App Deployment

The Streamlit app is deployed at https://bigdance-bracket.streamlit.app. It installs `bigdance` from PyPI via `app/requirements.txt`, so a new PyPI release is required for app changes that touch the `src/` package.

## Branching

Feature branches off `main`, PRs reviewed before merge. Releases tagged on `main` trigger PyPI publish.
