# dancing

<p align="center">
  <img src="assets/DancingHex.png" width="150" alt="dancing logo">
</p>

A Python package for NCAA March Madness bracket simulation combining real-time ratings from Warren Nolan with customizable tournament simulations.

## Installation

```bash
pip install dancing
```

## Quick Start

```python
from dancing import Standings, simulate_bracket_pool

# Get current team ratings
standings = Standings()

# Simulate a bracket pool
results = simulate_bracket_pool(standings, num_entries=100)
print(results)
```

## Features

- Real-time team ratings and rankings from Warren Nolan
- Customizable bracket simulation
- Tournament pool analysis
- Integration with common bracket pool formats

## Development Installation

To install the package for development:

```bash
git clone https://github.com/tefirman/dancing
cd dancing
pip install -e .
```
