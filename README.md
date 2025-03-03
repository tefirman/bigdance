<table>
<tr>
  <td><img src="https://github.com/tefirman/bigdance/blob/main/assets/DancingHex.png?raw=true" width="400" alt="bigdance logo"></td>
  <td>
    <h1>bigdance</h1>
    A Python package for NCAA March Madness bracket simulation combining real-time ratings from Warren Nolan with customizable tournament simulations.
  </td>
</tr>
</table>

## Installation

```bash
pip install bigdance
```

## Quick Start

```python
from bigdance import Standings, simulate_bracket_pool

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
git clone https://github.com/tefirman/bigdance
cd bigdance
pip install -e .
```
