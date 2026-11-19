# Scheduled betting jobs (macOS launchd)

Three jobs wrap `bigdance edges` (see `src/bigdance/betting.py`):

| Job | Schedule | What it does |
|---|---|---|
| `picks` | daily 10:00 | Logs +EV bets to `bets.csv`; records Elo vs market for every game in `snapshots.csv` |
| `closing` | every 15 min | Captures closing lines; captures closing lines for every tracked game (not just bets); skips the API call unless something tips within 35 min and credits remain above the reserve |
| `settle` | daily 06:00 | Grades finished bets, and records winners for every snapshot game, from Warren Nolan final scores |

Data and logs live in `~/bigdance-bets/`. The Odds API key is read from `~/.zshrc` by
`run_edges.sh` (launchd doesn't load shell profiles), so it never appears in a plist.

```bash
./install.sh              # install + load
./install.sh --uninstall  # unload + remove
bigdance edges --summary ~/bigdance-bets/bets.csv
```

Notes
- Times are local and assume US/Eastern (bets are dated by the local calendar day).
- The Mac must be awake. Missed calendar jobs run on wake; missed `closing` runs are lost.
- The wrapper expects `~/bigdance/.venv/bin/bigdance`; set `BIGDANCE_BIN` to change it.
- Jobs do nothing May-Oct.

Credits: tracking closing lines for every game costs roughly 25-40 Odds API calls per day (one per
15-minute run with a game about to tip). The free tier (500/month) can't sustain that all month, so
`--credit-reserve` (default 100) makes the closing job fall back to logged bets only once credits
run low, keeping enough for the daily picks run. The last-seen credit count is stored in
`snapshots.csv.quota`.
