#!/usr/bin/env python
"""
Pre-compute optimal upset strategy data for each supported pool size.

Run this script once before deploying the app (and again after Selection Sunday
with --use_espn to use the real bracket):

    # Pre-Selection Sunday (Warren Nolan hypothetical bracket):
    python app/generate_upset_analysis.py

    # Post-Selection Sunday (real ESPN bracket):
    python app/generate_upset_analysis.py --use_espn

Outputs one optimal_upset_strategy.csv per pool size under app/data/pool_{n}/.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from bigdance import Standings
from bigdance.bracket_analysis import BracketAnalysis

POOL_SIZES = [5, 10, 15, 20, 30, 50]
BASE_OBSERVATIONS = 1000
DATA_DIR = Path(__file__).parent / "data"


def run(use_espn: bool = False) -> None:
    source = "ESPN bracket" if use_espn else "Warren Nolan standings"
    print(f"Using {source} as tournament reference.\n")

    standings = None if use_espn else Standings()

    for pool_size in POOL_SIZES:
        num_pools = max(20, BASE_OBSERVATIONS // pool_size)
        print(f"=== Pool size {pool_size} ({num_pools} pools) ===")
        output_dir = DATA_DIR / f"pool_{pool_size}"
        output_dir.mkdir(parents=True, exist_ok=True)

        analyzer = BracketAnalysis(
            standings=standings,
            num_pools=num_pools,
            output_dir=str(output_dir),
            use_espn=use_espn,
        )
        analyzer.simulate_pools(entries_per_pool=pool_size)
        analyzer.plot_comparative_upset_distributions(save=True)
        strategy_df = analyzer.identify_optimal_upset_strategy()

        # identify_optimal_upset_strategy only covers rounds in the upset histogram
        # (First Round–Elite 8). Append Final Four and Championship directly from
        # winning_underdogs_by_round so all 6 rounds are present in the CSV.
        ROUND_ORDER = [
            "First Round", "Second Round", "Sweet 16",
            "Elite 8", "Final Four", "Championship",
        ]
        extra_rows = []
        for rnd in ["Final Four", "Championship"]:
            data = analyzer.winning_underdogs_by_round.get(rnd, [])
            if data:
                s = pd.Series(data)
                extra_rows.append({
                    "round": rnd,
                    "max_advantage_upsets": None,
                    "max_advantage": None,
                    "max_density_upsets": None,
                    "mode_upsets": float(s.mode()[0]),
                    "mean_upsets": s.mean(),
                    "std_upsets": s.std(),
                })
        if extra_rows:
            strategy_df = pd.concat(
                [strategy_df, pd.DataFrame(extra_rows)], ignore_index=True
            )
            # Re-sort to canonical round order, keeping Total Upsets at the end
            order = {r: i for i, r in enumerate(ROUND_ORDER)}
            order["Total Upsets"] = len(ROUND_ORDER)
            strategy_df["_order"] = strategy_df["round"].map(lambda r: order.get(r, 99))
            strategy_df = strategy_df.sort_values("_order").drop(columns="_order").reset_index(drop=True)

        # Add std_upsets for rounds already in the df that came from histogram data
        std_by_round = {
            rnd: np.std(analyzer.winning_underdogs_by_round.get(rnd, []))
            for rnd in ROUND_ORDER
        }
        std_by_round["Total Upsets"] = np.std(analyzer.winning_total_underdogs or [])
        # Only fill std_upsets where not already set by the extra_rows block above
        strategy_df["std_upsets"] = strategy_df.apply(
            lambda r: r["std_upsets"] if pd.notna(r.get("std_upsets")) else std_by_round.get(r["round"]),
            axis=1,
        )
        strategy_df["num_pools"] = num_pools
        strategy_df.to_csv(output_dir / "optimal_upset_strategy.csv", index=False)

        # Save Madness Score (negative log probability) stats for winners by round
        log_prob_rows = []
        for rnd in ROUND_ORDER:
            data = analyzer.winning_log_probs_by_round.get(rnd, [])
            if data:
                log_prob_rows.append({
                    "round": rnd,
                    "mean_madness": float(np.mean(data)),
                    "std_madness": float(np.std(data)),
                })
        pd.DataFrame(log_prob_rows).to_csv(output_dir / "log_prob_strategy.csv", index=False)

        # Save common underdogs in winning brackets
        analyzer.find_common_underdogs()

        print(f"  Saved to {output_dir}/\n")

    print("Done. All pool sizes complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate upset analysis data for the bracket app.")
    parser.add_argument(
        "--use_espn",
        action="store_true",
        help="Use real ESPN bracket (run after Selection Sunday).",
    )
    args = parser.parse_args()
    run(use_espn=args.use_espn)
