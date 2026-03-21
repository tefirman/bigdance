#!/usr/bin/env python
"""
Pre-compute optimal upset strategy data for each supported pool size.

Run this script once before deploying the app (and again after Selection Sunday
with --use_espn to use the real bracket):

    # Pre-Selection Sunday (Warren Nolan hypothetical bracket):
    python app/generate_upset_analysis.py
    python app/generate_upset_analysis.py --women

    # Post-Selection Sunday (real ESPN bracket):
    python app/generate_upset_analysis.py --use_espn
    python app/generate_upset_analysis.py --use_espn --women

Outputs one optimal_upset_strategy.csv per pool size under
app/data/men/pool_{n}/ or app/data/women/pool_{n}/.
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


def run(use_espn: bool = False, pool_sizes: list[int] | None = None, base_observations: int = BASE_OBSERVATIONS, gender: str = "men") -> None:
    women = gender == "women"
    source = "ESPN bracket" if use_espn else "Warren Nolan standings"
    print(f"Using {source} as tournament reference ({gender}).\n")

    standings = None if use_espn else Standings(women=women)

    for pool_size in (pool_sizes or POOL_SIZES):
        num_pools = max(20, base_observations // pool_size)
        print(f"=== Pool size {pool_size} ({num_pools} pools) ===")
        output_dir = DATA_DIR / gender / f"pool_{pool_size}"
        output_dir.mkdir(parents=True, exist_ok=True)

        analyzer = BracketAnalysis(
            standings=standings,
            num_pools=num_pools,
            output_dir=str(output_dir),
            use_espn=use_espn,
            women=(gender == "women"),
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
            strategy_df["_order"] = strategy_df["round"].map(lambda r, _order=order: _order.get(r, 99))
            strategy_df = strategy_df.sort_values("_order").drop(columns="_order").reset_index(drop=True)

        # Add std_upsets for rounds already in the df that came from histogram data
        std_by_round = {
            rnd: np.std(analyzer.winning_underdogs_by_round.get(rnd, []))
            for rnd in ROUND_ORDER
        }
        std_by_round["Total Upsets"] = np.std(analyzer.winning_total_underdogs or [])
        # Only fill std_upsets where not already set by the extra_rows block above
        strategy_df["std_upsets"] = strategy_df.apply(
            lambda r, _std=std_by_round: r["std_upsets"] if pd.notna(r.get("std_upsets")) else _std.get(r["round"]),
            axis=1,
        )

        # Add losers' (non-winning) upset stats per round
        for rnd_label, rnd_data_key in [*[(r, r) for r in ROUND_ORDER], ("Total Upsets", None)]:
            if rnd_data_key is not None:
                data = analyzer.non_winning_underdogs_by_round.get(rnd_data_key, [])
            else:
                data = analyzer.non_winning_total_underdogs or []
            mask = strategy_df["round"] == rnd_label
            if mask.any() and data:
                strategy_df.loc[mask, "losers_mean_upsets"] = float(np.mean(data))
                strategy_df.loc[mask, "losers_std_upsets"] = float(np.std(data))

        strategy_df["num_pools"] = num_pools
        strategy_df.to_csv(output_dir / "optimal_upset_strategy.csv", index=False)

        # Save Madness Score (negative log probability) stats for winners by round
        log_prob_rows = []
        for rnd in ROUND_ORDER:
            win_data = analyzer.winning_log_probs_by_round.get(rnd, [])
            lose_data = analyzer.non_winning_log_probs_by_round.get(rnd, [])
            if win_data:
                row = {
                    "round": rnd,
                    "mean_madness": float(np.mean(win_data)),
                    "std_madness": float(np.std(win_data)),
                }
                if lose_data:
                    row["losers_mean_madness"] = float(np.mean(lose_data))
                    row["losers_std_madness"] = float(np.std(lose_data))
                log_prob_rows.append(row)
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
    parser.add_argument(
        "--pool_size",
        type=int,
        default=None,
        help="Run for a single pool size only (e.g. --pool_size 10). Defaults to all pool sizes.",
    )
    parser.add_argument(
        "--base_observations",
        type=int,
        default=BASE_OBSERVATIONS,
        help=f"Total bracket observations to simulate across all pools (default: {BASE_OBSERVATIONS}).",
    )
    parser.add_argument(
        "--gender",
        choices=["men", "women"],
        default="men",
        help="Which tournament to generate data for (default: men).",
    )
    args = parser.parse_args()
    pool_sizes = [args.pool_size] if args.pool_size is not None else None
    run(use_espn=args.use_espn, pool_sizes=pool_sizes, base_observations=args.base_observations, gender=args.gender)
