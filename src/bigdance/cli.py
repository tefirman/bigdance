"""Unified CLI entry point for bigdance."""

import argparse
import sys
from typing import Optional

from bigdance import __version__


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bigdance",
        description="March Madness Bracket Pool Simulator and Analyzer",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser(
        "standings",
        help="Get current NCAA basketball team ratings and matchups",
        add_help=False,
    )

    subparsers.add_parser(
        "simulate",
        help="Simulate a hypothetical March Madness bracket pool",
        add_help=False,
    )

    subparsers.add_parser(
        "espn",
        help="Simulate an ESPN Tournament Challenge bracket pool",
        add_help=False,
    )

    subparsers.add_parser(
        "analyze",
        help="Analyze winning strategies across multiple simulated pools",
        add_help=False,
    )

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = create_parser()

    # Parse only the command name, pass remaining args to the subcommand
    args, remaining = parser.parse_known_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "standings":
        from bigdance.wn_cbb_scraper import main as standings_main

        standings_main(remaining)
        return 0

    elif args.command == "simulate":
        from bigdance.bigdance_integration import main as simulate_main

        simulate_main(remaining)
        return 0

    elif args.command == "espn":
        from bigdance.espn_tc_scraper import main as espn_main

        return espn_main(remaining) or 0

    elif args.command == "analyze":
        from bigdance.bracket_analysis import main as analyze_main

        return analyze_main(remaining) or 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
