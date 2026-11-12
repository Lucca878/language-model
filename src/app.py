"""CLI entry point."""

import argparse

from pipeline import run_project
from project_config import load_project_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one configured LLM task")
    parser.add_argument(
        "--config",
        default="project.yml",
        help="Project YAML file (default: project.yml)",
    )
    parser.add_argument("--n-statements", type=int, help="Override the configured sample size")
    parser.add_argument("--repetitions", type=int, help="Override repetitions per statement")
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run up to 3 statements once each and save results under a test folder",
    )
    args = parser.parse_args()

    config = load_project_config(args.config)
    run_project(
        config,
        n_statements=3 if args.test else args.n_statements,
        repetitions=1 if args.test else args.repetitions,
        test_mode=args.test,
    )


if __name__ == "__main__":
    main()
