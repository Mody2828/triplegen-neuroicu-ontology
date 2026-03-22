"""Run a matrix of experiment configs."""

from __future__ import annotations

import argparse
from pathlib import Path

from .run_experiments import run_one
from .config import load_config, load_strategies


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", required=True)
    args = parser.parse_args()
    for config_path in args.configs:
        config = load_config(config_path)
        for strategy in load_strategies(config):
            run_one(config, strategy)


if __name__ == "__main__":
    main()
