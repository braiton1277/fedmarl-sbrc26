"""
Entry point for FEDMARL experiments.

Usage:
    # Run an experiment defined in conf/experiments.yaml:
    python main.py --experiment exp1

    # Run all experiments with the canonical defaults:
    python main.py

    # List available experiments:
    python main.py --list
"""
import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml

from config import ExperimentConfig
from experiment import run_experiment


DEFAULT_CONFIG = "conf/experiments.yaml"


def parse_set_args(items: List[str]) -> Dict[str, Any]:
    """
    Parse `--set key=value` overrides. Values are decoded via yaml.safe_load,
    which handles ints, floats (incl. 1e-4), bools, null, and lists like [1,2,3].
    """
    out: Dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Bad --set entry {item!r}; expected key=value")
        k, v = item.split("=", 1)
        out[k.strip()] = yaml.safe_load(v)
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Run a FEDMARL experiment from a unified YAML config.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help="Path to the unified YAML config (defaults + experiments).",
    )
    parser.add_argument(
        "--experiment",
        default=None,
        help="Name of the experiment preset to apply (e.g. exp1). "
             "If omitted, only the defaults block is used.",
    )
    parser.add_argument(
        "--set",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="Ad-hoc overrides applied last, e.g. --set rounds=100 local_lr=5e-3.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List the experiments declared in the config and exit.",
    )
    args = parser.parse_args()

    if args.list:
        names = ExperimentConfig.list_experiments(args.config)
        print("Available experiments:")
        for name in names:
            print(f"  - {name}")
        sys.exit(0)

    cfg = ExperimentConfig.load(
        config_path=args.config,
        experiment=args.experiment,
        cli_overrides=parse_set_args(args.set),
    )

    
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg.dump(out_dir / f"{cfg.exp_name}_config.yaml")

    run_experiment(cfg)


if __name__ == "__main__":
    main()
