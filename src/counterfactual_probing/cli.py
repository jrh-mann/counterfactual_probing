"""
Command-line interface for counterfactual probing.

Provides the `cfprobe` command for running the pipeline.
"""

import argparse
import sys
from pathlib import Path


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="cfprobe",
        description="Counterfactual probing for language models",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run the counterfactual probing pipeline")
    run_parser.add_argument(
        "--config", "-c",
        required=True,
        help="Path to configuration JSON file",
    )

    # Init command - create default config
    init_parser = subparsers.add_parser("init", help="Create a default configuration file")
    init_parser.add_argument(
        "--output", "-o",
        default="config.json",
        help="Output path for configuration file (default: config.json)",
    )

    # Parse args
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "run":
        cmd_run(args)
    elif args.command == "init":
        cmd_init(args)


def cmd_run(args):
    """Execute the run command."""
    from .run import run

    config_path = args.config

    if not Path(config_path).exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    print(f"Running counterfactual probing with config: {config_path}")
    run(config_path)
    print("Done!")


def cmd_init(args):
    """Execute the init command."""
    import json

    from .config import create_default_config

    output_path = Path(args.output)

    if output_path.exists():
        print(f"Error: File already exists: {output_path}")
        print("Use a different output path or remove the existing file.")
        sys.exit(1)

    config = create_default_config()

    with open(output_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Created default configuration at: {output_path}")
    print("Edit this file to customize your pipeline settings.")


if __name__ == "__main__":
    main()
