#!/usr/bin/env python3
"""Teapot CLI — composition system for LLM training data."""

import sys


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print_help()
        sys.exit(0)

    if sys.argv[1] == "--version":
        from teapot import __version__
        print(f"teapot {__version__}")
        sys.exit(0)

    command = sys.argv[1]
    # Remove the subcommand from argv so each module's argparse works
    sys.argv = [f"teapot {command}"] + sys.argv[2:]

    if command == "compose":
        from teapot.compose import main as cmd_main
        cmd_main()
    elif command == "lock":
        from teapot.lockfile import main as cmd_main
        cmd_main()
    elif command == "train":
        from teapot.training_adapter import main as cmd_main
        cmd_main()
    elif command == "eval":
        from teapot.eval.orchestrator import main as cmd_main
        cmd_main()
    elif command == "sbom":
        from teapot.sbom import main as cmd_main
        cmd_main()
    elif command == "fetch":
        from teapot.data_fetch import main as cmd_main
        cmd_main()
    elif command == "validate":
        # Sub-dispatch: teapot validate module | teapot validate compose
        if len(sys.argv) < 2:
            print("Usage: teapot validate module [--all | PATH]")
            print("       teapot validate compose JSONL [--manifest M]")
            sys.exit(1)
        what = sys.argv[1]
        sys.argv = [f"teapot validate {what}"] + sys.argv[2:]
        if what == "module":
            from teapot.validate_module import main as cmd_main
            cmd_main()
        elif what == "compose":
            from teapot.validate_compose import main as cmd_main
            cmd_main()
        else:
            print(f"Unknown validate target: {what}")
            sys.exit(1)
    else:
        print(f"Unknown command: {command}")
        print()
        print_help()
        sys.exit(1)


def print_help():
    print("teapot — composition system for LLM training data")
    print()
    print("Commands:")
    print("  compose   CONFIG          Merge modules into training JSONL")
    print("  validate  module [--all]  Validate module.yaml files")
    print("  validate  compose JSONL   Validate composed output")
    print("  lock      generate|verify Lockfile for reproducibility")
    print("  train     CONFIG          Generate training framework config")
    print("  eval      CONFIG          Run evaluation pipeline")
    print("  sbom      MANIFEST        Generate SPDX 3.0 SBOM")
    print("  fetch     --type TYPE     Fetch data sources")
    print()
    print("Options:")
    print("  --version                 Show version")
    print("  -h, --help                Show this help")
    print()
    print("See docs/DESIGN.md for architecture.")


if __name__ == "__main__":
    main()
