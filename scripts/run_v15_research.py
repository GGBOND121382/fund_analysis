#!/usr/bin/env python3
"""Run all v1.5 layers together, without changing account or trading configuration."""
import sys

from run_v14_research import main

if __name__ == "__main__":
    if "--layered" not in sys.argv:
        sys.argv.append("--layered")
    if not any(x == "--output" or x.startswith("--output=") for x in sys.argv[1:]):
        sys.argv.extend(["--output", "reports/v15_layered"])
    raise SystemExit(main())
