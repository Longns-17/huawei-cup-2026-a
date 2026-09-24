"""Run from this project directory with its isolated Python environment."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

if __name__ == "__main__":
    from npu_schedule.cli import main

    main()
