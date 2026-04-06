#!/usr/bin/env python3

from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "birdclef_2026_utilities.py"
TARGETS = [
    ROOT / "kaggle" / "training" / "birdclef_2026_utilities.py",
    ROOT / "kaggle" / "inference" / "birdclef_2026_utilities.py",
]


def main() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(f"Missing source utility file: {SOURCE}")

    for target in TARGETS:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(SOURCE, target)
        print(f"Synced {SOURCE.name} -> {target}")


if __name__ == "__main__":
    main()
