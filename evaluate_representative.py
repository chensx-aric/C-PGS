"""No-argument entry point for the representative C-PGS model evaluation."""

from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from scripts.evaluate_representative import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
