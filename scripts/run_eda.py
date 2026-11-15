import sys
from pathlib import Path

# Locate the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import the main EDA execution function
from src.eda import run_eda

if __name__ == "__main__":
    run_eda()
