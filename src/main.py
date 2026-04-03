import sys
from pathlib import Path

from embedded_session import run_study_session_cli

if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    run_study_session_cli(root)
