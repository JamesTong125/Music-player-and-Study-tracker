from pathlib import Path

from embedded_session import run_collect_data_cli

LABEL = "distracted"

if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    run_collect_data_cli(root, LABEL)
