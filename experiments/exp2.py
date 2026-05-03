"""
Experimento 2 .

"""
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent

if __name__ == "__main__":
    sys.exit(subprocess.call([
        sys.executable,
        str(ROOT / "main.py"),
        "--experiment", "exp2",
        *sys.argv[1:],
    ]))
