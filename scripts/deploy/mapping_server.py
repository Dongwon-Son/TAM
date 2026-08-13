#!/usr/bin/env python3

import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.40")

from simadaptor.deploy.mapping_server import main


if __name__ == "__main__":
    main()
