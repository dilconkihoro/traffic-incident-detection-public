from __future__ import annotations

import sys
from pathlib import Path

# webapp/ is a standalone layer (decision D-25), not part of the installed trafficwatch package;
# its own repo root needs to be on the path for `from webapp.backend import ...` to resolve.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
