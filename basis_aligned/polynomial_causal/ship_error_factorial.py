"""Entry point for the preregistered full-ship replacement-group factorial.

The frozen ship construction remains in ship_error_attrib.py so this audit changes
only the evaluated arm cube and cell accounting.
"""

from __future__ import annotations

import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
BQ = HERE.parent / "bilinear_quotient"
sys.path.insert(0, str(BQ))

import ship_error_attrib  # noqa: E402


if __name__ == "__main__":
    ship_error_attrib.main(factorial=True)
