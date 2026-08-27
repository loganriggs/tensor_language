"""Entry point for the preregistered optimizer-free content oracle screen."""

from __future__ import annotations

import sys
from pathlib import Path


BQ = Path(__file__).resolve().parent.parent / "bilinear_quotient"
sys.path.insert(0, str(BQ))

import ship_error_attrib  # noqa: E402


if __name__ == "__main__":
    ship_error_attrib.main(oracle_content_screen=True)
