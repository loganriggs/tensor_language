"""Dependency-free authority contract shared by oracle entry points."""

from pathlib import Path
from typing import Any


def resolve_oracle_output(
    authority: str,
    row_sets: Any,
    output_path: Any,
    canonical_path: Path,
) -> Path:
    """Validate the authority/input/output cross-product and resolve output."""
    if authority not in ("preliminary_fineweb", "none"):
        raise ValueError(f"unknown oracle authority {authority!r}")
    canonical = canonical_path.resolve()
    output = canonical if output_path is None else Path(output_path).resolve()
    if authority == "preliminary_fineweb":
        if row_sets is not None:
            raise RuntimeError("preliminary FineWeb oracle may not accept explicit row_sets")
        if output != canonical:
            raise RuntimeError("preliminary FineWeb oracle must write its canonical preliminary path")
    else:
        if row_sets is None:
            raise RuntimeError("development oracle requires explicit frozen row_sets")
        if output == canonical:
            raise RuntimeError("development oracle may not write the canonical FineWeb result")
    return output
