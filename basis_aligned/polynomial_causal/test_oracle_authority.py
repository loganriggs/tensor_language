from pathlib import Path

import pytest

import oracle_authority as AUTH


def test_preliminary_fineweb_requires_implicit_rows_and_canonical_output(tmp_path):
    canonical = tmp_path / "canonical.json"
    assert AUTH.resolve_oracle_output(
        "preliminary_fineweb", None, None, canonical
    ) == canonical.resolve()
    with pytest.raises(RuntimeError, match="explicit row_sets"):
        AUTH.resolve_oracle_output("preliminary_fineweb", {}, None, canonical)
    with pytest.raises(RuntimeError, match="canonical preliminary"):
        AUTH.resolve_oracle_output(
            "preliminary_fineweb", None, tmp_path / "other.json", canonical
        )


def test_development_requires_explicit_rows_and_noncanonical_output(tmp_path):
    canonical = tmp_path / "canonical.json"
    other = tmp_path / "development.json"
    assert AUTH.resolve_oracle_output("none", {}, other, canonical) == other.resolve()
    with pytest.raises(RuntimeError, match="explicit frozen row_sets"):
        AUTH.resolve_oracle_output("none", None, other, canonical)
    with pytest.raises(RuntimeError, match="may not write"):
        AUTH.resolve_oracle_output("none", {}, canonical, canonical)
    with pytest.raises(ValueError, match="unknown oracle authority"):
        AUTH.resolve_oracle_output("canonical_fineweb", {}, other, canonical)
