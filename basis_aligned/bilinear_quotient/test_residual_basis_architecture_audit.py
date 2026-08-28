import json

from . import residual_basis_architecture_audit as audit


def test_pinned_architecture_closes_every_declared_source_obligation():
    result = audit.audit()
    assert result["sources_verified"] == 2
    contract = json.loads(audit.CONTRACT.read_text())
    expected_present = sum((audit.ROOT/row["path"]).exists()
                           for row in contract["sources"])
    assert result["source_files_reverified"] == expected_present
    assert result["source_excerpt_snapshot_verified"]
    assert result["runtime_obligations_verified"] == 7
    assert result["reference_obligations_verified"] == 12
    assert result["claims"]["exact_over_real_arithmetic"]
    assert not any(value for key, value in result["claims"].items()
                   if key != "exact_over_real_arithmetic")


def test_source_or_claim_drift_fails_closed(tmp_path):
    contract = json.loads(audit.CONTRACT.read_text())
    bad = dict(contract)
    bad["sources"] = [dict(row) for row in contract["sources"]]
    bad["sources"][0]["sha256"] = "0"*64
    path = tmp_path/"bad-source.json"; path.write_text(json.dumps(bad))
    try:
        audit.audit(path)
    except ValueError as error:
        assert "source" in str(error) and ("hash mismatch" in str(error)
                                           or "disagrees" in str(error))
    else:
        raise AssertionError("source drift was accepted")
    bad = dict(contract); bad["claims"] = dict(contract["claims"])
    bad["claims"]["finite_precision_logit_identity_certified"] = True
    path = tmp_path/"bad-claim.json"; path.write_text(json.dumps(bad))
    try:
        audit.audit(path)
    except ValueError as error:
        assert "claim boundary" in str(error)
    else:
        raise AssertionError("unsupported claim promotion was accepted")


def test_hash_bound_excerpt_mode_is_standalone(tmp_path):
    contract = json.loads(audit.CONTRACT.read_text())
    snapshot_source = audit.ROOT/contract["source_snapshot"]["path"]
    snapshot_target = tmp_path/contract["source_snapshot"]["path"]
    snapshot_target.parent.mkdir(parents=True)
    snapshot_target.write_bytes(snapshot_source.read_bytes())
    contract_path = tmp_path/"contract.json"
    contract_path.write_text(json.dumps(contract))
    result = audit.audit(contract_path, root=tmp_path)
    assert result["sources_verified"] == 2
    assert result["source_files_reverified"] == 0
    assert result["source_excerpt_snapshot_verified"]
