import hashlib
import json

import torch

from . import mlp4_signed_square_audit as audit
from . import mlp4_signed_square_codec as codec


def test_audit_binds_every_native_prefix_and_decodes():
    result = json.loads(audit.OUTPUT.read_text())
    artifact = torch.load(audit.OUTPUT_BYTES, map_location="cpu", weights_only=False)
    assert result["source_candidate_bytes_sha256"] == audit.sha(audit.BYTES)
    assert result["source_inventory_sha256"] == audit.sha(audit.INVENTORY)
    assert result["signed_square_candidate_bytes_sha256"] == audit.sha(audit.OUTPUT_BYTES)
    assert len(result["rows"]) == 5 and len(artifact["encoded"]) == 5
    for row in result["rows"]:
        stream = artifact["encoded"][row["candidate_id"]]
        assert "sha256:"+hashlib.sha256(stream).hexdigest() == row["signed_square_hash"]
        decoded = codec.decode(stream)
        assert decoded["U"].shape == (1152, row["components"])
        assert decoded["V"].shape == decoded["U"].shape
        assert decoded["C"].shape == (row["components"], 1152)
        assert row["signed_square_codec_bits"] == 8*len(stream)
        assert 0 <= row["relative_coefficient_tensor_frobenius_error"] < 1e-3
    assert not result["behavioral_roster_changed"]
    assert not result["validation_opened"]
