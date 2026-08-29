from __future__ import annotations

import json

import torch

import run_shared_output_rrr_real_v1 as base
import run_shared_output_rrr_real_v2_recovery as recovery


def test_recovery_parent_is_exact_and_has_no_scientific_result():
    recovery.verify_spent_parent()


def test_recovery_configuration_is_fresh_and_source_closed():
    try:
        recovery.configure_base()
        assert base.PROTOCOL_VERSION == "v2_recovery"
        assert base.RECOVERY_PARENT["scientific_metrics_observed"] is False
        assert base.AUTHORITY.name == "shared_output_rrr_real_v2_recovery_authority.json"
        assert str(recovery.RUNNER.relative_to(recovery.ROOT)) in base.SOURCE_PATHS
        assert str(recovery.TEST.relative_to(recovery.ROOT)) in base.SOURCE_PATHS
        assert str(recovery.PREREG.relative_to(recovery.ROOT)) in base.SOURCE_PATHS
        assert base.FILE_PINS[str(recovery.V1_AUTHORITY.relative_to(recovery.ROOT))] == (
            recovery.V1_AUTHORITY_FILE_SHA256
        )
    finally:
        recovery.restore_base_defaults()


def test_v2_protocol_equals_spent_v1_plus_recovery_lineage():
    source = {"commit": "c" * 40, "paths": {}, "sha256": "a" * 64}
    checkpoint = json.loads(recovery.V1_AUTHORITY.read_text())["checkpoint"]
    v1_protocol = json.loads(recovery.V1_AUTHORITY.read_text())["protocol"]
    try:
        recovery.configure_base()
        protocol = base.authority_payload(source, base.FILE_PINS, checkpoint)["protocol"]
        lineage = protocol.pop("recovery_parent")
        assert protocol == v1_protocol
        assert lineage == base.RECOVERY_PARENT
    finally:
        recovery.restore_base_defaults()


def test_coverage_partition_follows_non_cpu_token_device():
    covered = torch.ones(base.VOCAB, dtype=torch.bool, device="cpu")
    tokens = torch.empty((2, base.CONTEXT), dtype=torch.long, device="meta")
    observed = base.coverage_partition_mask(covered, tokens)
    assert observed.device.type == "meta"
    assert observed.shape == (2 * (base.CONTEXT - base.SCORE_START),)
