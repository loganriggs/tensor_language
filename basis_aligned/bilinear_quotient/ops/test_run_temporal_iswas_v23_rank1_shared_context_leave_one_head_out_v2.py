import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

import run_temporal_iswas_v23_rank1_shared_context_leave_one_head_out_v2 as target


def test_dryrun_is_v2_float64_zero_forward_and_authority_bound():
    environment = dict(os.environ, BQLIB_DRYRUN="1", BQLIB_NO_MODEL="1")
    completed = subprocess.run(
        [sys.executable, str(Path(target.__file__))],
        check=True, capture_output=True, text=True, env=environment,
    )
    report = json.loads(completed.stdout)
    assert report["candidate_id"].endswith("leave_one_head_out_v2")
    assert report["authority_ok"] is True
    assert report["arithmetic"] == "float64"
    assert report["gpu_accessed"] is False
    assert report["model_loaded"] is False
    assert report["queue_touched"] is False
    assert all(value == 0 for value in report["price"].values())


def test_random_control_preserves_float64_and_is_reproducible():
    value = torch.arange(35, dtype=torch.float64).reshape(7, 5)
    one = target.random_capture_q99(
        value, generator=torch.Generator().manual_seed(target.RANDOM_SEED)
    )
    two = target.random_capture_q99(
        value, generator=torch.Generator().manual_seed(target.RANDOM_SEED)
    )
    assert one == two
    assert all(np.isfinite(item) for item in one)


def test_immutable_map_precheck_keeps_float64_and_closes_certificate():
    frozen = np.load(target.ARRAYS, allow_pickle=False)
    maps = torch.from_numpy(frozen["contracted"]).to(torch.float64)
    report = target.translation.shared_context_leave_one_out(maps, rank=target.RANK)
    pooled = report["pooled_report"]
    certificate = max(
        [float(pooled["certificate_absolute_error"])]
        + [float(fold["training_certificate_absolute_error"])
           for fold in report["folds"]]
    )
    reconstruction = max(
        float((common + tail - value).abs().max())
        for common, tail, value in zip(
            pooled["common_maps"], pooled["private_tails"], maps
        )
    )
    assert pooled["basis"].dtype == torch.float64
    assert certificate <= target.BARS["instrument"]
    assert reconstruction <= target.BARS["instrument"]
