import json
import os
from pathlib import Path
import subprocess
import sys

import run_temporal_iswas_l11h3_source_writer_weight_ordered_greedy_v1 as target


def test_frozen_orders_and_price():
    assert target.ORDERS == {
        "temporal": ("L07H07", "L09H04", "L09H01", "L06H07", "L05H01"),
        "iswas": ("L07H07", "L09H04", "L06H07", "L09H01", "L03H04"),
    }
    assert target.PREFIX_ARMS == ("P1", "P2", "P3", "P4", "P5")
    assert target.PRICE["model_forwards"] == 26


def test_selection_and_validation_boundaries():
    good = {"signed_recovery": .5, "cosine": .9,
            "relative_residual": .75, "direction_agreement": .9}
    assert target.selection_qualified(good)
    assert not target.selection_qualified({**good, "signed_recovery": 1.50001})
    assert target.validation_qualified({**good, "signed_recovery": 1.75,
                                        "relative_residual": .9}, pooled=True)
    assert target.validation_qualified({**good, "signed_recovery": 0,
                                        "direction_agreement": .75}, pooled=False)


def test_dryrun_is_no_model_and_exact_authority():
    environment = dict(os.environ, BQLIB_DRYRUN="1")
    completed = subprocess.run([sys.executable, str(Path(target.__file__))],
        check=True, capture_output=True, text=True, env=environment)
    payload = json.loads(completed.stdout)
    assert payload["authority_ok"] is True
    assert payload["gpu_accessed"] is False
    assert payload["model_loaded"] is False
    assert payload["price"] == target.PRICE
