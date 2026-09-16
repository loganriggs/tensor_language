#!/usr/bin/env python3
"""Isolated replay check for the packaged contextual DCT node."""
import importlib.util
import json
import sys
from pathlib import Path

import torch

package = Path(sys.argv[1])
spec = importlib.util.spec_from_file_location("contextual_dct_execute", package / "execute.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
program = torch.load(package / "weights.pt", weights_only=True, map_location="cpu")
fixture = torch.load(package / "fixture.pt", weights_only=True, map_location="cpu")
actual = module.execute(fixture["state"], program).float()
expected = fixture["expected"].float()
relative = float((actual - expected).norm() / expected.norm().clamp_min(1e-30))
print(json.dumps({"schema": "mlp9_contextual_dct_node_v1_isolated_check", "relative_l2": relative, "shape": list(actual.shape), "passed": relative <= 2e-6}, sort_keys=True))
raise SystemExit(0 if relative <= 2e-6 else 1)
