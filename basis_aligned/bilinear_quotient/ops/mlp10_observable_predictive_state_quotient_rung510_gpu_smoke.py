#!/usr/bin/env python3
"""Managed no-outcome CUDA smoke for rung510."""

import os

os.environ["BQLIB_GPU_SMOKE"] = "1"

import mlp10_observable_predictive_state_quotient_rung510 as rung

rung.main()
