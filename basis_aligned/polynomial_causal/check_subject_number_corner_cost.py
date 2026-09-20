"""CPU audit: algebraic query price of the frozen subject-number graph."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
POLY = Path(__file__).resolve().parent
OPS = ROOT / 'basis_aligned/bilinear_quotient/ops'
sys.path.insert(0, str(OPS))
from disk_guard import guard_write


def main():
    source = POLY / 'extracted_circuits/subject_number_l11h3_sparse_graph_v1/node.py'
    spec = importlib.util.spec_from_file_location('frozen_subject_node', source)
    node = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(node)
    coefficients = Counter()
    for mask in node.MASKS:
        subset = mask
        while True:
            # Damage is minus the Mobius dividend.
            coefficients[subset] -= (-1) ** (mask.bit_count() - subset.bit_count())
            if subset == 0:
                break
            subset = (subset - 1) & mask
    coefficients = {k: v for k, v in sorted(coefficients.items()) if v}
    expected = {0: 1, 2: -1, 4: 1, 8: 2, 9: -1, 12: -1, 16: 1, 20: -1, 24: -1}
    assert coefficients == expected
    # Independent arbitrary corner values, including every coordinate basis:
    # this checks the linear functional, not only planted degree-two polynomials.
    rng = np.random.default_rng(20260920)
    values = np.concatenate([np.eye(32), rng.normal(size=(32, 1000))], axis=1)
    corners = {mask: values[mask] for mask in node.REQUIRED_CORNERS}
    reference = node.compose(node.decompose(corners))
    aggregate = sum(coef * values[mask] for mask, coef in coefficients.items())
    error = float(np.max(np.abs(reference - aggregate)))
    assert error < 1e-12
    # Dropping any retained query changes at least one basis response.
    essential = all(np.max(np.abs(reference - (aggregate - coef * values[mask]))) >= 1
                    for mask, coef in coefficients.items())
    assert essential
    receipt = json.loads((POLY / 'SUBJECT_NUMBER_SPARSE_GRAPH_TOKEN_EXTRACTION_V1_RESULT.json').read_text())
    executor = OPS / 'subject_number_sparse_graph_token_extraction_v1.py'
    bound = hashlib.sha256(executor.read_bytes()).hexdigest() == receipt['component_sha256']
    assert bound
    result = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'terminal': 'subject_number_corner_cost_audited',
        'native_execution': False,
        'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'executor_matches_original_receipt': bound,
        'aggregate_coefficients': coefficients,
        'basis_and_random_columns': values.shape[1],
        'max_absolute_replay_error': error,
        'all_retained_queries_essential_for_arbitrary_corner_oracle': essential,
        'price_block_evaluations_per_sequence': {
            'original_graph': 2 * 11 + 10 * 7,
            'aggregate_only_symbolic_plan': 2 * 11 + 9 * 7,
            'exact_joint_effect_two_corners': 2 * 11 + 2 * 7,
        },
        'scope': 'Block counts exclude embedding, readout, bookkeeping and batching; not FLOPs or measured latency. Aggregate plan not installed or natively replayed. No per-edge removability at nine-query boundary. Oracle minimality is not a native circuit lower bound.',
    }
    out = POLY / 'SUBJECT_NUMBER_CORNER_COST_AUDIT_2026-09-20.json'
    payload = json.dumps(result, indent=2) + '\n'
    guard_write(len(payload.encode()), str(out.parent), 'subject-number CPU audit')
    out.write_text(payload)
    print(payload)


if __name__ == '__main__':
    main()
