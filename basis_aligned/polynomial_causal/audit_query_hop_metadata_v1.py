"""Append-only metadata corrections; completed scores and artifacts stay intact."""
import hashlib
import json
from pathlib import Path
import signal
import time
import torch
import query_hop_fork_reference as H
import suffix_join_middle_match_reference as R


def main():
    signal.alarm(180); torch.set_num_threads(2); started = time.perf_counter()
    root = Path(__file__).resolve().parent; output = root/'QUERY_HOP_METADATA_CORRECTION_V1.json'
    assert not output.exists(); checks = H.controls(); assert checks['passed']
    data = R.populations(); results = {}
    for stem in ('JOIN_WRITE_READ_DEGREE_V1', 'MATCHER_PAIR_TRANSPORT_V1'):
        path = root/(stem+'_ROWS.pt'); saved = torch.load(path, map_location='cpu', weights_only=True)
        changed = {}; counts = {'answer': 0, 'expected_answer': 0}
        for pop, (tokens, _, metadata) in data.items():
            _, expected = H.fork(tokens[::4], metadata[::4]); actual = saved[pop]['metadata']
            assert len(expected) == len(actual); overlay = []
            for i, (a, b) in enumerate(zip(actual, expected)):
                assert all(a[k] == b[k] for k in ('world', 'order', 'case', 'orientation', 'hop'))
                fix = {k: b[k] for k in counts if a[k] != b[k]}
                for k in fix: counts[k] += 1
                if fix: overlay.append({'row': i, **fix})
            changed[pop] = overlay
        results[stem] = {'rows_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                         'changed_fields': counts, 'metadata_overlay': changed}
    result = {'scope': 'Apply overlays before answer-based downstream analysis; no logits or scientific scores changed.',
              'reason': 'Degree fork copied hop3 answer/expected_answer; pair fork recomputed answer but retained hop3 expected_answer.',
              'score_impact': 'None: registered exactness, centered effects, KL, degree and removal partitions do not consume these labels.',
              'controls': checks, 'artifacts': results, 'wall_seconds': time.perf_counter()-started}
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: v['changed_fields'] for k, v in results.items()}))


if __name__ == '__main__': main()
