"""Exact counterexample: factor interchange does not imply task-only output.

Hand-specified Boolean model, not learned-model evidence or a discovery algorithm.
Exhaust all natural worlds and producer/consumer-port interventions using integers.
"""
import hashlib
import itertools
import json
from pathlib import Path


def contraction(selector, payload):
    e = (1 - selector, selector)
    u = ((payload, 0), (1 - payload, 1))
    return tuple(sum(e[r] * u[r][j] for r in range(2)) for j in range(2))


def execute(target_selector, payload, context_selector):
    y = (1 - target_selector) * payload + target_selector * (1 - payload)
    c = context_selector
    return y, c, y * c


def main():
    worlds = list(itertools.product((0, 1), repeat=2))
    factor_cells = 0
    diagonal = []
    port_cells = 0
    for s, p in worlds:
        y, c = contraction(s, p)
        assert (y, c, y * c) == execute(s, p, s)
        # Changing both inputs preserves the target, but changes context.
        yp, cp = contraction(1 - s, 1 - p)
        assert yp == y and abs(cp - c) == 1
        diagonal.append(dict(s=s, p=p, target_unchanged=True,
                             context_change=cp - c,
                             nonlinear_consumer_change=yp * cp - y * c))
        for sd, pd in worlds:
            # Every factor interchange exactly implements its high-level edit.
            for ss, pp in ((sd, p), (s, pd), (sd, pd)):
                a, b = contraction(ss, pp)
                assert (a, b, a * b) == execute(ss, pp, ss)
                factor_cells += 1
            # Edit the selector's target-consumer port, retaining its context port.
            for sp, pc in itertools.product((s, sd), (p, pd)):
                out = execute(sp, pc, s)
                assert out[0] == (sp ^ pc) and out[1] == s
                assert out[2] == out[0] * s  # downstream consumer recomputes
                port_cells += 1
    # A decoder receiving only y cannot reproduce context: identical y, different c.
    collision = (execute(0, 0, 0), execute(1, 1, 1))
    assert collision[0][0] == collision[1][0] and collision[0][1] != collision[1][1]
    result = dict(experiment='shared_selector_intervention_counterexample_v1',
                  arithmetic='exact integers', natural_worlds=4,
                  factor_interchange_cells=factor_cells,
                  consumer_port_cells=port_cells, all_identities_passed=True,
                  joint_diagonals=diagonal,
                  target_only_decoder_collision=collision,
                  exact_factorization_compatible_with_diagonal_output_change=True,
                  hand_specified_fixture=True, trained_model_claim=False,
                  model_forwards=0,
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    path = Path(__file__).with_name('SHARED_SELECTOR_INTERVENTION_COUNTEREXAMPLE_V1_RESULT.json')
    with path.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
