"""Exact endpoint decomposition; a diagnostic, not independent circuit extraction."""
import json
from pathlib import Path
import numpy as np


def additive_corner(states):
    """states has axes [first switch, second switch, ..., residual coordinate]."""
    return states[1, 0] + states[0, 1] - states[0, 0]


def decompose(outputs, additive_output):
    """Use the registered total-minus-singletons sign convention."""
    interaction = outputs[1, 0] + outputs[0, 1] - outputs[0, 0] - outputs[1, 1]
    readout = outputs[1, 0] + outputs[0, 1] - outputs[0, 0] - additive_output
    internal = additive_output - outputs[1, 1]
    return dict(interaction=interaction, readout=readout, internal=internal)


def decode(x, weights, eps=1e-7):
    return 30 * np.tanh((x / np.sqrt(np.mean(x*x, axis=-1, keepdims=True) + eps)) @ weights.T / 30)


def controls():
    rng = np.random.default_rng(9100854)
    maximum = 0.0
    for _ in range(32):
        x = rng.normal(size=(2, 2, 7, 5)); w = rng.normal(size=(9, 5))
        a = additive_corner(x); outputs = decode(x, w)
        pieces = decompose(outputs, decode(a, w))
        maximum = max(maximum, float(np.max(np.abs(pieces['interaction'] - pieces['readout'] - pieces['internal']))))
        # Six Gram coefficients and three projected vectors suffice for this corner.
        vectors = np.stack([x[0, 0], x[1, 0]-x[0, 0], x[0, 1]-x[0, 0]])
        gram = np.einsum('aid,bid->iab', vectors, vectors)
        projected = np.einsum('aid,vd->aiv', vectors, w)
        folded = 30*np.tanh(projected.sum(0)/np.sqrt(gram.sum((1, 2))[:, None]/5+1e-7)/30)
        maximum = max(maximum, float(np.max(np.abs(folded-decode(a, w)))))
    # Independent additive writes, no internal interaction, substantial reader interaction.
    x = np.array([[[1., 0.], [1., 1.]], [[2., 0.], [2., 1.]]])
    w = np.array([[1., 0.]])
    pieces = decompose(decode(x, w), decode(additive_corner(x), w))
    assert np.max(np.abs(x[1, 1]-additive_corner(x))) == 0
    assert np.max(np.abs(pieces['internal'])) == 0
    assert np.linalg.norm(pieces['readout']) > .1
    reader_only = float(np.linalg.norm(pieces['readout']))
    # A linear reader cannot manufacture interaction; an internal joint write can.
    x[1, 1] += np.array([.5, 0.])
    pieces = decompose(x @ w.T, additive_corner(x) @ w.T)
    assert np.max(np.abs(pieces['readout'])) == 0
    assert float(pieces['internal'][0]) == -.5
    assert maximum < 1e-11
    return {'passed': True, 'random_fixtures': 32, 'maximum_closure_or_gram_error': maximum,
            'reader_only_interaction_norm': reader_only, 'linear_reader_internal_interaction': -.5,
            'scope': 'Exact real-arithmetic decomposition checked numerically in FP64; nonlinear decoder interaction does not prove internal cross-group computation. Native raw states and weights remain required.'}


if __name__ == '__main__':
    import sys
    result = controls()
    with Path(sys.argv[1]).open('x') as f:
        json.dump(result, f, indent=2); f.write('\n')
    print(json.dumps(result))
