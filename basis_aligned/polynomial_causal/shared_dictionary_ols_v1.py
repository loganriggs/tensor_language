"""Weight-space multi-output OLS; no data, sparsity-recovery theorem, or GPU run."""
import json
from pathlib import Path
import torch


def select(gram, cross, count):
    """gram=<features,features>; cross=<output target,features>.

    Select the feature giving greatest exact conditional least-squares gain.
    This is OLS, not correlation-only orthogonal matching pursuit.
    """
    g, c = gram.clone(), cross.clone()
    support, gains = [], []
    threshold = gram.diag().max() * 1e-12
    for _ in range(count):
        diagonal = g.diag()
        valid = diagonal > threshold
        if support:
            valid[support] = False
        if not bool(valid.any()):
            break
        scores = c.square().sum(0) / diagonal.clamp_min(threshold)
        scores[~valid] = -torch.inf
        j = int(scores.argmax())
        norm = diagonal[j].sqrt()
        v, z = g[j].clone() / norm, c[:, j].clone() / norm
        gains.append(float(z.square().sum()))
        support.append(j)
        g -= v[:, None] * v[None, :]
        c -= z[:, None] * v[None, :]
    return support, gains


def control():
    torch.manual_seed(941)
    torch.set_default_dtype(torch.float64)
    features = torch.randn(16, 40)
    target = torch.randn(5, 40)
    support, gains = select(features @ features.T, target @ features.T, 10)
    previous = target.square().sum()
    max_gain_error, selection_matches = 0., True
    for step, chosen in enumerate(support):
        prefix = support[:step]
        errors = {}
        for j in range(len(features)):
            if j in prefix:
                continue
            f = features[prefix + [j]]
            writer = torch.linalg.lstsq(f.T, target.T).solution.T
            errors[j] = float((target - writer @ f).square().sum())
        selection_matches &= min(errors, key=errors.get) == chosen
        current = errors[chosen]
        max_gain_error = max(max_gain_error, abs(float(previous) - current - gains[step]))
        previous = current
    orthogonal = torch.linalg.qr(features.T).Q.T
    planted_target = torch.randn(5, 3) @ orthogonal[[2, 7, 11]]
    recovered, _ = select(orthogonal @ orthogonal.T, planted_target @ orthogonal.T, 3)
    passed = selection_matches and max_gain_error < 1e-10 and set(recovered) == {2, 7, 11}
    result = dict(instrument_passed=passed, all_candidate_refit_choices_match=selection_matches,
                  max_absolute_gain_error=max_gain_error, planted_orthogonal_support=recovered,
                  scope="Synthetic coefficient-space control only. Correlated native dictionary recovery untested.")
    Path(__file__).with_name('SHARED_DICTIONARY_OLS_V1_CONTROL.json').write_text(json.dumps(result, indent=2)+'\n')
    assert passed, result
    return result


if __name__ == '__main__':
    print(json.dumps(control(), indent=2))
