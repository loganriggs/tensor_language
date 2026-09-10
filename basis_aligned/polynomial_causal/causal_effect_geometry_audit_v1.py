"""Read-only effect-geometry audit; no fitting, model loading, or new screening bars."""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def geometry(ratio, projection, local_error):
    # d = native intervention effect, t = natural paired effect.
    # p=<d,t>/||t||^2, r=||d||/||t||, e=||compiled-d||/||d||.
    squared = 1 + ratio * ratio - 2 * projection
    if squared < -1e-12:
        raise ValueError("Inconsistent norms and projection")
    error = math.sqrt(max(0., squared))
    radius = local_error * ratio
    return dict(native_relative_error=error,
                compiled_relative_error_lower=max(0., error-radius),
                compiled_relative_error_upper=error+radius,
                intervention_to_natural_norm_ratio=ratio,
                signed_projection=projection,
                compiler_error_in_natural_units=radius)


def main():
    path = ROOT / 'BILIN18_FIRST_ATTENTION_CUE_MESSAGE_V1_RESULT.json'
    raw = path.read_bytes()
    source = json.loads(raw)
    result = {}
    for panel, report in source['reports'].items():
        result[panel] = {}
        for side, cell in report['cells'].items():
            result[panel][side] = {
                'full_vector': geometry(
                    cell['direct_effect_norm']/cell['natural_effect_norm'],
                    cell['signed_paired_effect_projection'],
                    cell['compiled_relative_effect_error']),
                'answer_margin': geometry(
                    cell['direct_margin_norm']/cell['natural_margin_norm'],
                    cell['signed_paired_margin_projection'],
                    cell['compiled_relative_margin_error'])}
    # Independent coordinate check, including an exactly reproduced weak carrier.
    target, effect, compiled = (3., 4.), (-1., 2.), (-.8, 2.1)
    norm = lambda v: math.sqrt(sum(x*x for x in v))
    diff = lambda a,b: tuple(x-y for x,y in zip(a,b))
    g = geometry(norm(effect)/norm(target),
                 sum(x*y for x,y in zip(effect,target))/norm(target)**2,
                 norm(diff(compiled,effect))/norm(effect))
    direct = norm(diff(effect,target))/norm(target)
    actual = norm(diff(compiled,target))/norm(target)
    assert abs(g['native_relative_error']-direct) < 1e-14
    assert g['compiled_relative_error_lower'] <= actual <= g['compiled_relative_error_upper']
    weak = geometry(.01, .01, 0.)
    assert abs(weak['native_relative_error']-.99) < 1e-14
    out = dict(schema=1, source=path.name,
               source_sha256=hashlib.sha256(raw).hexdigest(),
               script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               scope='Post-result geometric audit, not a new candidate or changed verdict; opened rows only.',
               formula='E=sqrt(1+r^2-2p); compiled error in [max(0,E-e*r),E+e*r]',
               controls=dict(coordinate_identity_error=abs(g['native_relative_error']-direct),
                             compiled_triangle_bound_passed=True,
                             exact_weak_carrier_relative_error=weak['native_relative_error']),
               reports=result, model_forwards=0)
    output = ROOT / 'CAUSAL_EFFECT_GEOMETRY_AUDIT_V1_RESULT.json'
    with output.open('x') as f:
        json.dump(out, f, indent=2, allow_nan=False)
        f.write('\n')
    for panel, sides in result.items():
        print(panel)
        for side, frames in sides.items():
            print(side, {k: round(v['native_relative_error'],6) for k,v in frames.items()})


if __name__ == '__main__':
    main()
