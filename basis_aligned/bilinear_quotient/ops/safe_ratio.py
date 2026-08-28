"""One place to divide, with the two failure modes this arc actually hit.

LESSON 35 (§1740): `g['PROG'] / max(g['OAT'], 1e-9)` printed **308,477,470.98x** because the
denominator was -0.0008. A floor guard turns "this ratio is undefined here" into a large, confident,
printable number, and that number went into a registered prediction and made it unscoreable.

LESSON 32 (§1728, §1735): per-site ratios blew up to -3.6, -5.2 and -2.0 at three attention sites
whose denominator crossed zero, and a median over eighteen of them meant nothing. The fix there was
to report a DIFFERENCE instead, which is defined everywhere and, in nats, is the unit anyone cares
about.

The residual gap this module closes, found by a self-audit at §1760: every ratio helper written after
LESSON 35 guards `abs(den) > eps` and NOT `den > 0`. A near-zero denominator returns nan as intended;
a merely NEGATIVE one sails through and produces a large signed ratio that looks like data. Both
failures above were negative denominators, not tiny ones.

Use `ratio()` when a ratio is genuinely what you want and the denominator is genuinely positive.
Prefer `difference()` otherwise. Never floor a denominator.
"""

import math


def ratio(num, den, *, require_positive=True, eps=1e-9):
    """num/den, or None where the ratio is not defined.

    `require_positive=True` (the default) returns None for a denominator that is negative as well as
    one that is near zero, because a signed ratio across a sign change is not a comparable quantity.
    Set it False only when the denominator is a signed quantity whose sign you intend to propagate,
    and say so at the call site.
    """
    if den is None or num is None:
        return None
    if isinstance(den, float) and math.isnan(den):
        return None
    if abs(den) <= eps:
        return None
    if require_positive and den < 0:
        return None
    return num / den


def difference(a, b):
    """a - b. Defined everywhere, and in nats it is the unit the reader wants anyway."""
    return None if (a is None or b is None) else a - b


def fmt(x, spec='.4f', na='n/a'):
    """Render a possibly-None ratio. A missing row must LOOK missing, never print as 0.0000."""
    return na if x is None else format(x, spec)


def summarise(values, name='ratio'):
    """Median and range over the DEFINED entries, with the undefined count reported not dropped.

    §1728 took a median over eighteen per-site ratios, three of which had denominators crossing zero.
    The median existed and meant nothing. This makes the exclusion visible in the returned dict.
    """
    ok = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    n_undef = len(values) - len(ok)
    if not ok:
        return {'name': name, 'n': 0, 'n_undefined': n_undef, 'median': None,
                'min': None, 'max': None}
    s = sorted(ok)
    med = s[len(s) // 2] if len(s) % 2 else 0.5 * (s[len(s) // 2 - 1] + s[len(s) // 2])
    return {'name': name, 'n': len(ok), 'n_undefined': n_undef, 'median': med,
            'min': s[0], 'max': s[-1]}


def _self_test():
    """Known answers for every branch, run at import time in the scripts that use this."""
    assert ratio(1.0, 2.0) == 0.5
    assert ratio(1.0, -0.0008) is None, 'a negative denominator must not produce a ratio'
    assert ratio(1.0, 1e-12) is None, 'a near-zero denominator must not produce a ratio'
    assert ratio(1.0, -2.0, require_positive=False) == -0.5
    assert ratio(1.0, float('nan')) is None
    assert difference(1.0, 0.25) == 0.75
    assert fmt(None) == 'n/a' and fmt(0.5, '.2f') == '0.50'
    s = summarise([1.0, None, 3.0, 2.0])
    assert s['n'] == 3 and s['n_undefined'] == 1 and s['median'] == 2.0, s
    assert summarise([None, None])['median'] is None


_self_test()
