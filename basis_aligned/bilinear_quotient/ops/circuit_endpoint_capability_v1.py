"""Keep paired cue-effect reachability distinct from correct native endpoints."""
import math


def summarize(base_margins, donor_margins, epsilon=1e-6):
    """Each margin is oriented to that endpoint's OWN intended answer."""
    base,donor=list(base_margins),list(donor_margins)
    if not base or len(base)!=len(donor):raise ValueError('Need nonempty equal-length endpoint margins')
    if not all(math.isfinite(x) for x in base+donor):raise ValueError('Nonfinite endpoint margin')
    reachable=[b+d>epsilon for b,d in zip(base,donor)]
    correct=[b>0 and d>0 for b,d in zip(base,donor)]
    return {'pairs':len(base),'positive_effect_denominators':sum(reachable),
        'base_correct':sum(b>0 for b in base),'donor_correct':sum(d>0 for d in donor),
        'both_endpoints_correct':sum(correct),'both_endpoints_correct_fraction':sum(correct)/len(base),
        'reachable_but_incorrect_pairs':[i for i,(r,c) in enumerate(zip(reachable,correct)) if r and not c]}
