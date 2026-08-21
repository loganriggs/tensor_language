"""ARTICLE TRIGGER TRACE -- test 614's article-circuit triggers
(prepositions/be-verbs -> a/an, punctuation/sentence-start -> the)
causally, with the same front-attention-ablation tool that traced the
newline circuit (635). A second input-traced circuit, same shape.

614 (activation-based) claimed: the a/an-vs-the choice is driven by the
preceding token -- prepositions and be-verbs favor the indefinite a/an
(introducing a new noun), sentence boundaries favor the/The. This
measures P(a/an) and P(the) grouped by the current (preceding) token's
category, at baseline vs front [0-2] attention-ablated, to test the
triggers causally and confirm front attention carries them.

Token ids: ' a'=257 ' an'=281 ' the'=262 ' The'=383.

REGISTERED PREDICTIONS:
  (0) SANITY: overall P(article)=P(a/an)+P(the) is a few percent;
  (a) TRIGGER: the a/an-vs-the preference P(a/an)-P(the) is HIGHER after
      prepositions/be-verbs than after sentence-ending punctuation --
      prepositions favor the indefinite, punctuation favors the definite
      (614);
  (b) ATTENTION CARRIES IT: front-attention ablation REDUCES the spread
      in a/an-vs-the preference across the trigger groups (the context-
      dependent choice collapses toward a group-independent baseline);
  (c) report per-group P(a/an), P(the), and the preference;
  NULL: front-MLP ablation does not flatten the group spread the way
      front-attention ablation does -- the trigger is attention-carried
      (parallel to 635)."""
import json, time, torch
import torch.nn.functional as F
import numpy as np
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152
T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'article_trigger_trace_results.json'
NFRESH = 48
A_AN = [257, 281]
THE = [262, 383]

PREP = {'of', 'to', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as',
        'into', 'about', 'over', 'after', 'before', 'between', 'through'}
BE = {'is', 'was', 'are', 'were', 'be', 'been', 'being', 'am', "'s"}


def meanfill_hook(mo, i_, o_):
    return o_.mean(dim=(0, 1), keepdim=True).expand_as(o_)


def curcat(t):
    s = cl.d1(int(t)); st = s.strip().lower()
    if st in PREP:
        return 'prep'
    if st in BE:
        return 'be'
    if st and st[-1] in '.!?':
        return 'end_punct'
    if st and all(not c.isalnum() for c in st):
        return 'other_punct'
    return 'other'


@torch.no_grad()
def measure(fresh, blocks, kind):
    handles = []
    if kind is not None:
        for li in blocks:
            sub = (m.transformer.h[li].attn.c_proj if kind == 'attn'
                   else m.transformer.h[li].mlp)
            handles.append(sub.register_forward_hook(meanfill_hook))
    paa = torch.zeros(NFRESH, T); pth = torch.zeros(NFRESH, T)
    for i in range(0, NFRESH, 4):
        bb = fresh[i:i + 4, :257].to(DEV)
        idx = bb[:, :-1].contiguous(); B = bb.shape[0]
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for blk in m.transformer.h:
            x, v1 = blk(x, v1, x0)
        lg = (30 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30)).float()
        p = F.softmax(lg, dim=-1)
        paa[i:i + B] = (p[..., A_AN[0]] + p[..., A_AN[1]]).cpu()
        pth[i:i + B] = (p[..., THE[0]] + p[..., THE[1]]).cpu()
    for h in handles:
        h.remove()
    return paa.reshape(-1).numpy(), pth.reshape(-1).numpy()


@torch.no_grad()
def main():
    t0 = time.time()
    cl.use_state(PT + 'census_state_diverse.pt')
    fresh = cl.fineweb_rows(NFRESH)
    cur = fresh[:, :256].reshape(-1).numpy()
    cats = np.array([curcat(t) for t in cur])
    GROUPS = ['prep', 'be', 'end_punct', 'other_punct', 'other']

    conds = {'baseline': (None, None), 'front_attn_abl': ([0, 1, 2], 'attn'),
             'front_mlp_abl': ([0, 1, 2], 'mlp')}
    res = {name: measure(fresh, b, k) for name, (b, k) in conds.items()}

    out = {'overall': {}, 'by_group': {}}
    spreads = {}
    for name, (paa, pth) in res.items():
        out['overall'][name] = round(float((paa + pth).mean()), 5)
        g = {}
        for grp in GROUPS:
            msk = cats == grp
            if msk.sum() < 20:
                continue
            a = float(paa[msk].mean()); th = float(pth[msk].mean())
            g[grp] = {'P_a_an': round(a, 5), 'P_the': round(th, 5),
                      'pref': round(a - th, 5), 'n': int(msk.sum())}
        out['by_group'][name] = g
        prefs = [g[grp]['pref'] for grp in g]
        spreads[name] = max(prefs) - min(prefs)
        print(f'{name}:', flush=True)
        for grp in g:
            print(f'  {grp:12s} a/an {g[grp]["P_a_an"]:.4f}  the '
                  f'{g[grp]["P_the"]:.4f}  pref {g[grp]["pref"]:+.4f} '
                  f'(n={g[grp]["n"]})', flush=True)

    base = out['by_group']['baseline']
    p0 = 0.005 < out['overall']['baseline'] < 0.2
    pa = (base.get('prep', {}).get('pref', -9) > base.get('end_punct', {}).get('pref', 9)
          or base.get('be', {}).get('pref', -9) > base.get('end_punct', {}).get('pref', 9))
    pb = spreads['front_attn_abl'] < spreads['baseline']
    null_ok = spreads['front_attn_abl'] < spreads['front_mlp_abl']
    print(f'\n(0) overall plausible: {p0} ({out["overall"]["baseline"]})', flush=True)
    print(f'(a) prep/be pref > end_punct pref: {pa}', flush=True)
    print(f'(b) front-attn ablation shrinks group spread: {pb} '
          f'(base {spreads["baseline"]:.4f} -> attn {spreads["front_attn_abl"]:.4f})',
          flush=True)
    print(f'NULL attn spread < mlp spread: {"ok" if null_ok else "CHECK"} '
          f'(mlp {spreads["front_mlp_abl"]:.4f})', flush=True)

    out.update({'spreads': {k: round(v, 5) for k, v in spreads.items()},
                'pred_0': bool(p0), 'pred_a_trigger': bool(pa),
                'pred_b_attn_carries': bool(pb), 'null_ok': bool(null_ok),
                'runtime_s': time.time() - t0})
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\nwrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
