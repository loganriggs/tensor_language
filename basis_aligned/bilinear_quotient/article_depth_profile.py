"""ARTICLE DEPTH PROFILE -- across all 18 layers, which ones actually
build and maintain the a/an-vs-the decision? Traces the flagship
circuit's full depth profile, the whole-stack version of the "fold
into later layers" question.

The article circuit is fully traced at the front: token -> attn0
bigram -> mlp0 cluster 8 -> article logits (597/598), with a parallel
mlp1 echo (599). But the decision is committed early (mlp0 is context-
driven) and then must survive 16 more layers to the output. Does it
pass through untouched, get reinforced, or get refined/overwritten
downstream? This ablates each layer's MLP and each layer's attention
in turn (mean-fill its output over the corpus) and measures the change
in the whole-model article margin at article positions -- a depth
profile of where the a/an-vs-the choice actually lives.

Margin = P(a)+P(an) - P(the)+P(The), measured at real article-target
positions. For each component (mlp0..mlp17, attn0..attn17) mean-fill
its output and record the margin drop; large drop = that component is
load-bearing for the article decision.

REGISTERED PREDICTIONS:
  (0) IDENTITY: with no ablation the margin equals the true margin
      (sanity, exact);
  (a) FRONT-LOADED BUILD: mlp0 and/or attn0 are among the largest-
      magnitude contributors -- the decision is built at the front
      (597/598 established mlp0 cluster 8 + attn0 drive it);
  (b) NOT MIDDLE-CONCENTRATED: the middle-layer MLPs (mlp6-10, the
      abstract/tangled band, 607) contribute LESS on average than the
      front layers (mlp0-2) -- the article decision is not re-derived
      in the abstract middle;
  (c) THE PROFILE (no bar): report the full 18-layer profile for both
      MLP and attention, identifying where the decision is built and
      whether any LATE layer is a major readout/amplifier (a late
      spike would indicate the early decision is re-read near the
      output);
  NULL: mean-filling a component that should be irrelevant to articles
      (a late-middle attention, e.g. attn11) moves the margin far less
      than mlp0 does -- the profile reflects real article-specific
      dependence, not generic perturbation sensitivity."""
import json, time, torch
import torch.nn.functional as F
import numpy as np
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152
T = 256
NL = 18
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'article_depth_profile_results.json'
NFRESH = 64
TOK_A, TOK_AN, TOK_THE, TOK_THE2 = 257, 281, 262, 383


@torch.no_grad()
def comp_means(fresh):
    """Corpus-mean output of each mlp and attn (D-vector each)."""
    sums = {}
    cnt = 0
    hooks = []
    mods = {}
    for li in range(NL):
        mods[f'm{li}'] = m.transformer.h[li].mlp
        mods[f'a{li}'] = m.transformer.h[li].attn
    for key, mod in mods.items():
        sums[key] = torch.zeros(D, device=DEV)

        def mk(key=key):
            def h(mo, i_, o_):
                y = o_[0] if isinstance(o_, tuple) else o_
                sums[key] += y.detach().float().reshape(-1, D).sum(0)
            return h
        hooks.append(mod.register_forward_hook(mk()))
    for i in range(0, fresh.shape[0], 4):
        bb = fresh[i:i + 4, :257].to(DEV)
        idx = bb[:, :-1].contiguous()
        x = F.rms_norm(m.transformer.wte(idx), (D,))
        x0 = x
        v1 = None
        for blk in m.transformer.h:
            x, v1 = blk(x, v1, x0)
        cnt += idx.shape[0] * idx.shape[1]
    for h in hooks:
        h.remove()
    return {k: v / cnt for k, v in sums.items()}, mods


@torch.no_grad()
def margin_with_ablation(fresh, mods, means, key=None):
    """Article margin per position, optionally mean-filling one component."""
    hook = None
    if key is not None:
        mu = means[key]
        mod = mods[key]
        if key[0] == 'a':
            def fh(mo, i_, o_, mu=mu):
                y, v1 = o_
                return (mu.expand_as(y).to(y.dtype), v1)
        else:
            def fh(mo, i_, o_, mu=mu):
                return mu.expand_as(o_).to(o_.dtype)
        hook = mod.register_forward_hook(fh)
    out = []
    for i in range(0, fresh.shape[0], 4):
        bb = fresh[i:i + 4, :257].to(DEV)
        idx = bb[:, :-1].contiguous()
        x = F.rms_norm(m.transformer.wte(idx), (D,))
        x0 = x
        v1 = None
        for blk in m.transformer.h:
            x, v1 = blk(x, v1, x0)
        lg = (30 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30)).float()
        p = F.softmax(lg, dim=-1)
        mg = p[..., TOK_A] + p[..., TOK_AN] - p[..., TOK_THE] - p[..., TOK_THE2]
        out.append(mg.reshape(-1).cpu())
    if hook is not None:
        hook.remove()
    return torch.cat(out)


def main():
    t0 = time.time()
    cl.use_state(PT + 'census_state_diverse.pt')
    fresh = cl.fineweb_rows(NFRESH)
    nxt = fresh[:, 1:257].reshape(-1)
    art = ((nxt == TOK_A) | (nxt == TOK_AN) |
           (nxt == TOK_THE) | (nxt == TOK_THE2)).numpy()
    amask = torch.tensor(art)
    print(f'{art.sum()} article positions', flush=True)

    means, mods = comp_means(fresh)
    base = margin_with_ablation(fresh, mods, means, None)
    base_art = float(base[amask].mean())
    # identity: re-run with no hook must match
    base2 = margin_with_ablation(fresh, mods, means, None)
    p0 = abs(float(base2[amask].mean()) - base_art) < 1e-9
    print(f'(0) identity {p0}; baseline article margin {base_art:.4f}',
          flush=True)

    mlp_prof = {}
    attn_prof = {}
    for li in range(NL):
        mk = f'm{li}'
        ak = f'a{li}'
        mm = margin_with_ablation(fresh, mods, means, mk)
        aa = margin_with_ablation(fresh, mods, means, ak)
        mlp_prof[li] = round(float(mm[amask].mean()) - base_art, 5)
        attn_prof[li] = round(float(aa[amask].mean()) - base_art, 5)
        print(f'  L{li:>2}: mlp {mlp_prof[li]:+.5f}  attn {attn_prof[li]:+.5f}',
              flush=True)
        json.dump({'mlp': mlp_prof, 'attn': attn_prof, 'base': base_art},
                  open(OUT, 'w'), indent=1)

    front_mlp = np.mean([abs(mlp_prof[l]) for l in (0, 1, 2)])
    mid_mlp = np.mean([abs(mlp_prof[l]) for l in (6, 7, 8, 9, 10)])
    mlp_absmax = max(range(NL), key=lambda l: abs(mlp_prof[l]))
    attn_absmax = max(range(NL), key=lambda l: abs(attn_prof[l]))
    pa = mlp_absmax <= 2 or attn_absmax <= 2
    pb = front_mlp > mid_mlp
    print(f'\n(a) largest MLP contributor L{mlp_absmax}, largest attn '
          f'L{attn_absmax} (front-loaded if <=2): {"HELD" if pa else "FAILED"}',
          flush=True)
    print(f'(b) front mlp0-2 mean |delta| {front_mlp:.5f} > middle mlp6-10 '
          f'{mid_mlp:.5f}: {"HELD" if pb else "FAILED"}', flush=True)
    # late readout?
    late_mlp = max((6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17),
                   key=lambda l: abs(mlp_prof[l]))
    print(f'(c) largest late/mid MLP contributor: L{late_mlp} '
          f'({mlp_prof[late_mlp]:+.5f})', flush=True)
    null_ok = abs(attn_prof[11]) < abs(mlp_prof[0])
    print(f'NULL: attn11 |delta| {abs(attn_prof[11]):.5f} < mlp0 '
          f'{abs(mlp_prof[0]):.5f}: {"ok" if null_ok else "CHECK"}', flush=True)

    out = {'base_article_margin': base_art, 'pred_0': bool(p0),
           'mlp': mlp_prof, 'attn': attn_prof,
           'largest_mlp_layer': mlp_absmax, 'largest_attn_layer': attn_absmax,
           'front_mlp_mean': float(front_mlp), 'mid_mlp_mean': float(mid_mlp),
           'pred_a': bool(pa), 'pred_b': bool(pb),
           'largest_late_mlp': late_mlp, 'null_ok': bool(null_ok),
           'runtime_s': time.time() - t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\nwrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
