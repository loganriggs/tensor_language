"""MLP17 OUTPUT DIRECTIONS -- name the readout layer's actual
computational primitives. 615 found mlp17's output is rank-8 (8
directions carry 95% of its output variance); 612/616 established that
the NECESSARY structure lives in output DIRECTIONS, not units. So the
right object to characterize the readout layer is its ~8 output
directions: what token-class does each one read out?

Method: capture mlp17's output over real data, SVD to its top-8 output
PCA directions. For each direction v_k, project every position's output
onto v_k, and for each surface token CLASS of the next token (newline,
capitalized, digit, punct, space_word, determiner, preposition,
pronoun, subword), measure the mean projection at positions whose next
token is in that class vs the generic mean. The class with the largest
signed gap is what direction v_k reads out. This names the readout
layer's 8-dimensional output as interpretable token-class channels.

REGISTERED PREDICTIONS:
  (0) SANITY: the top-8 PCA directions capture >= 90% of mlp17's
      output variance (615 found rank-8 at 95%) -- VOIDS on failure;
  (a) INTERPRETABLE DIRECTIONS (the finding): at least 4 of the 8
      directions map to a DISTINCT token class -- the top class's
      |gap| is >= 2x the second-best class's -- so the readout
      layer's output directions are legible token-class channels, not
      an opaque mix;
  (b) THE MAP (no bar): report each of the 8 directions and its
      strongest token-class readout with the gap ratio;
  (c) COVERAGE: report how many distinct classes the 8 directions
      cover (do they read out 8 different classes, or several the
      same class?);
  NULL: 8 RANDOM orthogonal directions map to token classes far less
      sharply -- their best-class gap ratios average below the PCA
      directions' -- so the interpretability is real structure of the
      output basis, not a property of any 8 directions."""
import json, time, torch
import torch.nn.functional as F
import numpy as np
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152
T = 256
LJ = 17
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'mlp17_output_directions_results.json'
NFRESH = 64

DET = {'a', 'an', 'the', 'this', 'that', 'these', 'those', 'his', 'her',
       'its', 'their', 'our', 'your', 'my', 'some', 'any', 'no', 'each',
       'every', 'another', 'both', 'all'}
PREP = {'of', 'to', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as',
        'into', 'about', 'over', 'after', 'before', 'between', 'through',
        'during', 'under', 'against', 'without', 'within', 'onto'}
PRON = {'he', 'she', 'it', 'they', 'we', 'you', 'i', 'him', 'them', 'us',
        'me', 'who', 'which', 'what'}


def fine_class(s):
    t = s.strip().lower()
    if not t:
        return 'space'
    if t in DET:
        return 'determiner'
    if t in PREP:
        return 'preposition'
    if t in PRON:
        return 'pronoun'
    if t[0].isdigit():
        return 'digit'
    if all(not c.isalnum() for c in t):
        return 'punct'
    if s.strip()[:1].isupper():
        return 'capitalized'
    if s.startswith(' '):
        return 'space_word'
    return 'subword'


CLASSES = ['newline', 'space', 'determiner', 'preposition', 'pronoun',
           'digit', 'punct', 'capitalized', 'space_word', 'subword']


@torch.no_grad()
def main():
    t0 = time.time()
    cl.use_state(PT + 'census_state_diverse.pt')
    fresh = cl.fineweb_rows(NFRESH)
    mlp = m.transformer.h[LJ].mlp

    cap = []
    hk = mlp.register_forward_hook(
        lambda mo, i_, o_: cap.append(o_.detach().float().reshape(-1, D).cpu()))
    for i in range(0, NFRESH, 4):
        bb = fresh[i:i + 4, :257].to(DEV)
        idx = bb[:, :-1].contiguous()
        x = F.rms_norm(m.transformer.wte(idx), (D,))
        x0 = x
        v1 = None
        for blk in m.transformer.h:
            x, v1 = blk(x, v1, x0)
    hk.remove()
    O = torch.cat(cap, 0)                  # (Npos, D)
    mu = O.mean(0)
    Oc = O - mu
    U, S, Vt = torch.linalg.svd(Oc, full_matrices=False)
    var = (S ** 2)
    cum8 = float(var[:8].sum() / var.sum())
    p0 = cum8 >= 0.90
    print(f'top-8 directions capture {cum8:.3f} of output variance: '
          f'{"HELD" if p0 else "FAILED -- VOID"}', flush=True)
    if not p0:
        json.dump({'void': 'rank-8 does not capture 90%', 'cum8': cum8},
                   open(OUT, 'w'), indent=1)
        return

    # next-token class per position
    nxt = fresh[:, 1:257].reshape(-1)
    def classify(t):
        s = cl.d1(int(t))
        if chr(10) in s:
            return 'newline'
        return fine_class(s)
    cls = np.array([classify(t) for t in nxt.tolist()])

    # projections onto top-8 PCA directions
    V8 = Vt[:8]                             # (8, D)
    proj = Oc @ V8.T                        # (Npos, 8)
    proj = proj.numpy()

    def best_class(pvec):
        gm = pvec.mean()
        gaps = {}
        for c in CLASSES:
            msk = (cls == c)
            if msk.sum() >= 20:
                gaps[c] = float(pvec[msk].mean() - gm)
        ranked = sorted(gaps, key=lambda c: -abs(gaps[c]))
        top = ranked[0]
        second = ranked[1] if len(ranked) > 1 else None
        ratio = (abs(gaps[top]) / max(abs(gaps[second]), 1e-9)
                 if second else 999)
        return top, round(gaps[top], 2), round(ratio, 2)

    dirs = []
    ninterp = 0
    for k in range(8):
        top, gap, ratio = best_class(proj[:, k])
        interp = ratio >= 2.0
        ninterp += interp
        dirs.append({'dir': k, 'class': top, 'gap': gap, 'ratio': ratio,
                     'interpretable': interp})
        print(f'  dir {k}: reads out "{top}" (gap {gap:+.2f}, '
              f'{ratio}x 2nd) {"[clean]" if interp else ""}', flush=True)

    pa = ninterp >= 4
    covered = set(d['class'] for d in dirs if d['interpretable'])
    print(f'\n(a) {ninterp}/8 directions map to a distinct class (>=2x): '
          f'{"HELD" if pa else "FAILED"}', flush=True)
    print(f'(c) distinct classes covered by clean directions: '
          f'{sorted(covered)} ({len(covered)})', flush=True)

    # NULL: random orthogonal directions
    g = torch.Generator().manual_seed(0)
    Qr, _ = torch.linalg.qr(torch.randn(D, 8, generator=g))
    projr = (Oc @ Qr).numpy()
    rand_ratios = []
    for k in range(8):
        _, _, ratio = best_class(projr[:, k])
        rand_ratios.append(ratio)
    pca_ratios = [d['ratio'] for d in dirs]
    null_ok = np.median([r for r in pca_ratios if r < 999]) > \
        np.median([r for r in rand_ratios if r < 999])
    print(f'NULL: PCA median gap ratio {np.median([r for r in pca_ratios if r<999]):.2f} '
          f'> random {np.median([r for r in rand_ratios if r<999]):.2f}: '
          f'{"ok" if null_ok else "CHECK"}', flush=True)

    out = {'cum8_variance': cum8, 'pred_0': bool(p0), 'directions': dirs,
           'n_interpretable': int(ninterp), 'pred_a': bool(pa),
           'classes_covered': sorted(covered),
           'random_ratios': rand_ratios, 'null_ok': bool(null_ok),
           'runtime_s': time.time() - t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\nwrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
