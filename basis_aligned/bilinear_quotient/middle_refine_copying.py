"""MIDDLE REFINE COPYING -- connect the middle's content-word refinement
mechanism (665: attention + MLP balanced, context-dependent) to the
copying circuit. The induction reader heads (L5.H5, L8.H4/H6, L10.H8,
647) sit in the middle. Does the middle's refinement of WHICH content
word rely on COPYING the word from earlier context?

At space_word content-word targets, split by whether the target token
REPEATS earlier in the context (copyable via induction) vs is NOVEL
(never appeared). Mean-ablate the middle [7-15] and measure the drop in
P(correct token) for each subset. If the middle refines via copying, the
drop is LARGER for repeat targets (copying is disrupted); novel targets
must be computed from scratch and depend less on the middle's copy path.

REGISTERED PREDICTIONS:
  (0) SANITY: enough repeat and novel space_word targets (>=100 each);
  (a) COPYING CONTRIBUTES: middle ablation drops P(correct token) MORE
      for repeat targets than novel targets -- the middle's refinement
      uses copying for repeatable content words;
  (b) report P(correct token) full vs middle-ablated, for repeat vs
      novel space_word targets, and the two drops;
  NULL: ablating a LATE block [16] does not show the repeat>novel drop
      asymmetry (the copying refinement is a middle function)."""
import json, time, torch
import torch.nn.functional as F
import numpy as np
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152
T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'middle_refine_copying_results.json'
NFRESH = 48

DET = {'a', 'an', 'the', 'this', 'that', 'these', 'those', 'his', 'her',
       'its', 'their', 'our', 'your', 'my', 'some', 'any', 'no', 'all'}
PREP = {'of', 'to', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as',
        'into', 'about', 'over', 'after', 'before', 'between', 'through'}
PRON = {'he', 'she', 'it', 'they', 'we', 'you', 'i', 'him', 'them', 'us',
        'me', 'who', 'which', 'what'}


def classify(s):
    if chr(10) in s:
        return 'newline'
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


def meanfill(mo, i_, o_):
    return o_.mean(dim=(0, 1), keepdim=True).expand_as(o_)


@torch.no_grad()
def ptok(fresh, blocks, tok_of_pos):
    handles = [m.transformer.h[li].mlp.register_forward_hook(meanfill) for li in blocks] \
        + [m.transformer.h[li].attn.c_proj.register_forward_hook(meanfill) for li in blocks] \
        if blocks else []
    out = np.zeros(NFRESH * T)
    for i in range(0, NFRESH, 4):
        bb = fresh[i:i + 4, :257].to(DEV)
        idx = bb[:, :-1].contiguous(); B = bb.shape[0]
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for blk in m.transformer.h:
            x, v1 = blk(x, v1, x0)
        lg = (30 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30)).float()
        p = F.softmax(lg, dim=-1).reshape(-1, lg.shape[-1])
        npos = p.shape[0]; base = i * T
        tk = tok_of_pos[base:base + npos]
        out[base:base + npos] = p[np.arange(npos), tk].cpu().numpy()
    for h in handles:
        h.remove()
    return out


@torch.no_grad()
def main():
    t0 = time.time()
    cl.use_state(PT + 'census_state_diverse.pt')
    fresh = cl.fineweb_rows(NFRESH)
    toks = fresh.numpy()
    nxt = fresh[:, 1:257].reshape(-1).numpy()
    tok_of_pos = nxt.astype(np.int64)
    cls = np.array([classify(cl.d1(int(t))) for t in nxt])
    sw = cls == 'space_word'

    # is the target token a repeat of something earlier in its row's context?
    repeat = np.zeros(NFRESH * T, dtype=bool)
    for r in range(NFRESH):
        seen = set()
        for j in range(T):
            tgt = int(toks[r, j + 1])
            if tgt in seen:
                repeat[r * T + j] = True
            seen.add(int(toks[r, j]))
    rep_sw = sw & repeat
    nov_sw = sw & ~repeat
    print(f'{rep_sw.sum()} repeat space_word, {nov_sw.sum()} novel space_word',
          flush=True)

    MID = list(range(7, 16))
    base = ptok(fresh, None, tok_of_pos)
    mid = ptok(fresh, MID, tok_of_pos)
    late = ptok(fresh, [16], tok_of_pos)

    def st(p, mk):
        return float(p[mk].mean())
    rep_base, nov_base = st(base, rep_sw), st(base, nov_sw)
    rep_mid, nov_mid = st(mid, rep_sw), st(mid, nov_sw)
    rep_late, nov_late = st(late, rep_sw), st(late, nov_sw)
    dmid_rep = rep_base - rep_mid; dmid_nov = nov_base - nov_mid
    dlate_rep = rep_base - rep_late; dlate_nov = nov_base - nov_late
    print(f'repeat sw: base {rep_base:.4f} mid-abl {rep_mid:.4f} (drop {dmid_rep:.4f})',
          flush=True)
    print(f'novel  sw: base {nov_base:.4f} mid-abl {nov_mid:.4f} (drop {dmid_nov:.4f})',
          flush=True)
    print(f'late16 drops: repeat {dlate_rep:.4f} novel {dlate_nov:.4f}', flush=True)

    p0 = rep_sw.sum() >= 100 and nov_sw.sum() >= 100
    # relative drops (copying should hit repeat harder in RELATIVE terms)
    rel_rep = dmid_rep / (rep_base + 1e-9); rel_nov = dmid_nov / (nov_base + 1e-9)
    pa = rel_rep > rel_nov
    null_ok = (dlate_rep - dlate_nov) < 0.5 * (dmid_rep - dmid_nov) if (dmid_rep - dmid_nov) > 0 else True
    print(f'\n(0) enough: {p0}', flush=True)
    print(f'(a) middle copying (repeat rel-drop {rel_rep:.3f} > novel {rel_nov:.3f}): {pa}',
          flush=True)
    print(f'NULL late asymmetry << middle asymmetry: {null_ok}', flush=True)

    out = {'n_repeat_sw': int(rep_sw.sum()), 'n_novel_sw': int(nov_sw.sum()),
           'repeat_base': round(rep_base, 5), 'novel_base': round(nov_base, 5),
           'repeat_mid_drop': round(dmid_rep, 5), 'novel_mid_drop': round(dmid_nov, 5),
           'repeat_rel_drop': round(float(rel_rep), 4), 'novel_rel_drop': round(float(rel_nov), 4),
           'late_repeat_drop': round(dlate_rep, 5), 'late_novel_drop': round(dlate_nov, 5),
           'pred_0': bool(p0), 'pred_a_copying': bool(pa), 'null_ok': bool(null_ok),
           'runtime_s': time.time() - t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\nwrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
