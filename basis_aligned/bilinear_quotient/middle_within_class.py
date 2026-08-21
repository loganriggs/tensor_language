"""MIDDLE WITHIN CLASS -- test 630's synthesis directly: does the
middle refine WITHIN-class token identity rather than choosing the
class? If so, ablating the middle should degrade P(correct TOKEN) much
more than P(correct CLASS), while ablating the front kills both.

630 found the middle (blocks 6-16) is prediction-light, writes no class
(629), but its CE cost is rare-weighted -- suggesting it picks the
specific (usually rare/content) token within an already-decided class.
This measures, per position, both P(correct next token) and P(correct
class) (probability mass on all tokens of the target's class), with the
middle ablated vs the front ablated.

If the middle refines within-class identity: middle ablation drops
P(token) a lot but P(class) little (the class survives, the specific
token degrades). If the front decides the class: front ablation drops
BOTH P(token) and P(class).

REGISTERED PREDICTIONS:
  (0) SANITY: baseline P(class) > P(token) at each position (class mass
      >= its member token) -- trivially true, checks the machinery;
  (a) MIDDLE = WITHIN-CLASS: ablating the middle [6-16] reduces P(token)
      by a LARGER relative fraction than it reduces P(class) -- the
      class is preserved better than the specific token;
  (b) FRONT = CLASS: ablating the front [0-2] reduces P(class) by a
      relative fraction comparable to or larger than P(token) -- the
      front decides the class itself (no within-class sparing);
  (c) report the P(token) and P(class) relative drops for front vs
      middle, split by frequent/rare target;
  NULL/contrast: the middle's (P_class preserved / P_token lost) sparing
      ratio is LARGER than the front's -- within-class refinement is
      specific to the middle, not a property of any ablation."""
import json, time, torch
import torch.nn.functional as F
import numpy as np
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152
T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'middle_within_class_results.json'
NFRESH = 48
TOPK = 20

DET = {'a', 'an', 'the', 'this', 'that', 'these', 'those', 'his', 'her',
       'its', 'their', 'our', 'your', 'my', 'some', 'any', 'no', 'all'}
PREP = {'of', 'to', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as',
        'into', 'about', 'over', 'after', 'before', 'between', 'through'}
PRON = {'he', 'she', 'it', 'they', 'we', 'you', 'i', 'him', 'them', 'us',
        'me', 'who', 'which', 'what'}
BANDS = {'front': [0, 1, 2], 'middle': list(range(6, 17))}


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


@torch.no_grad()
def measure(fresh, ablate_set, tok_of_pos, tgt_class, class_to_vmask):
    """Return per-position P(correct token) and P(correct class). The
    per-batch class mask is built on the fly to avoid a full (npos, V)
    tensor."""
    ptok = np.zeros(len(tok_of_pos)); pcls = np.zeros(len(tok_of_pos))
    pos0 = 0
    for i in range(0, NFRESH, 4):
        bb = fresh[i:i + 4, :257].to(DEV)
        idx = bb[:, :-1].contiguous(); B = bb.shape[0]
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for li, blk in enumerate(m.transformer.h):
            x_in = x
            x, v1 = blk(x, v1, x0)
            if ablate_set is not None and li in ablate_set:
                delta = x - x_in
                x = x_in + delta.mean(dim=(0, 1), keepdim=True)
        lg = (30 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30)).float()
        p = F.softmax(lg, dim=-1).reshape(-1, lg.shape[-1])   # (B*T, V)
        npos = p.shape[0]
        tk = tok_of_pos[pos0:pos0 + npos]
        ptok[pos0:pos0 + npos] = p[np.arange(npos), tk].cpu().numpy()
        cm = torch.stack([class_to_vmask[c]
                          for c in tgt_class[pos0:pos0 + npos]]).to(DEV)  # (npos,V)
        pcls[pos0:pos0 + npos] = (p * cm).sum(-1).cpu().numpy()
        pos0 += npos
    return ptok, pcls


@torch.no_grad()
def main():
    t0 = time.time()
    cl.use_state(PT + 'census_state_diverse.pt')
    fresh = cl.fineweb_rows(NFRESH)
    V = m.lm_head.weight.shape[0]
    nxt = fresh[:, 1:257].reshape(-1).numpy()

    # class of each vocab token
    vocab_class = np.array([classify(cl.d1(t)) for t in range(V)])
    # per-position: target token and a boolean vocab-mask of its class
    tok_of_pos = nxt.astype(np.int64)
    tgt_class = np.array([classify(cl.d1(int(t))) for t in nxt])
    # build (npos, V) class mask lazily per class to save memory: precompute
    # class -> vocab bool, then index
    class_to_vmask = {c: torch.tensor(vocab_class == c, dtype=torch.float32)
                      for c in set(tgt_class.tolist())}

    freq = np.bincount(nxt, minlength=V)
    top_tokens = set(np.argsort(-freq)[:TOPK].tolist())
    is_freq = np.array([t in top_tokens for t in nxt])

    base_t, base_c = measure(fresh, None, tok_of_pos, tgt_class, class_to_vmask)
    print(f'baseline P(token) {base_t.mean():.4f}  P(class) {base_c.mean():.4f}',
          flush=True)

    def rel_drops(ab_t, ab_c, mask):
        bt, bc = base_t[mask].mean(), base_c[mask].mean()
        return (float(1 - ab_t[mask].mean() / (bt + 1e-9)),
                float(1 - ab_c[mask].mean() / (bc + 1e-9)))

    out = {'baseline': {'P_token': round(float(base_t.mean()), 4),
                        'P_class': round(float(base_c.mean()), 4)}, 'bands': {}}
    for name, blocks in BANDS.items():
        ab_t, ab_c = measure(fresh, set(blocks), tok_of_pos, tgt_class,
                             class_to_vmask)
        rt_all, rc_all = rel_drops(ab_t, ab_c, np.ones(len(nxt), bool))
        rt_r, rc_r = rel_drops(ab_t, ab_c, ~is_freq)
        rt_f, rc_f = rel_drops(ab_t, ab_c, is_freq)
        spare = rt_all - rc_all            # how much more token than class is lost
        out['bands'][name] = {
            'rel_drop_token_all': round(rt_all, 4), 'rel_drop_class_all': round(rc_all, 4),
            'sparing_all': round(spare, 4),
            'rel_drop_token_rare': round(rt_r, 4), 'rel_drop_class_rare': round(rc_r, 4),
            'rel_drop_token_freq': round(rt_f, 4), 'rel_drop_class_freq': round(rc_f, 4)}
        print(f'{name}: token-drop {rt_all:.3f} class-drop {rc_all:.3f} '
              f'(sparing {spare:+.3f}); rare tok {rt_r:.3f}/cls {rc_r:.3f}',
              flush=True)

    mid = out['bands']['middle']; fr = out['bands']['front']
    p0 = base_c.mean() >= base_t.mean()
    pa = mid['rel_drop_token_all'] > mid['rel_drop_class_all']
    pb = fr['rel_drop_class_all'] >= 0.5 * fr['rel_drop_token_all']
    null_ok = mid['sparing_all'] > fr['sparing_all']
    print(f'\n(0) P(class)>=P(token): {p0}', flush=True)
    print(f'(a) middle spares class over token: {pa} '
          f'(tok {mid["rel_drop_token_all"]:.3f} > cls {mid["rel_drop_class_all"]:.3f})',
          flush=True)
    print(f'(b) front drops class ~as much as token: {pb}', flush=True)
    print(f'NULL middle sparing {mid["sparing_all"]:+.3f} > front '
          f'{fr["sparing_all"]:+.3f}: {"ok" if null_ok else "CHECK"}', flush=True)

    out.update({'pred_0': bool(p0), 'pred_a_middle_within_class': bool(pa),
                'pred_b_front_class': bool(pb), 'null_ok': bool(null_ok),
                'runtime_s': time.time() - t0})
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\nwrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
