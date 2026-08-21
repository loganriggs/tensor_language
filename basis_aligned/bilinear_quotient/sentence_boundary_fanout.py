"""SENTENCE BOUNDARY FANOUT -- a fresh, well-powered circuit unifying the
newline (635/639) and capitalized (638) circuits under their shared
trigger. After a sentence-ending '.', the next token is one of: a
NEWLINE (new paragraph), a CAPITALIZED word (new sentence), or a
lowercase/other continuation. The '.' embedding-bigram fires the same at
every period; the blocks must ROUTE probability to the outcome the
context actually calls for. This tests that routing directly.

Method: at end-punct positions (current token . ! ?), measure P(newline)
and P(capitalized) from the direct embedding bigram vs the full model,
split by what ACTUALLY follows (newline / capitalized / other). If the
blocks route, the full model raises the probability of the outcome that
actually occurs, while the context-blind bigram does not vary across
outcomes.

REGISTERED PREDICTIONS:
  (0) SANITY: each actual-outcome bucket has >= 20 end-punct positions;
  (a) BIGRAM IS BLIND: the direct-path P(newline) and P(capitalized) at
      end-punct are nearly the same across the three actual-outcome
      buckets (the '.' bigram cannot tell which follows);
  (b) BLOCKS ROUTE: the full model raises P(newline) most in the
      newline-outcome bucket and P(capitalized) most in the
      capitalized-outcome bucket -- its prediction tracks the actual
      outcome;
  (c) report the 2 measures x 3 outcome-buckets x {direct,full} table;
  NULL: the direct path's variation across outcome buckets (its routing)
      is a small fraction of the full model's -- routing is added by the
      blocks."""
import json, time, torch
import torch.nn.functional as F
import numpy as np
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152
T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'sentence_boundary_fanout_results.json'
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


@torch.no_grad()
def measure(fresh, direct, nl_mask, cap_mask):
    pnl = torch.zeros(NFRESH, T); pcap = torch.zeros(NFRESH, T)
    for i in range(0, NFRESH, 4):
        bb = fresh[i:i + 4, :257].to(DEV)
        idx = bb[:, :-1].contiguous(); B = bb.shape[0]
        x = F.rms_norm(m.transformer.wte(idx), (D,))
        if not direct:
            x0 = x; v1 = None
            for blk in m.transformer.h:
                x, v1 = blk(x, v1, x0)
        lg = (30 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30)).float()
        p = F.softmax(lg, dim=-1).reshape(-1, lg.shape[-1])
        pnl[i:i + B] = (p @ nl_mask).cpu().view(B, T)
        pcap[i:i + B] = (p @ cap_mask).cpu().view(B, T)
    return pnl.reshape(-1).numpy(), pcap.reshape(-1).numpy()


@torch.no_grad()
def main():
    t0 = time.time()
    cl.use_state(PT + 'census_state_diverse.pt')
    fresh = cl.fineweb_rows(NFRESH)
    V = m.lm_head.weight.shape[0]
    nl_mask = torch.tensor([1.0 if chr(10) in cl.d1(t) else 0.0
                            for t in range(V)]).to(DEV)
    cap_mask = torch.tensor([1.0 if classify(cl.d1(t)) == 'capitalized' else 0.0
                             for t in range(V)]).to(DEV)

    cur = fresh[:, :256].reshape(-1).numpy()
    nxt = fresh[:, 1:257].reshape(-1).numpy()

    def is_end_punct(t):
        s = cl.d1(int(t)).strip()
        return len(s) > 0 and s[-1] in '.!?'
    endp = np.array([is_end_punct(t) for t in cur])
    ncls = np.array([classify(cl.d1(int(t))) for t in nxt])
    buckets = {'->newline': endp & (ncls == 'newline'),
               '->capitalized': endp & (ncls == 'capitalized'),
               '->other': endp & ~np.isin(ncls, ['newline', 'capitalized'])}
    print('end-punct outcome counts: '
          + ', '.join(f'{k}={int(v.sum())}' for k, v in buckets.items()), flush=True)

    out = {'counts': {k: int(v.sum()) for k, v in buckets.items()}, 'table': {}}
    routing = {}
    for cond, direct in [('direct', True), ('full', False)]:
        pnl, pcap = measure(fresh, direct, nl_mask, cap_mask)
        out['table'][cond] = {}
        for k, msk in buckets.items():
            out['table'][cond][k] = {'P_newline': round(float(pnl[msk].mean()), 4),
                                     'P_capitalized': round(float(pcap[msk].mean()), 4)}
        # routing = how much P(newline) varies between ->newline and ->cap buckets
        nl_route = (out['table'][cond]['->newline']['P_newline']
                    - out['table'][cond]['->capitalized']['P_newline'])
        cap_route = (out['table'][cond]['->capitalized']['P_capitalized']
                     - out['table'][cond]['->newline']['P_capitalized'])
        routing[cond] = {'nl_route': round(nl_route, 4), 'cap_route': round(cap_route, 4)}
        print(f'{cond}:', flush=True)
        for k in buckets:
            print(f'  {k:14s} P(nl) {out["table"][cond][k]["P_newline"]:.4f}  '
                  f'P(cap) {out["table"][cond][k]["P_capitalized"]:.4f}', flush=True)
        print(f'  routing: nl {nl_route:+.4f}  cap {cap_route:+.4f}', flush=True)

    out['routing'] = routing
    p0 = all(v.sum() >= 20 for v in buckets.values())
    # (b) full model routes: nl higher in ->newline, cap higher in ->cap
    tf = out['table']['full']
    pb = (tf['->newline']['P_newline'] > tf['->capitalized']['P_newline'] and
          tf['->capitalized']['P_capitalized'] > tf['->newline']['P_capitalized'])
    # (a) direct is blind: its routing is small; NULL: direct routing << full
    null_ok = (abs(routing['direct']['nl_route']) < 0.4 * abs(routing['full']['nl_route'])
               and abs(routing['direct']['cap_route']) < 0.4 * abs(routing['full']['cap_route']))
    print(f'\n(0) enough per bucket: {p0}', flush=True)
    print(f'(b) full model routes to actual outcome: {pb}', flush=True)
    print(f'NULL direct routing << full routing: {null_ok}', flush=True)

    out.update({'pred_0': bool(p0), 'pred_b_blocks_route': bool(pb),
                'null_ok': bool(null_ok), 'runtime_s': time.time() - t0})
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\nwrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
