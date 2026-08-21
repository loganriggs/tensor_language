"""ROUTING ATTENTION LOCALIZATION -- where is the sentence-boundary
routing (643) computed? 643 showed the 18 blocks route P among newline/
capitalized after a '.', while the embedding bigram is blind. 634/635
found the context that sets class identity enters at the FRONT attention.
This tests whether the newline-vs-capitalized ROUTING is carried by
front attention specifically: mean-ablate attention at the front [0-2]
vs the late-middle [10-12] and see which collapses the routing.

Routing metrics (from 643): nl_route = P(nl | actual->newline) -
P(nl | actual->capitalized); cap_route = P(cap | actual->capitalized) -
P(cap | actual->newline). Baseline routing is +0.248 / +0.212; the
context-blind bigram's is ~0.

REGISTERED PREDICTIONS:
  (0) SANITY: baseline routing is clearly positive (reproduces 643);
  (a) FRONT ATTENTION CARRIES ROUTING: ablating front [0-2] attention
      substantially reduces nl_route and cap_route toward the blind-
      bigram level;
  (b) LATE ATTENTION DOES NOT: ablating late [10-12] attention reduces
      the routing far less than front attention does;
  (c) report routing under baseline / front-attn / late-attn / front-mlp
      ablation;
  NULL: late-attention routing reduction is a small fraction of the
      front-attention reduction -- routing context enters at the front."""
import json, time, torch
import torch.nn.functional as F
import numpy as np
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152
T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'routing_attention_localization_results.json'
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


def meanfill_hook(mo, i_, o_):
    return o_.mean(dim=(0, 1), keepdim=True).expand_as(o_)


@torch.no_grad()
def measure(fresh, blocks, kind, nl_mask, cap_mask):
    handles = []
    if kind is not None:
        for li in blocks:
            sub = (m.transformer.h[li].attn.c_proj if kind == 'attn'
                   else m.transformer.h[li].mlp)
            handles.append(sub.register_forward_hook(meanfill_hook))
    pnl = torch.zeros(NFRESH, T); pcap = torch.zeros(NFRESH, T)
    for i in range(0, NFRESH, 4):
        bb = fresh[i:i + 4, :257].to(DEV)
        idx = bb[:, :-1].contiguous(); B = bb.shape[0]
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for blk in m.transformer.h:
            x, v1 = blk(x, v1, x0)
        lg = (30 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30)).float()
        p = F.softmax(lg, dim=-1).reshape(-1, lg.shape[-1])
        pnl[i:i + B] = (p @ nl_mask).cpu().view(B, T)
        pcap[i:i + B] = (p @ cap_mask).cpu().view(B, T)
    for h in handles:
        h.remove()
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
    to_nl = endp & (ncls == 'newline')
    to_cap = endp & (ncls == 'capitalized')

    conds = {'baseline': (None, None), 'front_attn': ([0, 1, 2], 'attn'),
             'late_attn': ([10, 11, 12], 'attn'), 'front_mlp': ([0, 1, 2], 'mlp')}
    out = {'routing': {}}
    for name, (b, k) in conds.items():
        pnl, pcap = measure(fresh, b, k, nl_mask, cap_mask)
        nl_route = float(pnl[to_nl].mean() - pnl[to_cap].mean())
        cap_route = float(pcap[to_cap].mean() - pcap[to_nl].mean())
        out['routing'][name] = {'nl_route': round(nl_route, 4),
                                'cap_route': round(cap_route, 4)}
        print(f'{name:12s} nl_route {nl_route:+.4f}  cap_route {cap_route:+.4f}',
              flush=True)

    r = out['routing']
    base = r['baseline']
    def total(x):
        return x['nl_route'] + x['cap_route']
    front_red = total(base) - total(r['front_attn'])
    late_red = total(base) - total(r['late_attn'])
    p0 = base['nl_route'] > 0.1 and base['cap_route'] > 0.1
    pa = front_red > 0.4 * total(base)
    pb = late_red < 0.5 * front_red
    null_ok = pb
    print(f'\n(0) baseline routing positive: {p0}', flush=True)
    print(f'(a) front-attn ablation reduces routing (>40%): {pa} '
          f'(reduction {front_red:.4f} of {total(base):.4f})', flush=True)
    print(f'(b) late-attn reduction << front-attn: {pb} '
          f'(late {late_red:.4f} vs front {front_red:.4f})', flush=True)

    out.update({'front_reduction': round(front_red, 4),
                'late_reduction': round(late_red, 4),
                'baseline_total_routing': round(total(base), 4),
                'pred_0': bool(p0), 'pred_a_front_carries': bool(pa),
                'pred_b_late_not': bool(pb), 'null_ok': bool(null_ok),
                'runtime_s': time.time() - t0})
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\nwrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
