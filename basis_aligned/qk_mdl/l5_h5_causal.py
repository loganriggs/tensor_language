"""Causal check of WW-1: if L5.H5 is bilin18's induction head, disabling it
should specifically hurt SECOND-half CE on repeated sequences (seq = A + A,
random-token A of length 256) and barely touch natural text.
Arms per config: natural-audit dCE (16 pile chunks) and repeat-second-half dCE.
Configs: H5 zeroed; H5 cond-mean tabled (from all17 tables); H7 same, as the
non-induction contextual contrast."""
import json
import torch
import torch.nn.functional as F
import sys
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, build_eval_tokens, reference_forward
from tier2_folding import scores_from_factors

torch.manual_seed(0)
DEV = 'cuda'
QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
L = 5
OUT = f'{QK}/l5_h5_causal.json'
m, cfg = load_elriggs('bilin18')
NH, HD = cfg['n_head'], cfg['n_embd'] // cfg['n_head']
V = cfg['vocab_size']
AUDIT = build_eval_tokens(n_chunks=20, seq_len=513)[4:20]
g = torch.Generator(); g.manual_seed(7)
REP = torch.randint(0, V, (16, 256), generator=g).repeat(1, 2)   # (16, 512)

raw = torch.load(f'{QK}/all17_tables.pt')
tabs = {n: raw[f'{L}_{n}'].float().to(DEV) for n in ('q1', 'k1', 'q2', 'k2')}
del raw


def make_patch(head, mode):
    def patch(li, s1, s2):
        if li != L:
            return s1, s2
        if mode == 'zero':
            z1, z2 = s1.clone(), s2.clone()
            z1[:, head] = 0.0
            z2[:, head] = 0.0
            return z1, z2
        n1 = scores_from_factors(tabs['q1'], tabs['k1'], patch.idx, HD).to(s1.dtype)
        n2 = scores_from_factors(tabs['q2'], tabs['k2'], patch.idx, HD).to(s2.dtype)
        z1, z2 = s1.clone(), s2.clone()
        z1[:, head] = n1[:, head]
        z2[:, head] = n2[:, head]
        return z1, z2
    return patch


@torch.no_grad()
def ce_eval(tokens, patch=None, second_half_only=False):
    tot, n = 0.0, 0
    for i in range(0, len(tokens), 4):
        b = tokens[i:i + 4].to(DEV)
        idx = b[:, :-1]
        if patch is not None:
            patch.idx = idx
        logits = reference_forward(m, idx, 'bf16', score_patch=patch).float()
        tgt = b[:, 1:]
        if second_half_only:
            logits = logits[:, 256:]
            tgt = tgt[:, 256:]
        ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1))
        tot += ce.item() * tgt.numel()
        n += tgt.numel()
    return tot / n


res = {}
base_nat = ce_eval(AUDIT)
base_rep = ce_eval(REP, second_half_only=True)
res['baseline'] = {'natural': base_nat, 'repeat_2nd_half': base_rep}
print(f'baseline: natural {base_nat:.4f}, repeat-2nd-half {base_rep:.4f}', flush=True)
for head in (5, 7):
    for mode in ('zero', 'table'):
        p = make_patch(head, mode)
        dn = ce_eval(AUDIT, p) - base_nat
        dr = ce_eval(REP, p, second_half_only=True) - base_rep
        res[f'H{head} {mode}'] = {'d_natural': dn, 'd_repeat_2nd_half': dr}
        print(f'H{head} {mode}: natural {dn:+.4f} · repeat-2nd-half {dr:+.4f}', flush=True)
        with open(OUT, 'w') as fh:
            json.dump(res, fh, indent=2)
print('l5 h5 causal done', flush=True)
