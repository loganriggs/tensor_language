"""MECHANISM->FUNCTION BRIDGE (tick 178): score the Stage-1 mechanism-ledger
reconstructions on the FUNCTION metric (held-out FineWeb dCE, standard 307k audit).

The mechanism pipeline (qk_stage1_triple) compressed y_t = [k1|k2|v] per head with a
triple SAE gated on the sketched third-moment residual — it was never scored in nats.
Here: rebuild k1_hat, k2_hat from the winner codes (unigram+nonneg, m=512, k=6), patch
the layer-0 KEY tables (queries exact — the mechanism ledger has no query-side codes),
and audit. Arms:
  base        : no patch (reference CE)
  mech9       : all 9 heads' keys replaced by SAE reconstructions
  mech7       : only the 7 moment-gated heads replaced (h0/h4 exact)
Bits reported for context: triple-SAE cost (atoms+bias+codes, all 9 heads) and the raw
query-side cost that an end-to-end ledger would still owe.
"""
import json
import sys
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, reference_forward
from tier2_folding import branch_factors, scores_from_factors

torch.manual_seed(0)
DEV = 'cuda'
QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
M_ATOMS, K_CODE = 512, 6

m, cfg = load_elriggs('bilin18')
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
V = cfg['vocab_size']
FINEWEB = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))

TAB = {}
for br in (1, 2):
    qh, kh = branch_factors(m, br)
    TAB[f'q{br}'], TAB[f'k{br}'] = qh.float().to(DEV), kh.float().to(DEV)

blob = torch.load(f'{QK}/qk_stage1_triple.pt', map_location=DEV)


def unit_rms(t):
    return t * (t.pow(2).mean(-1, keepdim=True).clamp_min(1e-12).rsqrt())


def recon_head(h):
    key = f'h{h}_unigram_nonneg'
    Dn = blob[f'{key}_Dn'].to(DEV)
    b = blob[f'{key}_b'].to(DEV)
    idx = blob[f'{key}_idx'].long().to(DEV)
    coeff = blob[f'{key}_coeff'].to(DEV)
    rec = b + (coeff.unsqueeze(-1) * Dn[idx]).sum(1)           # (V, 384)
    return rec[:, :HD], rec[:, HD:2 * HD]                      # k1_hat, k2_hat


def tables_for(heads):
    out = {n: TAB[n].clone() for n in TAB}
    for h in heads:
        k1h, k2h = recon_head(h)
        out['k1'][:, h] = k1h
        out['k2'][:, h] = k2h
    out['k1'] = unit_rms(out['k1'])
    out['k2'] = unit_rms(out['k2'])
    return out


@torch.no_grad()
def audit_fw(tabs, batch=4):
    tot, n = 0.0, 0
    for i in range(0, len(FINEWEB), batch):
        b = FINEWEB[i:i + batch].to(DEV)
        idx = b[:, :-1]

        def patch(li, s1, s2):
            if li != 0:
                return s1, s2
            n1 = scores_from_factors(tabs['q1'], tabs['k1'], idx, HD)
            n2 = scores_from_factors(tabs['q2'], tabs['k2'], idx, HD)
            return n1.to(s1.dtype), n2.to(s2.dtype)

        logits = reference_forward(m, idx, 'bf16', score_patch=None if tabs is None else patch).float()
        ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), b[:, 1:].reshape(-1))
        tot += ce.item() * b[:, 1:].numel()
        n += b[:, 1:].numel()
    return tot / n


sae_bits = NH * (M_ATOMS * 3 * HD * 32 + 3 * HD * 32 + V * K_CODE * (32 + 9))
query_raw_bits = 2 * NH * V * HD * 32
res = {'sae_bits_Mbit': round(sae_bits / 1e6, 1),
       'query_raw_bits_Mbit': round(query_raw_bits / 1e6, 1)}

base = audit_fw(None)
res['base_ce'] = round(base, 6)
print(f'base CE {base:.6f}', flush=True)
for name, heads in (('mech9', list(range(NH))), ('mech7', [1, 2, 3, 5, 6, 7, 8])):
    ce = audit_fw(tables_for(heads))
    res[name] = {'ce': round(ce, 6), 'dce': round(ce - base, 6)}
    print(f'{name}: CE {ce:.6f} dCE {ce - base:+.6f}', flush=True)
    json.dump(res, open(f'{QK}/qk_mech_bridge.json', 'w'), indent=2)
print('BRIDGE DONE', flush=True)
