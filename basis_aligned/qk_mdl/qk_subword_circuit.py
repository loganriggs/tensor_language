"""SECOND end-to-end circuit: SUBWORD CONTINUATION (complete a multi-token word from its prefix).
Target positions = where the NEXT token is a genuine mid-word BPE piece (no leading-space marker,
alphabetic). General circuit metric (contrast with induction): task_score = CE reduction on target
positions relative to the mean-ablate-ALL floor, i.e. how much the model's computation lowers loss
on this specific task. Backward-elimination finds the minimal sufficient circuit; necessity tests
name the load-bearing components. Prior work flags l1-h1 as the subword-continuation causal giant,
so we expect an EARLY, few-head circuit -- contrasting with induction's MLP1 + distributed core.
"""
import json, sys
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot
torch.manual_seed(0)
DEV = 'cuda'; QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
m, cfg = load_elriggs('bilin18')
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
V = cfg['vocab_size']; NL = len(m.transformer.h)
FINEWEB = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))

# subword-continuation vocab mask: no leading-space marker AND starts with a lowercase letter
tok = AutoTokenizer.from_pretrained('gpt2')
is_cont = torch.zeros(V, dtype=torch.bool)
for i in range(50257):
    s = tok.convert_ids_to_tokens(i)
    if s is not None and not s.startswith('Ġ') and len(s) and s[0].isalpha() and s[0].islower():
        is_cont[i] = True
is_cont = is_cont.to(DEV)
print(f"continuation-type tokens: {int(is_cont.sum())}", flush=True)

NSEQ, T0 = 48, 128
EV = FINEWEB[:NSEQ, :T0].to(DEV)   # natural text (not repeated)
tgt = EV[:, 1:]                    # next-token targets
CONT = is_cont[tgt]               # (NSEQ, T0-1) boolean: is target a subword continuation


@torch.no_grad()
def forward(keep=None, collect_mean=False):
    idx = EV[:, :-1]; B, T = idx.shape
    dt = m.transformer.wte.weight.dtype; x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(T, HD, DEV, dt, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool)); means = {}
    for li in range(NL):
        blk = m.transformer.h[li]; x = blk.lambdas[0]*x + blk.lambdas[1]*x0; a = blk.attn; hcur = F.rms_norm(x, (D,))
        def qk(lin): z = F.rms_norm(lin(hcur).view(B, T, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
        v = a.c_v(hcur).view(B, T, NH, HD)
        if v1 is None: v1 = v
        v = (1-a.lamb)*v + a.lamb*v1.view_as(v)
        q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k)/HD; s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD
        pat = (s1*s2).masked_fill(~mask, 0.0); yh4 = torch.einsum('bhqk,bkhd->bqhd', pat, v)
        if collect_mean: means[('h', li)] = yh4.mean((0, 1))
        if keep is not None:
            for h in range(NH):
                if ('h', li, h) not in keep: yh4[:, :, h, :] = MEAN[('h', li)][h]
        x = x + a.c_proj(yh4.reshape(B, T, -1)); mo = blk.mlp(F.rms_norm(x, (D,)))
        if collect_mean: means[('m', li)] = mo.mean((0, 1))
        if keep is not None and ('m', li) not in keep: mo = MEAN[('m', li)].expand_as(mo)
        x = x + mo
    lg = 30*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30).float()
    ce = F.cross_entropy(lg.reshape(-1, V), tgt.reshape(-1), reduction='none').view(B, T)
    return ce[CONT].mean().item(), means   # mean CE on continuation targets (LOWER = better)

_, MEAN = forward(None, True)
CE_FULL, _ = forward(None)          # full-model CE on continuation targets
CE_FLOOR, _ = forward(set())        # mean-ablate-all floor
SCORE = CE_FLOOR - CE_FULL          # task score: CE reduction the circuit provides
print(f"continuation CE: full {CE_FULL:.4f} | floor {CE_FLOOR:.4f} | task score (reduction) {SCORE:.4f}", flush=True)
ALL = [('h', li, h) for li in range(NL) for h in range(NH)] + [('m', li) for li in range(NL)]
def red(ce): return (CE_FLOOR - ce) / SCORE   # fraction of task score retained

# knockout importance (CE increase when removed from full)
KO = {c: forward(set(ALL) - {c})[0] - CE_FULL for c in ALL}
rank_imp = sorted(ALL, key=lambda c: -KO[c])
print("top-12 knockout (CE rise on continuation when removed):", flush=True)
for c in rank_imp[:12]:
    print(f"  {c}: +{KO[c]:.4f}", flush=True)

# backward elimination: remove least-important-first, keep >=90% of task score
THRESH = CE_FLOOR - 0.90 * SCORE    # kept CE must stay <= this
order = sorted(ALL, key=lambda c: KO[c])   # least important first
live = set(ALL)
for c in order:
    if forward(live - {c})[0] <= THRESH: live = live - {c}
ce_live, _ = forward(live)
minimal = sorted(live, key=lambda c: (c[1], c[0]))
print(f"MINIMAL SUFFICIENT: {len(live)} comps, CE {ce_live:.4f} ({red(ce_live):.1%} of task score)", flush=True)
print("  kept:", [str(c) for c in minimal], flush=True)

# necessity of the top candidates
nec = {}
for c in rank_imp[:6]:
    ce_c, _ = forward(set(ALL) - {c}); nec[str(c)] = round(red(ce_c), 4)
res = {'ce_full': round(CE_FULL, 4), 'ce_floor': round(CE_FLOOR, 4), 'task_score': round(SCORE, 4),
       'minimal_size': len(live), 'minimal_ce': round(ce_live, 4), 'minimal_retention': round(red(ce_live), 4),
       'minimal_components': [str(c) for c in minimal],
       'top_knockout': [(str(c), round(KO[c], 4)) for c in rank_imp[:15]],
       'necessity_single': nec}
json.dump(res, open(f'{QK}/qk_subword_circuit.json', 'w'), indent=2)
print("QK SUBWORD CIRCUIT DONE", flush=True)
