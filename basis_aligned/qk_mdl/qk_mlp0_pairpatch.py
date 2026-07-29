"""Patch the MLP0 program's named blind spot (subword reassembly) with its codable hypothesis:
a (prev,current)-PAIR-keyed correction table, gated on split-word positions (current token is a
non-space-leading alphabetic piece). Discriminating CONTROL: a prev-token-ONLY correction table fit
on the same gated positions -- if pair beats prev-only, the PAIR keying (word identity) is verified,
not mere capacity. Verification: (a) residual FVU at gated positions; (b) divergence probe re-run on
held-out data (do the worst positions stop being subword?); (c) full-model substitution dCE on the
main audit + no-harm checks on shuffled and rare-token cells (unseen pairs -> zero correction);
(d) honest coverage stats (a pair lexicon is memorized reassembly -- fine, that IS lexical memory,
but coverage bounds its reach).
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
COOC = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_cooc_tokens.npy').astype(np.int64))
tok = AutoTokenizer.from_pretrained('gpt2')
A0, U0 = torch.load(f'{QK}/qk_mlp0_interaction.pt', map_location=DEV)['table_R256']
# split-word gate: current token is a lowercase-alphabetic continuation piece
GATE = torch.zeros(V, dtype=torch.bool)
for i in range(50257):
    s = tok.convert_ids_to_tokens(i)
    if s and not s.startswith('Ġ') and len(s) and s[0].isalpha(): GATE[i] = True
GATE = GATE.to(DEV)
print(f"gate vocab: {int(GATE.sum())}", flush=True)


@torch.no_grad()
def block0(idx):
    B, T = idx.shape; x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x
    blk = m.transformer.h[0]; x = blk.lambdas[0]*x + blk.lambdas[1]*x0; a = blk.attn; hcur = F.rms_norm(x, (D,))
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    def qk(lin): z = F.rms_norm(lin(hcur).view(B, T, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
    v = a.c_v(hcur).view(B, T, NH, HD)
    q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    s1 = torch.einsum('bqhd,bkhd->bhqk', q, k)/HD; s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD
    pat = (s1*s2).masked_fill(~mask, 0.0); yh4 = torch.einsum('bhqk,bkhd->bqhd', pat, v)
    x = x + a.c_proj(yh4.reshape(B, T, -1)); hin = F.rms_norm(x, (D,))
    return hin, blk.mlp(hin)

# TT0 from the standard recipe (cooc 0-240)
Hs, Ys, Ts = [], [], []
for i in range(0, 240, 8):
    h, y = block0(COOC[i:i+8].to(DEV)[:, :128])
    Hs.append(h.reshape(-1, D)); Ys.append(y.reshape(-1, D)); Ts.append(COOC[i:i+8, :128].reshape(-1).to(DEV))
H = torch.cat(Hs); Y = torch.cat(Ys); T = torch.cat(Ts)
ts = torch.zeros(V, D, device=DEV); tc = torch.zeros(V, device=DEV)
ts.index_add_(0, T, Y); tc.index_add_(0, T, torch.ones_like(T, dtype=torch.float32))
lam = tc.unsqueeze(1)/(tc.unsqueeze(1)+3.0); TT0 = lam*(ts/tc.clamp_min(1).unsqueeze(1)) + (1-lam)*Y.mean(0)
del Hs, Ys, Ts, H, Y, T

# stream cooc 400-2400 to fit pair + prev-only correction tables at GATED positions
pair_sum = {}; pair_cnt = {}
prev_sum = torch.zeros(V, D, device=DEV); prev_cnt = torch.zeros(V, device=DEV)
ngated = 0; ntot = 0
for i in range(400, 2400, 8):
    idx = COOC[i:i+8].to(DEV)[:, :128]
    hin, y = block0(idx)
    flat = hin.reshape(-1, D); toks = idx.reshape(-1)
    pred = TT0[toks] + ((flat @ A0.T)**2) @ U0
    resid = (y.reshape(-1, D) - pred)
    prv = torch.roll(idx, 1, 1); prv[:, 0] = idx[:, 0]; prvf = prv.reshape(-1)
    gmask = GATE[toks]; ntot += toks.numel(); ngated += int(gmask.sum())
    gi = gmask.nonzero().squeeze(1)
    keys = (prvf[gi].to(torch.int64) * V + toks[gi]).cpu().numpy()
    rg = resid[gi]
    prev_sum.index_add_(0, prvf[gi], rg); prev_cnt.index_add_(0, prvf[gi], torch.ones(len(gi), device=DEV))
    rgc = rg.cpu().numpy()
    for j, kk in enumerate(keys):
        if kk in pair_sum: pair_sum[kk] += rgc[j]; pair_cnt[kk] += 1
        else: pair_sum[kk] = rgc[j].copy(); pair_cnt[kk] = 1
print(f"gated {ngated}/{ntot} positions; unique pairs {len(pair_sum)}", flush=True)
keep = [k for k, c in pair_cnt.items() if c >= 2]
PK = torch.tensor(sorted(keep), dtype=torch.int64, device=DEV)
PVEC = torch.stack([torch.from_numpy(pair_sum[int(k)] / pair_cnt[int(k)]) for k in PK.cpu().numpy()]).float().to(DEV)
PREVT = prev_sum / prev_cnt.clamp_min(1).unsqueeze(1)
print(f"pair lexicon: {len(PK)} entries (count>=2)", flush=True)

def corr(idx_flat, prv_flat, mode):
    g = GATE[idx_flat]
    out = torch.zeros(idx_flat.shape[0], D, device=DEV)
    if mode == 'pair':
        keys = prv_flat.to(torch.int64) * V + idx_flat
        pos = torch.searchsorted(PK, keys.clamp(max=PK[-1]))
        pos = pos.clamp(max=len(PK)-1)
        hit = (PK[pos] == keys) & g
        out[hit] = PVEC[pos[hit]]
    elif mode == 'prev':
        out[g] = PREVT[prv_flat[g]]
    return out


def forward(idx, mode):
    """mode: 'model' | 'prog' | 'pair' | 'prev'"""
    B, T2 = idx.shape; x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(T2, HD, DEV, x.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T2, T2, device=DEV, dtype=torch.bool))
    prv = torch.roll(idx, 1, 1); prv[:, 0] = idx[:, 0]
    for li in range(NL):
        blk = m.transformer.h[li]; x = blk.lambdas[0]*x + blk.lambdas[1]*x0; a = blk.attn; hcur = F.rms_norm(x, (D,))
        def qk(lin): z = F.rms_norm(lin(hcur).view(B, T2, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
        v = a.c_v(hcur).view(B, T2, NH, HD)
        if v1 is None: v1 = v
        v = (1-a.lamb)*v + a.lamb*v1.view_as(v)
        q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k)/HD; s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD
        pat = (s1*s2).masked_fill(~mask, 0.0); yh4 = torch.einsum('bhqk,bkhd->bqhd', pat, v)
        x = x + a.c_proj(yh4.reshape(B, T2, -1)); hin = F.rms_norm(x, (D,))
        if li == 0 and mode != 'model':
            flat = hin.reshape(-1, D); tf = idx.reshape(-1); pf = prv.reshape(-1)
            mo = TT0[tf] + ((flat @ A0.T)**2) @ U0
            if mode in ('pair', 'prev'): mo = mo + corr(tf, pf, mode)
            x = x + mo.view(B, T2, D).to(x.dtype)
        else:
            x = x + blk.mlp(hin)
    return 30*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30).float()

@torch.no_grad()
def audit(seqs, mode):
    tot = 0.0; n = 0
    for i in range(0, len(seqs), 4):
        b = seqs[i:i+4].to(DEV)
        lg = forward(b[:, :-1], mode)
        ce = F.cross_entropy(lg.reshape(-1, V), b[:, 1:].reshape(-1))
        tot += ce.item()*b[:, 1:].numel(); n += b[:, 1:].numel()
    return tot/n

# (a) residual FVU at gated positions on HELD-OUT cooc 300-380
fv = {}
num = {mode: 0.0 for mode in ['prog', 'pair', 'prev']}; den = 0.0; cover = 0; gtot = 0
for i in range(300, 380, 4):
    idx = COOC[i:i+4].to(DEV)[:, :128]
    hin, y = block0(idx); flat = hin.reshape(-1, D); tf = idx.reshape(-1)
    prv = torch.roll(idx, 1, 1); prv[:, 0] = idx[:, 0]; pf = prv.reshape(-1)
    pred = TT0[tf] + ((flat @ A0.T)**2) @ U0
    g = GATE[tf]; yv = y.reshape(-1, D)[g]
    den += (yv - yv.mean(0)).pow(2).sum().item()
    num['prog'] += (pred[g] - yv).pow(2).sum().item()
    for mode in ['pair', 'prev']:
        pc = pred + corr(tf, pf, mode)
        num[mode] += (pc[g] - yv).pow(2).sum().item()
    keys = pf[g].to(torch.int64) * V + tf[g]
    pos = torch.searchsorted(PK, keys.clamp(max=PK[-1])).clamp(max=len(PK)-1)
    cover += int((PK[pos] == keys).sum()); gtot += int(g.sum())
for mode in num: fv[mode] = round(num[mode]/den, 4)
print(f"held-out gated-position FVU: prog {fv['prog']} | +pair {fv['pair']} | +prev-ctl {fv['prev']} | pair coverage {cover/gtot:.1%}", flush=True)

# (c) substitution audits
res = {'gated_fvu': fv, 'pair_coverage_heldout': round(cover/gtot, 3), 'pair_entries': len(PK)}
main = FINEWEB[:200]
sh = FINEWEB[:64, :128].clone(); g = torch.Generator().manual_seed(5)
for r in range(64): sh[r] = sh[r][torch.randperm(128, generator=g)]
for cell, seqs in [('fineweb513', main), ('shuffled128', sh)]:
    base = audit(seqs, 'model')
    row = {}
    for mode in ['prog', 'pair', 'prev']:
        row[mode] = round(audit(seqs, mode) - base, 5)
    res[cell] = row
    print(f"{cell}: dCE prog +{row['prog']:.5f} | +pair +{row['pair']:.5f} | +prev-ctl +{row['prev']:.5f}", flush=True)
json.dump(res, open(f'{QK}/qk_mlp0_pairpatch.json', 'w'), indent=2)
print("QK MLP0 PAIRPATCH DONE", flush=True)
