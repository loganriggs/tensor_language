"""ADVERSARIAL testing of the MLP0 program (Logan: 'really try to adversarially test it... robustly
define the first attn and MLP layer so future layers can use them as input variables').
Stress cells for the 97.9% claim (fit on FineWeb cooc, natural distribution):
  (1) OOD corpora: Pile (different corpus), SHUFFLED text (no n-gram structure), rare-token-dense
      sequences (bottom-frequency vocab) -- substitution dCE vs each cell's own base.
  (2) MAX-DIVERGENCE probe: hunt the positions where program and MLP0 disagree most (relative L2);
      characterize the worst 0.5% (which tokens, which contexts) -- where does the program NOT
      capture MLP0?
  (3) COMPOSITION: program output fed to ALL downstream layers at long context (T=512, fit was
      T<=128 activations) -- already implicit in the 513-token audit; re-verify plus Pile-513.
"""
import json, sys
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot, build_eval_tokens
torch.manual_seed(0)
DEV = 'cuda'; QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
m, cfg = load_elriggs('bilin18')
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
V = cfg['vocab_size']; NL = len(m.transformer.h)
FINEWEB = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))
COOC = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_cooc_tokens.npy').astype(np.int64))
tok = AutoTokenizer.from_pretrained('gpt2')
A0, U0 = torch.load(f'{QK}/qk_mlp0_interaction.pt', map_location=DEV)['table_R256']


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

# rebuild token table (train recipe)
Hs, Ys, Ts = [], [], []
for i in range(0, 240, 8):
    h, y = block0(COOC[i:i+8].to(DEV)[:, :128])
    Hs.append(h.reshape(-1, D)); Ys.append(y.reshape(-1, D)); Ts.append(COOC[i:i+8, :128].reshape(-1).to(DEV))
H = torch.cat(Hs); Y = torch.cat(Ys); T = torch.cat(Ts)
ts = torch.zeros(V, D, device=DEV); tc = torch.zeros(V, device=DEV)
ts.index_add_(0, T, Y); tc.index_add_(0, T, torch.ones_like(T, dtype=torch.float32))
lam = tc.unsqueeze(1)/(tc.unsqueeze(1)+3.0); TT0 = lam*(ts/tc.clamp_min(1).unsqueeze(1)) + (1-lam)*Y.mean(0)


def forward(idx, use_prog):
    B, T2 = idx.shape; x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(T2, HD, DEV, x.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T2, T2, device=DEV, dtype=torch.bool))
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
        if li == 0 and use_prog:
            flat = hin.reshape(-1, D)
            x = x + (TT0[idx.reshape(-1)] + ((flat @ A0.T)**2) @ U0).view(B, T2, D).to(x.dtype)
        else:
            x = x + blk.mlp(hin)
    return 30*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30).float()

@torch.no_grad()
def cell_dce(seqs):
    tb = tp = 0.0; n = 0
    for i in range(0, len(seqs), 4):
        b = seqs[i:i+4].to(DEV)
        for use, acc in [(False, 'b'), (True, 'p')]:
            lg = forward(b[:, :-1], use)
            ce = F.cross_entropy(lg.reshape(-1, V), b[:, 1:].reshape(-1)).item()
            if use: tp += ce*b[:, 1:].numel()
            else: tb += ce*b[:, 1:].numel()
        n += b[:, 1:].numel()
    return tb/n, tp/n - tb/n

# (1) stress cells
cells = {}
cells['fineweb_128 (in-dist)'] = FINEWEB[:64, :128]
cells['fineweb_513 (long)'] = FINEWEB[:48, :513]
print("building pile eval...", flush=True)
PILE = build_eval_tokens(n_chunks=48, seq_len=129)
cells['pile_128 (ood corpus)'] = PILE
sh = FINEWEB[:64, :128].clone(); g = torch.Generator().manual_seed(5)
for r in range(64): sh[r] = sh[r][torch.randperm(128, generator=g)]
cells['shuffled_128 (no ngrams)'] = sh
# rare-token-dense: sample from bottom-frequency half of vocab seen in fineweb
freq = torch.zeros(V); freq.index_add_(0, FINEWEB.reshape(-1), torch.ones(FINEWEB.numel()))
seen = (freq > 0).nonzero().squeeze(1)
rare = seen[freq[seen].argsort()[:len(seen)//2]]
gg = torch.Generator().manual_seed(6)
cells['raretoken_128 (adversarial vocab)'] = rare[torch.randint(0, len(rare), (48, 128), generator=gg)]

res = {'cells': {}}
for name, seqs in cells.items():
    base, d = cell_dce(seqs)
    res['cells'][name] = {'base_CE': round(base, 4), 'dCE': round(d, 5)}
    print(f"{name}: base {base:.3f} | substitution dCE +{d:.5f}", flush=True)

# (2) max-divergence probe on natural data
divs, toks, prevs = [], [], []
for i in range(300, 380, 4):
    idx = COOC[i:i+4].to(DEV)[:, :128]
    hin, y = block0(idx)
    flat = hin.reshape(-1, D)
    pred = TT0[idx.reshape(-1)] + ((flat @ A0.T)**2) @ U0
    yf = y.reshape(-1, D)
    d = (pred - yf).norm(dim=1) / yf.norm(dim=1).clamp_min(1e-6)
    divs.append(d); toks.append(idx.reshape(-1))
    pr = torch.roll(idx, 1, 1); pr[:, 0] = idx[:, 0]; prevs.append(pr.reshape(-1))
DIV = torch.cat(divs); TOKS = torch.cat(toks); PREVS = torch.cat(prevs)
q995 = DIV.quantile(0.995)
worst = (DIV > q995).nonzero().squeeze(1)
wt = [tok.convert_ids_to_tokens(int(t)) for t in TOKS[worst][:30]]
wp = [tok.convert_ids_to_tokens(int(t)) for t in PREVS[worst][:30]]
res['divergence'] = {'median_rel': round(float(DIV.median()), 4), 'p99.5_rel': round(float(q995), 4),
                     'worst_tokens': wt, 'worst_prev_tokens': wp}
print(f"divergence: median {float(DIV.median()):.3f} p99.5 {float(q995):.3f}", flush=True)
print("worst-position tokens:", wt, flush=True)
print("worst-position PREV tokens:", wp, flush=True)
json.dump(res, open(f'{QK}/qk_mlp0_adversarial.json', 'w'), indent=2)
print("QK MLP0 ADVERSARIAL DONE", flush=True)
