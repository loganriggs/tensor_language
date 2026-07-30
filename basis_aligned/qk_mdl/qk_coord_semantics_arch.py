"""MECHANISM-LEDGER NAMING TEST: the same class battery + meaning gate as qk_coord_semantics_l0,
applied to the 144 ARCHETYPE coordinates (16/head value-space directions from the mechanism arc)
instead of the anonymous PCA coordinates. Hypothesis: nameability lives in mechanism space --
archetype coordinates should be class-codable where PCA coordinates were 3/576.
Original header: COORDINATE SEMANTICS, layer 0, under the meaning gate. Each layer-0 PCA coordinate c of head h
has an EXACT weight-derived vocabulary spectrum S_h[t, c] = <basis_c, c_proj_h v_h(e_t)> (the value
written to that coordinate when attending to token t). Naming hypothesis: 'coordinate c aggregates
class C' coded as S_hat[t, c] = alpha*1[t in C] + beta, with C from an INDEPENDENT hand library of
grammar/orthography classes. Frequency-weighted spectrum R^2 ranks codability; the MEANING GATE is
substitution: run the model with layer-0 attention writes replaced by (a) exact spectra (identity
reference) and (b) coded spectra for codable coordinates + exact for the rest; if (b) ~ (a), the
names are verified causally. SEs on the held-back slice FW[448:600].
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
FW = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))
HELD = FW[448:600]
COOC = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_cooc_tokens.npy').astype(np.int64))
tok = AutoTokenizer.from_pretrained('gpt2')
B0 = m.transformer.h[0]
wte = m.transformer.wte.weight.detach().float().to(DEV)
E = F.rms_norm(wte, (D,))

# layer-0 PCA-64/head basis (standard recipe)
acc = torch.zeros(NH, HD, HD, device=DEV, dtype=torch.float64)
@torch.no_grad()
def l0_heads(idx):
    B, T = idx.shape; x0 = F.rms_norm(m.transformer.wte(idx), (D,))
    x = (B0.lambdas[0]+B0.lambdas[1])*x0; a = B0.attn; hcur = F.rms_norm(x, (D,))
    cos, sin = rope_tables(T, HD, DEV, x0.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    def qk(lin): z = F.rms_norm(lin(hcur).view(B, T, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
    v = a.c_v(hcur).view(B, T, NH, HD)
    q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    pat = ((torch.einsum('bqhd,bkhd->bhqk', q, k)/HD)*(torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD)).masked_fill(~mask, 0.0)
    return torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(-1, NH, HD)
for i in range(0, 64, 8):
    f = l0_heads(COOC[i:i+8].to(DEV)[:, :128]).double()
    acc += torch.einsum('nhd,nhe->hde', f, f)
cw = B0.attn.c_proj.weight.detach().float()
mh_ = torch.load(f'{QK}/qk_minimal_heads.pt', map_location=DEV)
pol0 = torch.load(f'{QK}/qk_h0_polish_g025.pt', map_location=DEV); pol4 = torch.load(f'{QK}/qk_h04_polish.pt', map_location=DEV)
BASIS = []   # per head: (D,16) ORTHONORMAL archetype write basis
NC = 16
for hh in range(NH):
    if hh in (0, 4):
        bb = pol0 if hh == 0 else pol4; Dv = bb[f'h{hh}_v_Dm'].to(DEV); Dv = Dv/Dv.norm(dim=1, keepdim=True).clamp_min(1e-8)
        Vd = Dv.T @ bb[f'h{hh}_CJ'][:, :16].to(DEV)
    else:
        Pp = mh_[f'h{hh}']; Dn_ = Pp['Dm'].to(DEV); Dn_ = Dn_/Dn_.norm(dim=1, keepdim=True).clamp_min(1e-8)
        Vd = Dn_[:, 2*HD:].T @ Pp['U'].to(DEV)[:, :16]
    if Vd.shape[1] < 16: Vd = torch.cat([Vd, torch.zeros(HD, 16-Vd.shape[1], device=DEV)], 1)
    W_ = cw[:, hh*HD:(hh+1)*HD] @ Vd
    Qh, _ = torch.linalg.qr(W_)
    BASIS.append(Qh.contiguous())

# exact weight-derived spectra: S[h][t, c] = <basis_hc, c_proj_h v_h(e_t)>
cv = B0.attn.c_v.weight.detach().float()
S = []
for hh in range(NH):
    vh = E @ cv[hh*HD:(hh+1)*HD, :].T                     # (V, HD) head-h value of each token
    wr = vh @ cw[:, hh*HD:(hh+1)*HD].T                    # (V, D) write vectors
    S.append(wr @ BASIS[hh])                              # (V, NC)
print("spectra built (exact, weight-derived)", flush=True)

# unigram weights (frequency for R^2 weighting only)
cnt = torch.zeros(V, device=DEV); cnt.index_add_(0, COOC[:400, :128].reshape(-1).to(DEV), torch.ones(400*128, device=DEV))
p = (cnt/cnt.sum())


def varimax(Phi, w, iters=60):
    # Kaiser varimax on frequency-weighted spectra: rotate to sparsify columns
    Wt = torch.sqrt(w).unsqueeze(1)
    X = (Phi * Wt)
    k = X.shape[1]; R = torch.eye(k, device=DEV)
    for _ in range(iters):
        L = X @ R
        u, sv, vt = torch.linalg.svd(X.T @ (L**3 - L @ torch.diag((L**2).mean(0))), full_matrices=False)
        R = u @ vt
    return R

print("archetype directions used AS-IS (no varimax: they are the mechanism arc's named objects)", flush=True)


# independent class library
import string as _string
_P = set(_string.punctuation)
GR = {
 'conjunction': {'and','or','but','because','while','as','though','although','whereas','nor','so','yet','since','if','when','unless','whether'},
 'preposition': {'to','in','on','for','with','at','by','from','of','into','over','under','about','through','between','during','against','among'},
 'pronoun': {'you','your','they','we','my','he','i','she','it','us','them','our','his','her','me','him','their','its'},
 'determiner': {'a','an','the','this','that','these','those','some','any','no','each','every'},
 'be_verb': {'is','are','was','were','be','been','being','am'},
 'modal': {'will','would','can','could','may','might','must','shall','should'},
}
def build_classes():
    cls = {}
    for name, words in GR.items():
        v = torch.zeros(V, dtype=torch.bool)
        for i in range(50257):
            s = tok.convert_ids_to_tokens(i)
            if s and s.replace('Ġ','').lower() in words: v[i] = True
        cls[name] = v
    extra = {k: torch.zeros(V, dtype=torch.bool) for k in
             ['punct','digit','capitalized','subword_frag','space_lower','newline','quote','bracket','suffix_ing','suffix_ed','suffix_s','allcaps']}
    for i in range(50257):
        s = tok.convert_ids_to_tokens(i)
        if s is None: continue
        core = s.replace('Ġ',''); lead = s.startswith('Ġ')
        if len(core) and all(c in _P for c in core): extra['punct'][i] = True
        if len(core) and all(c.isdigit() for c in core): extra['digit'][i] = True
        if lead and len(core) and core[0].isupper(): extra['capitalized'][i] = True
        if not lead and len(core) and core[0].isalpha() and core[0].islower(): extra['subword_frag'][i] = True
        if lead and len(core) and core[0].islower(): extra['space_lower'][i] = True
        if 'Ċ' in s: extra['newline'][i] = True
        if core in ('"', "'", '"""', "''"): extra['quote'][i] = True
        if core in ('(',')','[',']','{','}'): extra['bracket'][i] = True
        if core.endswith('ing'): extra['suffix_ing'][i] = True
        if core.endswith('ed'): extra['suffix_ed'][i] = True
        if len(core) > 2 and core.endswith('s'): extra['suffix_s'][i] = True
        if len(core) > 1 and core.isupper(): extra['allcaps'][i] = True
    cls.update(extra)
    # frequency band
    top1k = torch.zeros(V, dtype=torch.bool); top1k[cnt.topk(1000).indices.cpu()] = True
    cls['top1k_frequent'] = top1k
    return {k: v.to(DEV) for k, v in cls.items()}
CLS = build_classes()
print(f"class library: {len(CLS)} classes", flush=True)

# per-coordinate best class fit (frequency-weighted R^2), coded spectra
w = p.clamp_min(1e-9)
results = []; SHAT = [s.clone() for s in S]; codable = 0
for hh in range(NH):
    for c in range(NC):
        y = S[hh][:, c]
        mu = float((w*y).sum()/w.sum()); var = float((w*(y-mu)**2).sum())
        best = (None, 0.0, 0.0, mu)
        for name, mask in CLS.items():
            mk = mask.float()
            wm = float((w*mk).sum());
            if wm < 1e-6: continue
            a_in = float((w*mk*y).sum()/wm)
            w0 = float((w*(1-mk)).sum()); a_out = float((w*(1-mk)*y).sum()/w0)
            pred = mk*a_in + (1-mk)*a_out
            r2 = 1 - float((w*(y-pred)**2).sum())/max(var, 1e-12)
            if r2 > best[1]: best = (name, r2, a_in, a_out)
        results.append({'head': hh, 'coord': c, 'class': best[0], 'r2': round(best[1], 3)})
        if best[1] >= 0.8:
            codable += 1
            mk = CLS[best[0]].float()
            SHAT[hh][:, c] = mk*best[2] + (1-mk)*best[3]
frac = codable/ (NH*NC)
r2s = [r['r2'] for r in results]
print(f"codable at R2>=0.8: {codable}/144 ({frac:.0%}); median spectrum R2 {np.median(r2s):.3f}", flush=True)
from collections import Counter
top = Counter(r['class'] for r in results if r['r2'] >= 0.8).most_common(8)
print("codable classes:", top, flush=True)


@torch.no_grad()
def forward_spec(idx, spectra):
    """layer-0 attention writes replaced by pattern-weighted SPECTRA (exact or coded); rest real."""
    B, T2 = idx.shape
    x0 = F.rms_norm(m.transformer.wte(idx), (D,)); x = None; v1 = None
    cos, sin = rope_tables(T2, HD, DEV, x0.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T2, T2, device=DEV, dtype=torch.bool))
    for li in range(NL):
        blk = m.transformer.h[li]; a = blk.attn
        x = (blk.lambdas[0]+blk.lambdas[1])*x0 if li == 0 else blk.lambdas[0]*x + blk.lambdas[1]*x0
        hcur = F.rms_norm(x, (D,))
        def qk(lin): z = F.rms_norm(lin(hcur).view(B, T2, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
        v = a.c_v(hcur).view(B, T2, NH, HD)
        if v1 is None: v1 = v
        v = (1-a.lamb)*v + a.lamb*v1.view_as(v)
        q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
        pat = ((torch.einsum('bqhd,bkhd->bhqk', q, k)/HD)*(torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD)).masked_fill(~mask, 0.0)
        if li == 0 and spectra is not None:
            aout = torch.zeros(B, T2, D, device=DEV)
            for hh in range(NH):
                sp = spectra[hh][idx.reshape(-1)].view(B, T2, NC)      # (B,T,64) per-key coord writes
                Z = torch.einsum('bqk,bkc->bqc', pat[:, hh], sp)       # pattern-weighted coord sums
                aout = aout + Z @ BASIS[hh].T
        else:
            aout = a.c_proj(torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, T2, -1))
        x = x + aout; x = x + blk.mlp(F.rms_norm(x, (D,)))
    return 30*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30).float()

@torch.no_grad()
def percetok(spectra):
    ces = []
    for i in range(0, len(HELD), 4):
        b = HELD[i:i+4].to(DEV)
        lg = forward_spec(b[:, :-1], spectra)
        ce = F.cross_entropy(lg.reshape(-1, V), b[:, 1:].reshape(-1), reduction='none')
        ces.append(ce.cpu())
    return torch.cat(ces)

ce_real = percetok(None)
ce_exact = percetok(S)
ce_coded = percetok(SHAT)
def rep(name, ce):
    dif = ce - ce_real
    return {'dCE': round(float(dif.mean()), 5), 'SE': round(float(dif.std()/np.sqrt(dif.numel())), 6)}
res = {'codable_frac': round(frac, 3), 'codable_n': codable, 'median_r2': round(float(np.median(r2s)), 3),
       'codable_classes': top,
       'exact_spectra': rep('exact', ce_exact), 'coded_spectra': rep('coded', ce_coded)}
gate = res['coded_spectra']['dCE'] - res['exact_spectra']['dCE']
dif2 = (ce_coded - ce_exact)
res['gate_coded_minus_exact'] = {'dCE': round(float(dif2.mean()), 5), 'SE': round(float(dif2.std()/np.sqrt(dif2.numel())), 6)}
print(f"exact spectra (identity ref): dCE +{res['exact_spectra']['dCE']:.5f} (SE {res['exact_spectra']['SE']})", flush=True)
print(f"coded spectra ({codable} coords named): dCE +{res['coded_spectra']['dCE']:.5f}", flush=True)
print(f"MEANING GATE (coded - exact): +{res['gate_coded_minus_exact']['dCE']:.5f} (SE {res['gate_coded_minus_exact']['SE']})", flush=True)
json.dump(res, open(f'{QK}/qk_coord_semantics_arch.json', 'w'), indent=2)
json.dump(results, open(f'{QK}/qk_coord_semantics_arch_table.json', 'w'), indent=2)
print("QK COORD SEMANTICS ARCH DONE", flush=True)
