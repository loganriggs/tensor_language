"""§34 CONTENT-NAMEABILITY PROTOCOL, LAYER 3 (verbatim copy of qk_l1_content_gate.py,
layer index 1->3). Each layer-3 head-value coordinate c of head h has a mean-residual-derived
vocabulary spectrum S_h[t, c] = <basis_hc, c_proj3_h vs_h(t)>, where vs_h(t) is the layer-3
effective head value of token t built from the MEAN-RESIDUAL TABLE (the per-token mean of the
layer-3 input residual, shrunk tau=8 toward the embedding prior; the residual is the block-3
lambda-mixed input, estimated exactly as qk_l4_port.py's block4_input generalization) mixed with
the layer-0 value (v1 mixing). BASIS is the per-head PCA-64 orthonormal write basis of the actual
layer-3 head outputs (varimax-sparsified). Naming hypotheses, both from §34: (a) BROAD-CLASS code
S_hat[t,c] = alpha*1[t in C] + beta with C an INDEPENDENT hand library of grammar/orthography
classes; (b) 8-SPIKE code (top-8 tokens exact, rest = weighted off-spike mean) for concentrated
coordinates. Frequency-weighted spectrum R^2 / concentration rank codability. The MEANING GATE is
substitution on LAYER 3: run the model with layer-3 attention writes replaced by (a) the exact
mean-residual spectra (identity reference) and (b) coded spectra; if coded ~ exact the names are
causally verified. L3H8 is the model's strongest head; answers whether its content stays spectral.
SEs on the held-back slice FW[448:600].

DERIVATION: copied VERBATIM from qk_l1_content_gate.py with these layer-index-only edits:
  - l1_heads -> l3_heads: forward loop range(2)->range(4), captured/returned at li==3 (was li==1);
    PCA-64 basis on layer-3 outputs; write projection cw = h[3].attn.c_proj (was h[1]);
  - block1_input -> block3_input: runs blocks 0,1,2 (with v1-mixing) then returns block-3's
    lambda-mixed input, using qk_l4_port.py's validated block4_input structure (loop (0,1,2),
    return h[3] mix). block1_input's single-block form only worked because block-0 v-mixing is a
    no-op; deeper layers require the v1 term, so the multi-layer form is used;
  - value spectra: a1 -> a3 = h[3].attn; vs = (1-a3.lamb)*c_v3(table) + a3.lamb*c_v0(embedding);
    a0 = h[0].attn UNCHANGED (v1 is always the layer-0 value);
  - the substitution branch fires at li == 3 (was li == 1);
  - PCA / mean-residual batches reduced 8->4 for memory safety; NC = 64, denominators NH*NC = 576.
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
N_EST, TAU = 1024, 8.0
m, cfg = load_elriggs('bilin18')
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
V = cfg['vocab_size']; NL = len(m.transformer.h)
FW = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))
HELD = FW[448:600]
COOC = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_cooc_tokens.npy').astype(np.int64))
tok = AutoTokenizer.from_pretrained('gpt2')
NC = 64
wte = m.transformer.wte.weight.detach().float().to(DEV)


# ---- layer-3 head outputs (full forward captured at layer 3), verbatim forward loop ----
@torch.no_grad()
def l3_heads(idx):
    B, T = idx.shape
    x0 = F.rms_norm(m.transformer.wte(idx), (D,)); x = None; v1 = None
    cos, sin = rope_tables(T, HD, DEV, x0.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    for li in range(4):
        blk = m.transformer.h[li]; a = blk.attn
        x = (blk.lambdas[0]+blk.lambdas[1])*x0 if li == 0 else blk.lambdas[0]*x + blk.lambdas[1]*x0
        hcur = F.rms_norm(x, (D,))
        def qk(lin): z = F.rms_norm(lin(hcur).view(B, T, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
        v = a.c_v(hcur).view(B, T, NH, HD)
        if v1 is None: v1 = v
        v = (1-a.lamb)*v + a.lamb*v1.view_as(v)
        q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
        pat = ((torch.einsum('bqhd,bkhd->bhqk', q, k)/HD)*(torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD)).masked_fill(~mask, 0.0)
        yh = torch.einsum('bhqk,bkhd->bqhd', pat, v)
        if li == 3:
            return yh.reshape(-1, NH, HD)
        x = x + a.c_proj(yh.reshape(B, T, -1)); x = x + blk.mlp(F.rms_norm(x, (D,)))


# per-head PCA-64 basis on layer-3 head outputs (standard recipe, three layers up)
acc = torch.zeros(NH, HD, HD, device=DEV, dtype=torch.float64)
for i in range(0, 64, 4):
    f = l3_heads(COOC[i:i+4].to(DEV)[:, :128]).double()
    acc += torch.einsum('nhd,nhe->hde', f, f)
cw = m.transformer.h[3].attn.c_proj.weight.detach().float()
BASIS = []   # per head: (D,NC) ORTHONORMAL layer-3 write basis (QR of top-NC PCA dirs)
for hh in range(NH):
    ev, evec = torch.linalg.eigh(acc[hh])
    W_ = cw[:, hh*HD:(hh+1)*HD] @ evec[:, ev.argsort(descending=True)[:NC]].float()
    Qh, _ = torch.linalg.qr(W_)
    BASIS.append(Qh.contiguous())
print("layer-3 PCA-64 basis built", flush=True)


# ---- mean-residual table (block3_input + shrink); block3_input = qk_l4_port block4_input at L=3 ----
@torch.no_grad()
def block3_input(idx):
    dt = m.transformer.wte.weight.dtype
    x = m.transformer.wte(idx)
    x = F.rms_norm(x, (x.size(-1),))
    x0 = x
    v1 = None
    B, T = idx.shape
    cos, sin = rope_tables(T, HD, idx.device, dt, 'bf16')
    cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=idx.device, dtype=torch.bool))
    for li in (0, 1, 2):
        blk = m.transformer.h[li]
        x = blk.lambdas[0] * x + blk.lambdas[1] * x0
        a = blk.attn
        hcur = F.rms_norm(x, (x.size(-1),))

        def qk(lin):
            z = F.rms_norm(lin(hcur).view(B, T, NH, HD), (HD,))
            return apply_rot(z, cosb, sinb)

        v = a.c_v(hcur).view(B, T, NH, HD)
        if v1 is None:
            v1 = v
        v = (1 - a.lamb) * v + a.lamb * v1.view_as(v)
        q, k = qk(a.c_q), qk(a.c_k)
        q2, k2 = qk(a.c_q2), qk(a.c_k2)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k) / HD
        s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2) / HD
        pat = (s1 * s2).masked_fill(~mask, 0.0)
        yh = torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, T, -1)
        x = x + a.c_proj(yh)
        x = x + blk.mlp(F.rms_norm(x, (x.size(-1),)))
    blk3 = m.transformer.h[3]
    return blk3.lambdas[0] * x + blk3.lambdas[1] * x0


print('estimating mean-residual table (shrink tau=8)...', flush=True)
sum_x = torch.zeros(V, D, device=DEV)
cnt = torch.zeros(V, device=DEV)
with torch.no_grad():
    for i in range(0, N_EST, 4):
        b = COOC[i:i + 4].to(DEV)
        idx = b[:, :-1]
        x = block3_input(idx).float().reshape(-1, D)
        ids = idx.reshape(-1)
        sum_x.index_add_(0, ids, x)
        cnt.index_add_(0, ids, torch.ones_like(ids, dtype=torch.float))
mean_x = torch.where((cnt > 0)[:, None], sum_x / cnt[:, None].clamp_min(1), wte)
shrunk = (cnt / (cnt + TAU))[:, None] * mean_x + (TAU / (cnt + TAU))[:, None] * wte
del sum_x, mean_x
torch.cuda.empty_cache()
a0 = m.transformer.h[0].attn
a3 = m.transformer.h[3].attn
with torch.no_grad():
    Vv0 = a0.c_v(F.rms_norm(wte, (D,))).view(V, NH, HD).float()
    xn = F.rms_norm(shrunk, (D,))
    v_new = a3.c_v(xn).view(V, NH, HD).float()
    vs = (1 - a3.lamb) * v_new + a3.lamb * Vv0.view_as(v_new)
del shrunk, xn, v_new
torch.cuda.empty_cache()

# layer-3 head-value spectra from the mean-residual table: S[h][t,c] = <basis_hc, c_proj3_h vs_h(t)>
S = []
for hh in range(NH):
    wr = vs[:, hh] @ cw[:, hh*HD:(hh+1)*HD].T             # (V, D) write vectors
    S.append(wr @ BASIS[hh])                              # (V, NC)
del vs
torch.cuda.empty_cache()
print("layer-3 spectra built (mean-residual table)", flush=True)

# unigram weights (frequency for R^2 weighting only)
fcnt = torch.zeros(V, device=DEV); fcnt.index_add_(0, COOC[:400, :128].reshape(-1).to(DEV), torch.ones(400*128, device=DEV))
p = (fcnt/fcnt.sum())


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

# rotate each head's retained span to spectrum-sparse coordinates (span unchanged)
for hh in range(NH):
    R_ = varimax(S[hh], p)
    S[hh] = S[hh] @ R_
    BASIS[hh] = BASIS[hh] @ R_
print("varimax rotation applied (span-preserving)", flush=True)


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
    top1k = torch.zeros(V, dtype=torch.bool); top1k[fcnt.topk(1000).indices.cpu()] = True
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
print(f"codable at R2>=0.8: {codable}/{NH*NC} ({frac:.0%}); median spectrum R2 {np.median(r2s):.3f}", flush=True)
from collections import Counter
top = Counter(r['class'] for r in results if r['r2'] >= 0.8).most_common(8)
print("codable classes:", top, flush=True)


@torch.no_grad()
def forward_spec(idx, spectra):
    """layer-3 attention writes replaced by pattern-weighted SPECTRA (exact or coded); rest real."""
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
        if li == 3 and spectra is not None:
            aout = torch.zeros(B, T2, D, device=DEV)
            for hh in range(NH):
                sp = spectra[hh][idx.reshape(-1)].view(B, T2, NC)      # (B,T,NC) per-key coord writes
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
# ---- spike-sparsity analysis on the SAME spectra ----
def concentration(Svec, k=8):
    mass = (w * Svec.abs())
    tot = float(mass.sum())
    return float(mass.topk(k).values.sum())/max(tot, 1e-12)

conc = []
SPIKE = [s_.clone() for s_ in S]
spikey = 0
for hh in range(NH):
    for c in range(NC):
        y = S[hh][:, c]
        cc = concentration(y)
        conc.append(cc)
        if cc >= 0.8:
            spikey += 1
            mass = (w * y.abs()); topi = mass.topk(8).indices
            off = float((w*y).sum() - (w[topi]*y[topi]).sum()) / max(float(w.sum() - w[topi].sum()), 1e-9)
            newv = torch.full_like(y, off); newv[topi] = y[topi]
            SPIKE[hh][:, c] = newv
conc = np.array(conc)
print(f"spike concentration (top-8 weighted mass): median {np.median(conc):.3f}, frac>=0.8: {float((conc>=0.8).mean()):.2%} ({spikey} coords)", flush=True)
ce_spike = percetok(SPIKE)
dif3 = (ce_spike - ce_exact)
print(f"SPIKE GATE ({spikey} coords 8-spike-coded): coded - exact = +{float(dif3.mean()):.5f} (SE {float(dif3.std()/np.sqrt(dif3.numel())):.6f})", flush=True)
res['spike'] = {'median_concentration': round(float(np.median(conc)), 3),
                'frac_ge_0.8': round(float((conc>=0.8).mean()), 3), 'n_spike': spikey,
                'gate_dCE': round(float(dif3.mean()), 5), 'gate_SE': round(float(dif3.std()/np.sqrt(dif3.numel())), 6)}
print(f"SUMMARY L3 CONTENT: class-nameable {codable}/{NH*NC} (median R2 {np.median(r2s):.3f}); "
      f"class gate +{res['gate_coded_minus_exact']['dCE']:.5f}; spike {spikey}/{NH*NC} gate +{res['spike']['gate_dCE']:.5f} nats/tok", flush=True)
json.dump(res, open(f'{QK}/qk_l3_content_gate.json', 'w'), indent=2)
json.dump(results, open(f'{QK}/qk_l3_content_gate_table.json', 'w'), indent=2)
print("QK L3 CONTENT GATE DONE", flush=True)
