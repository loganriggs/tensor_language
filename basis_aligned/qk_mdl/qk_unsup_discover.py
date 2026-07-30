"""UNSUPERVISED circuit/algorithm-discovery scanner over bilin18's decomposition.

Idea (Logan): a *path* (a single attention head's output, or a single MLP output
direction) that fires on a SHARP trigger and pushes a SHARP set of output tokens
IS an algorithm. Find the cleanest such input->output maps WITHOUT pre-specifying
any behavior, then rank by cleanliness gated by non-trivial magnitude.

Candidate paths:
  (A) each of the 162 attention HEADS (li,h): output contribution to the residual
      = c_proj applied to that head's yh (that head's slot only). Effect direction
      = the (mean, over its own top-K firing positions) output vector pushed through
      the final rms_norm + lm_head (first-order direct-to-logits proxy).
  (B) each block's top-4 MLP output right-singular-vectors (from the uncentred gram
      of that block's MLP output over the data). A direction's activation at a
      position = projection of that MLP's output onto the direction; its effect =
      the (sign-oriented) direction pushed through lm_head.

Per candidate on the DISCOVERY data (FineWeb TRAIN slice FW[0:256,:128]):
  1. TRIGGER: top-K=50 positions by |activation|. Characterise by (a) current token,
     (b) heads: attended-source token = token at argmax|attention| key, (c) coarse
     classes. trigger-purity = concentration (1 - normalised entropy) of the top-K
     current tokens and (heads) attended-source tokens; head uses max of the two axes.
  2. EFFECT: effect direction -> lm_head -> top boosted next-tokens. effect-purity =
     concentration of the top-M boosted-token softmax (unit direction => scale-free).
  3. CLEANLINESS = trigger-purity * effect-purity. Also report mean |activation| as a
     residual fraction (load-bearing) so tiny-but-pure paths don't rank up.

FORWARD CONVENTIONS COPIED VERBATIM from qk_conditional_redirect.py / qk_bracket_patch.py
(tier2_model.reference_forward): bilin18 two-branch pattern (q1k1/HD)(q2k2/HD) masked
UNNORMALISED, per-head QK rms_norm THEN RoPE, v-lerp via a.lamb (block-0 v cache),
30*tanh logits. Discovery pass only -- causal verification is a separate later step,
so every hit is a "candidate algorithm, UNVERIFIED".
"""
import json, sys, math
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
print(f"bilin18: NL={NL} NH={NH} HD={HD} D={D} V={V}  -> {NL*NH} head-paths", flush=True)

# model exposes no tokenizer via load_elriggs; use GPT-2 (matches qk_bracket_patch.py) -- FLAGGED
tok = AutoTokenizer.from_pretrained('gpt2')

FINEWEB = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))
NSEQ, SEQL = 256, 128
IDX = FINEWEB[0:NSEQ, :SEQL].to(DEV)         # discovery TRAIN slice; FW[448:600] held back, untouched
BATCH = 8
K = 50                                       # top-K firing positions per path
M_EFF = 64                                   # top-M boosted tokens for effect purity
N_SVD = 4                                    # MLP right-singular-vecs per block
Npos = NSEQ * SEQL
idx_flat = IDX.reshape(-1).cpu().numpy()     # current token per flat position

# special/degenerate tokens: document-separator + UTF-8 replacement/byte-fallback tokens.
# Top MLP singular directions are dominated by the massive-norm endoftext/BOS outlier, giving
# trivially "pure" (single-token) triggers that are sinks, not algorithms. Mask from selection.
_special = {tok.eos_token_id}
for _t in range(V):
    if '�' in tok.decode([_t]): _special.add(_t)
SPECIAL = np.array(sorted(_special))
is_special = np.isin(idx_flat, SPECIAL)      # per flat position
print(f"{len(SPECIAL)} special token ids masked from trigger selection (eos={tok.eos_token_id})", flush=True)

# c_proj weight -> per-head map: Wr[o, h, c]  (o=D_out, h=head, c=HD)
Wcp = m.transformer.h[0].attn.c_proj.weight  # (D_out, NH*HD)  -- checked below all blocks share shape
LMW = m.lm_head.weight                        # (V, D)

# -------------------------------------------------------------------------------------
# storage
head_act = np.zeros((NL * NH, Npos), dtype=np.float32)      # ||head output vector||
head_src = np.zeros((NL * NH, Npos), dtype=np.int32)        # argmax|attn| key position
resid_nrm = np.zeros((NL, Npos), dtype=np.float32)          # ||residual|| after each block
gram = [torch.zeros(D, D, device=DEV) for _ in range(NL)]   # uncentred MLP-output gram per block


@torch.no_grad()
def forward_passA(idx, p0):
    """VERBATIM forward; accumulate head-output norms, argmax|attn| source, resid norm, MLP gram."""
    B, T = idx.shape
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    for li in range(NL):
        blk = m.transformer.h[li]; x = blk.lambdas[0]*x + blk.lambdas[1]*x0; a = blk.attn; hcur = F.rms_norm(x, (D,))
        def qk(lin): z = F.rms_norm(lin(hcur).view(B, T, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
        v = a.c_v(hcur).view(B, T, NH, HD)
        if v1 is None: v1 = v
        v = (1-a.lamb)*v + a.lamb*v1.view_as(v)
        q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k)/HD; s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD
        pat = (s1*s2).masked_fill(~mask, 0.0)                 # (B,NH,T,T) UNNORMALISED bilinear
        yh4 = torch.einsum('bhqk,bkhd->bqhd', pat, v)         # (B,T,NH,HD)
        # per-head output contribution vector (that head's slot through c_proj)
        Wr = a.c_proj.weight.view(D, NH, HD)
        comp = torch.einsum('bthc,ohc->btho', yh4, Wr)        # (B,T,NH,D)
        hn = comp.norm(dim=-1)                                # (B,T,NH)
        srcpos = pat.abs().argmax(-1)                         # (B,NH,T) argmax|attn| key position
        f0, f1 = p0, p0 + B*T
        for h in range(NH):
            head_act[li*NH + h, f0:f1] = hn[:, :, h].reshape(-1).float().cpu().numpy()
            head_src[li*NH + h, f0:f1] = srcpos[:, h, :].reshape(-1).cpu().numpy()
        x = x + a.c_proj(yh4.reshape(B, T, -1))
        mo = blk.mlp(F.rms_norm(x, (D,)))
        gram[li] += torch.einsum('btd,bte->de', mo, mo)       # uncentred gram of MLP output
        x = x + mo
        resid_nrm[li, f0:f1] = x.norm(dim=-1).reshape(-1).float().cpu().numpy()


print("PASS A: head norms + attn-source + resid + MLP gram ...", flush=True)
for i in range(0, NSEQ, BATCH):
    forward_passA(IDX[i:i+BATCH], i*SEQL)
    if i % 64 == 0: print(f"  seq {i}/{NSEQ}", flush=True)

# MLP directions: top-N_SVD eigenvectors of each block's MLP-output gram (== top right-singular-vecs)
mlp_dirs = torch.zeros(NL, N_SVD, D, device=DEV)
for li in range(NL):
    evals, evecs = torch.linalg.eigh(gram[li])                # ascending
    mlp_dirs[li] = evecs[:, -N_SVD:].T.flip(0)                # top-N_SVD, descending
print("MLP directions computed.", flush=True)

# head top-K positions known now (exclude t=0: trivial self-attention)
pos_t = np.tile(np.arange(SEQL), NSEQ)                         # within-seq position of each flat idx
head_topk = {}                                                # (li*NH+h) -> array of K flat positions
head_mask = np.zeros((NL*NH, Npos), dtype=bool)
for p in range(NL*NH):
    a = head_act[p].copy(); a[(pos_t == 0) | is_special] = -1.0
    tk = np.argpartition(a, -K)[-K:]
    head_topk[p] = tk; head_mask[p, tk] = True

# -------------------------------------------------------------------------------------
mlp_proj = np.zeros((NL * N_SVD, Npos), dtype=np.float32)      # projection of MLP output onto each dir
head_eff_sum = torch.zeros(NL*NH, D, device=DEV)              # sum of head output vecs over its top-K


@torch.no_grad()
def forward_passB(idx, p0):
    """VERBATIM forward; compute MLP-dir projections, and gather head output vectors at their top-K."""
    B, T = idx.shape
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    f0, f1 = p0, p0 + B*T
    for li in range(NL):
        blk = m.transformer.h[li]; x = blk.lambdas[0]*x + blk.lambdas[1]*x0; a = blk.attn; hcur = F.rms_norm(x, (D,))
        def qk(lin): z = F.rms_norm(lin(hcur).view(B, T, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
        v = a.c_v(hcur).view(B, T, NH, HD)
        if v1 is None: v1 = v
        v = (1-a.lamb)*v + a.lamb*v1.view_as(v)
        q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k)/HD; s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD
        pat = (s1*s2).masked_fill(~mask, 0.0)
        yh4 = torch.einsum('bhqk,bkhd->bqhd', pat, v)
        Wr = a.c_proj.weight.view(D, NH, HD)
        comp = torch.einsum('bthc,ohc->btho', yh4, Wr).reshape(B*T, NH, D)  # (npos,NH,D)
        for h in range(NH):
            sel = head_mask[li*NH + h, f0:f1]
            if sel.any():
                head_eff_sum[li*NH + h] += comp[torch.from_numpy(sel).to(DEV), h, :].sum(0)
        x = x + a.c_proj(yh4.reshape(B, T, -1))
        mo = blk.mlp(F.rms_norm(x, (D,)))                      # (B,T,D)
        pr = torch.einsum('btd,nd->btn', mo, mlp_dirs[li])     # (B,T,N_SVD)
        for kk in range(N_SVD):
            mlp_proj[li*N_SVD + kk, f0:f1] = pr[:, :, kk].reshape(-1).float().cpu().numpy()
        x = x + mo


print("PASS B: MLP projections + head effect directions ...", flush=True)
for i in range(0, NSEQ, BATCH):
    forward_passB(IDX[i:i+BATCH], i*SEQL)
    if i % 64 == 0: print(f"  seq {i}/{NSEQ}", flush=True)

# -------------------------------------------------------------------------------------
# concentration = 1 - normalised entropy of a multiset of category ids
def conc_ids(ids):
    ids = np.asarray(ids)
    if len(ids) <= 1: return 0.0
    _, c = np.unique(ids, return_counts=True)
    p = c / c.sum(); H = -(p * np.log(p)).sum()
    return float(1.0 - H / math.log(len(ids)))

def top_counts(ids, n):
    v, c = np.unique(np.asarray(ids), return_counts=True)
    o = np.argsort(-c)[:n]
    return [(int(v[i]), int(c[i])) for i in o]

def dec(t):                       # human-readable decoded token
    return repr(tok.decode([int(t)]))

def coarse_class(t):
    s = tok.decode([int(t)]); raw = tok.convert_ids_to_tokens(int(t))
    st = s.strip()
    cl = []
    if raw.startswith('Ġ') or (s[:1] == ' '): cl.append('space_pref')
    if 'Ċ' in raw or '\n' in s: cl.append('newline')
    if st and any(ch.isdigit() for ch in st): cl.append('digit')
    if st[:1].isupper(): cl.append('capital')
    if st and all(not ch.isalnum() for ch in st): cl.append('punct')
    return cl

def class_dist(ids):
    from collections import Counter
    c = Counter()
    for t in ids:
        for cl in coarse_class(t): c[cl] += 1
    return {k: round(v/len(ids), 2) for k, v in c.most_common()}

@torch.no_grad()
def effect_from_dir(dir_vec):
    d = dir_vec / (dir_vec.norm() + 1e-9)                      # unit dir => scale-free
    logits = (LMW @ d).float()                                 # (V,) direct-to-logits proxy
    logits = logits - logits.mean()
    top = torch.topk(logits, M_EFF)
    v = top.values                                            # descending, top-M boosted logits
    w = (v - v.min())                                         # excess above the M-th boosted token
    if w.sum() <= 1e-9:
        eff_pur = 0.0
    else:
        p = w / w.sum()                                       # scale-free weights over the leaders
        H = -(p * (p + 1e-12).log()).sum().item()
        eff_pur = float(1.0 - H / math.log(M_EFF))            # 1 => one token dominates; 0 => flat top-M
    boosted = top.indices[:8].cpu().numpy().tolist()
    return eff_pur, boosted

records = []
# --- head paths ---
for li in range(NL):
    for h in range(NH):
        p = li*NH + h; tk = head_topk[p]
        cur = idx_flat[tk]
        seq_of = tk // SEQL
        srcpos = head_src[p, tk]
        src_tok = IDX.cpu().numpy()[seq_of, srcpos]            # token at attended-source key
        cur_conc = conc_ids(cur); src_conc = conc_ids(src_tok)
        src_special_frac = float(np.isin(src_tok, SPECIAL).mean())   # attention-sink source share
        trig_pur = max(cur_conc, src_conc)
        eff_dir = head_eff_sum[p] / max(1, head_mask[p].sum())
        eff_pur, boosted = effect_from_dir(eff_dir)
        mag = float(np.mean(head_act[p, tk] / (resid_nrm[li, tk] + 1e-9)))   # residual-fraction (load-bearing)
        records.append({
            'comp': f"h.L{li}.{h}", 'kind': 'head', 'li': li, 'h': h,
            'trigger_purity': round(trig_pur, 4), 'cur_purity': round(cur_conc, 4),
            'src_purity': round(src_conc, 4), 'src_special_frac': round(src_special_frac, 3),
            'effect_purity': round(eff_pur, 4),
            'cleanliness': round(trig_pur * eff_pur, 4),
            'mag_resid_frac': round(mag, 4), 'mean_act_topk': round(float(head_act[p, tk].mean()), 3),
            'top5_cur': [dec(t) for t, _ in top_counts(cur, 5)],
            'top5_cur_counts': [c for _, c in top_counts(cur, 5)],
            'top5_src': [dec(t) for t, _ in top_counts(src_tok, 5)],
            'top5_src_counts': [c for _, c in top_counts(src_tok, 5)],
            'top8_boost': [dec(t) for t in boosted],
            'cur_classes': class_dist(cur), 'src_classes': class_dist(src_tok),
        })
# --- MLP direction paths ---
for li in range(NL):
    for kk in range(N_SVD):
        p = li*N_SVD + kk; pr = mlp_proj[p].copy()
        a = np.abs(pr); a[(pos_t == 0) | is_special] = -1.0
        tk = np.argpartition(a, -K)[-K:]
        cur = idx_flat[tk]
        cur_conc = conc_ids(cur); trig_pur = cur_conc
        sgn = np.sign(pr[tk].mean()) or 1.0
        eff_dir = mlp_dirs[li, kk] * float(sgn)
        eff_pur, boosted = effect_from_dir(eff_dir)
        mag = float(np.mean(np.abs(pr[tk]) / (resid_nrm[li, tk] + 1e-9)))
        records.append({
            'comp': f"mlp.L{li}.d{kk}", 'kind': 'mlp', 'li': li, 'k': kk,
            'trigger_purity': round(trig_pur, 4), 'cur_purity': round(cur_conc, 4),
            'src_purity': None, 'effect_purity': round(eff_pur, 4),
            'cleanliness': round(trig_pur * eff_pur, 4),
            'mag_resid_frac': round(mag, 4), 'mean_act_topk': round(float(np.abs(pr[tk]).mean()), 3),
            'top5_cur': [dec(t) for t, _ in top_counts(cur, 5)],
            'top5_cur_counts': [c for _, c in top_counts(cur, 5)],
            'top5_src': None, 'top5_src_counts': None,
            'top8_boost': [dec(t) for t in boosted],
            'cur_classes': class_dist(cur), 'src_classes': None,
        })

# magnitude gate: drop the lowest-magnitude tail so tiny-but-pure paths don't top the board.
mags = np.array([r['mag_resid_frac'] for r in records])
gate = float(np.percentile(mags, 25))
for r in records:
    r['passes_mag_gate'] = bool(r['mag_resid_frac'] >= gate)

ranked = sorted(records, key=lambda r: (-r['cleanliness'] if r['passes_mag_gate'] else 1e9 - r['cleanliness']))
# stable: gated-in first by cleanliness desc, then gated-out
gated_in = sorted([r for r in records if r['passes_mag_gate']], key=lambda r: -r['cleanliness'])
gated_out = sorted([r for r in records if not r['passes_mag_gate']], key=lambda r: -r['cleanliness'])
ranked = gated_in + gated_out

out = {
    'meta': {
        'model': 'bilin18', 'tokenizer': 'gpt2 (model exposes none via load_elriggs) -- FLAGGED',
        'discovery_data': 'FineWeb FW[0:256,:128] TRAIN; FW[448:600] held back untouched',
        'n_head_paths': NL*NH, 'n_mlp_paths': NL*N_SVD, 'K_topk': K, 'M_effect': M_EFF,
        'N_svd_per_block': N_SVD, 'mag_gate_pctile25': round(gate, 4),
        'attended_source': 'token at argmax|attention| key', 'note': 'DISCOVERY PASS -- all hits UNVERIFIED',
        'purity': 'concentration = 1 - normalised entropy; head trigger = max(current, source); effect = softmax of top-64 unit-dir logits',
    },
    'ranking': ranked,
}
json.dump(out, open(f'{QK}/qk_unsup_paths.json', 'w'), indent=2)

print("\n===== TOP 15 CANDIDATE PATHS (gated by magnitude, ranked by cleanliness) =====", flush=True)
for r in ranked[:15]:
    print(f"\n{r['comp']:12s} clean={r['cleanliness']:.3f} trig={r['trigger_purity']:.3f}"
          f"(cur={r['cur_purity']:.2f} src={r['src_purity']}) eff={r['effect_purity']:.3f}"
          f" mag={r['mag_resid_frac']:.3f} meanact={r['mean_act_topk']}", flush=True)
    print(f"   cur : {r['top5_cur']}", flush=True)
    if r['top5_src']: print(f"   src : {r['top5_src']}", flush=True)
    print(f"   BOOST: {r['top8_boost']}", flush=True)
    print(f"   classes cur={r['cur_classes']} src={r['src_classes']}", flush=True)
print("\nSaved qk_unsup_paths.json  (full ranking).", flush=True)
print("QK UNSUP DISCOVER DONE", flush=True)
