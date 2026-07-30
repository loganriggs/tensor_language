"""UNSUPERVISED SUPPRESSION-CIRCUIT discovery scanner over bilin18's decomposition.

WHY A DIFFERENT TOOL (Logan: different circuit types need different tools).
  qk_unsup_discover.py ranks paths by their POSITIVE effect: a path is interesting
  if its output direction pushes a SHARP set of next-tokens UP (boost). That pipeline
  is structurally blind to the opposite mechanism -- a SUPPRESSION / inhibition
  circuit that pushes a concentrated set of tokens DOWN (e.g. "don't repeat the token
  you just attended to"; suppress digits/capitals in the wrong context). The boost
  tool's effect_purity is the concentration of the top (most positive) logits; a pure
  suppressor has a FLAT, uninteresting positive side and would never surface.

THE MIRROR SIGNATURE (this tool):
  * DISCOVERY: for each path (attention head output, or top MLP output direction),
    take the SAME effect direction pushed through lm_head, but score the concentration
    of the MOST-NEGATIVE logit contributions (the sharply pushed-DOWN tokens).
    suppression-purity = concentration(bottom-M logits); rank by suppression-purity x
    magnitude (residual-fraction, load-bearing).
  * ANTI-COPY probe (heads only): does the path suppress the token at the ATTENDED
    SOURCE position? suppress-what-you-attend = anti-repetition. And self-suppression:
    does it push down the CURRENT token? Measured as the source/current token's rank in
    the path's own logit distribution at its top-activating positions.
  * CAUSAL VERIFY (top ~5, held-out FW[448:600]): mean-ablate the path at its
    top-activating positions and check the discovered suppressed tokens RISE (positive
    dLogit -- the MIRROR of a boost head, where ablation makes the boosted set DROP),
    with a control set and paired SEs. For anti-copy heads, check the attended-source
    token itself rises. CE story: a real anti-repetition head, when ablated, should
    HELP CE where repetition is actually correct (next==source) and HURT CE where it is
    wrong -- the crisp inhibition signature.

HONESTY (§56 / greater-of-two / KEY_newline lessons): a genuine suppression MECHANISM
  must be distinguished from (a) a path with a merely diffuse negative tail, and (b) the
  negative side of a BOOST head (sharp positive + incidental sharp negative). We report
  boost-purity alongside suppress-purity and a suppress/boost dominance ratio, and flag
  bidirectional paths. "No strong suppression circuits found / suppression is diffuse"
  is a valid, valuable verdict.

FORWARD copied VERBATIM from qk_bracket_patch.py / qk_unsup_verify2.py
(tier2_model.reference_forward): bilin18 two-branch UNNORMALISED pattern (s1*s2), per-head
QK rms_norm THEN RoPE, v-lerp via a.lamb (block-0 v cache), 30*tanh logits.
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
LMW = m.lm_head.weight                         # (V, D)
print(f"bilin18: NL={NL} NH={NH} HD={HD} D={D} V={V}  -> {NL*NH} head-paths", flush=True)

tok = AutoTokenizer.from_pretrained('gpt2')    # model exposes none via load_elriggs -- FLAGGED

FW = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))
NSEQ, SEQL = 256, 128
IDX = FW[0:NSEQ, :SEQL].to(DEV)                # discovery TRAIN slice; FW[448:600] held back
BATCH = 4
K = 50                                         # top-K firing positions per path
M_EFF = 64                                     # bottom-M suppressed tokens (mirror of boost top-M)
N_SVD = 4                                       # MLP right-singular-vecs per block
Npos = NSEQ * SEQL
idx_flat = IDX.reshape(-1).cpu().numpy()

# special/degenerate tokens: masked from selection (same rule as boost discovery).
_special = {tok.eos_token_id}
for _t in range(V):
    if '�' in tok.decode([_t]): _special.add(_t)
SPECIAL = np.array(sorted(_special))
SPECIAL_SET = set(SPECIAL.tolist())
is_special = np.isin(idx_flat, SPECIAL)
print(f"{len(SPECIAL)} special token ids masked (eos={tok.eos_token_id})", flush=True)


# ============================================================================
# PASS A: head-output norms, argmax|attn| source, resid norm, MLP-output gram
# ============================================================================
head_act = np.zeros((NL * NH, Npos), dtype=np.float32)
head_src = np.zeros((NL * NH, Npos), dtype=np.int32)
resid_nrm = np.zeros((NL, Npos), dtype=np.float32)
gram = [torch.zeros(D, D, device=DEV) for _ in range(NL)]


@torch.no_grad()
def forward_passA(idx, p0):
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
        pat = (s1*s2).masked_fill(~mask, 0.0)
        yh4 = torch.einsum('bhqk,bkhd->bqhd', pat, v)
        Wr = a.c_proj.weight.view(D, NH, HD)
        comp = torch.einsum('bthc,ohc->btho', yh4, Wr)        # (B,T,NH,D) per-head output vec
        hn = comp.norm(dim=-1)
        srcpos = pat.abs().argmax(-1)                         # (B,NH,T) argmax|attn| key pos
        f0, f1 = p0, p0 + B*T
        for h in range(NH):
            head_act[li*NH + h, f0:f1] = hn[:, :, h].reshape(-1).float().cpu().numpy()
            head_src[li*NH + h, f0:f1] = srcpos[:, h, :].reshape(-1).cpu().numpy()
        x = x + a.c_proj(yh4.reshape(B, T, -1))
        mo = blk.mlp(F.rms_norm(x, (D,)))
        gram[li] += torch.einsum('btd,bte->de', mo, mo)
        x = x + mo
        resid_nrm[li, f0:f1] = x.norm(dim=-1).reshape(-1).float().cpu().numpy()


print("PASS A ...", flush=True)
for i in range(0, NSEQ, BATCH):
    forward_passA(IDX[i:i+BATCH], i*SEQL)
    if i % 64 == 0: print(f"  seq {i}/{NSEQ}", flush=True)

mlp_dirs = torch.zeros(NL, N_SVD, D, device=DEV)
for li in range(NL):
    _, evecs = torch.linalg.eigh(gram[li])
    mlp_dirs[li] = evecs[:, -N_SVD:].T.flip(0)
print("MLP directions computed.", flush=True)

pos_t = np.tile(np.arange(SEQL), NSEQ)
head_topk = {}; head_mask = np.zeros((NL*NH, Npos), dtype=bool)
for p in range(NL*NH):
    a = head_act[p].copy(); a[(pos_t == 0) | is_special] = -1.0
    tk = np.argpartition(a, -K)[-K:]
    head_topk[p] = tk; head_mask[p, tk] = True


# ============================================================================
# PASS B: MLP-dir projections + head effect (mean output) directions
# ============================================================================
mlp_proj = np.zeros((NL * N_SVD, Npos), dtype=np.float32)
head_eff_sum = torch.zeros(NL*NH, D, device=DEV)


@torch.no_grad()
def forward_passB(idx, p0):
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
        comp = torch.einsum('bthc,ohc->btho', yh4, Wr).reshape(B*T, NH, D)
        for h in range(NH):
            sel = head_mask[li*NH + h, f0:f1]
            if sel.any():
                head_eff_sum[li*NH + h] += comp[torch.from_numpy(sel).to(DEV), h, :].sum(0)
        x = x + a.c_proj(yh4.reshape(B, T, -1))
        mo = blk.mlp(F.rms_norm(x, (D,)))
        pr = torch.einsum('btd,nd->btn', mo, mlp_dirs[li])
        for kk in range(N_SVD):
            mlp_proj[li*N_SVD + kk, f0:f1] = pr[:, :, kk].reshape(-1).float().cpu().numpy()
        x = x + mo


print("PASS B ...", flush=True)
for i in range(0, NSEQ, BATCH):
    forward_passB(IDX[i:i+BATCH], i*SEQL)
    if i % 64 == 0: print(f"  seq {i}/{NSEQ}", flush=True)


# ============================================================================
# scoring helpers
# ============================================================================
def conc_ids(ids):
    ids = np.asarray(ids)
    if len(ids) <= 1: return 0.0
    _, c = np.unique(ids, return_counts=True); p = c / c.sum()
    return float(1.0 - (-(p*np.log(p)).sum()) / math.log(len(ids)))

def top_counts(ids, n):
    v, c = np.unique(np.asarray(ids), return_counts=True); o = np.argsort(-c)[:n]
    return [(int(v[i]), int(c[i])) for i in o]

def dec(t): return repr(tok.decode([int(t)]))

def coarse_class(t):
    s = tok.decode([int(t)]); raw = tok.convert_ids_to_tokens(int(t)); st = s.strip(); cl = []
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


def concentration_of(vals):
    """vals: nonneg excess-mass weights over the leaders. 1 => one dominates, 0 => flat."""
    s = vals.sum()
    if s <= 1e-9: return 0.0
    p = vals / s
    Hh = -(p * (p + 1e-12).log()).sum().item()
    return float(1.0 - Hh / math.log(len(vals)))


@torch.no_grad()
def suppress_boost_from_dir(dir_vec):
    """Push a (signed) effect direction through lm_head; score BOTH tails.
    Returns: suppress_purity (concentration of bottom-M most-negative logits),
             boost_purity (concentration of top-M), suppressed token ids, boosted ids,
             suppress_dominance = |min|/max (>1 => suppression tail sharper),
             plus the full centered logits (unit dir) for per-token lookups."""
    d = dir_vec / (dir_vec.norm() + 1e-9)                  # unit dir => scale-free
    logits = (LMW @ d).float(); logits = logits - logits.mean()
    # suppression side: most-negative -> take bottom-M
    bot = torch.topk(-logits, M_EFF)                       # bot.values = -logits (descending in suppression)
    neg = bot.values                                       # >0, most-suppressed first
    w_s = neg - neg.min()                                  # excess above the M-th most-suppressed
    supp_pur = concentration_of(w_s)
    supp_ids = bot.indices[:12].cpu().numpy().tolist()
    # boost side (mirror, for honesty / bidirectional flag)
    top = torch.topk(logits, M_EFF); pos = top.values
    w_b = pos - pos.min(); boost_pur = concentration_of(w_b)
    boost_ids = top.indices[:8].cpu().numpy().tolist()
    dominance = float((-logits.min()) / (logits.max() + 1e-9))
    return dict(supp_pur=supp_pur, boost_pur=boost_pur, supp_ids=supp_ids,
                boost_ids=boost_ids, dominance=dominance, logits=logits)


def pctile_of_token(logits, tok_ids):
    """fraction of vocab with logit strictly below each token's logit (low => deep in
    suppressed tail). Returns mean logit at those tokens and mean percentile."""
    tok_ids = np.asarray([t for t in tok_ids if 0 <= t < V])
    if tok_ids.size == 0: return None, None
    lg = logits.cpu().numpy()
    tv = lg[tok_ids]
    pct = np.array([(lg < v).mean() for v in tv])
    return float(tv.mean()), float(pct.mean())


records = []
# ---- head paths ----
for li in range(NL):
    for h in range(NH):
        p = li*NH + h; tk = head_topk[p]
        cur = idx_flat[tk]; seq_of = tk // SEQL; srcpos = head_src[p, tk]
        src_tok = IDX.cpu().numpy()[seq_of, srcpos]
        eff_dir = head_eff_sum[p] / max(1, head_mask[p].sum())
        sb = suppress_boost_from_dir(eff_dir)
        mag = float(np.mean(head_act[p, tk] / (resid_nrm[li, tk] + 1e-9)))
        # anti-copy / self-suppression: where does the ATTENDED-SOURCE / CURRENT token sit
        # in THIS path's own logit distribution, averaged over top-activating positions?
        src_nonspec = src_tok[~np.isin(src_tok, SPECIAL)]
        ac_logit, ac_pct = pctile_of_token(sb['logits'], src_nonspec)
        ss_logit, ss_pct = pctile_of_token(sb['logits'], cur)
        records.append({
            'comp': f"h.L{li}.{h}", 'kind': 'head', 'li': li, 'h': h,
            'suppress_purity': round(sb['supp_pur'], 4),
            'boost_purity': round(sb['boost_pur'], 4),
            'suppress_dominance': round(sb['dominance'], 3),
            'score': round(sb['supp_pur'] * mag, 5),
            'mag_resid_frac': round(mag, 4), 'mean_act_topk': round(float(head_act[p, tk].mean()), 3),
            'anti_copy_src_logit': None if ac_logit is None else round(ac_logit, 3),
            'anti_copy_src_pctile': None if ac_pct is None else round(ac_pct, 3),
            'self_supp_cur_logit': None if ss_logit is None else round(ss_logit, 3),
            'self_supp_cur_pctile': None if ss_pct is None else round(ss_pct, 3),
            'top5_cur': [dec(t) for t, _ in top_counts(cur, 5)],
            'top5_src': [dec(t) for t, _ in top_counts(src_tok, 5)],
            'top8_suppressed': [dec(t) for t in sb['supp_ids'][:8]],
            'top6_boosted': [dec(t) for t in sb['boost_ids'][:6]],
            'suppress_ids': sb['supp_ids'], 'boost_ids': sb['boost_ids'],
            'cur_classes': class_dist(cur), 'src_classes': class_dist(src_tok),
        })
# ---- MLP direction paths ----
for li in range(NL):
    for kk in range(N_SVD):
        p = li*N_SVD + kk; pr = mlp_proj[p].copy()
        a = np.abs(pr); a[(pos_t == 0) | is_special] = -1.0
        tk = np.argpartition(a, -K)[-K:]
        cur = idx_flat[tk]
        sgn = np.sign(pr[tk].mean()) or 1.0
        eff_dir = mlp_dirs[li, kk] * float(sgn)
        sb = suppress_boost_from_dir(eff_dir)
        mag = float(np.mean(np.abs(pr[tk]) / (resid_nrm[li, tk] + 1e-9)))
        ss_logit, ss_pct = pctile_of_token(sb['logits'], cur)
        records.append({
            'comp': f"mlp.L{li}.d{kk}", 'kind': 'mlp', 'li': li, 'k': kk,
            'suppress_purity': round(sb['supp_pur'], 4),
            'boost_purity': round(sb['boost_pur'], 4),
            'suppress_dominance': round(sb['dominance'], 3),
            'score': round(sb['supp_pur'] * mag, 5),
            'mag_resid_frac': round(mag, 4), 'mean_act_topk': round(float(np.abs(pr[tk]).mean()), 3),
            'anti_copy_src_logit': None, 'anti_copy_src_pctile': None,
            'self_supp_cur_logit': None if ss_logit is None else round(ss_logit, 3),
            'self_supp_cur_pctile': None if ss_pct is None else round(ss_pct, 3),
            'top5_cur': [dec(t) for t, _ in top_counts(cur, 5)], 'top5_src': None,
            'top8_suppressed': [dec(t) for t in sb['supp_ids'][:8]],
            'top6_boosted': [dec(t) for t in sb['boost_ids'][:6]],
            'suppress_ids': sb['supp_ids'], 'boost_ids': sb['boost_ids'],
            'cur_classes': class_dist(cur), 'src_classes': None,
        })

# magnitude gate (drop lowest-magnitude quartile so tiny-but-pure paths don't top the board)
mags = np.array([r['mag_resid_frac'] for r in records])
gate = float(np.percentile(mags, 25))
for r in records:
    r['passes_mag_gate'] = bool(r['mag_resid_frac'] >= gate)
    # bidirectional flag: sharp on BOTH tails => likely a boost head, not a pure suppressor
    r['bidirectional'] = bool(r['boost_purity'] >= 0.5 * r['suppress_purity'] and r['boost_purity'] > 0.15)

gated_in = sorted([r for r in records if r['passes_mag_gate']], key=lambda r: -r['score'])
gated_out = sorted([r for r in records if not r['passes_mag_gate']], key=lambda r: -r['score'])
ranked = gated_in + gated_out

print("\n===== TOP 15 SUPPRESSION CANDIDATES (gated by magnitude, ranked by suppress-purity x mag) =====", flush=True)
for r in ranked[:15]:
    print(f"\n{r['comp']:12s} score={r['score']:.4f} supp_pur={r['suppress_purity']:.3f}"
          f" boost_pur={r['boost_purity']:.3f} dom={r['suppress_dominance']:.2f}"
          f" mag={r['mag_resid_frac']:.3f} {'[BIDIR]' if r['bidirectional'] else ''}", flush=True)
    print(f"   cur : {r['top5_cur']}", flush=True)
    if r['top5_src']: print(f"   src : {r['top5_src']}  anti_copy_pct={r['anti_copy_src_pctile']} logit={r['anti_copy_src_logit']}", flush=True)
    print(f"   self_supp_cur_pct={r['self_supp_cur_pctile']} logit={r['self_supp_cur_logit']}", flush=True)
    print(f"   SUPPRESSED: {r['top8_suppressed']}", flush=True)
    print(f"   (boosted) : {r['top6_boosted']}", flush=True)


# ============================================================================
# CAUSAL VERIFICATION of the top candidates on held-out FW[448:600]
# ============================================================================
TEST = FW[448:600, :SEQL].to(DEV)
NT = TEST.shape[0]; NposT = NT * SEQL
pos_tT = np.tile(np.arange(SEQL), NT)
test_cur = TEST.cpu().numpy().reshape(-1)
is_special_test = np.isin(test_cur, SPECIAL)
nxt = np.full(NposT, -1, dtype=np.int64)
tc2 = TEST.cpu().numpy(); nxt.reshape(NT, SEQL)[:, :-1] = tc2[:, 1:]
rng = np.random.RandomState(0)
CTRL = np.array([t for t in rng.choice(V, 500, replace=False) if t not in SPECIAL_SET][:200])
N_CAUSAL = 400


@torch.no_grad()
def forward_v(idx, ablate=None, mean_yh=None, mlp_dir=None, mean_proj=None,
              pos_mask=None, capture_head=None, capture_mlpdir=None):
    """VERBATIM bilin18 forward with optional mean-ablation (head or mlpdir) on pos_mask,
    and optional capture of head-output norm + argmax|attn| source, or MLP-dir projection."""
    B, Tt = idx.shape
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(Tt, HD, DEV, x.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(Tt, Tt, device=DEV, dtype=torch.bool)); cap = {}
    for li in range(NL):
        blk = m.transformer.h[li]; x = blk.lambdas[0]*x + blk.lambdas[1]*x0; a = blk.attn; hc = F.rms_norm(x, (D,))
        def qk(l): z = F.rms_norm(l(hc).view(B, Tt, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
        v = a.c_v(hc).view(B, Tt, NH, HD)
        if v1 is None: v1 = v
        v = (1 - a.lamb) * v + a.lamb * v1.view_as(v)
        q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k) / HD; s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2) / HD
        pat = (s1 * s2).masked_fill(~mask, 0.0)
        yh = torch.einsum('bhqk,bkhd->bqhd', pat, v)
        if capture_head is not None and capture_head[0] == li:
            hh = capture_head[1]; Wr = a.c_proj.weight.view(D, NH, HD)
            comp = torch.einsum('btc,oc->bto', yh[:, :, hh], Wr[:, hh])
            cap['hnorm'] = comp.norm(dim=-1); cap['src'] = pat[:, hh].abs().argmax(-1)
        if ablate is not None and ablate[0] == 'head' and ablate[1] == li:
            hh = ablate[2]; yh = yh.clone()
            repl = mean_yh.to(yh.dtype).view(1, 1, HD).expand(B, Tt, HD)
            yh[:, :, hh] = torch.where(pos_mask.unsqueeze(-1), repl, yh[:, :, hh])
        x = x + a.c_proj(yh.reshape(B, Tt, -1))
        mo = blk.mlp(F.rms_norm(x, (D,)))
        if capture_mlpdir is not None and capture_mlpdir[0] == li:
            cap['proj'] = torch.einsum('btd,d->bt', mo, capture_mlpdir[1])
        if ablate is not None and ablate[0] == 'mlpdir' and ablate[1] == li:
            proj = torch.einsum('btd,d->bt', mo, mlp_dir)
            delta = torch.where(pos_mask, mean_proj - proj, torch.zeros_like(proj))
            mo = mo + delta.unsqueeze(-1) * mlp_dir.view(1, 1, D)
        x = x + mo
    logits = 30 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30)
    return logits, cap


@torch.no_grad()
def mean_head_yh_test(li, h):
    s = torch.zeros(HD, device=DEV); n = 0
    for i in range(0, NT, BATCH):
        idx = TEST[i:i+BATCH]; B, Tt = idx.shape
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        cos, sin = rope_tables(Tt, HD, DEV, x.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
        mask = torch.tril(torch.ones(Tt, Tt, device=DEV, dtype=torch.bool))
        for L in range(li+1):
            blk = m.transformer.h[L]; x = blk.lambdas[0]*x + blk.lambdas[1]*x0; a = blk.attn; hc = F.rms_norm(x, (D,))
            def qk(l): z = F.rms_norm(l(hc).view(B, Tt, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
            vv = a.c_v(hc).view(B, Tt, NH, HD)
            if v1 is None: v1 = vv
            vv = (1-a.lamb)*vv + a.lamb*v1.view_as(vv)
            q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
            s1 = torch.einsum('bqhd,bkhd->bhqk', q, k)/HD; s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD
            pat = (s1*s2).masked_fill(~mask, 0.0); yh = torch.einsum('bhqk,bkhd->bqhd', pat, vv)
            if L == li:
                s += yh[:, :, h].reshape(-1, HD).sum(0); n += B*Tt; break
            x = x + a.c_proj(yh.reshape(B, Tt, -1)); x = x + blk.mlp(F.rms_norm(x, (D,)))
    return s / n


def pse(v): return round(float(v.std(ddof=1) / math.sqrt(len(v))), 4) if len(v) > 1 else None


def verify(rec):
    r = {'comp': rec['comp'], 'kind': rec['kind'], 'suppress_purity': rec['suppress_purity'],
         'boost_purity': rec['boost_purity'], 'bidirectional': rec['bidirectional'],
         'discovered_suppressed': rec['top8_suppressed'], 'discovered_boosted': rec['top6_boosted']}
    supp_ids = np.array([t for t in rec['suppress_ids'] if t not in SPECIAL_SET])
    # ---- capture activation on held-out ----
    act = np.zeros(NposT, dtype=np.float32); src = np.zeros(NposT, dtype=np.int64)
    if rec['kind'] == 'head':
        for i in range(0, NT, BATCH):
            idx = TEST[i:i+BATCH]; f0 = i*SEQL; f1 = f0 + idx.shape[0]*SEQL
            _, cap = forward_v(idx, capture_head=(rec['li'], rec['h']))
            act[f0:f1] = cap['hnorm'].reshape(-1).float().cpu().numpy()
            srcpos = cap['src'].cpu().numpy()
            src[f0:f1] = np.take_along_axis(idx.cpu().numpy(), srcpos, axis=1).reshape(-1)
        myh = mean_head_yh_test(rec['li'], rec['h'])
        abl = {'ablate': ('head', rec['li'], rec['h']), 'mean_yh': myh}
    else:
        du = mlp_dirs[rec['li'], rec['k']]; du = du / (du.norm()+1e-9)
        sp = 0.0; cn = 0
        for i in range(0, NT, BATCH):
            idx = TEST[i:i+BATCH]; f0 = i*SEQL; f1 = f0 + idx.shape[0]*SEQL
            _, cap = forward_v(idx, capture_mlpdir=(rec['li'], du))
            pr = cap['proj'].reshape(-1).float().cpu().numpy(); act[f0:f1] = pr; sp += pr.sum(); cn += pr.size
        # orient the direction the way it fires (sign of mean proj at top-|act| positions), as in discovery
        aa0 = np.abs(act).copy(); aa0[(pos_tT == 0) | is_special_test] = -1.0
        tk0 = np.argpartition(aa0, -K)[-K:]; sgn = np.sign(act[tk0].mean()) or 1.0
        du = du * float(sgn); mean_proj = float(sp / cn) * float(sgn)
        abl = {'ablate': ('mlpdir', rec['li']), 'mlp_dir': du, 'mean_proj': torch.tensor(mean_proj, device=DEV)}

    aa = np.abs(act).copy(); aa[(pos_tT == 0) | is_special_test] = -1.0
    tk = np.argpartition(aa, -K)[-K:]; cur = test_cur[tk]
    r['heldout_top50_cur'] = [(dec(t), int(c)) for t, c in
        sorted(zip(*np.unique(cur, return_counts=True)), key=lambda z: -z[1])[:6]]
    r['heldout_cur_purity'] = round(conc_ids(cur), 3)
    if rec['kind'] == 'head':
        r['heldout_top50_src'] = [(dec(t), int(c)) for t, c in
            sorted(zip(*np.unique(src[tk], return_counts=True)), key=lambda z: -z[1])[:6]]

    # causal positions = top-|act| (the path's top-activating positions)
    causal_idx = np.argpartition(aa, -N_CAUSAL)[-N_CAUSAL:]
    causal_idx = np.sort(causal_idx[aa[causal_idx] > 0])
    glob_med = float(np.median(np.abs(act[(pos_tT != 0) & ~is_special_test])) + 1e-9)
    r['selectivity_trig/median'] = round(float(np.abs(act[causal_idx]).mean() / glob_med), 2)
    r['n_causal'] = int(len(causal_idx))

    def run(ablate_kw):
        outB = []
        for i in range(0, NT, BATCH):
            idx = TEST[i:i+BATCH]; B = idx.shape[0]; f0 = i*SEQL
            flat = np.arange(f0, f0 + B*SEQL); sel = np.isin(flat, causal_idx)
            if not sel.any(): continue
            pm = torch.from_numpy(sel.reshape(B, SEQL)).to(DEV)
            lg, _ = forward_v(idx, pos_mask=pm, **ablate_kw)
            lg = lg.reshape(B*SEQL, V)
            outB.append((flat[sel], lg[torch.from_numpy(sel).to(DEV)].float().cpu().numpy()))
        pos = np.concatenate([o[0] for o in outB]); L = np.concatenate([o[1] for o in outB])
        o = np.argsort(pos); return pos[o], L[o]

    posb, Lbase = run({}); posa, Labl = run(abl); assert np.array_equal(posb, posa)
    dLog = Labl - Lbase                                    # ablated - baseline; RISE (>0) = un-suppression
    mean_dLog = dLog.mean(0)
    most_risen = np.argsort(-mean_dLog)[:12]
    r['most_risen_on_ablation'] = [(dec(t), round(float(mean_dLog[t]), 3)) for t in most_risen]
    r['suppressed_in_top12_risen'] = int(len(set(most_risen.tolist()) & set(supp_ids.tolist())))

    # discovered suppressed set should RISE; control ~ 0
    dsupp = dLog[:, supp_ids].mean(1); dctrl = dLog[:, CTRL].mean(1)
    r['dLogit_suppressed_mean'] = round(float(dsupp.mean()), 4); r['dLogit_suppressed_se'] = pse(dsupp)
    r['dLogit_control_mean'] = round(float(dctrl.mean()), 4); r['dLogit_control_se'] = pse(dctrl)
    diff = dsupp - dctrl
    r['supp_minus_control_mean'] = round(float(diff.mean()), 4)
    r['supp_minus_control_z'] = round(float(diff.mean()/(diff.std(ddof=1)/math.sqrt(len(diff))+1e-12)), 2)
    r['dLogit_allvocab_mean'] = round(float(mean_dLog.mean()), 4)

    # ANTI-COPY (heads): does the attended-SOURCE token itself rise when we ablate?
    if rec['kind'] == 'head':
        src_c = src[posb]; valid = (src_c >= 0) & ~np.isin(src_c, SPECIAL)
        if valid.sum() >= 5:
            d_src = dLog[np.arange(len(posb))[valid], src_c[valid]]
            r['anti_copy_src_dLogit_mean'] = round(float(d_src.mean()), 4)
            r['anti_copy_src_dLogit_se'] = pse(d_src)
            r['anti_copy_src_z'] = round(float(d_src.mean()/(d_src.std(ddof=1)/math.sqrt(len(d_src))+1e-12)), 2)
            r['n_anti_copy'] = int(valid.sum())
            # CE split: positions where the ACTUAL next token IS the attended source (genuine repeat)
            nxt_c = nxt[posb]
            rep = valid & (nxt_c == src_c)                 # repetition is CORRECT here
            norep = valid & (nxt_c != src_c) & (nxt_c >= 0) # repetition would be WRONG
            def ce_delta(msk):
                if msk.sum() < 3: return None, None, int(msk.sum())
                lb = torch.log_softmax(torch.from_numpy(Lbase[msk]), -1).numpy()
                la = torch.log_softmax(torch.from_numpy(Labl[msk]), -1).numpy()
                tg = nxt_c[msk]
                d = (-la[np.arange(len(tg)), tg]) - (-lb[np.arange(len(tg)), tg])  # CE_abl - CE_base
                return round(float(d.mean()), 4), pse(d), int(msk.sum())
            r['CE_delta_where_repeat_correct'], r['CE_delta_repeat_se'], r['n_repeat_correct'] = ce_delta(rep)
            r['CE_delta_where_repeat_wrong'], r['CE_delta_norepeat_se'], r['n_repeat_wrong'] = ce_delta(norep)

    # CE on positions whose ACTUAL next token is in the discovered suppressed set
    nxt_c = nxt[posb]; insupp = np.isin(nxt_c, supp_ids)
    r['n_next_in_suppressed'] = int(insupp.sum())
    if insupp.sum() >= 3:
        lb = torch.log_softmax(torch.from_numpy(Lbase[insupp]), -1).numpy()
        la = torch.log_softmax(torch.from_numpy(Labl[insupp]), -1).numpy()
        tg = nxt_c[insupp]
        d = (-la[np.arange(len(tg)), tg]) - (-lb[np.arange(len(tg)), tg])
        r['CE_delta_where_next_is_suppressed'] = round(float(d.mean()), 4)  # <0 => ablation HELPS (removing suppression right)
        r['CE_delta_next_supp_se'] = pse(d)
    else:
        r['CE_delta_where_next_is_suppressed'] = None
    # overall CE change on all causal positions (does the path help or hurt LM here?)
    validn = nxt_c >= 0
    lb = torch.log_softmax(torch.from_numpy(Lbase[validn]), -1).numpy()
    la = torch.log_softmax(torch.from_numpy(Labl[validn]), -1).numpy()
    tg = nxt_c[validn]
    d = (-la[np.arange(len(tg)), tg]) - (-lb[np.arange(len(tg)), tg])
    r['CE_delta_overall_causal'] = round(float(d.mean()), 4)  # >0 => ablation HURTS => path helps LM here
    r['CE_delta_overall_se'] = pse(d)
    return r


# choose top candidates to verify: prefer gated-in, non-bidirectional, highest score;
# but ALSO force-include the strongest anti-copy head (most negative source pctile) so the
# anti-repetition hypothesis is tested even if it doesn't top the purity board.
cands = [r for r in ranked if r['passes_mag_gate']]
verify_list = cands[:5]
head_ac = [r for r in cands if r['kind'] == 'head' and r['anti_copy_src_pctile'] is not None]
if head_ac:
    best_ac = min(head_ac, key=lambda r: r['anti_copy_src_pctile'])
    if best_ac['comp'] not in [x['comp'] for x in verify_list]:
        verify_list.append(best_ac)
# and the strongest self-suppression path
self_s = [r for r in cands if r['self_supp_cur_pctile'] is not None]
if self_s:
    best_ss = min(self_s, key=lambda r: r['self_supp_cur_pctile'])
    if best_ss['comp'] not in [x['comp'] for x in verify_list]:
        verify_list.append(best_ss)

print(f"\n===== CAUSAL VERIFICATION on held-out FW[448:600] ({len(verify_list)} paths) =====", flush=True)
vresults = []
for rec in verify_list:
    print(f"\n=== verifying {rec['comp']} (supp_pur={rec['suppress_purity']}, bidir={rec['bidirectional']}) ===", flush=True)
    vr = verify(rec); vresults.append(vr)
    print(json.dumps(vr, indent=1), flush=True)

out = {
    'meta': {
        'tool': 'qk_unsup_suppress -- unsupervised SUPPRESSION/inhibition circuit discovery',
        'model': 'bilin18', 'tokenizer': 'gpt2 (model exposes none) -- FLAGGED',
        'discovery_data': 'FineWeb FW[0:256,:128] TRAIN; causal verify on held-out FW[448:600,:128]',
        'n_head_paths': NL*NH, 'n_mlp_paths': NL*N_SVD, 'K_topk': K, 'M_suppress': M_EFF,
        'ranking_key': 'suppress_purity * mag_resid_frac (gated by mag pctile-25)',
        'suppress_purity': 'concentration (1 - norm entropy) of the bottom-M (most-negative) unit-dir logits',
        'anti_copy': 'source/current token percentile in the path own logit distribution (low => deep in suppressed tail)',
        'causal': 'mean-ablate at top-activating held-out positions; suppressed set should RISE (dLogit>0); '
                  'anti-copy: attended-source token rises; CE split by repeat-correct vs repeat-wrong',
        'honesty': 'boost_purity + suppress_dominance reported; bidirectional paths flagged (likely boost-head negative side)',
        'mag_gate_pctile25': round(gate, 4),
    },
    'ranking': ranked,
    'verification': vresults,
}
json.dump(out, open(f'{QK}/qk_unsup_suppress.json', 'w'), indent=2)
print("\nSAVED qk_unsup_suppress.json", flush=True)
print("QK UNSUP SUPPRESS DONE", flush=True)
