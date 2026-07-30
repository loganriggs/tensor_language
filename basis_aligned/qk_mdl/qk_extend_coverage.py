"""EXTEND COVERAGE, PART A -- characterize the causally-important-but-UNNAMED single paths.

Background: §67 (per-path causal importance, cleanliness), §68 (causal class-level
detector), §71 (coverage ledger: named ~44% of the SINGLE-PATH-expressible effect; the
rest is 55% single-path-expressible-but-UNNAMED, 82% of that in the hard low-cleanliness
region). This script pushes the NAMED fraction UP by running the top ~30 UNNAMED
causally-important paths through the full class-level battery and classifying each.

For each target path (unnamed, ranked by census trigger-position delta cross-entropy):
  1. trigger-position delta cross-entropy (recomputed; positive control vs the census).
  2. CLASS-SUMMED delta-logit signature (§68): pushed class + SIGN (push vs suppress),
     class-level concentration, top-8 class movements.
  3. output entropy of the mean delta-logit boost (distributed vs sharp) + top-token share.
  4. trigger CURRENT-token CLASS distribution + purity (what fires the path).
  5. for HEADS: attended-source-token copy signal (does ablation drop the source-token
     logit -> copy/induction) + source-token purity.
  6. CAUSAL VERIFICATION (§68): class-summed movement of the pushed class at FIRING vs a
     matched INACTIVE control, independent standard errors -> specificity.
  7. CLASSIFY: class-pusher / class-suppressor / copy-or-induction / sharp-token /
     positional-candidate / irreducibly-diffuse (fails the causal-clearness bar, §74).
  8. UPDATED single-path named fraction (GLOBAL delta cross-entropy scale, §71): the newly
     nameable paths' positive global delta cross-entropy added to the named numerator.

FORWARD + mean-ablation + class library copied VERBATIM from qk_unsup_classpush.py /
qk_census_difficulty.py. Held-back FW[448:600,:128]. Batch 6, GPU guard, <4GB footprint.
"""
import json, sys, math, time, subprocess
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot
torch.manual_seed(0)
DEV = 'cuda'; QK = '/workspace/tensor_language/basis_aligned/qk_mdl'

# ---------------- GPU GUARD (verbatim from census/classpush) ----------------
def gpu_guard(min_free=4500, tries=45, sleep=20):
    for _ in range(tries):
        free = int(subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits']
        ).decode().split('\n')[0].strip())
        if free >= min_free:
            print(f"GPU guard: {free} MiB free -- proceeding.", flush=True); return
        print(f"GPU guard: only {free} MiB free (<{min_free}); sleeping {sleep}s ...", flush=True)
        time.sleep(sleep)
    raise RuntimeError("GPU guard timed out waiting for free memory")
gpu_guard()

m, cfg = load_elriggs('bilin18')
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
V = cfg['vocab_size']; NL = len(m.transformer.h); N_SVD = 4
tok = AutoTokenizer.from_pretrained('gpt2')
def dec(t): return repr(tok.decode([int(t)]))
print(f"bilin18 NL={NL} NH={NH} HD={HD} D={D} V={V}; {NL*NH} head-paths + {NL*N_SVD} mlp-paths", flush=True)

FINEWEB = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))
SEQL = 128
TRAIN = FINEWEB[0:256, :SEQL].to(DEV)        # discovery slice -- ONLY to recompute MLP dirs (path defn)
HELD = FINEWEB[448:600, :SEQL].to(DEV)       # held-back verification slice
NHELD = HELD.shape[0]
BATCH = 6
KCAUSAL = 200                                # top-K activation positions per path (matches census)

# special/degenerate tokens excluded from trigger selection (matches census/classpush)
_special = {tok.eos_token_id}
for _t in range(V):
    if '�' in tok.decode([_t]): _special.add(_t)
SPECIAL = np.array(sorted(_special))
print(f"{len(SPECIAL)} special token ids masked from trigger selection", flush=True)

# =====================================================================================
# LEXICAL CLASS LIBRARY -- VERBATIM from qk_unsup_classpush.py (lex1 / VOCAB_CLASS).
# =====================================================================================
BRACKETS_OPEN  = set("([{<")
BRACKETS_CLOSE = set(")]}>")
QUOTE_OPEN     = set("“‘`")
QUOTE_CLOSE    = set("”’")
QUOTE_STRAIGHT = set("\"'")
PUNCT  = set(".,;:!?—–-…*|/\\~@#%^&+=_")
COORDINATORS = {"and","or","but","nor","yet","so"}
DETERMINERS  = {"the","a","an","this","that","these","those","some","any","each",
                "every","no","another","such"}
PRONOUNS     = {"i","we","you","he","she","it","they","them","us","me","him","her","which","who"}

def lex1(s):
    if s == "": return 'other'
    if ('�' in s) or (s == tok.eos_token or '<|endoftext|>' in s): return 'special'
    if '\n' in s: return 'newline'
    body = s.strip(); low = body.lower()
    if body == "": return 'other'
    if all(ch in QUOTE_OPEN for ch in body): return 'quote_open'
    if all(ch in QUOTE_CLOSE for ch in body): return 'quote_close'
    if all(ch in QUOTE_STRAIGHT for ch in body): return 'quote'
    if all(ch in BRACKETS_OPEN for ch in body): return 'bracket_open'
    if all(ch in BRACKETS_CLOSE for ch in body): return 'bracket_close'
    if any(ch.isdigit() for ch in body): return 'digit'
    if all((ch in PUNCT or ch in QUOTE_STRAIGHT or ch in QUOTE_OPEN or ch in QUOTE_CLOSE
            or ch in BRACKETS_OPEN or ch in BRACKETS_CLOSE) for ch in body): return 'punct'
    if low in DETERMINERS: return 'determiner'
    if low in COORDINATORS: return 'coordinator'
    if low in PRONOUNS: return 'pronoun'
    if body[0].isupper(): return 'capital'
    lead_space = s.startswith(' ')
    if lead_space and body.isalpha() and len(body) > 1: return 'word'
    if (not lead_space) and body.isalpha() and body[0].islower(): return 'subword'
    return 'other'

VOCAB_CLASS = np.array([lex1(tok.decode([t])) for t in range(V)], dtype=object)
CLASS_LIST = sorted(set(VOCAB_CLASS.tolist()))
CIDX = {c: i for i, c in enumerate(CLASS_LIST)}
CMAT = torch.zeros(len(CLASS_LIST), V, device=DEV)
for t in range(V):
    CMAT[CIDX[VOCAB_CLASS[t]], t] = 1.0
CLASS_SIZE = {c: int((VOCAB_CLASS == c).sum()) for c in CLASS_LIST}
PUSH_EXCLUDE = {'special'}
print(f"lexical classes ({len(CLASS_LIST)}): {CLASS_LIST}", flush=True)

# =====================================================================================
# MLP directions: recompute top-4 SVD dirs per block from the TRAIN gram (VERBATIM).
# =====================================================================================
gram = [torch.zeros(D, D, device=DEV) for _ in range(NL)]
@torch.no_grad()
def fwd_gram(idx):
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
        pat = (s1*s2).masked_fill(~mask, 0.0); yh4 = torch.einsum('bhqk,bkhd->bqhd', pat, v)
        x = x + a.c_proj(yh4.reshape(B, T, -1)); mo = blk.mlp(F.rms_norm(x, (D,)))
        gram[li] += torch.einsum('btd,bte->de', mo, mo); x = x + mo
print("Recomputing MLP SVD directions from TRAIN gram ...", flush=True)
for i in range(0, TRAIN.shape[0], BATCH): fwd_gram(TRAIN[i:i+BATCH])
mlp_dirs = torch.zeros(NL, N_SVD, D, device=DEV)
for li in range(NL):
    _evals, _evecs = torch.linalg.eigh(gram[li])
    mlp_dirs[li] = _evecs[:, -N_SVD:].T.flip(0)
del gram
print("MLP directions ready.", flush=True)

# =====================================================================================
# Core forward (VERBATIM from classpush) with single-path mean-ablation + collect.
# collect returns per-head hnorm AND per-head argmax-source token id (for copy check).
# =====================================================================================
@torch.no_grad()
def forward(idx, ablate=None, yhmeans=None, projmeans=None, collect=False):
    B, T = idx.shape
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    out = {} if collect else None
    for li in range(NL):
        blk = m.transformer.h[li]; x = blk.lambdas[0]*x + blk.lambdas[1]*x0; a = blk.attn; hcur = F.rms_norm(x, (D,))
        def qk(lin): z = F.rms_norm(lin(hcur).view(B, T, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
        v = a.c_v(hcur).view(B, T, NH, HD)
        if v1 is None: v1 = v
        v = (1-a.lamb)*v + a.lamb*v1.view_as(v)
        q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k)/HD; s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD
        pat = (s1*s2).masked_fill(~mask, 0.0)                 # (B,NH,T,T) UNNORMALISED
        yh4 = torch.einsum('bhqk,bkhd->bqhd', pat, v)         # (B,T,NH,HD)
        if collect:
            Wr = a.c_proj.weight.view(D, NH, HD)
            YH_SUM[li] += yh4.sum(0)
            comp = torch.einsum('bthc,ohc->btho', yh4, Wr).norm(dim=-1)  # (B,T,NH)
            out[('hnorm', li)] = comp.cpu().numpy()
            sp = pat.abs().argmax(-1).permute(0, 2, 1)        # (B,T,NH) argmax key pos
            srcid = torch.gather(idx.unsqueeze(-1).expand(B, T, NH), 1, sp.clamp(max=T-1))
            out[('src', li)] = srcid.cpu().numpy().astype(np.int32)   # (B,T,NH)
        if ablate is not None and ablate[0] == 'head' and ablate[1] == li:
            yh4 = yh4.clone(); yh4[:, :, ablate[2]] = yhmeans[li][:, ablate[2]].unsqueeze(0)
        x = x + a.c_proj(yh4.reshape(B, T, -1))
        mo = blk.mlp(F.rms_norm(x, (D,)))
        if collect:
            pr = torch.einsum('btd,nd->btn', mo, mlp_dirs[li])   # (B,T,N_SVD)
            PROJ_SUM[li] += pr.sum(0)
            out[('mproj', li)] = pr.cpu().numpy()
        if ablate is not None and ablate[0] == 'mlp' and ablate[1] == li:
            kk = ablate[2]
            pr = torch.einsum('btd,d->bt', mo, mlp_dirs[li, kk])
            mo = mo - (pr - projmeans[li][:, kk].unsqueeze(0)).unsqueeze(-1) * mlp_dirs[li, kk]
        x = x + mo
    logits = 30*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30)
    return (logits, out) if collect else logits

# =====================================================================================
# PASS A: per-position means + per-path activation magnitudes + per-head source tokens.
# =====================================================================================
YH_SUM = {li: torch.zeros(SEQL, NH, HD, device=DEV) for li in range(NL)}
PROJ_SUM = {li: torch.zeros(SEQL, N_SVD, device=DEV) for li in range(NL)}
head_act = np.zeros((NL*NH, NHELD, SEQL), np.float32)
mlp_act = np.zeros((NL*N_SVD, NHELD, SEQL), np.float32)
head_src = np.zeros((NL*NH, NHELD, SEQL), np.int32)
print("PASS A: collect activation magnitudes + per-position means + source tokens ...", flush=True)
for i in range(0, NHELD, BATCH):
    _, out = forward(HELD[i:i+BATCH], collect=True)
    b = HELD[i:i+BATCH].shape[0]
    for li in range(NL):
        hn = out[('hnorm', li)]; sr = out[('src', li)]
        for h in range(NH):
            head_act[li*NH + h, i:i+b] = hn[:, :, h]
            head_src[li*NH + h, i:i+b] = sr[:, :, h]
        pj = np.abs(out[('mproj', li)])
        for kk in range(N_SVD): mlp_act[li*N_SVD + kk, i:i+b] = pj[:, :, kk]
YHMEAN = {li: YH_SUM[li] / NHELD for li in range(NL)}
PROJMEAN = {li: PROJ_SUM[li] / NHELD for li in range(NL)}
del YH_SUM, PROJ_SUM
print("PASS A done.", flush=True)

held_np = HELD.cpu().numpy()
pos_t = np.tile(np.arange(SEQL), NHELD).reshape(NHELD, SEQL)
is_special = np.isin(held_np, SPECIAL)
valid_next = pos_t < (SEQL - 1)
bad_trigger = (pos_t == 0) | is_special | ~valid_next

PATHS = []
for li in range(NL):
    for h in range(NH): PATHS.append((f"h.L{li}.{h}", 'head', li, h))
for li in range(NL):
    for kk in range(N_SVD): PATHS.append((f"mlp.L{li}.d{kk}", 'mlp', li, kk))
PATH_META = {c: (kind, li, ix) for (c, kind, li, ix) in PATHS}

def act_of(kind, li, ix):
    return head_act[li*NH + ix] if kind == 'head' else mlp_act[li*N_SVD + ix]

# =====================================================================================
# NAMED SET (§71 coverage ledger, verbatim from qk_coverage_ledger.py NAMED_EXT) + census
# =====================================================================================
NAMED = sorted(set([
    'h.L8.3', 'h.L8.4', 'h.L13.0', 'h.L14.7', 'h.L7.3', 'h.L5.5', 'h.L8.7', 'h.L13.8',
    'h.L0.3', 'h.L1.1', 'h.L11.2', 'mlp.L17.d1', 'mlp.L17.d2', 'mlp.L17.d3', 'mlp.L16.d2',
    'mlp.L16.d0', 'mlp.L15.d2', 'mlp.L16.d1',
    'h.L9.6', 'h.L6.0', 'h.L3.3', 'h.L8.2', 'h.L6.7', 'h.L4.0', 'h.L5.7', 'mlp.L1.d3',
]))
census = json.load(open(f'{QK}/qk_census_difficulty.json'))
cen_rec = {r['comp']: r for r in census['records']}
# global-dCE single-path denominators (§71 scale): sum of positive global_dCE over all 234
GLOBAL_DENOM = sum(max(cen_rec[c]['global_dCE'], 0.0) for c in cen_rec)
NAMED_OLD_NUM = sum(max(cen_rec[c]['global_dCE'], 0.0) for c in NAMED)
print(f"single-path denom (sum positive global dCE) = {GLOBAL_DENOM:.4f}; "
      f"named-old numerator = {NAMED_OLD_NUM:.4f} ({NAMED_OLD_NUM/GLOBAL_DENOM:.3f})", flush=True)

# TARGETS: top-30 UNNAMED by census trigger-position delta cross-entropy
unnamed_ranked = [r for r in census['records'] if r['comp'] not in NAMED]  # census sorted by -trigger_dCE
TARGETS = [r['comp'] for r in unnamed_ranked[:30]]
print(f"PART A targets ({len(TARGETS)} unnamed, by trigger dCE): {TARGETS}", flush=True)

# top-KCAUSAL activation trigger mask + matched INACTIVE control mask (VERBATIM classpush)
trig_mask = {}; ctrl_mask = {}
for comp in TARGETS:
    kind, li, ix = PATH_META[comp]
    a = act_of(kind, li, ix).copy().reshape(-1); a[bad_trigger.reshape(-1)] = -1e30
    tk = np.argpartition(a, -KCAUSAL)[-KCAUSAL:]
    mk = np.zeros(NHELD*SEQL, bool); mk[tk] = True
    trig_mask[comp] = mk.reshape(NHELD, SEQL)
    a2 = act_of(kind, li, ix).copy().reshape(-1); a2[bad_trigger.reshape(-1)] = 1e30
    bk = np.argpartition(a2, KCAUSAL)[:KCAUSAL]
    mc = np.zeros(NHELD*SEQL, bool); mc[bk] = True
    ctrl_mask[comp] = mc.reshape(NHELD, SEQL)

def stats(s, sq, n):
    if n <= 1: return 0.0, 0.0
    mean = s/n; var = max(sq/n - mean*mean, 0.0)*n/(n-1)
    return mean, math.sqrt(var/n)

def conc_ids(ids):
    ids = np.asarray(ids)
    if len(ids) <= 1: return 0.0
    _, c = np.unique(ids, return_counts=True); p = c/c.sum()
    return float(1 - (-(p*np.log(p)).sum())/math.log(len(ids)))

# =====================================================================================
# PASS B: per target path -- ablate; accumulate trigger dCE, mean delta-logit at firing
# (full vocab), per-position class-summed movement at FIRING and at CONTROL, source copy.
# =====================================================================================
t_sum = {c: 0.0 for c in TARGETS}; t_sq = {c: 0.0 for c in TARGETS}; t_n = {c: 0 for c in TARGETS}
t_pos = {c: 0 for c in TARGETS}
g_sum = {c: 0.0 for c in TARGETS}; g_sq = {c: 0.0 for c in TARGETS}; g_n = {c: 0 for c in TARGETS}
dl_fire_sum = {c: torch.zeros(V, device=DEV) for c in TARGETS}    # sum delta-logit over firing positions
nfire = {c: 0 for c in TARGETS}
cs_fire = {c: [] for c in TARGETS}     # per-position class-summed movement at firing (n, n_classes)
cs_ctrl = {c: [] for c in TARGETS}     # per-position class-summed movement at control
src_dl = {c: [] for c in TARGETS}      # per firing position: delta-logit at attended source token (heads)
src_tok = {c: [] for c in TARGETS}     # attended source token at firing (heads)

tgt_all = torch.from_numpy(held_np).to(DEV)
print(f"PASS B: {len(TARGETS)} single-path ablations (dCE + class movement + copy) ...", flush=True)
t0 = time.time()
for bi, i in enumerate(range(0, NHELD, BATCH)):
    sb = slice(i, min(i+BATCH, NHELD))
    idx = HELD[sb]; b = idx.shape[0]
    base = forward(idx).float()                            # (b,T,V)
    tgt = tgt_all[sb]
    logp = F.log_softmax(base[:, :SEQL-1], dim=-1)
    base_ce = -logp.gather(-1, tgt[:, 1:].unsqueeze(-1)).squeeze(-1)   # (b,T-1)
    del logp
    vmask = torch.from_numpy(valid_next[sb, :SEQL-1]).to(DEV)
    for comp in TARGETS:
        kind, li, ix = PATH_META[comp]
        abl = forward(idx, ablate=(kind, li, ix), yhmeans=YHMEAN, projmeans=PROJMEAN).float()
        alogp = F.log_softmax(abl[:, :SEQL-1], dim=-1)
        abl_ce = -alogp.gather(-1, tgt[:, 1:].unsqueeze(-1)).squeeze(-1)
        del alogp
        dce = (abl_ce - base_ce)                           # (b,T-1) ablated - base
        dg = dce[vmask]
        g_sum[comp] += float(dg.sum()); g_sq[comp] += float((dg*dg).sum()); g_n[comp] += int(dg.numel())
        # FIRING trigger positions (query t<SEQL-1)
        tm = torch.from_numpy(trig_mask[comp][sb, :SEQL-1]).to(DEV)
        if tm.any():
            dt = dce[tm]
            t_sum[comp] += float(dt.sum()); t_sq[comp] += float((dt*dt).sum())
            t_n[comp] += int(dt.numel()); t_pos[comp] += int((dt > 0).sum())
            dl = (base[:, :SEQL-1] - abl[:, :SEQL-1])[tm]  # (nf,V) positive => path boosts
            dl_fire_sum[comp] += dl.sum(0)
            nfire[comp] += int(dl.shape[0])
            cs_fire[comp].append((CMAT @ dl.T).T.cpu().numpy())   # (nf,n_classes)
            if kind == 'head':
                # attended-source token at these firing positions + delta-logit at that token
                fm_np = trig_mask[comp][sb, :SEQL-1]
                rows, cols = np.where(fm_np)
                st = head_src[li*NH + ix][sb][rows, cols]         # source token ids (nf,)
                st_t = torch.from_numpy(st.astype(np.int64)).to(DEV)
                src_dl[comp].append(dl.gather(-1, st_t.unsqueeze(-1)).squeeze(-1).cpu().numpy())
                src_tok[comp].append(st)
        # CONTROL inactive positions
        cmask = torch.from_numpy(ctrl_mask[comp][sb, :SEQL-1]).to(DEV)
        if cmask.any():
            dlc = (base[:, :SEQL-1] - abl[:, :SEQL-1])[cmask]
            cs_ctrl[comp].append((CMAT @ dlc.T).T.cpu().numpy())
        del abl, dce
    if bi % 4 == 0:
        print(f"  batch {bi+1}/{(NHELD+BATCH-1)//BATCH}  elapsed {time.time()-t0:.0f}s", flush=True)
    del base, base_ce
print(f"PASS B done in {time.time()-t0:.0f}s", flush=True)

# =====================================================================================
# ASSEMBLE + CLASSIFY
# =====================================================================================
DCE_Z_BAR = 3.0; SPEC_Z_BAR = 3.0; SHARP_TOP_SHARE = 0.35
records = []
newly_nameable = []
for comp in TARGETS:
    kind, li, ix = PATH_META[comp]
    tm_, tse = stats(t_sum[comp], t_sq[comp], t_n[comp])
    gm, _gse = stats(g_sum[comp], g_sq[comp], g_n[comp])
    tz = tm_/tse if tse > 0 else 0.0
    csf = np.concatenate(cs_fire[comp], 0) if cs_fire[comp] else np.zeros((0, len(CLASS_LIST)))
    csc = np.concatenate(cs_ctrl[comp], 0) if cs_ctrl[comp] else np.zeros((0, len(CLASS_LIST)))
    mean_cs = csf.mean(0) if len(csf) else np.zeros(len(CLASS_LIST))    # mean class-summed movement
    abs_total = float(np.abs(mean_cs).sum())
    order = np.argsort(-np.abs(mean_cs))
    pushed = next((j for j in order if CLASS_LIST[j] not in PUSH_EXCLUDE), int(order[0]))
    push_class = CLASS_LIST[pushed]; push_val = float(mean_cs[pushed])
    concentration = float(abs(mean_cs[pushed]) / abs_total) if abs_total > 0 else 0.0
    class_top8 = {CLASS_LIST[j]: round(float(mean_cs[j]), 3) for j in order[:8]}
    # output distribution (mean delta-logit boost)
    dmean = (dl_fire_sum[comp] / max(1, nfire[comp])).cpu().numpy()
    pos = np.clip(dmean, 0, None); Z = pos.sum()
    if Z > 0:
        p = pos / Z; nz = p[p > 0]
        ent = float(-(nz*np.log(nz)).sum()); top_share = float(p.max())
    else:
        ent = 0.0; top_share = 0.0
    ent_norm = ent / math.log(V)
    top_boost_ids = np.argsort(-dmean)[:6]
    top_boost = [(dec(int(t)), round(float(dmean[t]), 3)) for t in top_boost_ids]
    # trigger current-token class distribution
    cur_ids = held_np[trig_mask[comp]]
    cur_cls = VOCAB_CLASS[cur_ids]
    cc, ccn = np.unique(cur_cls, return_counts=True)
    cord = np.argsort(-ccn)
    trig_cls_top = [(str(cc[j]), int(ccn[j])) for j in cord[:5]]
    trig_cls_purity = round(conc_ids(cur_ids), 4)
    # source copy (heads)
    copy_src_dlogit = None; copy_src_z = None; copy_src_purity = None
    if kind == 'head' and src_dl[comp]:
        sd = np.concatenate(src_dl[comp]); stall = np.concatenate(src_tok[comp])
        copy_src_dlogit = round(float(sd.mean()), 4)
        copy_src_z = round(float(sd.mean()/(sd.std(ddof=1)/math.sqrt(len(sd)))), 2) if len(sd) > 1 and sd.std() > 0 else 0.0
        copy_src_purity = round(conc_ids(stall), 4)
    # VERIFICATION: pushed-class movement at firing vs control (independent SE, §68)
    fvec = csf[:, pushed] if len(csf) else np.zeros(0)
    cvec = csc[:, pushed] if len(csc) else np.zeros(0)
    def msd(v): n = len(v); return (float(v.mean()) if n else 0.0,
                                    float(v.std(ddof=1)/math.sqrt(n)) if n > 1 else 0.0, n)
    fmean, fse, fn = msd(fvec); cmean, cse, cn = msd(cvec)
    spec = fmean - cmean; spec_se = math.sqrt(fse*fse + cse*cse)
    fz = fmean/(fse+1e-9); sz = spec/(spec_se+1e-9)

    # ---- CLASSIFY ----
    # causal-clearness bar: the path's own trigger delta cross-entropy must be clear (z>=3)
    causal_clear = tz >= DCE_Z_BAR
    cls = None; nameable = False
    if not causal_clear:
        cls = 'irreducibly-diffuse'      # fails the causal-clearness bar (like the §74 MLP1 tail)
    else:
        # copy/induction: source-token strongly boosted, diverse sources
        if kind == 'head' and copy_src_dlogit is not None and copy_src_dlogit > 0.5 \
                and copy_src_z is not None and copy_src_z >= 3.0 and (copy_src_purity is None or copy_src_purity < 0.5):
            cls = 'copy-or-induction'; nameable = True
        # sharp-token booster: output concentrated on a small token set
        elif top_share >= SHARP_TOP_SHARE:
            cls = 'sharp-token'; nameable = True
        # class-pusher: pushed-class movement positive + specific vs control
        elif push_val > 0 and fz >= SPEC_Z_BAR and sz >= SPEC_Z_BAR:
            cls = f'class-pusher:{push_class}'; nameable = True
        # class-suppressor: pushed-class movement negative + specific vs control
        elif push_val < 0 and fz <= -SPEC_Z_BAR and sz <= -SPEC_Z_BAR:
            cls = f'class-suppressor:{push_class}'; nameable = True
        else:
            # causally clear but no clean class/copy/sharp signature -> distributed but real
            cls = 'positional-or-diffuse'   # needs the §62 positional tool; not named here
    rec = {
        'comp': comp, 'kind': kind, 'li': li, 'idx': ix,
        'cleanliness': cen_rec[comp]['cleanliness'],
        'trigger_dCE': round(tm_, 5), 'trigger_dCE_SE': round(tse, 5), 'trigger_dCE_z': round(tz, 2),
        'census_trigger_dCE': cen_rec[comp]['trigger_dCE'],
        'global_dCE': round(gm, 6), 'census_global_dCE': cen_rec[comp]['global_dCE'],
        'trigger_frac_positive': round(t_pos[comp]/max(1, t_n[comp]), 3),
        'pushed_class': push_class, 'pushed_class_movement': round(push_val, 3),
        'pushed_class_sign': '+' if push_val >= 0 else '-',
        'class_concentration': round(concentration, 4), 'class_summed_top8': class_top8,
        'output_entropy_nats': round(ent, 3), 'output_entropy_norm': round(ent_norm, 4),
        'output_top_token_share': round(top_share, 4), 'top_boost': top_boost,
        'trigger_class_top': trig_cls_top, 'trigger_class_purity': trig_cls_purity,
        'copy_src_dlogit': copy_src_dlogit, 'copy_src_z': copy_src_z, 'copy_src_purity': copy_src_purity,
        'verify_pushclass_firing': round(fmean, 3), 'verify_pushclass_firing_SE': round(fse, 3),
        'verify_pushclass_firing_z': round(fz, 2),
        'verify_pushclass_control': round(cmean, 3),
        'verify_specificity': round(spec, 3), 'verify_specificity_SE': round(spec_se, 3),
        'verify_specificity_z': round(sz, 2),
        'causal_clear_z3': bool(causal_clear),
        'classification': cls, 'newly_nameable': bool(nameable),
    }
    records.append(rec)
    if nameable:
        newly_nameable.append(comp)
    print(f"  {comp:11s} tdCE={tm_:+.3f}(z{tz:.1f}) push={rec['pushed_class_sign']}{push_class:8s} "
          f"fireZ={fz:+.1f} specZ={sz:+.1f} ent={ent_norm:.2f} topshare={top_share:.3f} "
          f"copy={copy_src_dlogit} -> {cls}", flush=True)

# positive control: recomputed trigger dCE matches census
mad = [abs(r['trigger_dCE'] - r['census_trigger_dCE']) for r in records]
print(f"\nPOSITIVE CONTROL: recomputed vs census trigger dCE max abs diff = {max(mad):.5f}", flush=True)

# =====================================================================================
# UPDATED single-path named fraction (GLOBAL delta cross-entropy scale, §71)
# =====================================================================================
newly_num = sum(max(cen_rec[c]['global_dCE'], 0.0) for c in newly_nameable)
named_new_num = NAMED_OLD_NUM + newly_num
by_class = {}
for r in records:
    key = r['classification'].split(':')[0] if r['newly_nameable'] else r['classification']
    if r['newly_nameable']:
        key = r['classification']
    by_class.setdefault(key, []).append(r['comp'])
name_type_counts = {}
for r in records:
    if r['newly_nameable']:
        t = r['classification'].split(':')[0]
        name_type_counts[t] = name_type_counts.get(t, 0) + 1

summary = {
    'n_targets': len(TARGETS), 'targets': TARGETS,
    'n_newly_nameable': len(newly_nameable), 'newly_nameable': newly_nameable,
    'newly_nameable_types': name_type_counts,
    'classification_counts': {k: len(v) for k, v in by_class.items()},
    'single_path_scale': 'GLOBAL mean-ablation delta cross-entropy, positive part, summed (matches §71)',
    'single_path_denom': round(GLOBAL_DENOM, 4),
    'named_old_numerator': round(NAMED_OLD_NUM, 4),
    'named_old_fraction': round(NAMED_OLD_NUM/GLOBAL_DENOM, 4),
    'newly_nameable_numerator_added': round(newly_num, 4),
    'named_new_numerator': round(named_new_num, 4),
    'named_new_fraction': round(named_new_num/GLOBAL_DENOM, 4),
    'positive_control_recompute_vs_census_maxabsdiff': round(float(max(mad)), 5),
    'bar': {'causal_clear_trigger_dCE_z': DCE_Z_BAR, 'specificity_z': SPEC_Z_BAR,
            'sharp_token_top_share': SHARP_TOP_SHARE},
}
out = {
    'meta': {
        'model': 'bilin18', 'part': 'A -- extend single-path naming', 'held_slice': 'FW[448:600,:128]',
        'KCAUSAL': KCAUSAL, 'BATCH': BATCH, 'n_classes': len(CLASS_LIST), 'classes': CLASS_LIST,
        'named_set_size': len(NAMED),
        'method': 'Top-30 UNNAMED paths (by census trigger-position delta cross-entropy) through the '
                  'full class-level battery: trigger delta cross-entropy (recomputed), class-summed '
                  'delta-logit signature + SIGN, output entropy, trigger-class distribution, source-copy '
                  '(heads), and firing-vs-inactive-control specificity of the pushed class. Forward + '
                  'mean-ablation + class library VERBATIM from qk_unsup_classpush / qk_census_difficulty.',
        'classify': 'newly-nameable if causally clear (own trigger dCE z>=3) AND a clean output '
                    'signature: copy-or-induction (source-token boosted, diverse sources), sharp-token '
                    '(top-share>=0.35), class-pusher / class-suppressor (pushed-class movement specific vs '
                    'control, |z|>=3). causally-clear-but-no-clean-signature = positional-or-diffuse '
                    '(needs §62); fails causal-clearness = irreducibly-diffuse (§74).',
    },
    'summary': summary,
    'records': sorted(records, key=lambda r: -r['trigger_dCE']),
}
json.dump(out, open(f'{QK}/qk_extend_coverage.json', 'w'), indent=2)

print("\n===== PART A SUMMARY =====", flush=True)
print(f"targets: {len(TARGETS)} unnamed causally-important paths", flush=True)
print(f"newly nameable: {len(newly_nameable)} {newly_nameable}", flush=True)
print(f"types: {name_type_counts}", flush=True)
print(f"classification counts: {summary['classification_counts']}", flush=True)
print(f"named-old single-path fraction: {summary['named_old_fraction']} "
      f"({NAMED_OLD_NUM:.4f}/{GLOBAL_DENOM:.4f})", flush=True)
print(f"named-NEW single-path fraction: {summary['named_new_fraction']} "
      f"({named_new_num:.4f}/{GLOBAL_DENOM:.4f})  [+{newly_num:.4f} nats added]", flush=True)
print("Saved qk_extend_coverage.json", flush=True)
print("QK EXTEND COVERAGE PART A DONE", flush=True)
