"""DEPTH-FIRST ALGORITHM ARC #5 -- WHAT DOES HEAD h.L7.0 DO?

Full toolbox battery on the largest completely uncharacterized attention head
(census global mean-ablation delta cross-entropy ~0.017; census fingerprint only).
Layer 7 sits in the DISTRIBUTED region of the flow map.

All forwards + mean-ablation copied VERBATIM from qk_census_difficulty.py /
qk_unsup_classpush.py / qk_unsup_copy.py / qk_unsup_discover.py. Held-back
FW[448:600,:128]. Paired standard errors. Batch<=6. GPU guard.

Outputs qk_arc_h70.json.
"""
import json, sys, math, time, subprocess
import numpy as np
import torch
import torch.nn.functional as F
from collections import Counter
from transformers import AutoTokenizer
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot
torch.manual_seed(0)
DEV = 'cuda'; QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
LI_T, H_T = 7, 0                                   # TARGET head h.L7.0

# ---------------- GPU GUARD (verbatim from census) ----------------
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
V = cfg['vocab_size']; NL = len(m.transformer.h)
tok = AutoTokenizer.from_pretrained('gpt2')
def dec(t): return repr(tok.decode([int(t)]))
Wu = m.lm_head.weight.detach().float()             # (V,D) unembedding
print(f"bilin18 NL={NL} NH={NH} HD={HD} D={D} V={V}; target h.L{LI_T}.{H_T}", flush=True)

FINEWEB = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))
SEQL = 128
HELD = FINEWEB[448:600, :SEQL].to(DEV)             # held-back verification slice
NHELD = HELD.shape[0]
BATCH = 6
KCAUSAL = 200

_special = {tok.eos_token_id}
for _t in range(V):
    if '�' in tok.decode([_t]): _special.add(_t)
SPECIAL = np.array(sorted(_special))
print(f"{len(SPECIAL)} special token ids masked from trigger selection", flush=True)

# ------- LEXICAL CLASS LIBRARY -- VERBATIM from qk_unsup_classpush.py (lex1) -------
BRACKETS_OPEN  = set("([{<"); BRACKETS_CLOSE = set(")]}>")
QUOTE_OPEN = set("“‘`"); QUOTE_CLOSE = set("”’"); QUOTE_STRAIGHT = set("\"'")
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
for t in range(V): CMAT[CIDX[VOCAB_CLASS[t]], t] = 1.0
print(f"lexical classes ({len(CLASS_LIST)}): {CLASS_LIST}", flush=True)

# =====================================================================================
# Core forward (VERBATIM from qk_census_difficulty.py) with single-head mean-ablation of
# the TARGET head, plus (a) collect for target head stats, (b) capture of the post-attn
# residual at layer LI_T and the pre-final-norm residual (for the direct-vs-mediated split).
# =====================================================================================
@torch.no_grad()
def forward(idx, ablate_target=False, yhmean_t=None, collect=False, capture=False):
    B, T = idx.shape
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    out = {}
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
        if collect and li == LI_T:
            Wr = a.c_proj.weight.view(D, NH, HD)
            comp = torch.einsum('bthc,ohc->btho', yh4[:, :, H_T:H_T+1], Wr[:, H_T:H_T+1, :])  # (B,T,1,D)
            rh = comp[:, :, 0, :]                             # (B,T,D) target head residual write
            out['hnorm'] = rh.norm(dim=-1).cpu().numpy()      # (B,T)
            out['rh'] = rh                                    # keep on device
            out['srcpos'] = pat[:, H_T].abs().argmax(-1).cpu().numpy()   # (B,T)
            YH_SUM[:, H_T] += yh4[:, :, H_T].sum(0)           # (T,HD)
        if ablate_target and li == LI_T:
            yh4 = yh4.clone(); yh4[:, :, H_T] = yhmean_t.unsqueeze(0)
        x = x + a.c_proj(yh4.reshape(B, T, -1))
        if capture and li == LI_T:
            out['a7'] = x.clone()                             # post-attention residual at layer 7
        mo = blk.mlp(F.rms_norm(x, (D,)))
        x = x + mo
    xf = x                                                    # pre-final-norm residual
    if capture: out['xf'] = xf.clone()
    logits = 30*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30)
    return logits, out

# =====================================================================================
# PASS A: target-head norm + argmax source + per-position yh mean (census-identical).
# =====================================================================================
YH_SUM = torch.zeros(SEQL, NH, HD, device=DEV)
head_act = np.zeros((NHELD, SEQL), np.float32)
src_pos = np.zeros((NHELD, SEQL), np.int32)
rh_store = np.zeros((NHELD, SEQL, D), np.float32)             # target head residual write per position
print("PASS A: target-head norm + source + yh mean ...", flush=True)
for i in range(0, NHELD, BATCH):
    _, out = forward(HELD[i:i+BATCH], collect=True)
    b = out['hnorm'].shape[0]
    head_act[i:i+b] = out['hnorm']; src_pos[i:i+b] = out['srcpos']
    rh_store[i:i+b] = out['rh'].cpu().numpy()
YHMEAN_T = (YH_SUM[:, H_T] / NHELD)                           # (T,HD) per-position mean, target head
print("PASS A done.", flush=True)

held_np = HELD.cpu().numpy()
pos_t = np.tile(np.arange(SEQL), NHELD).reshape(NHELD, SEQL)
is_special = np.isin(held_np, SPECIAL)
valid_next = pos_t < (SEQL - 1)                               # census GLOBAL validity
bad_trigger = (pos_t == 0) | is_special | ~valid_next

# top-KCAUSAL activation trigger mask + matched inactive control (VERBATIM census/classpush)
a = head_act.copy().reshape(-1); a[bad_trigger.reshape(-1)] = -1e30
tk = np.argpartition(a, -KCAUSAL)[-KCAUSAL:]
trig_mask = np.zeros(NHELD*SEQL, bool); trig_mask[tk] = True; trig_mask = trig_mask.reshape(NHELD, SEQL)
a2 = head_act.copy().reshape(-1); a2[bad_trigger.reshape(-1)] = 1e30
bk = np.argpartition(a2, KCAUSAL)[:KCAUSAL]
ctrl_mask = np.zeros(NHELD*SEQL, bool); ctrl_mask[bk] = True; ctrl_mask = ctrl_mask.reshape(NHELD, SEQL)

# =====================================================================================
# FINGERPRINT (held-back) at top firing positions
# =====================================================================================
fseq, fpos = np.where(trig_mask)
cur_tok = held_np[fseq, fpos]
srcp = src_pos[fseq, fpos]
src_tok = held_np[fseq, srcp]
offset = (fpos - srcp)
def class_dist(ids):
    c = Counter(lex1(tok.decode([int(t)])) for t in ids)
    n = len(ids)
    return {k: round(v/n, 3) for k, v in c.most_common()}
fp_cur_classes = class_dist(cur_tok)
fp_src_classes = class_dist(src_tok)
def top_counts(ids, n=8):
    v, c = np.unique(np.asarray(ids), return_counts=True)
    o = np.argsort(-c)[:n]
    return [(dec(int(v[i])), int(c[i])) for i in o]
# offset profile
off_hist = Counter(int(o) for o in offset)
off_profile = {str(k): v for k, v in sorted(off_hist.items())}
def conc(ids):
    ids = np.asarray(ids)
    if len(ids) <= 1: return 0.0
    _, c = np.unique(ids, return_counts=True); p = c/c.sum()
    return float(1 - (-(p*np.log(p)).sum())/math.log(len(ids)))
offset_conc = conc(offset); self_frac = float((offset == 0).mean()); prev_frac = float((offset == 1).mean())
sink_frac = float((srcp == 0).mean())

# head effect direction = mean r_h over firing positions -> unembed boost/suppress
eff_dir = torch.from_numpy(rh_store[fseq, fpos].mean(0)).to(DEV)   # (D,)
eff_unit = eff_dir / (eff_dir.norm() + 1e-9)
eff_logits = (Wu.to(DEV) @ eff_unit).float(); eff_logits = eff_logits - eff_logits.mean()
boost_top = torch.topk(eff_logits, 8).indices.cpu().numpy().tolist()
suppress_top = torch.topk(-eff_logits, 8).indices.cpu().numpy().tolist()
fp_boost = [dec(t) for t in boost_top]; fp_suppress = [dec(t) for t in suppress_top]

# =====================================================================================
# COPY TEST (held-back) -- per-position DLA alignment of r_h with SOURCE / SUCCESSOR
# token vs vocab distribution (VERBATIM copy machinery restricted to target head).
# =====================================================================================
Wu_dev = Wu.to(DEV)
succp = np.clip(srcp + 1, 0, SEQL-1)
succ_tok = held_np[fseq, succp]
rh_fire = torch.from_numpy(rh_store[fseq, fpos]).to(DEV)      # (Nfire,D)
dla = rh_fire @ Wu_dev.T                                      # (Nfire,V)
mu = dla.mean(-1, keepdim=True); sd = dla.std(-1, keepdim=True); z = (dla - mu)/sd
srcz = float(z[torch.arange(len(fseq)), torch.from_numpy(src_tok.astype(np.int64)).to(DEV)].mean())
sucz = float(z[torch.arange(len(fseq)), torch.from_numpy(succ_tok.astype(np.int64)).to(DEV)].mean())
top10 = torch.topk(dla, 10, dim=-1).indices.cpu().numpy()
verb = np.array([src_tok[j] in top10[j] for j in range(len(fseq))])
succ = np.array([succ_tok[j] in top10[j] for j in range(len(fseq))])
srank = np.array([int((dla[j] > dla[j, int(src_tok[j])]).sum().item()) for j in range(len(fseq))])
copy_test = {
    'copy_purity': round(float((verb | succ).mean()), 3),
    'verbatim_purity': round(float(verb.mean()), 3),
    'successor_purity': round(float(succ.mean()), 3),
    'src_logit_z': round(srcz, 3), 'succ_logit_z': round(sucz, 3),
    'src_rank_median': int(np.median(srank)),
    'offset_median': int(np.median(offset)), 'offset_conc': round(offset_conc, 3),
    'self_frac': round(self_frac, 3), 'prev_frac': round(prev_frac, 3), 'sink_frac': round(sink_frac, 3),
}
del dla, z
print("fingerprint + copy test done.", flush=True)

# =====================================================================================
# PASS B: mean-ablate target head over HELD. Accumulate GLOBAL + TRIGGER dCE (paired),
# DIRECT-only dCE (direct-vs-mediated), CLASS-SUMMED delta-logit (classpush), dCE by class,
# and per-firing-position records for concrete examples.
# =====================================================================================
g_sum=g_sq=g_n=0.0; g_n=0
t_sum=t_sq=0.0; t_n=0; t_pos=0
d_sum=d_sq=0.0; d_n=0                                         # DIRECT-only dCE at trigger positions
ft_sum=ft_sq=0.0; ft_n=0                                     # FULL dCE at trigger positions (paired w/ direct)
classsum = torch.zeros(len(CLASS_LIST), device=DEV); nfire = 0
classsum_ctrl = torch.zeros(len(CLASS_LIST), device=DEV); nctrl = 0
# dCE by current-token class over ALL valid positions
cls_dce = {c: [0.0, 0] for c in CLASS_LIST}
he_curcls = np.array([[CIDX[VOCAB_CLASS[held_np[s, p]]] for p in range(SEQL)] for s in range(NHELD)])
ex_records = []                                              # for concrete examples

tgt_all = torch.from_numpy(held_np).to(DEV)
print("PASS B: mean-ablation (global/trigger/direct dCE + class push) ...", flush=True)
t0 = time.time()
for bi, i in enumerate(range(0, NHELD, BATCH)):
    sb = slice(i, min(i+BATCH, NHELD)); idx = HELD[sb]; b = idx.shape[0]
    base, capb = forward(idx, capture=True)
    base = base.float()
    abl, capa = forward(idx, ablate_target=True, yhmean_t=YHMEAN_T, capture=True)
    abl = abl.float()
    # DIRECT-only logits: propagate the layer-7 post-attn residual change straight to logits
    delta7 = capa['a7'] - capb['a7']
    direct_res = capb['xf'] + delta7
    direct = (30*torch.tanh(m.lm_head(F.rms_norm(direct_res, (D,)))/30)).float()
    tgt = tgt_all[sb]
    logp = F.log_softmax(base[:, :SEQL-1], -1)
    base_ce = -logp.gather(-1, tgt[:, 1:].unsqueeze(-1)).squeeze(-1)
    alogp = F.log_softmax(abl[:, :SEQL-1], -1)
    abl_ce = -alogp.gather(-1, tgt[:, 1:].unsqueeze(-1)).squeeze(-1)
    dlogp = F.log_softmax(direct[:, :SEQL-1], -1)
    dir_ce = -dlogp.gather(-1, tgt[:, 1:].unsqueeze(-1)).squeeze(-1)
    dce = abl_ce - base_ce                                    # (b,T-1)
    dce_dir = dir_ce - base_ce
    vmask = torch.from_numpy(valid_next[sb, :SEQL-1]).to(DEV)
    dg = dce[vmask]
    g_sum += float(dg.sum()); g_sq += float((dg*dg).sum()); g_n += int(dg.numel())
    # dCE by current class (all valid)
    curcls_b = he_curcls[sb, :SEQL-1]
    dce_np = dce.cpu().numpy(); vm_np = valid_next[sb, :SEQL-1]
    for c in range(len(CLASS_LIST)):
        msk = (curcls_b == c) & vm_np
        if msk.any():
            cls_dce[CLASS_LIST[c]][0] += float(dce_np[msk].sum()); cls_dce[CLASS_LIST[c]][1] += int(msk.sum())
    # trigger positions
    tm = torch.from_numpy(trig_mask[sb, :SEQL-1]).to(DEV)
    if tm.any():
        dt = dce[tm]; dd = dce_dir[tm]
        t_sum += float(dt.sum()); t_sq += float((dt*dt).sum()); t_n += int(dt.numel()); t_pos += int((dt>0).sum())
        ft_sum += float(dt.sum()); ft_sq += float((dt*dt).sum()); ft_n += int(dt.numel())
        d_sum += float(dd.sum()); d_sq += float((dd*dd).sum()); d_n += int(dd.numel())
        dl = (base[:, :SEQL-1] - abl[:, :SEQL-1])[tm]         # (nfire,V) base-ablated = what head adds
        classsum += CMAT @ dl.sum(0); nfire += int(dl.shape[0])
    # control positions (inactive) for class-push specificity
    cm = torch.from_numpy(ctrl_mask[sb, :SEQL-1]).to(DEV)
    if cm.any():
        dlc = (base[:, :SEQL-1] - abl[:, :SEQL-1])[cm]
        classsum_ctrl += CMAT @ dlc.sum(0); nctrl += int(dlc.shape[0])
    # concrete-example records at trigger positions in this batch
    tmn = trig_mask[sb, :SEQL-1]
    for bb in range(b):
        for pp in np.where(tmn[bb])[0]:
            gp = i + bb
            ex_records.append({'seq': int(gp), 'pos': int(pp),
                               'dce': float(dce[bb, pp].item()),
                               'dce_dir': float(dce_dir[bb, pp].item())})
    del base, abl, direct, capa, capb, delta7, direct_res
    if bi % 5 == 0:
        print(f"  batch {bi+1}/{(NHELD+BATCH-1)//BATCH} elapsed {time.time()-t0:.0f}s", flush=True)
print(f"PASS B done in {time.time()-t0:.0f}s", flush=True)

def stat(s, sq, n):
    if n <= 1: return 0.0, 0.0
    mean = s/n; var = max(sq/n - mean*mean, 0.0)*n/(n-1)
    return mean, math.sqrt(var/n)
gm, gse = stat(g_sum, g_sq, g_n)
tm_, tse = stat(t_sum, t_sq, t_n)
dm, dse = stat(d_sum, d_sq, d_n)
fm, fse = stat(ft_sum, ft_sq, ft_n)

# class-push assembly (VERBATIM logic from classpush)
cs = (classsum / max(1, nfire)).cpu().numpy()                # mean class-summed dlogit at firing
cs_ctrl = (classsum_ctrl / max(1, nctrl)).cpu().numpy()
abs_total = float(np.abs(cs).sum())
order = np.argsort(-np.abs(cs))
class_summed = {CLASS_LIST[j]: round(float(cs[j]), 3) for j in order[:8]}
class_summed_ctrl = {CLASS_LIST[j]: round(float(cs_ctrl[j]), 3) for j in order[:8]}
pushed_j = order[0]
pushed_class = CLASS_LIST[pushed_j]
class_concentration = round(abs(float(cs[pushed_j]))/max(abs_total, 1e-9), 3)

# dCE-by-class table
dce_by_class = {}
for c in CLASS_LIST:
    s, n = cls_dce[c]
    if n >= 20: dce_by_class[c] = {'dCE': round(s/n, 5), 'n': n}
dce_by_class = dict(sorted(dce_by_class.items(), key=lambda kv: -kv[1]['dCE']))

# =====================================================================================
# CONCRETE EXAMPLES: pick 4 highest-|dCE| firing positions, decode context/source/effect.
# =====================================================================================
ex_records.sort(key=lambda r: -abs(r['dce']))
examples = []
seen_seq = set()
for r in ex_records:
    if len(examples) >= 4: break
    s, p = r['seq'], r['pos']
    if s in seen_seq: continue                               # diversify across sequences
    seen_seq.add(s)
    sp = int(src_pos[s, p])
    ctx_lo = max(0, p-12)
    context = tok.decode([int(t) for t in held_np[s, ctx_lo:p+1]])
    cur = dec(held_np[s, p]); srct = dec(held_np[s, sp]); tgt_tok = dec(held_np[s, p+1])
    # per-position head effect at this position
    rh_here = torch.from_numpy(rh_store[s, p]).to(DEV)
    el = (Wu_dev @ rh_here).float()
    boost_here = [dec(t) for t in torch.topk(el, 6).indices.cpu().numpy()]
    supp_here = [dec(t) for t in torch.topk(-el, 6).indices.cpu().numpy()]
    examples.append({
        'seq': s, 'pos': p, 'src_pos': sp, 'offset': int(p-sp),
        'context_...cur': context, 'cur_tok': cur, 'attended_src_tok': srct,
        'true_next_tok': tgt_tok, 'dCE_full': round(r['dce'], 4), 'dCE_direct': round(r['dce_dir'], 4),
        'head_boosts_here': boost_here, 'head_suppresses_here': supp_here,
    })

# =====================================================================================
OUT = {
    'meta': {'model': 'bilin18', 'target': f'h.L{LI_T}.{H_T}', 'held_slice': 'FW[448:600,:128]',
             'BATCH': BATCH, 'KCAUSAL': KCAUSAL,
             'census_global_dCE_expected': 0.017004, 'census_trigger_dCE_expected': 0.02898},
    'GATE_reproduce_census': {
        'global_dCE': round(gm, 6), 'global_dCE_SE': round(gse, 6), 'global_dCE_z': round(gm/gse, 2) if gse>0 else 0,
        'trigger_dCE': round(tm_, 5), 'trigger_dCE_SE': round(tse, 5), 'trigger_dCE_z': round(tm_/tse, 2) if tse>0 else 0,
        'trigger_frac_pos': round(t_pos/max(1, t_n), 3), 'n_global': g_n, 'n_trigger': t_n,
    },
    'fingerprint': {
        'cur_token_classes_at_firing': fp_cur_classes,
        'attended_source_classes': fp_src_classes,
        'top_cur_tokens': top_counts(cur_tok), 'top_src_tokens': top_counts(src_tok),
        'source_offset_profile_top': dict(sorted(off_profile.items(), key=lambda kv: -kv[1])[:12]),
        'offset_median': int(np.median(offset)), 'offset_conc': round(offset_conc, 3),
        'self_frac': round(self_frac, 3), 'prev_frac': round(prev_frac, 3), 'sink_frac': round(sink_frac, 3),
        'effect_dir_boosts': fp_boost, 'effect_dir_suppresses': fp_suppress,
    },
    'type_test_copy': copy_test,
    'type_test_classpush': {
        'pushed_class': pushed_class, 'class_concentration': class_concentration,
        'class_summed_dlogit_at_firing': class_summed,
        'class_summed_dlogit_at_control': class_summed_ctrl,
        'note': 'base-ablated: positive => head normally BOOSTS class; negative => head SUPPRESSES class',
    },
    'causal_direct_vs_mediated': {
        'full_dCE_trigger': round(fm, 5), 'full_dCE_trigger_SE': round(fse, 5),
        'direct_dCE_trigger': round(dm, 5), 'direct_dCE_trigger_SE': round(dse, 5),
        'direct_fraction': round(dm/fm, 3) if abs(fm) > 1e-9 else None,
        'note': 'direct = layer-7 post-attention residual change propagated straight to logits; '
                'mediated = full - direct (routed through blocks 8-17 + block-7 MLP nonlinearity).',
    },
    'dce_by_current_class_global': dce_by_class,
    'concrete_examples': examples,
}
json.dump(OUT, open(f'{QK}/qk_arc_h70.json', 'w'), indent=2)
print("\n===== GATE =====", flush=True)
print(json.dumps(OUT['GATE_reproduce_census'], indent=1), flush=True)
print("\n===== FINGERPRINT =====", flush=True)
print(json.dumps(OUT['fingerprint'], indent=1), flush=True)
print("\n===== COPY =====", flush=True); print(json.dumps(OUT['type_test_copy'], indent=1), flush=True)
print("\n===== CLASSPUSH =====", flush=True); print(json.dumps(OUT['type_test_classpush'], indent=1), flush=True)
print("\n===== DIRECT vs MEDIATED =====", flush=True); print(json.dumps(OUT['causal_direct_vs_mediated'], indent=1), flush=True)
print("\n===== dCE BY CLASS =====", flush=True); print(json.dumps(dce_by_class, indent=1), flush=True)
print("\n===== EXAMPLES =====", flush=True); print(json.dumps(examples, indent=1), flush=True)
print("\nQK ARC H70 DONE", flush=True)
