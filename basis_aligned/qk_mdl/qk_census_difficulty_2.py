"""CHARACTERIZE the LOW-clean / HIGH-causal missed-hard circuits from the census.

For the top missed-hard paths (qk_census_difficulty.json, quadrant LH), determine
WHY each is HARD for the cleanliness-ranked easy-loop, picking from:
  (1) impure / multi-class TRIGGER   -- fires across several token classes
  (2) distributed / multi-class OUTPUT -- boosts many tokens, no sharp effect dir
  (3) deep COMPOSITION               -- its own contribution is BUILT from upstream
        (test: mean-ablate the whole previous block's attention; does the path's
         output-vector magnitude at its trigger positions collapse? does its causal
         trigger-dCE change?)
  (4) REDUNDANCY (connect to §61)    -- only matters jointly; family joint-ablation
        redundancy ratio = joint_dCE / sum(solo_dCE) at a FIXED position set.

FORWARD verbatim from qk_census_difficulty.py (extended to multi-component ablation).
All causal measurement on held-back FW[448:600], paired standard errors.
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

def gpu_guard(min_free=4500, tries=45, sleep=20):
    for _ in range(tries):
        free = int(subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits']
        ).decode().split('\n')[0].strip())
        if free >= min_free:
            print(f"GPU guard: {free} MiB free -- proceeding.", flush=True); return
        print(f"GPU guard: only {free} MiB free; sleeping {sleep}s ...", flush=True); time.sleep(sleep)
    raise RuntimeError("GPU guard timed out")
gpu_guard()

m, cfg = load_elriggs('bilin18')
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
V = cfg['vocab_size']; NL = len(m.transformer.h); N_SVD = 4
tok = AutoTokenizer.from_pretrained('gpt2')
def dec(t): return repr(tok.decode([int(t)]))

FINEWEB = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))
SEQL = 128
TRAIN = FINEWEB[0:256, :SEQL].to(DEV)
HELD = FINEWEB[448:600, :SEQL].to(DEV); NHELD = HELD.shape[0]
BATCH = 6; KCAUSAL = 200

_special = {tok.eos_token_id}
for _t in range(V):
    if '�' in tok.decode([_t]): _special.add(_t)
SPECIAL = np.array(sorted(_special))

def coarse_class(t):
    s = tok.decode([int(t)]); raw = tok.convert_ids_to_tokens(int(t)); st = s.strip(); cl = []
    if raw.startswith('Ġ') or (s[:1] == ' '): cl.append('space_pref')
    if 'Ċ' in raw or '\n' in s: cl.append('newline')
    if st and any(ch.isdigit() for ch in st): cl.append('digit')
    if st[:1].isupper(): cl.append('capital')
    if st and all(not ch.isalnum() for ch in st): cl.append('punct')
    return cl
def class_hist(ids):
    from collections import Counter
    c = Counter()
    for t in ids:
        for cl in coarse_class(t): c[cl] += 1
    return {k: round(v/len(ids), 2) for k, v in c.most_common()}

# ---- MLP dirs from TRAIN gram (verbatim) ----
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
for i in range(0, TRAIN.shape[0], BATCH): fwd_gram(TRAIN[i:i+BATCH])
mlp_dirs = torch.zeros(NL, N_SVD, D, device=DEV)
for li in range(NL):
    _, ev = torch.linalg.eigh(gram[li]); mlp_dirs[li] = ev[:, -N_SVD:].T.flip(0)
del gram
print("MLP dirs ready.", flush=True)

# ---- forward with MULTI-component mean-ablation + optional collect of selected activations ----
# ablations: list of ('head',li,h) | ('mlp',li,kk) | ('attnlayer',li)  (whole block attn)
@torch.no_grad()
def forward(idx, ablations=(), yhmeans=None, projmeans=None, collect_heads=None, collect_mlps=None):
    B, T = idx.shape
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    coll = {}
    ab_heads = {}; ab_mlps = {}; ab_attn = set()
    for a_ in ablations:
        if a_[0] == 'head': ab_heads.setdefault(a_[1], []).append(a_[2])
        elif a_[0] == 'mlp': ab_mlps.setdefault(a_[1], []).append(a_[2])
        elif a_[0] == 'attnlayer': ab_attn.add(a_[1])
    for li in range(NL):
        blk = m.transformer.h[li]; x = blk.lambdas[0]*x + blk.lambdas[1]*x0; a = blk.attn; hcur = F.rms_norm(x, (D,))
        def qk(lin): z = F.rms_norm(lin(hcur).view(B, T, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
        v = a.c_v(hcur).view(B, T, NH, HD)
        if v1 is None: v1 = v
        v = (1-a.lamb)*v + a.lamb*v1.view_as(v)
        q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k)/HD; s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD
        pat = (s1*s2).masked_fill(~mask, 0.0); yh4 = torch.einsum('bhqk,bkhd->bqhd', pat, v)
        if collect_heads and li in collect_heads:
            Wr = a.c_proj.weight.view(D, NH, HD)
            for h in collect_heads[li]:
                comp = torch.einsum('btc,oc->bto', yh4[:, :, h], Wr[:, h])
                coll[('hnorm', li, h)] = comp.norm(dim=-1).cpu().numpy()
        if li in ab_attn:
            yh4 = yhmeans[li].unsqueeze(0).expand(B, -1, -1, -1).clone()   # whole-layer attn -> mean
        elif li in ab_heads:
            yh4 = yh4.clone()
            for h in ab_heads[li]: yh4[:, :, h] = yhmeans[li][:, h].unsqueeze(0)
        x = x + a.c_proj(yh4.reshape(B, T, -1))
        mo = blk.mlp(F.rms_norm(x, (D,)))
        if collect_mlps and li in collect_mlps:
            for kk in collect_mlps[li]:
                coll[('mproj', li, kk)] = torch.einsum('btd,d->bt', mo, mlp_dirs[li, kk]).cpu().numpy()
        if li in ab_mlps:
            for kk in ab_mlps[li]:
                pr = torch.einsum('btd,d->bt', mo, mlp_dirs[li, kk])
                mo = mo - (pr - projmeans[li][:, kk].unsqueeze(0)).unsqueeze(-1) * mlp_dirs[li, kk]
        x = x + mo
    logits = 30*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30)
    return logits, coll

# ---- PASS A: means + activation magnitudes (all paths) for trigger selection ----
YH_SUM = {li: torch.zeros(SEQL, NH, HD, device=DEV) for li in range(NL)}
PROJ_SUM = {li: torch.zeros(SEQL, N_SVD, device=DEV) for li in range(NL)}
head_act = np.zeros((NL*NH, NHELD, SEQL), np.float32)
mlp_act = np.zeros((NL*N_SVD, NHELD, SEQL), np.float32)
@torch.no_grad()
def collect_pass(idx, p0):
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
        Wr = a.c_proj.weight.view(D, NH, HD)
        YH_SUM[li] += yh4.sum(0)
        comp = torch.einsum('bthc,ohc->btho', yh4, Wr).norm(dim=-1)   # (B,T,NH)
        f0 = p0; b = B
        for h in range(NH): head_act[li*NH+h, f0:f0+b] = comp[:, :, h].cpu().numpy()
        x = x + a.c_proj(yh4.reshape(B, T, -1)); mo = blk.mlp(F.rms_norm(x, (D,)))
        pr = torch.einsum('btd,nd->btn', mo, mlp_dirs[li]); PROJ_SUM[li] += pr.sum(0)
        for kk in range(N_SVD): mlp_act[li*N_SVD+kk, f0:f0+b] = pr[:, :, kk].abs().cpu().numpy()
        x = x + mo
print("PASS A ...", flush=True)
for i in range(0, NHELD, BATCH): collect_pass(HELD[i:i+BATCH], i//BATCH*BATCH if False else i)
YHMEAN = {li: YH_SUM[li]/NHELD for li in range(NL)}
PROJMEAN = {li: PROJ_SUM[li]/NHELD for li in range(NL)}
del YH_SUM, PROJ_SUM
print("PASS A done.", flush=True)

held_np = HELD.cpu().numpy()
pos_t = np.tile(np.arange(SEQL), NHELD).reshape(NHELD, SEQL)
is_special = np.isin(held_np, SPECIAL)
valid_next = pos_t < (SEQL-1)
bad_trigger = (pos_t == 0) | is_special | ~valid_next

def act_of(kind, li, ix):
    return head_act[li*NH+ix] if kind == 'head' else mlp_act[li*N_SVD+ix]
def trigger_mask(kind, li, ix):
    a = act_of(kind, li, ix).copy().reshape(-1); a[bad_trigger.reshape(-1)] = -1e30
    tk = np.argpartition(a, -KCAUSAL)[-KCAUSAL:]
    mk = np.zeros(NHELD*SEQL, bool); mk[tk] = True
    return mk.reshape(NHELD, SEQL)

# ---- select missed-hard targets ----
census = json.load(open(f'{QK}/qk_census_difficulty.json'))
LH = census['summary']['missed_hard_LH_top']
TARGETS = LH[:10]                       # top-10 missed-hard by trigger_dCE
print("TARGETS:", [t['comp'] for t in TARGETS], flush=True)

def parse(comp):
    if comp.startswith('h.'):
        _, l, h = comp.split('.'); return 'head', int(l[1:]), int(h)
    else:
        _, l, d = comp.split('.'); return 'mlp', int(l[1:]), int(d[1:])

def stats(dvec):
    dvec = np.asarray(dvec); n = len(dvec)
    if n <= 1: return 0.0, 0.0, 0.0
    mean = dvec.mean(); se = dvec.std(ddof=1)/math.sqrt(n)
    return float(mean), float(se), float((dvec > 0).mean())

# ---- per-target causal measurement over trigger positions ----
@torch.no_grad()
def dce_at_positions(mask, ablations):
    """paired dCE at the given (NHELD,SEQL) query-position mask; return per-position array + mean Δlogit."""
    dce_list = []; dlogit_sum = torch.zeros(V, device=DEV); ndl = 0
    tgt_all = torch.from_numpy(held_np).to(DEV)
    seqs_needed = np.where(mask[:, :SEQL-1].any(axis=1))[0]
    for i in range(0, len(seqs_needed), BATCH):
        sb = seqs_needed[i:i+BATCH]; idx = HELD[sb]
        base, _ = forward(idx, ablations=())
        abl, _ = forward(idx, ablations=ablations, yhmeans=YHMEAN, projmeans=PROJMEAN)
        base = base.float(); abl = abl.float()
        tgt = tgt_all[sb]
        blp = F.log_softmax(base[:, :SEQL-1], -1); alp = F.log_softmax(abl[:, :SEQL-1], -1)
        bce = -blp.gather(-1, tgt[:, 1:].unsqueeze(-1)).squeeze(-1)
        ace = -alp.gather(-1, tgt[:, 1:].unsqueeze(-1)).squeeze(-1)
        mk = torch.from_numpy(mask[sb, :SEQL-1]).to(DEV)
        d = (ace - bce)[mk]
        dce_list.append(d.cpu().numpy())
        # mean Δlogit (base - ablated) at masked positions
        dl = (base[:, :SEQL-1] - abl[:, :SEQL-1])[mk]     # (n_sel, V)
        dlogit_sum += dl.sum(0); ndl += dl.shape[0]
    dce = np.concatenate(dce_list) if dce_list else np.zeros(0)
    return dce, (dlogit_sum/max(1, ndl))

RESULTS = {}
for T_ in TARGETS:
    comp = T_['comp']; kind, li, ix = parse(comp)
    mask = trigger_mask(kind, li, ix)
    print(f"\n=== {comp} (kind={kind} L{li}) ===", flush=True)
    # trigger current-token + (heads) attended-source distributions from PASS A already? need src.
    tk_flat = np.where(mask.reshape(-1))[0]
    cur = held_np.reshape(-1)[tk_flat]
    trig_cur_hist = class_hist(cur); trig_cur_purity = None
    _, cc = np.unique(cur, return_counts=True); p = cc/cc.sum()
    trig_cur_purity = float(1 - (-(p*np.log(p)).sum())/math.log(len(cur)))
    n_classes = len([k for k, v in trig_cur_hist.items() if v >= 0.1])

    # (A) SOLO causal + output distribution
    solo_dce, dmean = dce_at_positions(mask, [(kind, li, ix)])
    smean, sse, sfrac = stats(solo_dce)
    dpos = dmean.clamp(min=0)                    # positive boosts
    if dpos.sum() > 0:
        pw = (dpos/dpos.sum()).cpu().numpy(); pw.sort(); pwd = pw[::-1]
        top1 = float(pwd[0]); cum = np.cumsum(pwd)
        tok80 = int(np.searchsorted(cum, 0.8)+1)
        out_entropy = float(-(pwd[pwd > 0]*np.log(pwd[pwd > 0])).sum())
    else:
        top1, tok80, out_entropy = 0.0, 0, 0.0
    top_boost = torch.topk(dmean, 12)
    boost_ids = top_boost.indices.cpu().numpy()
    boost_hist = class_hist(boost_ids)

    # (B) DEEP COMPOSITION: mean-ablate whole previous block's attention; re-measure
    #     (b1) the path's own activation magnitude at its trigger positions
    #     (b2) the path's causal trigger-dCE (does its importance survive upstream lesion?)
    comp_res = {'testable': li >= 1}
    if li >= 1:
        # activation magnitude intact vs upstream-lesioned, at trigger positions
        seqs_needed = np.where(mask[:, :SEQL-1].any(axis=1))[0]
        intact_act, les_act = [], []
        ch = {li: [ix]} if kind == 'head' else None
        cm = {li: [ix]} if kind == 'mlp' else None
        for i in range(0, len(seqs_needed), BATCH):
            sb = seqs_needed[i:i+BATCH]; idx = HELD[sb]
            _, c0 = forward(idx, ablations=(), collect_heads=ch, collect_mlps=cm)
            _, c1 = forward(idx, ablations=[('attnlayer', li-1)], yhmeans=YHMEAN, collect_heads=ch, collect_mlps=cm)
            key = ('hnorm', li, ix) if kind == 'head' else ('mproj', li, ix)
            a0 = np.abs(c0[key]); a1 = np.abs(c1[key])
            mk = mask[sb, :]
            intact_act.append(a0[mk[:, :a0.shape[1]]]); les_act.append(a1[mk[:, :a1.shape[1]]])
        ia = np.concatenate(intact_act); la = np.concatenate(les_act)
        comp_res['act_intact_mean'] = round(float(ia.mean()), 4)
        comp_res['act_upstream_lesioned_mean'] = round(float(la.mean()), 4)
        comp_res['act_retained_frac'] = round(float(la.mean()/(ia.mean()+1e-9)), 3)
        # causal trigger-dCE of the path WITH upstream attn lesioned
        joint_dce, _ = dce_at_positions(mask, [('attnlayer', li-1), (kind, li, ix)])
        up_dce, _ = dce_at_positions(mask, [('attnlayer', li-1)])
        jm, _, _ = stats(joint_dce); um, _, _ = stats(up_dce)
        # marginal effect of the path GIVEN upstream lesioned = joint - upstream_only
        comp_res['path_dCE_given_upstream_lesion'] = round(jm - um, 4)
        comp_res['path_dCE_intact'] = round(smean, 4)
        comp_res['effect_retained_frac_under_lesion'] = round((jm-um)/(smean+1e-9), 3)

    # (C) REDUNDANCY (family joint vs Σsolo at target's trigger positions), §61 method
    if kind == 'head':
        family = [('head', li, h) for h in range(NH)]
    else:
        family = [('mlp', li, kk) for kk in range(N_SVD)]
    solo_means = {}
    for (fk, fl, fi) in family:
        d, _ = dce_at_positions(mask, [(fk, fl, fi)])
        solo_means[f"{fk}.L{fl}.{fi}"] = stats(d)[0]
    joint_d, _ = dce_at_positions(mask, family)
    joint_mean = stats(joint_d)[0]
    sum_solo = sum(v for v in solo_means.values())
    redundancy_ratio = float(joint_mean/(sum_solo+1e-9))

    RESULTS[comp] = {
        'kind': kind, 'li': li, 'cleanliness': T_['cleanliness'],
        'trigger_purity_census': T_['trigger_purity'], 'effect_purity_census': T_['effect_purity'],
        'solo_trigger_dCE': round(smean, 4), 'solo_trigger_dCE_SE': round(sse, 4),
        'solo_frac_pos': round(sfrac, 3),
        'trigger': {'cur_purity': round(trig_cur_purity, 4), 'cur_class_hist': trig_cur_hist,
                    'n_classes_ge10pct': n_classes},
        'output': {'boost_top1_share': round(top1, 3), 'boost_tokens_to_80pct': tok80,
                   'boost_entropy_nats': round(out_entropy, 3), 'boost_class_hist': boost_hist,
                   'top_boost_tokens': [dec(t) for t in boost_ids[:8]]},
        'deep_composition': comp_res,
        'redundancy': {'family_size': len(family), 'sum_solo_dCE': round(sum_solo, 4),
                       'joint_dCE': round(joint_mean, 4), 'redundancy_ratio_joint_over_sumsolo': round(redundancy_ratio, 3)},
    }
    r = RESULTS[comp]
    print(f"  solo dCE {smean:.3f}+-{sse:.3f} | trig cur_purity {trig_cur_purity:.2f} "
          f"classes {trig_cur_hist}", flush=True)
    print(f"  OUTPUT top1 {top1:.2f} tok80 {tok80} entropy {out_entropy:.2f} boostclasses {boost_hist}", flush=True)
    if comp_res.get('testable'):
        print(f"  COMPOSITION act_retained {comp_res['act_retained_frac']} "
              f"effect_retained_under_upstream_lesion {comp_res['effect_retained_frac_under_lesion']}", flush=True)
    print(f"  REDUNDANCY ratio joint/Σsolo = {redundancy_ratio:.2f} (joint {joint_mean:.3f} Σsolo {sum_solo:.3f})", flush=True)

# ---- provisional type assignment (rule-based from the evidence) ----
def assign_type(r):
    reasons = []
    if r['trigger']['cur_purity'] < 0.2 or r['trigger']['n_classes_ge10pct'] >= 3:
        reasons.append('impure/multi-class-trigger')
    if r['output']['boost_top1_share'] < 0.15 or r['output']['boost_tokens_to_80pct'] >= 12:
        reasons.append('distributed/multi-class-output')
    dc = r['deep_composition']
    if dc.get('testable') and (dc.get('act_retained_frac', 1) < 0.6 or dc.get('effect_retained_frac_under_lesion', 1) < 0.5):
        reasons.append('deep-composition(upstream-dependent)')
    if r['redundancy']['redundancy_ratio_joint_over_sumsolo'] >= 1.5:
        reasons.append('redundant(joint>>Σsolo)')
    return reasons or ['none-of-catalog(hard-region residual)']
for comp, r in RESULTS.items():
    r['provisional_hardness_types'] = assign_type(r)

out = {
    'meta': {'model': 'bilin18', 'held_slice': 'FW[448:600,:128]', 'KCAUSAL': KCAUSAL,
             'n_targets': len(RESULTS),
             'method': 'top missed-hard (LOW-clean/HIGH-causal) paths characterized on 4 hardness axes: '
                       'impure trigger, distributed output, deep composition (prev-block-attn lesion), '
                       'redundancy (family joint/Σsolo, §61).'},
    'targets': RESULTS,
}
json.dump(out, open(f'{QK}/qk_census_difficulty_2.json', 'w'), indent=2)
print("\n===== HARDNESS TYPE ASSIGNMENTS =====", flush=True)
for comp, r in RESULTS.items():
    print(f"  {comp:12s} dCE={r['solo_trigger_dCE']:.3f} clean={r['cleanliness']:.3f} -> {r['provisional_hardness_types']}", flush=True)
print("\nSaved qk_census_difficulty_2.json", flush=True)
print("QK CENSUS DIFFICULTY 2 DONE", flush=True)
