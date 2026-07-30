"""GREEDY JOINT / SUBSET ABLATION -- a new tool for REDUNDANT / DISTRIBUTED circuits.

WHY A NEW TOOL (different circuit types need different detectors).
The single-component ablation pipeline (qk_unsup_verify / qk_unsup_copy §60) ranks
each head by its OWN mean-ablation dCE in ISOLATION. That test is BLIND to redundancy:
if a function is DUPLICATED across several heads, removing any ONE is masked by the
others, so every member reads dCE~=0 and the pipeline calls them all "causally null."
§60 found a family of genuine VERBATIM-COPY heads with exactly this signature -- large
source-specific logit drops but dCE~=0 solo (L8H4, L13H0, L14H7, L7H3, L5H5; plus the
one load-bearing member L8H3). §56/§57 flagged the same pattern for structure/newline
heads (h.L1.2, h.L6.0, h.L13.1). Single ablation cannot tell a TRULY-NULL head from a
REDUNDANT one.

THE TECHNIQUE (what this file adds):
  On a fixed held-back position set, mean-ablate a SET of heads jointly (in-distribution
  per-position mean = zero point), and compare:
    solo dCE_h            (ablate head h alone)
    joint dCE            (ablate the whole set at once)
    greedy build-up       (add heads one at a time, largest-marginal-gain first)
    REDUNDANCY = joint_dCE / sum_h(solo_dCE)          [ >>1 redundant/distributed ;
                                                        ~1 independent/additive ]
    minimal subset that recovers >=80% of the joint effect
  Copy-relevant readout: the attended-SOURCE-token logit and whether the source token is
  still the model's TOP-1 prediction, under base / solo / joint -- does the copy OUTPUT
  survive solo ablation but COLLAPSE only jointly?
  HONESTY controls: (i) a set can be jointly NULL too (truly unimportant) -> joint~=0;
  (ii) a large joint dCE could just mean "removed a lot of capacity" -> compared against
  R RANDOM head-sets of the SAME size ablated on the SAME positions.

FORWARD copied VERBATIM from qk_bracket_patch.py (tier2_model.reference_forward): the
load-bearing computation lines (x-mix, per-head QK rms_norm THEN RoPE, v-lerp via a.lamb,
two-branch pattern s1*s2 UNNORMALISED, 30*tanh logits) are IDENTICAL; only the ablation
hook is generalised from one head to a SET, plus a copy-stat collector, exactly as §60 did.
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
tok = AutoTokenizer.from_pretrained('gpt2')
Wu = m.lm_head.weight.detach().float()

FINEWEB = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))
SEQL = 128
HELD = FINEWEB[448:600, :SEQL].to(DEV)                 # held-back slice (untouched by discovery)
NHELD = HELD.shape[0]
BATCH = 4
KPOS = 150                                             # top activating positions per head -> shared set
KCAUSAL = 200                                          # §60-style per-head own-position causal set
NRAND = 40                                             # random same-size control draws
print(f"bilin18 NL={NL} NH={NH} HD={HD} D={D} V={V}; held={NHELD}", flush=True)

def dec(t): return repr(tok.decode([int(t)]))
def pid(hn):  # 'L8H3' -> (8,3)
    return int(hn[1:hn.index('H')]), int(hn[hn.index('H') + 1:])

_special = {tok.eos_token_id}
for _t in range(V):
    if '�' in tok.decode([_t]): _special.add(_t)
SPECIAL = np.array(sorted(_special))

# ===== the two circuits under test =====
COPY_FAMILY = ['L8H3', 'L8H4', 'L13H0', 'L14H7', 'L7H3', 'L5H5']   # §60 verbatim-copy family
DIFFUSE_SET = ['L1H0', 'L1H2', 'L0H2', 'L0H1', 'L0H5', 'L1H8']     # cluster-3 structure/newline heads
FAMILY_HEADS = set(pid(h) for h in COPY_FAMILY) | set(pid(h) for h in DIFFUSE_SET)


# =====================================================================================
# Forward -- VERBATIM computation lines from qk_bracket_patch.py; ablation hook
# generalised to a SET of heads, plus a copy-stat collector (as §60).
# =====================================================================================
@torch.no_grad()
def forward(idx, ablate=None, means=None, want_yh=False, collect=False):
    """ablate: None | ('heads', set_of_(li,h))   -- mean-ablate every listed head.
    means: {'yh': [ (T,NH,HD) per layer ]} per-position held mean (in-distribution zero).
    want_yh: accumulate per-layer yh sum (mean collection).
    collect: return per-head hnorm / argmax-src pos / src token id (for position sets)."""
    B, Tt = idx.shape
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(Tt, HD, DEV, x.dtype, 'bf16')
    cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(Tt, Tt, device=DEV, dtype=torch.bool))
    acc = {'yh': []} if want_yh else None
    out = {} if collect else None
    ab = {}
    if ablate is not None and ablate[0] == 'heads':
        for (li, h) in ablate[1]:
            ab.setdefault(li, []).append(h)
    for li in range(NL):
        blk = m.transformer.h[li]; x = blk.lambdas[0] * x + blk.lambdas[1] * x0
        a = blk.attn; hc = F.rms_norm(x, (D,))
        def qk(l): z = F.rms_norm(l(hc).view(B, Tt, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
        v = a.c_v(hc).view(B, Tt, NH, HD)
        if v1 is None:
            v1 = v
        lamb = a.lamb
        v = (1 - lamb) * v + lamb * v1.view_as(v)
        q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k) / HD
        s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2) / HD
        pat = (s1 * s2).masked_fill(~mask, 0.0)
        yh = torch.einsum('bhqk,bkhd->bqhd', pat, v)          # (B,Tt,NH,HD)
        if collect:
            Wp = a.c_proj.weight.view(D, NH, HD)
            comp = torch.einsum('bthc,dhc->bthd', yh, Wp)     # (B,Tt,NH,D)
            sp = pat.abs().argmax(-1).permute(0, 2, 1)        # (B,Tt,NH) argmax key pos
            srcid = torch.gather(idx.unsqueeze(-1).expand(B, Tt, NH), 1, sp.clamp(max=Tt - 1))
            out[li] = {'hnorm': comp.norm(dim=-1).cpu().numpy(),
                       'srcpos': sp.cpu().numpy().astype(np.int16),
                       'srcid': srcid.cpu().numpy().astype(np.int32)}
            del comp
        if li in ab:                                          # <-- SET ablation (generalised hook)
            yh = yh.clone()
            for h in ab[li]:
                yh[:, :, h] = means['yh'][li][:, h].unsqueeze(0)
        if want_yh: acc['yh'].append(yh.sum(0))
        attn = a.c_proj(yh.reshape(B, Tt, -1))
        x = x + attn
        mo = blk.mlp(F.rms_norm(x, (D,)))
        x = x + mo
    logits = 30 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30)
    if want_yh: return logits, acc
    if collect: return logits, out
    return logits


# =====================================================================================
# PASS 1: per-position held mean (zero point) + per-head hnorm / src stats over held.
# =====================================================================================
print("PASS 1: held-slice means + per-head copy stats ...", flush=True)
YH_SUM = {li: torch.zeros(SEQL, NH, HD, device=DEV) for li in range(NL)}
hnorm = np.zeros((NL, NH, NHELD, SEQL), np.float32)
srcpos = np.zeros((NL, NH, NHELD, SEQL), np.int16)
srcid = np.zeros((NL, NH, NHELD, SEQL), np.int32)
for i in range(0, NHELD, BATCH):
    (_, acc) = forward(HELD[i:i + BATCH], want_yh=True)
    for li in range(NL):
        YH_SUM[li] += acc['yh'][li]
    _, out = forward(HELD[i:i + BATCH], collect=True)
    b = out[0]['hnorm'].shape[0]
    for li in range(NL):
        o = out[li]
        hnorm[li, :, i:i + b] = o['hnorm'].transpose(2, 0, 1)
        srcpos[li, :, i:i + b] = o['srcpos'].transpose(2, 0, 1)
        srcid[li, :, i:i + b] = o['srcid'].transpose(2, 0, 1)
YHMEAN = {li: YH_SUM[li] / NHELD for li in range(NL)}       # in-distribution per-position zero

held_np = HELD.cpu().numpy()
pos_th = np.tile(np.arange(SEQL), NHELD).reshape(NHELD, SEQL)
is_special_h = np.isin(held_np, SPECIAL)
valid_h = (pos_th > 0) & (pos_th < SEQL - 1) & ~is_special_h


# =====================================================================================
# Shared position set for a circuit = union of each member's top-KPOS activating
# positions; per position record the dominant member (max hnorm) and its source token.
# =====================================================================================
def build_posset(heads, kpos=KPOS):
    best = {}   # (seq,pos) -> (hnorm, dom_head_index, srctok)
    for hi, (li, h) in enumerate(heads):
        a = hnorm[li, h].reshape(-1).copy(); a[~valid_h.reshape(-1)] = -1e30
        tk = np.argpartition(a, -kpos)[-kpos:]
        for f in tk:
            s, p = divmod(int(f), SEQL)
            hv = float(hnorm[li, h, s, p])
            if (s, p) not in best or hv > best[(s, p)][0]:
                best[(s, p)] = (hv, hi, int(srcid[li, h, s, p]))
    seqs = np.array([k[0] for k in best], np.int64)
    poss = np.array([k[1] for k in best], np.int64)
    dom = np.array([v[1] for v in best.values()], np.int64)
    srct = np.array([v[2] for v in best.values()], np.int64)
    tgts = held_np[seqs, poss + 1].astype(np.int64)
    return {'seqs': seqs, 'poss': poss, 'dom': dom, 'srctok': srct, 'tgts': tgts, 'n': len(seqs)}


def se(x): return float(np.asarray(x).std(ddof=1) / math.sqrt(len(x)))


# ===== evaluate a config (head-set ablation) on a FIXED position set -> per-pos CE etc =====
@torch.no_grad()
def eval_config(ps, ablate_heads):
    seqs, poss, srct, tgts = ps['seqs'], ps['poss'], ps['srctok'], ps['tgts']
    N = ps['n']
    ce = np.zeros(N); srcl = np.zeros(N); srank = np.zeros(N); top1 = np.zeros(N)
    ab = ('heads', set(ablate_heads)) if ablate_heads else None
    by_seq = {}
    for j, s in enumerate(seqs):
        by_seq.setdefault(int(s), []).append(j)
    uniq = np.array(sorted(by_seq))
    for i in range(0, len(uniq), BATCH):
        sb = uniq[i:i + BATCH]
        logits = forward(HELD[sb], ablate=ab, means={'yh': YHMEAN}).float()
        loc = {int(s): bi for bi, s in enumerate(sb)}
        rows, ps_, tg_, sr_, jj_ = [], [], [], [], []
        for s in sb:
            for j in by_seq[int(s)]:
                rows.append(loc[int(s)]); ps_.append(int(poss[j]))
                tg_.append(int(tgts[j])); sr_.append(int(srct[j])); jj_.append(j)
        rows = torch.tensor(rows, device=DEV); ps_t = torch.tensor(ps_, device=DEV)
        lg = logits[rows, ps_t]                                # (npos,V)
        tg = torch.tensor(tg_, device=DEV); sr = torch.tensor(sr_, device=DEV)
        ce_b = F.cross_entropy(lg, tg, reduction='none').cpu().numpy()
        sl = lg[torch.arange(len(sr)), sr]
        rk = (lg > sl.unsqueeze(1)).sum(1).cpu().numpy()
        t1 = (lg.argmax(1) == sr).cpu().numpy()
        sl = sl.cpu().numpy()
        for kk, j in enumerate(jj_):
            ce[j] = ce_b[kk]; srcl[j] = sl[kk]; srank[j] = rk[kk]; top1[j] = t1[kk]
    return {'ce': ce, 'srcl': srcl, 'srank': srank, 'top1': top1}


# ===== §60-style: solo dCE at a head's OWN top-KCAUSAL positions (reproduce §60 ~0) =====
@torch.no_grad()
def solo_own_dCE(li, h):
    ps = build_posset([(li, h)], kpos=KCAUSAL)
    base = eval_config(ps, [])
    abl = eval_config(ps, [(li, h)])
    dce = abl['ce'] - base['ce']
    return {'n': ps['n'], 'dCE': round(float(dce.mean()), 4), 'dCE_SE': round(se(dce), 4)}


# =====================================================================================
# MAIN per-circuit analysis
# =====================================================================================
def analyse(name, head_names):
    heads = [pid(h) for h in head_names]
    ps = build_posset(heads)
    base = eval_config(ps, [])
    print(f"\n===== {name}: {head_names}  (shared posset n={ps['n']}) =====", flush=True)

    # --- solos on shared set + §60-style own-position solo ---
    solos = {}
    for hn, hp in zip(head_names, heads):
        cfg_ = eval_config(ps, [hp])
        dce = cfg_['ce'] - base['ce']
        own = solo_own_dCE(*hp)
        solos[hn] = {
            'solo_dCE_shared': round(float(dce.mean()), 4), 'solo_dCE_shared_SE': round(se(dce), 4),
            'solo_dCE_own': own['dCE'], 'solo_dCE_own_SE': own['dCE_SE'], 'own_n': own['n'],
            'src_logit_drop': round(float((base['srcl'] - cfg_['srcl']).mean()), 4),
            'src_top1_frac': round(float(cfg_['top1'].mean()), 3),
        }
        print(f"  solo {hn:6s} dCE_shared={solos[hn]['solo_dCE_shared']:+.4f}"
              f"±{solos[hn]['solo_dCE_shared_SE']:.4f}  dCE_own(§60)={solos[hn]['solo_dCE_own']:+.4f}"
              f"±{solos[hn]['solo_dCE_own_SE']:.4f}  src_drop={solos[hn]['src_logit_drop']:+.3f}"
              f"  src_top1={solos[hn]['src_top1_frac']:.2f}", flush=True)

    # --- joint (whole set) ---
    jc = eval_config(ps, heads)
    jdce = jc['ce'] - base['ce']
    joint = {'joint_dCE': round(float(jdce.mean()), 4), 'joint_dCE_SE': round(se(jdce), 4),
             'src_logit_drop': round(float((base['srcl'] - jc['srcl']).mean()), 4),
             'src_top1_base': round(float(base['top1'].mean()), 3),
             'src_top1_joint': round(float(jc['top1'].mean()), 3),
             'src_rank_med_base': int(np.median(base['srank'])),
             'src_rank_med_joint': int(np.median(jc['srank']))}
    sum_solo = sum(max(solos[hn]['solo_dCE_shared'], 0.0) for hn in head_names)
    sum_solo_signed = sum(solos[hn]['solo_dCE_shared'] for hn in head_names)
    redundancy = round(joint['joint_dCE'] / sum_solo, 2) if sum_solo > 1e-6 else None
    redundancy_signed = round(joint['joint_dCE'] / sum_solo_signed, 2) if abs(sum_solo_signed) > 1e-6 else None
    print(f"  JOINT dCE={joint['joint_dCE']:+.4f}±{joint['joint_dCE_SE']:.4f}"
          f"  sum(solo+)={sum_solo:.4f}  REDUNDANCY={redundancy}"
          f"  src_top1 base={joint['src_top1_base']:.2f}->joint={joint['src_top1_joint']:.2f}"
          f"  src_rank_med base={joint['src_rank_med_base']}->joint={joint['src_rank_med_joint']}", flush=True)

    # --- greedy build-up (largest marginal dCE first) ---
    remaining = list(head_names); chosen = []; curve = []
    while remaining:
        best_h, best_dce, best_cfg = None, -1e9, None
        for hn in remaining:
            trial = [pid(x) for x in chosen + [hn]]
            c_ = eval_config(ps, trial)
            d_ = float((c_['ce'] - base['ce']).mean())
            if d_ > best_dce:
                best_dce, best_h, best_cfg = d_, hn, c_
        chosen.append(best_h); remaining.remove(best_h)
        dvec = best_cfg['ce'] - base['ce']
        curve.append({'added': best_h, 'set': list(chosen),
                      'cum_dCE': round(best_dce, 4), 'cum_dCE_SE': round(se(dvec), 4),
                      'cum_src_top1': round(float(best_cfg['top1'].mean()), 3)})
        print(f"  greedy +{best_h:6s} -> cum_dCE={best_dce:+.4f}±{se(dvec):.4f}"
              f"  src_top1={curve[-1]['cum_src_top1']:.2f}  set={chosen}", flush=True)
    # minimal subset >=80% of joint
    thr = 0.8 * joint['joint_dCE']
    minimal = None
    if joint['joint_dCE'] > 0:
        for c in curve:
            if c['cum_dCE'] >= thr:
                minimal = {'subset': c['set'], 'size': len(c['set']),
                           'cum_dCE': c['cum_dCE'], 'frac_of_joint': round(c['cum_dCE'] / joint['joint_dCE'], 2)}
                break

    # --- random same-size control on the SAME positions ---
    rng = np.random.RandomState(0)
    allheads = [(li, h) for li in range(NL) for h in range(NH) if (li, h) not in FAMILY_HEADS]
    rand_dce = []
    for _ in range(NRAND):
        pick = [allheads[i] for i in rng.choice(len(allheads), len(heads), replace=False)]
        c_ = eval_config(ps, pick)
        rand_dce.append(float((c_['ce'] - base['ce']).mean()))
    rand_dce = np.array(rand_dce)
    control = {'n_draws': NRAND, 'rand_joint_dCE_mean': round(float(rand_dce.mean()), 4),
               'rand_joint_dCE_std': round(float(rand_dce.std(ddof=1)), 4),
               'rand_joint_dCE_max': round(float(rand_dce.max()), 4),
               'rand_joint_dCE_p95': round(float(np.percentile(rand_dce, 95)), 4),
               'family_over_random_z': round(float((joint['joint_dCE'] - rand_dce.mean()) /
                                                   (rand_dce.std(ddof=1) + 1e-9)), 2),
               'family_exceeds_all_random': bool(joint['joint_dCE'] > rand_dce.max())}
    print(f"  RANDOM control (same size, same posset): mean={control['rand_joint_dCE_mean']:+.4f}"
          f" max={control['rand_joint_dCE_max']:+.4f} p95={control['rand_joint_dCE_p95']:+.4f}"
          f" | family joint z={control['family_over_random_z']} exceeds_all={control['family_exceeds_all_random']}", flush=True)

    # interpretation
    verdict = interpret(joint, redundancy, control, minimal)
    print(f"  VERDICT: {verdict}", flush=True)
    return {'heads': head_names, 'posset_n': ps['n'], 'solos': solos, 'joint': joint,
            'sum_solo_positive': round(sum_solo, 4), 'sum_solo_signed': round(sum_solo_signed, 4),
            'redundancy_ratio': redundancy, 'redundancy_ratio_signed': redundancy_signed,
            'greedy_curve': curve, 'minimal_subset_80pct': minimal,
            'random_control': control, 'verdict': verdict}


def interpret(joint, redundancy, control, minimal):
    jd, jse = joint['joint_dCE'], joint['joint_dCE_SE']
    if jd < 3 * jse:
        return 'JOINTLY NULL too -> set is genuinely unimportant (not redundant, not load-bearing)'
    if not control['family_exceeds_all_random'] and control['family_over_random_z'] < 3:
        return ('JOINT effect NOT specific -> a random same-size set is comparably damaging '
                '(joint dCE reflects removed capacity, not a redundant circuit)')
    tag = 'REDUNDANT / DISTRIBUTED' if (redundancy is not None and redundancy >= 2) else \
          'JOINTLY LOAD-BEARING (partly additive)'
    sub = (f"; minimal {minimal['size']}-head subset recovers {int(100*minimal['frac_of_joint'])}% of joint"
           if minimal else '')
    return (f"{tag}: joint dCE={jd:+.3f} >> sum(solo+); redundancy={redundancy}; "
            f"specific vs random (z={control['family_over_random_z']}){sub}")


RESULTS = {
    'technique': 'GREEDY JOINT / SUBSET ABLATION for redundant/distributed circuits: on a '
                 'fixed held-back position set (union of members top-KPOS activating positions), '
                 'mean-ablate a SET of heads jointly (in-distribution per-position mean zero point) '
                 'and compare solo dCE vs joint dCE vs a greedy largest-marginal-gain build-up. '
                 'REDUNDANCY = joint_dCE / sum(solo_dCE) (>>1 = duplicated/distributed). Honesty: a '
                 'jointly-null set is truly unimportant; a random same-size set on the same positions '
                 'controls for removed-capacity. Complements single-ablation, which oversells '
                 'redundant members as causally null.',
    'params': {'held_slice': 'FW[448:600,:128]', 'ablation': 'mean (in-distribution per-position)',
               'KPOS': KPOS, 'KCAUSAL': KCAUSAL, 'NRAND': NRAND, 'BATCH': BATCH},
    'copy_family': analyse('COPY FAMILY (§60 verbatim-copy)', COPY_FAMILY),
    'diffuse_set': analyse('DIFFUSE structure/newline (cluster-3)', DIFFUSE_SET),
}
json.dump(RESULTS, open(f'{QK}/qk_unsup_redundant.json', 'w'), indent=2)
print("\nQK UNSUP REDUNDANT DONE", flush=True)
