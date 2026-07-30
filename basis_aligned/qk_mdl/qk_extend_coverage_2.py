"""EXTEND COVERAGE, PART B -- first pass at the MULTI-PATH combination structure.

§71 found a whole-model super-additivity of 2.87x: joint ablation of all 234 paths
(3.376 nats) is 2.87x the sum of positive single-path importances (1.177) -- most of
the model's computation lives in path COMBINATIONS, not single paths. This script takes
the top ~20 causally-important paths and uses greedy joint-ablation grouping (§61 method)
to find load-bearing GROUPS whose JOINT effect exceeds the SUM of their solo effects
(super-additive combinations), with same-size RANDOM-set controls.

Method (§61 verbatim conventions -- greedy joint/subset ablation, redundancy ratio,
random control): on a FIXED held-back position set (union of the members' top-KCAUSAL
firing positions), mean-ablate a SET jointly (in-distribution per-position mean = zero
point) and compare solo dCE vs joint dCE.
  - whole top-20 joint vs sum-of-solos  -> global super-additivity + multi-path residual.
  - pairwise super-additivity matrix among the strongest paths.
  - greedy agglomerative grouping: seed from the strongest super-additive pair, grow by
    max super-additive marginal (joint(S+c) - joint(S) - solo(c)); close when no positive
    marginal; then next group from the remaining paths.
  - each group named by its members' §68 class-push functions (copy heads / capital
    integrators / word integrators / positional), with a same-size random control.

FORWARD + multi-component mean-ablation + class library copied VERBATIM from
qk_unsup_classpush.py. Held-back FW[448:600,:128]. Batch 6, GPU guard, <4GB footprint.
"""
import json, sys, math, time, subprocess
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from transformers import AutoTokenizer
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
        print(f"GPU guard: only {free} MiB free (<{min_free}); sleeping {sleep}s ...", flush=True)
        time.sleep(sleep)
    raise RuntimeError("GPU guard timed out waiting for free memory")
gpu_guard()

m, cfg = load_elriggs('bilin18')
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
V = cfg['vocab_size']; NL = len(m.transformer.h); N_SVD = 4
tok = AutoTokenizer.from_pretrained('gpt2')
print(f"bilin18 NL={NL} NH={NH} HD={HD} D={D} V={V}", flush=True)

FINEWEB = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))
SEQL = 128
TRAIN = FINEWEB[0:256, :SEQL].to(DEV)
HELD = FINEWEB[448:600, :SEQL].to(DEV)
NHELD = HELD.shape[0]
BATCH = 6
KCAUSAL = 200
NRAND = 25
print(f"held={NHELD}  KCAUSAL={KCAUSAL} NRAND={NRAND}", flush=True)

_special = {tok.eos_token_id}
for _t in range(V):
    if '�' in tok.decode([_t]): _special.add(_t)
SPECIAL = np.array(sorted(_special))

# ---- MLP directions (VERBATIM) ----
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
    _e, _v = torch.linalg.eigh(gram[li]); mlp_dirs[li] = _v[:, -N_SVD:].T.flip(0)
del gram
print("MLP directions ready.", flush=True)

# ---- forward with MULTI-component mean-ablation + collect (VERBATIM classpush) ----
@torch.no_grad()
def forward(idx, ablations=(), yhmeans=None, projmeans=None, collect=False):
    B, T = idx.shape
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    ab_heads = {}; ab_mlps = {}
    for a_ in ablations:
        if a_[0] == 'head': ab_heads.setdefault(a_[1], []).append(a_[2])
        elif a_[0] == 'mlp': ab_mlps.setdefault(a_[1], []).append(a_[2])
    out = {} if collect else None
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
        if collect:
            Wr = a.c_proj.weight.view(D, NH, HD)
            YH_SUM[li] += yh4.sum(0)
            out[('hnorm', li)] = torch.einsum('bthc,ohc->btho', yh4, Wr).norm(dim=-1).cpu().numpy()
        if li in ab_heads:
            yh4 = yh4.clone()
            for h in ab_heads[li]: yh4[:, :, h] = yhmeans[li][:, h].unsqueeze(0)
        x = x + a.c_proj(yh4.reshape(B, T, -1))
        mo = blk.mlp(F.rms_norm(x, (D,)))
        if collect:
            PROJ_SUM[li] += torch.einsum('btd,nd->btn', mo, mlp_dirs[li]).sum(0)
            out[('mproj', li)] = torch.einsum('btd,nd->btn', mo, mlp_dirs[li]).cpu().numpy()
        if li in ab_mlps:
            for kk in ab_mlps[li]:
                pr = torch.einsum('btd,d->bt', mo, mlp_dirs[li, kk])
                mo = mo - (pr - projmeans[li][:, kk].unsqueeze(0)).unsqueeze(-1) * mlp_dirs[li, kk]
        x = x + mo
    logits = 30*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30)
    return (logits, out) if collect else logits

# ---- PASS A: means + activation magnitudes ----
YH_SUM = {li: torch.zeros(SEQL, NH, HD, device=DEV) for li in range(NL)}
PROJ_SUM = {li: torch.zeros(SEQL, N_SVD, device=DEV) for li in range(NL)}
head_act = np.zeros((NL*NH, NHELD, SEQL), np.float32)
mlp_act = np.zeros((NL*N_SVD, NHELD, SEQL), np.float32)
print("PASS A: means + activation magnitudes ...", flush=True)
for i in range(0, NHELD, BATCH):
    _, out = forward(HELD[i:i+BATCH], collect=True)
    b = HELD[i:i+BATCH].shape[0]
    for li in range(NL):
        hn = out[('hnorm', li)]
        for h in range(NH): head_act[li*NH + h, i:i+b] = hn[:, :, h]
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

def spec_of(comp):
    p = comp.split('.'); li = int(p[1][1:])
    if comp.startswith('h.'): return ('head', li, int(p[2]))
    return ('mlp', li, int(p[2][1:]))
def act_of(kind, li, ix):
    return head_act[li*NH + ix] if kind == 'head' else mlp_act[li*N_SVD + ix]

# ---- TOP-20 causally-important paths (by census trigger dCE) ----
census = json.load(open(f'{QK}/qk_census_difficulty.json'))
cen_rec = {r['comp']: r for r in census['records']}
TOP = [r['comp'] for r in census['records'][:20]]
SPEC = {c: spec_of(c) for c in TOP}
print(f"TOP-20 causally-important: {TOP}", flush=True)

# class-push signatures for naming (§68)
cp = json.load(open(f'{QK}/qk_unsup_classpush.json'))
cp_rec = {r['comp']: r for r in cp['records']}
def sig(c):
    r = cp_rec.get(c, {})
    return f"{r.get('pushed_class_sign','?')}{r.get('pushed_class','?')}"

# ---- per-path top-KCAUSAL firing mask + fixed union eval set ----
trig_mask = {}
for c in TOP:
    kind, li, ix = SPEC[c]
    a = act_of(kind, li, ix).copy().reshape(-1); a[bad_trigger.reshape(-1)] = -1e30
    tk = np.argpartition(a, -KCAUSAL)[-KCAUSAL:]
    mk = np.zeros(NHELD*SEQL, bool); mk[tk] = True
    trig_mask[c] = mk.reshape(NHELD, SEQL)
union = np.zeros((NHELD, SEQL), bool)
for c in TOP: union |= trig_mask[c]
union[:, SEQL-1] = False; union[bad_trigger] = False
# flat (seq,pos,tgt) list for the fixed eval set
seqs_u, poss_u = np.where(union)
tgts_u = held_np[seqs_u, poss_u + 1]
uniq_seq = np.unique(seqs_u)
print(f"fixed eval set: {len(seqs_u)} positions over {len(uniq_seq)} sequences", flush=True)

# ---- precompute per-position BASE cross-entropy ONCE over the eval set ----
tgt_all = torch.from_numpy(held_np).to(DEV)
base_ce_map = {}   # (seq,pos) -> base CE
for i in range(0, len(uniq_seq), BATCH):
    sb = uniq_seq[i:i+BATCH]
    base = forward(HELD[sb]).float()
    blp = F.log_softmax(base[:, :SEQL-1], -1)
    bce = -blp.gather(-1, tgt_all[sb][:, 1:].unsqueeze(-1)).squeeze(-1)   # (b,T-1)
    for bi, s in enumerate(sb):
        for p in poss_u[seqs_u == s]:
            base_ce_map[(int(s), int(p))] = float(bce[bi, p])
    del base, blp, bce
base_ce_vec = np.array([base_ce_map[(int(s), int(p))] for s, p in zip(seqs_u, poss_u)])
print("base CE precomputed.", flush=True)

# ---- eval a config: mean/SE dCE over the FIXED eval set (only ablated forward) ----
@torch.no_grad()
def eval_joint(ablations):
    abl_ce_vec = np.zeros(len(seqs_u))
    pos_by_seq = {int(s): np.where(seqs_u == s)[0] for s in uniq_seq}
    for i in range(0, len(uniq_seq), BATCH):
        sb = uniq_seq[i:i+BATCH]
        abl = forward(HELD[sb], ablations=ablations, yhmeans=YHMEAN, projmeans=PROJMEAN).float()
        alp = F.log_softmax(abl[:, :SEQL-1], -1)
        ace = -alp.gather(-1, tgt_all[sb][:, 1:].unsqueeze(-1)).squeeze(-1)
        for bi, s in enumerate(sb):
            idxs = pos_by_seq[int(s)]
            for j in idxs:
                abl_ce_vec[j] = float(ace[bi, poss_u[j]])
        del abl, alp, ace
    dce = abl_ce_vec - base_ce_vec
    n = len(dce)
    return float(dce.mean()), float(dce.std(ddof=1)/math.sqrt(n)) if n > 1 else 0.0

# ---- SOLOS on the fixed set ----
print("SOLOS on fixed eval set ...", flush=True)
solo = {}
for c in TOP:
    mn, se = eval_joint([SPEC[c]])
    solo[c] = {'dCE': round(mn, 4), 'SE': round(se, 4)}
    print(f"  solo {c:11s} dCE={mn:+.4f}+-{se:.4f}  sig={sig(c)}", flush=True)
sum_solo_pos = sum(max(solo[c]['dCE'], 0.0) for c in TOP)
sum_solo_signed = sum(solo[c]['dCE'] for c in TOP)

# ---- WHOLE-20 joint + multi-path residual ----
whole_mn, whole_se = eval_joint([SPEC[c] for c in TOP])
whole_ratio = whole_mn / (sum_solo_pos + 1e-9)
multipath_residual = whole_mn - sum_solo_pos
print(f"\nWHOLE-20 joint dCE={whole_mn:+.4f}+-{whole_se:.4f}  sum(solo+)={sum_solo_pos:.4f}"
      f"  super-additivity={whole_ratio:.2f}  multi-path residual={multipath_residual:+.4f}", flush=True)

# ---- pairwise super-additivity among the strongest 10 ----
STRONG = TOP[:10]
print("\nPAIRWISE super-additivity (strongest 10) ...", flush=True)
pair_joint = {}
pairs = []
for ii in range(len(STRONG)):
    for jj in range(ii+1, len(STRONG)):
        a, b = STRONG[ii], STRONG[jj]
        mn, se = eval_joint([SPEC[a], SPEC[b]])
        pair_joint[(a, b)] = mn
        superadd = mn - max(solo[a]['dCE'], 0) - max(solo[b]['dCE'], 0)
        pairs.append({'pair': [a, b], 'joint_dCE': round(mn, 4),
                      'sum_solo': round(max(solo[a]['dCE'],0)+max(solo[b]['dCE'],0), 4),
                      'super_additive': round(superadd, 4),
                      'ratio': round(mn/(max(solo[a]['dCE'],0)+max(solo[b]['dCE'],0)+1e-9), 2),
                      'sigs': [sig(a), sig(b)]})
pairs.sort(key=lambda p: -p['super_additive'])
print("  top super-additive pairs:", flush=True)
for p in pairs[:8]:
    print(f"    {p['pair'][0]:11s}+{p['pair'][1]:11s} joint={p['joint_dCE']:+.4f} "
          f"sumsolo={p['sum_solo']:+.4f} superadd={p['super_additive']:+.4f} ratio={p['ratio']} {p['sigs']}", flush=True)

# ---- greedy agglomerative grouping: seed strongest pair, grow by max super-additive marginal ----
print("\nGREEDY agglomerative grouping ...", flush=True)
assigned = set()
groups = []
def joint_of(members):
    return eval_joint([SPEC[c] for c in members])[0]
# order candidate seeds by pair super-additivity
seed_order = [p for p in pairs if p['super_additive'] > 0]
for seedp in seed_order:
    a, b = seedp['pair']
    if a in assigned or b in assigned: continue
    group = [a, b]; jS = seedp['joint_dCE']
    assigned.update(group)
    # grow
    while True:
        best_c, best_marg, best_j = None, 1e-4, None   # require positive super-additive marginal
        for c in TOP:
            if c in assigned: continue
            jSc = joint_of(group + [c])
            marg = jSc - jS - max(solo[c]['dCE'], 0.0)   # super-additive marginal beyond c's solo
            if marg > best_marg:
                best_marg, best_c, best_j = marg, c, jSc
        if best_c is None: break
        group.append(best_c); assigned.add(best_c); jS = best_j
    sum_s = sum(max(solo[c]['dCE'], 0.0) for c in group)
    groups.append({'members': group, 'joint_dCE': round(jS, 4), 'sum_solo': round(sum_s, 4),
                   'super_additive_residual': round(jS - sum_s, 4),
                   'super_additivity_ratio': round(jS/(sum_s+1e-9), 2),
                   'frac_of_multipath_residual': round((jS - sum_s)/(multipath_residual+1e-9), 3),
                   'sigs': [sig(c) for c in group]})
    print(f"  GROUP {group} joint={jS:+.4f} sumsolo={sum_s:+.4f} "
          f"ratio={groups[-1]['super_additivity_ratio']} sigs={groups[-1]['sigs']}", flush=True)
# leftover singletons
leftover = [c for c in TOP if c not in assigned]

# ---- random same-size control for the biggest group (by |members|) ----
groups_sorted = sorted(groups, key=lambda g: -len(g['members']))
control = None
if groups_sorted:
    big = groups_sorted[0]; k = len(big['members'])
    rng = np.random.RandomState(0)
    FAMILY = set(SPEC[c] for c in TOP)
    allpaths = [('head', li, h) for li in range(NL) for h in range(NH)] + \
               [('mlp', li, kk) for li in range(NL) for kk in range(N_SVD)]
    allpaths = [p for p in allpaths if p not in FAMILY]
    rand = []
    for _ in range(NRAND):
        pick = [allpaths[i] for i in rng.choice(len(allpaths), k, replace=False)]
        rand.append(eval_joint(pick)[0])
    rand = np.array(rand)
    control = {'group': big['members'], 'k': k,
               'group_joint_dCE': big['joint_dCE'],
               'rand_joint_mean': round(float(rand.mean()), 4),
               'rand_joint_std': round(float(rand.std(ddof=1)), 4),
               'rand_joint_max': round(float(rand.max()), 4),
               'rand_joint_p95': round(float(np.percentile(rand, 95)), 4),
               'group_over_random_z': round(float((big['joint_dCE'] - rand.mean())/(rand.std(ddof=1)+1e-9)), 2),
               'group_exceeds_all_random': bool(big['joint_dCE'] > rand.max())}
    print(f"\nRANDOM control (size {k}, same positions): mean={control['rand_joint_mean']:+.4f} "
          f"max={control['rand_joint_max']:+.4f} | group joint={big['joint_dCE']:+.4f} "
          f"z={control['group_over_random_z']} exceeds_all={control['group_exceeds_all_random']}", flush=True)

# ---- assemble ----
def name_group(g):
    caps = sum(1 for s in g['sigs'] if 'capital' in s)
    words = sum(1 for s in g['sigs'] if 'word' in s or 'subword' in s)
    heads = sum(1 for c in g['members'] if c.startswith('h.'))
    mlps = sum(1 for c in g['members'] if c.startswith('mlp.'))
    parts = []
    if caps: parts.append(f"{caps} capital")
    if words: parts.append(f"{words} word/subword")
    parts.append(f"{heads} heads/{mlps} feed-forward")
    return "; ".join(parts)
for g in groups: g['name'] = name_group(g)

summary = {
    'top20': TOP, 'top20_sigs': {c: sig(c) for c in TOP},
    'fixed_eval_positions': int(len(seqs_u)), 'fixed_eval_sequences': int(len(uniq_seq)),
    'solos': solo,
    'sum_solo_positive': round(sum_solo_pos, 4), 'sum_solo_signed': round(sum_solo_signed, 4),
    'whole20_joint_dCE': round(whole_mn, 4), 'whole20_joint_SE': round(whole_se, 4),
    'whole20_super_additivity_ratio': round(whole_ratio, 2),
    'whole20_multipath_residual': round(multipath_residual, 4),
    'top_super_additive_pairs': pairs[:10],
    'greedy_groups': groups, 'leftover_singletons': leftover,
    'biggest_group_random_control': control,
    'note_71_whole_model': 'whole-234 super-additivity was 2.87x (joint 3.376 vs sum-solo 1.177). '
                           'Here we resolve WHERE the top-20 combination residual concentrates.',
}
out = {
    'meta': {
        'model': 'bilin18', 'part': 'B -- multi-path combination structure', 'held_slice': 'FW[448:600,:128]',
        'KCAUSAL': KCAUSAL, 'BATCH': BATCH, 'NRAND': NRAND,
        'method': 'Greedy joint-ablation grouping (§61 conventions) over the top-20 causally-important '
                  'paths on a fixed union firing-position set: solos + whole-20 joint (global super-'
                  'additivity + multi-path residual); pairwise super-additivity; greedy agglomerative '
                  'groups seeded from the strongest super-additive pair and grown by max super-additive '
                  'marginal; same-size random control on the same positions. Forward + multi-ablation '
                  'VERBATIM from qk_unsup_classpush.py; groups named by §68 class-push signatures.',
    },
    'summary': summary,
}
json.dump(out, open(f'{QK}/qk_extend_coverage_2.json', 'w'), indent=2)

print("\n===== PART B SUMMARY =====", flush=True)
print(f"whole-20: joint {whole_mn:+.4f} vs sum-solo {sum_solo_pos:+.4f} -> super-additivity {whole_ratio:.2f}"
      f", multi-path residual {multipath_residual:+.4f} nats", flush=True)
print(f"discovered {len(groups)} super-additive groups; leftover singletons {leftover}", flush=True)
for g in groups:
    print(f"  {g['members']} ratio={g['super_additivity_ratio']} "
          f"residual={g['super_additive_residual']:+.4f} ({g['frac_of_multipath_residual']*100:.0f}% of top-20 residual) [{g['name']}]", flush=True)
print("Saved qk_extend_coverage_2.json", flush=True)
print("QK EXTEND COVERAGE PART B DONE", flush=True)
