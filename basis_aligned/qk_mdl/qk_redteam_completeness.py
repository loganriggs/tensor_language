"""ADVERSARIAL RED-TEAM of §74 ("MLP1 high-rank tail is IRREDUCIBLY DISTRIBUTED":
0/32 single-direction nameable, 76% of effect only under joint removal).

THE KEY CONFOUND under test: super-additivity under mean-ablation is GENERIC
(jointly ablating many orthogonal things exceeds the sum of solos because of
downstream interaction). So "76% only under joint removal" may be a trivial
mean-ablation artifact, NOT special superposition. We compute the SAME
joint-vs-sum-of-solos signature for CONTROLS:
  (1) MLP layer 1 top-32 SVD dirs           -- reproduce §74
  (2) MLP layer 16 top-32 SVD dirs          -- §73 says late layers are LOW-rank
  (3) MLP layer 17 top-32 SVD dirs          -- §73 says late layers are LOW-rank
  (4) MLP layer 1, 32 RANDOM orthogonal dirs
  (5)/(6) MLP layer 1, top-32 SVD subspace RANDOMLY ROTATED (basis-dependence)
  (7) MLP layer 1, top-32 NEURONS (bilinear hidden units, the computational
      primitive basis)  -- basis-dependence of nameability
If the 76% joint-only signature appears EQUALLY for the low-rank late layer and
for random dirs, it is a generic interaction, not evidence of special
superposition. If MLP1 shows MUCH more joint-only effect, §74 survives.
And if the NEURON basis (or a rotation) makes MLP1 dirs individually nameable,
"irreducibly distributed" is basis-dependent.

FORWARD + mean-ablation + MLP-SVD-dir construction copied VERBATIM from
qk_mlp_superposition.py / qk_mlp1_tail.py. Class-clearness bar (trigger_dCE_z>=3
& trigger_dCE>=0.02 at top-KCAUSAL firing positions) VERBATIM from qk_mlp1_tail.py.
Held-back FW[448:600,:128], paired standard errors.
"""
import json, sys, math, time, subprocess
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot
torch.manual_seed(0)
DEV = 'cuda'; QK = '/workspace/tensor_language/basis_aligned/qk_mdl'

# ---------------- GPU GUARD (verbatim) ----------------
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
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('gpt2')
K32 = 32
KCAUSAL = 200
print(f"bilin18 NL={NL} NH={NH} HD={HD} D={D} V={V}", flush=True)

FINEWEB = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))
SEQL = 128
TRAIN = FINEWEB[0:256, :SEQL].to(DEV)     # ONLY to recompute MLP dirs / neuron energies
HELD = FINEWEB[448:600, :SEQL].to(DEV)    # held-back verification slice
NHELD = HELD.shape[0]
BATCH = 6

# special tokens excluded from trigger selection (verbatim census/tail)
_special = {tok.eos_token_id}
for _t in range(V):
    if '�' in tok.decode([_t]): _special.add(_t)
SPECIAL = np.array(sorted(_special))

LAYERS_FOR_GRAM = [1, 16, 17]
gram = {li: torch.zeros(D, D, device=DEV) for li in LAYERS_FOR_GRAM}
# neuron energy for L1: E[h_i^2] * ||Down[:,i]||^2  (output-energy contribution)
mlp1 = m.transformer.h[1].mlp
NHID = mlp1.Down.weight.shape[1]                      # 4*D
down_col_norm2 = (mlp1.Down.weight.data ** 2).sum(0)  # (NHID,) ||Down[:,i]||^2
neuron_h2_sum = torch.zeros(NHID, device=DEV)
neuron_n = 0

@torch.no_grad()
def fwd_gram(idx):
    global neuron_n
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
        x = x + a.c_proj(yh4.reshape(B, T, -1))
        zin = F.rms_norm(x, (D,))
        mo = blk.mlp(zin)
        if li in LAYERS_FOR_GRAM:
            gram[li].add_(torch.einsum('btd,bte->de', mo, mo))
        if li == 1:
            h = mlp1.Left(zin) * mlp1.Right(zin)          # (B,T,NHID) bilinear hidden
            neuron_h2_sum.add_((h*h).sum(dim=(0, 1)))
            neuron_n += B*T
        x = x + mo

print("Recomputing grams (L1,L16,L17) + L1 neuron energies from TRAIN ...", flush=True)
for i in range(0, TRAIN.shape[0], BATCH):
    fwd_gram(TRAIN[i:i+BATCH])

# top-32 SVD dirs per layer
SVD = {}
gram_eval = {}
for li in LAYERS_FOR_GRAM:
    ev, evec = torch.linalg.eigh(gram[li])
    SVD[li] = evec[:, -K32:].T.flip(0).contiguous()      # (32,D) descending
    gram_eval[li] = ev.cpu().numpy()
del gram

# neuron energy ranking for L1
neuron_energy = (neuron_h2_sum/neuron_n) * down_col_norm2   # (NHID,)
top_neurons = torch.argsort(neuron_energy, descending=True)[:K32].cpu().numpy()
NEURON_DIRS_out = mlp1.Down.weight.data[:, top_neurons].T.contiguous()  # (32,D) output directions (NOT orthonormal)

# random orthonormal in D
g = torch.Generator(device=DEV); g.manual_seed(1234)
A = torch.randn(D, K32, device=DEV, generator=g)
RAND = torch.linalg.qr(A)[0].T.contiguous()              # (32,D)

# random rotations WITHIN the L1 top-32 SVD subspace
def rotated_subspace(seed):
    gg = torch.Generator(device=DEV); gg.manual_seed(seed)
    R = torch.linalg.qr(torch.randn(K32, K32, device=DEV, generator=gg))[0]  # (32,32) orthogonal
    return (R @ SVD[1]).contiguous()                     # (32,D) span == SVD[1] span
ROT1 = rotated_subspace(7); ROT2 = rotated_subspace(99)
print("Bases built.", flush=True)

# =====================================================================================
# Forward with layer-LI intervention. mode='proj': project-out mean-ablate a set of
# orthonormal dirs (verbatim tail). mode='neuron': mean-ablate hidden units of L1.
# =====================================================================================
@torch.no_grad()
def forward(idx, LI=None, mode=None, dirs=None, PM=None, neurons=None, neuron_mean=None, collect_which=None):
    """collect_which: 'proj'(dirs) returns (B,T,k) projection coeffs of layer-LI mo;
                      'neuron' returns (B,T,k) hidden activations of selected neurons."""
    B, T = idx.shape
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    collected = None
    for li in range(NL):
        blk = m.transformer.h[li]; x = blk.lambdas[0]*x + blk.lambdas[1]*x0; a = blk.attn; hcur = F.rms_norm(x, (D,))
        def qk(lin): z = F.rms_norm(lin(hcur).view(B, T, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
        v = a.c_v(hcur).view(B, T, NH, HD)
        if v1 is None: v1 = v
        v = (1-a.lamb)*v + a.lamb*v1.view_as(v)
        q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k)/HD; s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD
        pat = (s1*s2).masked_fill(~mask, 0.0)             # attention ALWAYS intact
        yh4 = torch.einsum('bhqk,bkhd->bqhd', pat, v)
        x = x + a.c_proj(yh4.reshape(B, T, -1))
        zin = F.rms_norm(x, (D,))
        if li == LI and mode == 'neuron':
            mlp = blk.mlp
            h = mlp.Left(zin) * mlp.Right(zin)            # (B,T,NHID); gated=False for bilin18
            if collect_which == 'neuron':
                collected = h[:, :, neurons].detach().clone()
            if neurons is not None and neuron_mean is not None:
                h = h.clone()
                h[:, :, neurons] = neuron_mean.unsqueeze(0).to(h.dtype)   # (T,k)->(B,T,k)
            mo = mlp.Down(h) + mlp.Down_bias
        else:
            mo = blk.mlp(zin)
            if li == LI and collect_which == 'proj':
                collected = torch.einsum('btd,kd->btk', mo, dirs).detach().clone()
            if li == LI and mode == 'proj' and dirs is not None and PM is not None:
                pr = torch.einsum('btd,kd->btk', mo, dirs)
                coeff = pr - PM.unsqueeze(0)
                mo = mo - torch.einsum('btk,kd->btd', coeff, dirs)
        x = x + mo
    logits = 30*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30)
    return (logits, collected) if collect_which else logits

# trigger-selection scaffolding (verbatim tail)
held_np = HELD.cpu().numpy()
tgt_all = torch.from_numpy(held_np).to(DEV)
pos_t = np.tile(np.arange(SEQL), NHELD).reshape(NHELD, SEQL)
is_special = np.isin(held_np, SPECIAL)
valid_next = pos_t < (SEQL - 1)
bad_trigger = (pos_t == 0) | is_special | ~valid_next
NPOS_valid = int(valid_next[:, :SEQL-1].sum())

def stats(s, sq, n):
    if n <= 1: return 0.0, 0.0
    mean = s/n; var = max(sq/n - mean*mean, 0.0)*n/(n-1)
    return mean, math.sqrt(var/n)

def run_basis(name, LI, mode, dirs=None, neurons=None):
    """Return dict of joint-vs-solo + nameability metrics for one 32-direction basis."""
    print(f"\n[{name}] LI={LI} mode={mode} ...", flush=True)
    # ---- PASS A: per-direction activation magnitudes + per-position means for ablation ----
    act = np.zeros((K32, NHELD, SEQL), np.float32)
    if mode == 'proj':
        PSUM = torch.zeros(SEQL, K32, device=DEV)
        for i in range(0, NHELD, BATCH):
            sb = slice(i, min(i+BATCH, NHELD))
            _, coll = forward(HELD[sb], LI=LI, mode='proj', dirs=dirs, PM=None, collect_which='proj')
            PSUM += coll.sum(0)
            cc = coll.cpu().numpy()
            for kk in range(K32): act[kk, i:i+cc.shape[0]] = np.abs(cc[:, :, kk])
        PM = PSUM / NHELD                          # (T,32)
        nmean = None
    else:  # neuron
        NSUM = torch.zeros(SEQL, K32, device=DEV)
        for i in range(0, NHELD, BATCH):
            sb = slice(i, min(i+BATCH, NHELD))
            _, coll = forward(HELD[sb], LI=LI, mode='neuron', neurons=neurons, collect_which='neuron')
            NSUM += coll.sum(0)
            cc = coll.cpu().numpy()
            for kk in range(K32): act[kk, i:i+cc.shape[0]] = np.abs(cc[:, :, kk])
        nmean = NSUM / NHELD                        # (T,32) per-position neuron means
        PM = None
    # trigger masks
    trig_mask = {}
    for kk in range(K32):
        aa = act[kk].reshape(-1).copy(); aa[bad_trigger.reshape(-1)] = -1e30
        tk = np.argpartition(aa, -KCAUSAL)[-KCAUSAL:]
        mk = np.zeros(NHELD*SEQL, bool); mk[tk] = True; trig_mask[kk] = mk.reshape(NHELD, SEQL)
    # ---- PASS B: base + joint + 32 solos ----
    g_sum = {k: 0.0 for k in range(K32)}; g_sq = {k: 0.0 for k in range(K32)}; g_n = {k: 0 for k in range(K32)}
    t_sum = {k: 0.0 for k in range(K32)}; t_sq = {k: 0.0 for k in range(K32)}; t_n = {k: 0 for k in range(K32)}
    j_sum = 0.0; j_sq = 0.0; j_n = 0
    t0 = time.time()
    for bi, i in enumerate(range(0, NHELD, BATCH)):
        sb = slice(i, min(i+BATCH, NHELD)); idx = HELD[sb]
        tgt = tgt_all[sb]
        base = forward(idx).float()
        blp = F.log_softmax(base[:, :SEQL-1], -1)
        bce = -blp.gather(-1, tgt[:, 1:].unsqueeze(-1)).squeeze(-1); del blp
        vm = torch.from_numpy(valid_next[sb, :SEQL-1]).to(DEV)
        # joint (all 32)
        if mode == 'proj':
            jabl = forward(idx, LI=LI, mode='proj', dirs=dirs, PM=PM).float()
        else:
            jabl = forward(idx, LI=LI, mode='neuron', neurons=neurons, neuron_mean=nmean).float()
        jlp = F.log_softmax(jabl[:, :SEQL-1], -1)
        jce = -jlp.gather(-1, tgt[:, 1:].unsqueeze(-1)).squeeze(-1); del jlp, jabl
        dj = (jce - bce)[vm]
        j_sum += float(dj.sum()); j_sq += float((dj*dj).sum()); j_n += int(dj.numel())
        # solos
        for kk in range(K32):
            if mode == 'proj':
                abl = forward(idx, LI=LI, mode='proj', dirs=dirs[kk:kk+1], PM=PM[:, kk:kk+1]).float()
            else:
                abl = forward(idx, LI=LI, mode='neuron', neurons=neurons[kk:kk+1],
                              neuron_mean=nmean[:, kk:kk+1]).float()
            alp = F.log_softmax(abl[:, :SEQL-1], -1)
            ace = -alp.gather(-1, tgt[:, 1:].unsqueeze(-1)).squeeze(-1); del alp, abl
            dce = (ace - bce)
            dg = dce[vm]
            g_sum[kk] += float(dg.sum()); g_sq[kk] += float((dg*dg).sum()); g_n[kk] += int(dg.numel())
            tm = torch.from_numpy(trig_mask[kk][sb, :SEQL-1]).to(DEV)
            if tm.any():
                dt = dce[tm]
                t_sum[kk] += float(dt.sum()); t_sq[kk] += float((dt*dt).sum()); t_n[kk] += int(dt.numel())
            del dce
        del base, bce
        if bi % 6 == 0:
            print(f"  batch {bi+1}/{(NHELD+BATCH-1)//BATCH} elapsed {time.time()-t0:.0f}s", flush=True)
    joint_mean, joint_se = stats(j_sum, j_sq, j_n)
    solos = []
    for kk in range(K32):
        gm, gse = stats(g_sum[kk], g_sq[kk], g_n[kk])
        tm_, tse = stats(t_sum[kk], t_sq[kk], t_n[kk])
        z = tm_/tse if tse > 0 else 0.0
        clear = (z >= 3.0) and (tm_ >= 0.02)
        solos.append({'rank': kk, 'solo_global_dCE': round(gm, 6), 'solo_global_SE': round(gse, 6),
                      'trigger_dCE': round(tm_, 5), 'trigger_dCE_SE': round(tse, 5), 'trigger_dCE_z': round(z, 2),
                      'nameable': bool(clear)})
    solo_sum = sum(max(0.0, s['solo_global_dCE']) for s in solos)
    solo_sum_raw = sum(s['solo_global_dCE'] for s in solos)
    n_nameable = sum(s['nameable'] for s in solos)
    max_z = max(s['trigger_dCE_z'] for s in solos)
    res = {
        'name': name, 'LI': LI, 'mode': mode,
        'joint32_global_dCE': round(joint_mean, 6), 'joint32_global_SE': round(joint_se, 6),
        'solo_sum_clamped': round(solo_sum, 6), 'solo_sum_raw': round(solo_sum_raw, 6),
        'solo_frac_of_joint': round(solo_sum/joint_mean, 4) if joint_mean > 1e-9 else None,
        'joint_only_frac': round(1 - solo_sum/joint_mean, 4) if joint_mean > 1e-9 else None,
        'n_nameable': int(n_nameable), 'max_trigger_z': round(max_z, 2),
        'solos': solos,
    }
    print(f"  joint32={joint_mean:.4f}+-{joint_se:.4f}  solo_sum={solo_sum:.4f}  "
          f"solo_frac={res['solo_frac_of_joint']}  joint_only={res['joint_only_frac']}  "
          f"n_nameable={n_nameable}  max_z={max_z:.2f}", flush=True)
    return res

BASES = [
    ('MLP1_SVD32',  1,  'proj',   SVD[1],  None),
    ('MLP16_SVD32', 16, 'proj',   SVD[16], None),
    ('MLP17_SVD32', 17, 'proj',   SVD[17], None),
    ('MLP1_RAND32', 1,  'proj',   RAND,    None),
    ('MLP1_ROT32_a',1,  'proj',   ROT1,    None),
    ('MLP1_ROT32_b',1,  'proj',   ROT2,    None),
    ('MLP1_NEURON32',1, 'neuron', None,    top_neurons),
]
results = []
for (name, LI, mode, dirs, neurons) in BASES:
    results.append(run_basis(name, LI, mode, dirs=dirs, neurons=neurons))

# gram energy top-4 fraction per layer (rank descriptor)
def top4_energy(li):
    e = gram_eval[li]; return float(e[-4:].sum()/e.sum())

out = {
    'meta': {
        'model': 'bilin18', 'held_slice': 'FW[448:600,:128]', 'K_dirs': K32, 'KCAUSAL': KCAUSAL, 'BATCH': BATCH,
        'n_valid_positions': NPOS_valid,
        'attack': '§74 red-team: is "76% only under joint removal / 0 of 32 nameable" a GENERIC mean-ablation '
                  'interaction (appears equally for low-rank late layer + random dirs) or special superposition?',
        'currency': 'mean-ablation (per-position held mean) delta cross-entropy per valid held position (nats), paired SE',
        'forward': 'VERBATIM qk_mlp_superposition.py / qk_mlp1_tail.py (project-out mean-ablation); '
                   'neuron mode mean-ablates bilinear hidden units of L1',
        'nameable_rule': 'trigger_dCE_z>=3 AND trigger_dCE>=0.02 at top-200 firing positions (VERBATIM §74)',
        'gram_top4_energy_frac': {str(li): round(top4_energy(li), 4) for li in LAYERS_FOR_GRAM},
    },
    'results': results,
    'summary_table': [
        {'basis': r['name'], 'joint32': r['joint32_global_dCE'], 'joint_SE': r['joint32_global_SE'],
         'solo_sum': r['solo_sum_clamped'], 'solo_frac_of_joint': r['solo_frac_of_joint'],
         'joint_only_frac': r['joint_only_frac'], 'n_nameable': r['n_nameable'], 'max_z': r['max_trigger_z']}
        for r in results
    ],
}
json.dump(out, open(f'{QK}/qk_redteam_completeness.json', 'w'), indent=2)
print("\n===== §74 RED-TEAM SUMMARY =====", flush=True)
print(f"{'basis':16s} {'joint32':>9s} {'solo_sum':>9s} {'solo%':>7s} {'jointonly%':>10s} {'nameable':>9s} {'maxz':>6s}", flush=True)
for r in results:
    print(f"{r['name']:16s} {r['joint32_global_dCE']:9.4f} {r['solo_sum_clamped']:9.4f} "
          f"{100*r['solo_frac_of_joint']:6.1f}% {100*r['joint_only_frac']:9.1f}% "
          f"{r['n_nameable']:9d} {r['max_trigger_z']:6.2f}", flush=True)
print("\nSaved qk_redteam_completeness.json", flush=True)
print("QK REDTEAM COMPLETENESS (attack 1) DONE", flush=True)
