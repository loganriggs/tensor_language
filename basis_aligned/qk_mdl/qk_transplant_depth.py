"""BOUNDING THE SECTION-94 EDITING LAW: mid-stack context transplants -- part 1 of 2 (FORCE
at three depths + same-target layer-17 reference).

Background: section 94 established that transplanting a coarse-group vector at LAYER 17's
feed-forward input achieves surgical overrides (force capital +0.0280 for attention-earlier,
+0.2474 for MLP-EARLIER donors; suppress -0.0703) with EXACTLY zero collateral -- but layer
17 is the LAST layer, so a per-position input edit affects only that position's prediction:
collateral is free BY CONSTRUCTION. THIS SCRIPT bounds the scope: repeat the section-94
force test (MLP-EARLIER group, the group section 94 found carries the forceable boundary
context) at L15 (history-reader, own attention dead per sections 89/90), L12 (early history
regime), L8 (the crossover), plus L17 at the SAME targets as the apples-to-apples reference.
Mid-stack the edit changes that layer's MLP output at position p, which propagates to
position p's own later layers AND (via later attention reading position p) to positions
AFTER p. MEASURE: (a) target effect -- capital-probability change AT edited positions (washout
question: do downstream layers compensate, cf section 83's redundancy?); (b) collateral --
delta cross-entropy at NON-edited positions split into BEFORE the first preceding edit
(causal gate: must be exactly zero) and AFTER (propagation cost), with the distance profile;
(c) surgery score = target gain / after-position collateral, versus L17's infinite score.

EDIT (verbatim section 94, LI now a parameter): at target positions p,
x_mlp_in(p) = x_pre(p) + (Me_donor - Me_own(p)); blk.mlp recomputes; the edited mlp output
enters the residual and propagates through the remaining layers. NOTE the groups are defined
PER LAYER (qk_allterm_census.py): Me at layer L is the lambda-decayed sum of mlp outputs
before the immediately preceding one -- donor vectors are collected at the SAME layer.

MACHINERY VERBATIM: 5-group coarse-group-accumulator forward from qk_allterm_census.py /
qk_stream_transplant.py (groups = [cE*x0, SA, aout, SM, MR] at LI, x is x_pre); capital-class
metric, lexical classes, position splits, donor assignment (same-sequence boundary donors ->
not-due targets), zero-collateral gate check from qk_stream_transplant.py. GATES: group-sum
identity per layer (<1e-4 rel); base cross-entropy 3.4946; base P(capital) sentence-punct
0.34902 / newline 0.55269; before-positions max abs logit difference exactly zero.
Held-back FW[448:600,:128], paired standard errors, batch 6, <4GB, GPU guard.
Output: qk_transplant_depth.json (partial; part 2 adds placebo controls + washout analysis)
and qk_transplant_depth_state.pt.
"""
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import json, sys, math, time, subprocess
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot
torch.manual_seed(0); np.random.seed(0)
DEV = 'cuda'; QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
OUT = f'{QK}/qk_transplant_depth.json'
STATE = f'{QK}/qk_transplant_depth_state.pt'

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
tok = AutoTokenizer.from_pretrained('gpt2')   # model exposes none -- FLAGGED (matches prior sections)
FW = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))
SEQL = 128
HELD = FW[448:600, :SEQL].to(DEV); B0 = 6
S_, T_ = HELD.shape
NHELD = S_
LAYERS = [8, 12, 15, 17]   # crossover, early-history, history-reader (attention dead), readout reference
print(f"bilin18 NL={NL} D={D} NH={NH} held {S_}x{T_} layers {LAYERS}", flush=True)

# group naming (VERBATIM qk_allterm_census.py): groups = [E, Ae, Ar, Me, Mr]
GNAMES = ['E', 'Ae', 'Ar', 'Me', 'Mr']
GI_AE = 1   # attention-earlier accumulator (SA)
GI_ME = 3   # mlp-earlier accumulator (SM)

@torch.no_grad()
def fwd(idx, LI=17, mode=None, EM=None, DON=None, gi=None, dose=1.0, stats=None):
    """Forward verbatim from qk_allterm_census.py / qk_stream_transplant.py
    (coarse-group stream accumulators to LI); RETURNS LOGITS. LI is now a parameter.
    mode: None (full model) | 'collect' (store Me group grid + group-sum gate at LI)
        | 'transplant' (x_mlp_in = x_pre + EM * dose * (DON - groups[gi]); mlp recomputes;
          the edited mlp output enters the residual and propagates downstream)."""
    B, T = idx.shape
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    track = mode is not None
    if track:
        cE = torch.ones((), device=DEV)
        SA = torch.zeros_like(x); SM = torch.zeros_like(x); MR = torch.zeros_like(x)
    for li in range(NL):
        blk = m.transformer.h[li]; x = blk.lambdas[0]*x + blk.lambdas[1]*x0; a = blk.attn; hcur = F.rms_norm(x, (D,))
        if track and li <= LI:
            cE = blk.lambdas[0]*cE + blk.lambdas[1]
            SA = blk.lambdas[0]*SA; SM = blk.lambdas[0]*SM; MR = blk.lambdas[0]*MR
        def qk(l): z = F.rms_norm(l(hcur).view(B, T, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
        v = a.c_v(hcur).view(B, T, NH, HD)
        if v1 is None: v1 = v
        v = (1-a.lamb)*v + a.lamb*v1.view_as(v)
        q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k)/HD; s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD
        pat = (s1*s2).masked_fill(~mask, 0.0); yh = torch.einsum('bhqk,bkhd->bqhd', pat, v)
        aout = a.c_proj(yh.reshape(B, T, -1)); x = x + aout
        mo = blk.mlp(F.rms_norm(x, (D,)))
        if track and li == LI:
            groups = [cE*x0, SA, aout, SM, MR]     # E, Ae, Ar, Me, Mr; x is x_pre here
            if mode == 'collect':
                gs = groups[0] + groups[1] + groups[2] + groups[3] + groups[4]
                num = (gs - x).norm(dim=-1); den = x.norm(dim=-1).clamp_min(1e-8)
                stats['maxrel'] = max(stats['maxrel'], float((num/den).max()))
                stats['fro_num'] += float((gs - x).pow(2).sum()); stats['fro_den'] += float(x.pow(2).sum())
                stats['ME'].append(groups[GI_ME].cpu())
            elif mode == 'transplant':
                x_in = x + EM.unsqueeze(-1) * (dose * (DON - groups[gi]))
                mo = blk.mlp(F.rms_norm(x_in, (D,)))
            del groups
        x = x + mo
        if track and li < LI:
            SA = SA + aout; SM = SM + MR; MR = mo
    return 30*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30)

# --- lexical classes copied VERBATIM from qk_stream_transplant.py / qk_arc_caps.py ---
BRACKETS_OPEN=set("([{<"); BRACKETS_CLOSE=set(")]}>")
QUOTE_OPEN=set("“‘`"); QUOTE_CLOSE=set("”’"); QUOTE_STRAIGHT=set("\"'")
PUNCT=set(".,;:!?—–-…*|/\\~@#%^&+=_")
COORDINATORS={"and","or","but","nor","yet","so"}
DETERMINERS={"the","a","an","this","that","these","those","some","any","each","every","no","another","such"}
PRONOUNS={"i","we","you","he","she","it","they","them","us","me","him","her","which","who"}
def lex1(s):
    if s=="": return 'other'
    if ('�' in s) or (s==tok.eos_token or '<|endoftext|>' in s): return 'special'
    if '\n' in s: return 'newline'
    body=s.strip(); low=body.lower()
    if body=="": return 'other'
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
    lead=s.startswith(' ')
    if lead and body.isalpha() and len(body)>1: return 'word'
    if (not lead) and body.isalpha() and body[0].islower(): return 'subword'
    return 'other'
VOCAB_CLASS = np.array([lex1(tok.decode([t])) for t in range(V)], dtype=object)
CAP_MASK = torch.from_numpy((VOCAB_CLASS=='capital')).to(DEV)
SENT_END = set('.!?…')
_special={tok.eos_token_id}
for _t in range(V):
    if '�' in tok.decode([_t]): _special.add(_t)
SPECIAL=np.array(sorted(_special))
NEWLINE_BOOL = np.array([('\n' in tok.decode([t])) for t in range(V)])
def _is_sent_end_tok(t):
    s=tok.decode([int(t)]).strip(); return len(s)>0 and all(ch in SENT_END for ch in s)
SENTEND_BOOL = np.array([_is_sent_end_tok(t) for t in range(V)])
CAP_BOOL     = (VOCAB_CLASS=='capital')
PUNCT_BOOL   = (VOCAB_CLASS=='punct')
WORD_BOOL    = np.isin(VOCAB_CLASS, np.array(['word','subword']))

def dsn_grid(tokens):
    N,T=tokens.shape; nl=NEWLINE_BOOL[tokens]
    seg=np.zeros((N,T),np.int64); cur=np.full(N,-1)
    for t in range(T):
        seg[:,t]=np.where(cur>=0,cur,0)
        cur=np.where(nl[:,t],t,cur)
    return np.arange(T)[None,:]-seg
held_np = HELD.cpu().numpy()
HE_DSN  = dsn_grid(held_np)

# ---------------- position-type grids (VERBATIM qk_stream_transplant.py) ----------------
posmat=np.tile(np.arange(SEQL),NHELD).reshape(NHELD,SEQL)
held_special=np.isin(held_np,SPECIAL)
cur_class=VOCAB_CLASS[held_np]
nxt=np.zeros_like(held_np); nxt[:,:-1]=held_np[:,1:]
next_special=np.zeros_like(held_special); next_special[:,:-1]=held_special[:,1:]
valid=(posmat>0)&(posmat<SEQL-1)&~held_special&~next_special
cur_sentend=SENTEND_BOOL[held_np]
cur_newline=NEWLINE_BOOL[held_np]
next_cap=CAP_BOOL[nxt]
next_word=WORD_BOOL[nxt]

M_sentend  = valid & cur_sentend
M_newline  = valid & cur_newline
M_boundary = (M_sentend | M_newline)                           # boundary = capital contextually DUE
M_target_notdue = valid & next_word & (HE_DSN>=8) & ~cur_sentend & ~cur_newline
print(f"counts: valid={valid.sum()} boundary={M_boundary.sum()} "
      f"target_notdue={M_target_notdue.sum()}", flush=True)

# ---------------- target + donor assignment (VERBATIM machinery, n=300) ----------------
rng = np.random.default_rng(0)
def sample_targets(mask, n):
    pool = np.argwhere(mask)
    sel = pool[rng.permutation(len(pool))[:min(n, len(pool))]]
    return sel   # (n,2) rows (s,t)

def assign_donors(targets, donor_mask):
    """for each target (s,t): random donor position from the SAME sequence if the donor
    pool has one (excluding the target position itself), else from the global pool."""
    by_seq = [np.flatnonzero(donor_mask[s]) for s in range(S_)]
    glob = np.argwhere(donor_mask)
    out = np.zeros_like(targets); same = 0
    for r, (s, t) in enumerate(targets):
        pool = by_seq[s]; pool = pool[pool != t]
        if len(pool) > 0:
            out[r] = (s, int(rng.choice(pool))); same += 1
        else:
            out[r] = glob[int(rng.integers(len(glob)))]
    return out, same/len(targets)

N_TGT = 300
TGT_FORCE = sample_targets(M_target_notdue, N_TGT)
DON_FORCE, frac_same_force = assign_donors(TGT_FORCE, M_boundary)   # boundary donors
print(f"targets: force {len(TGT_FORCE)} (same-seq donors {frac_same_force:.2f})", flush=True)

def mask_of(targets):
    g = np.zeros((S_, T_), bool)
    g[targets[:,0], targets[:,1]] = True
    return g
EM_FORCE = mask_of(TGT_FORCE)

# ---------------- BEFORE/AFTER split + distance-to-nearest-preceding-edit grid ----------------
# an edit at position p can causally reach only position p itself and positions t > p in the
# SAME sequence (later attention reads position p's keys/values). BEFORE = non-edited positions
# with no edited position strictly before them (includes all positions of unedited sequences):
# their logits must be BIT-IDENTICAL (the causal gate). AFTER = non-edited positions with at
# least one edit strictly before: the propagation cost lives here.
DIST = np.full((S_, T_), -1, np.int64)   # distance to nearest strictly-preceding edit
for s in range(S_):
    last = -1
    for t in range(T_):
        if last >= 0: DIST[s, t] = t - last
        if EM_FORCE[s, t]: last = t
M_after_all  = (DIST > 0) & ~EM_FORCE
M_before_all = (DIST < 0) & ~EM_FORCE
M_before = valid & M_before_all
M_after  = valid & M_after_all
n_seq_edited = int(np.unique(TGT_FORCE[:,0]).size)
print(f"split: before={M_before.sum()} after={M_after.sum()} edited={EM_FORCE.sum()} "
      f"(sequences touched {n_seq_edited}/{S_})", flush=True)
DIST_BINS = [(1,2),(3,4),(5,8),(9,16),(17,32),(33,64),(65,127)]

# ---------------- stats helpers (VERBATIM qk_stream_transplant.py) ----------------
def se(x):
    x=np.asarray(x,float); n=len(x)
    return float(x.std(ddof=1)/math.sqrt(n)) if n>1 else float('nan')
def stat(vals):
    vals=np.asarray(vals,float)
    return {'mean':round(float(vals.mean()),6) if len(vals) else None,
            'SE':round(se(vals),6) if len(vals)>1 else None, 'n':int(len(vals))}
def paired(a_grid, b_grid, mask):   # a - b at mask, paired per position
    d=(a_grid[mask]-b_grid[mask]); return stat(d)

@torch.no_grad()
def grids_tr(spec, LI):
    """spec: None (natural model) | dict(gi, EMg (S,T) bool np, DONg (S,T,D) cpu tensor, dose).
    Returns Pcap, CE grids; for spec != None also the gate numbers now split BEFORE/AFTER:
    max abs logit difference over before-positions (must be exactly zero), over
    after-positions (the propagation), and over edited positions."""
    Pcap=np.full((NHELD,SEQL),np.nan,np.float32); CE=np.full((NHELD,SEQL),np.nan,np.float32)
    max_bf = 0.0; bf_zero = 0; bf_tot = 0
    max_af = 0.0; af_zero = 0; af_tot = 0; max_ed = 0.0
    for i in range(0,NHELD,B0):
        idx=HELD[i:i+B0]; b=idx.shape[0]
        if spec is None:
            lg = fwd(idx).float()
        else:
            EM = torch.from_numpy(spec['EMg'][i:i+b]).to(DEV)
            DON = spec['DONg'][i:i+b].to(DEV)
            lg = fwd(idx, LI=LI, mode='transplant', EM=EM.float(), DON=DON,
                     gi=spec['gi'], dose=spec['dose']).float()
            lgb = fwd(idx).float()
            dmax = (lg-lgb).abs().amax(-1)                      # (b,T)
            bf = torch.from_numpy(M_before_all[i:i+b]).to(DEV)
            af = torch.from_numpy(M_after_all[i:i+b]).to(DEV)
            if bf.any():
                max_bf = max(max_bf, float(dmax[bf].max()))
                bf_zero += int((dmax[bf]==0).sum()); bf_tot += int(bf.sum())
            if af.any():
                max_af = max(max_af, float(dmax[af].max()))
                af_zero += int((dmax[af]==0).sum()); af_tot += int(af.sum())
            if EM.any(): max_ed = max(max_ed, float(dmax[EM].max()))
            del lgb, dmax, EM, DON, bf, af
        sm=torch.softmax(lg,-1)
        pc=sm[...,CAP_MASK].sum(-1)
        lp=F.log_softmax(lg[:,:-1],-1); tgt=idx[:,1:]
        nll=-lp.gather(-1,tgt.unsqueeze(-1)).squeeze(-1)
        Pcap[i:i+b]=pc.cpu().numpy(); CE[i:i+b,:-1]=nll.cpu().numpy()
        del lg,sm,pc,lp,nll
    if spec is None: return Pcap, CE
    return Pcap, CE, {
        'max_abs_logit_diff_before': max_bf,
        'frac_before_exactly_zero': round(bf_zero/max(bf_tot,1), 6),
        'max_abs_logit_diff_after': max_af,
        'frac_after_exactly_zero': round(af_zero/max(af_tot,1), 6),
        'max_abs_logit_diff_edited': max_ed}

# ---------------- natural reference + known-answer gates ----------------
print("natural (exact full model) grids ...", flush=True)
Pnat, Cnat = grids_tr(None, None)
base_ce = float(np.nanmean(Cnat))
print(f"base CE mean {base_ce:.4f} (expect 3.4946)", flush=True)
assert abs(base_ce - 3.4946) < 0.002, "base cross-entropy gate FAILED"
p_se = float(Pnat[M_sentend].mean()); p_nl = float(Pnat[M_newline].mean())
print(f"GATE base P(capital): sentence-punct {p_se:.5f} (expect 0.34902) "
      f"newline {p_nl:.5f} (expect 0.55269)", flush=True)
assert abs(p_se-0.34902) < 0.005 and abs(p_nl-0.55269) < 0.005, "arc_caps step-1 gate FAILED"

def build_spec(targets, donors, src_all, gi, dose):
    EMg = np.zeros((S_, T_), bool)
    DONg = torch.zeros(S_, T_, D)
    for (s, t), (ds, dt) in zip(targets, donors):
        EMg[s, t] = True
        DONg[s, t] = src_all[ds, dt]
    return {'gi': gi, 'EMg': EMg, 'DONg': DONg, 'dose': dose}

def config_report(name, Pg, Cg, gate, EMg):
    tgt = EMg
    rep = {'n_edited': int(tgt.sum()),
           'P_capital_before_edit': stat(Pnat[tgt]), 'P_capital_after_edit': stat(Pg[tgt]),
           'dP_capital_edited': paired(Pg, Pnat, tgt),
           'dCE_edited': paired(Cg, Cnat, tgt),
           'dCE_before_positions': paired(Cg, Cnat, M_before),
           'dCE_after_positions': paired(Cg, Cnat, M_after),
           'dP_capital_after_positions': paired(Pg, Pnat, M_after),
           'global_dCE_all_predict_positions':
               round(float(np.mean(Cg[:, :T_-1] - Cnat[:, :T_-1])), 6),
           'zero_collateral_gate': gate}
    # distance profile of the after-position collateral
    prof = {}
    for lo, hi in DIST_BINS:
        mb = M_after & (DIST >= lo) & (DIST <= hi)
        prof[f'{lo}-{hi}'] = paired(Cg, Cnat, mb)
    rep['dCE_after_by_distance'] = prof
    gain = rep['dP_capital_edited']['mean']; coll = rep['dCE_after_positions']['mean']
    rep['surgery_score_gain_over_after_collateral'] = (
        round(gain / coll, 4) if (coll is not None and abs(coll) > 1e-9) else None)
    rep['collateral_per_unit_gain'] = (
        round(coll / gain, 6) if (gain is not None and abs(gain) > 1e-9) else None)
    print(f"[{name}] P(cap) {rep['P_capital_before_edit']['mean']:.4f} -> "
          f"{rep['P_capital_after_edit']['mean']:.4f} "
          f"(dP {rep['dP_capital_edited']['mean']:+.4f} +- {rep['dP_capital_edited']['SE']:.4f}); "
          f"dCE@edited {rep['dCE_edited']['mean']:+.4f}; "
          f"dCE@before {rep['dCE_before_positions']['mean']:+.6f} "
          f"(gate max|dlogit| {gate['max_abs_logit_diff_before']:.2e}, "
          f"frac zero {gate['frac_before_exactly_zero']:.4f}); "
          f"dCE@after {rep['dCE_after_positions']['mean']:+.6f} "
          f"+- {rep['dCE_after_positions']['SE']:.6f} "
          f"(max|dlogit|after {gate['max_abs_logit_diff_after']:.3f}); "
          f"global dCE {rep['global_dCE_all_predict_positions']:+.6f}", flush=True)
    print(f"   distance profile dCE@after: " + "  ".join(
        f"{k}:{v['mean']:+.5f}(n={v['n']})" for k, v in prof.items()), flush=True)
    return rep

# ---------------- per-layer: collect Me grid at LI, transplant, report ----------------
layers_out = {}
donor_geom = {}
own_me_norms = {}   # per layer, for part-2 placebo
don_me_vecs = {}
for LI in LAYERS:
    print(f"\n===== LAYER {LI}: collect mlp-earlier group grid =====", flush=True)
    st = {'ME': [], 'maxrel': 0.0, 'fro_num': 0.0, 'fro_den': 0.0}
    for i in range(0, S_, B0): fwd(HELD[i:i+B0], LI=LI, mode='collect', stats=st)
    ME_ALL = torch.cat(st['ME'], 0)    # (S,T,D) mlp-earlier group vectors at layer LI, cpu
    gate_group = (st['fro_num']/st['fro_den'])**0.5
    print(f"GATE group-sum identity (L{LI}): global rel {gate_group:.2e} "
          f"maxpos {st['maxrel']:.2e}", flush=True)
    assert gate_group < 1e-4, f"five-group sum identity gate FAILED at layer {LI}"

    own_me = ME_ALL[TGT_FORCE[:,0], TGT_FORCE[:,1]]
    don_me = ME_ALL[DON_FORCE[:,0], DON_FORCE[:,1]]
    cosod = F.cosine_similarity(own_me, don_me, dim=-1)
    donor_geom[str(LI)] = {'own_Me_norm': stat(own_me.norm(dim=-1).numpy()),
                           'donor_Me_norm': stat(don_me.norm(dim=-1).numpy()),
                           'cos_own_donor': stat(cosod.numpy()),
                           'group_sum_identity_rel_global': gate_group,
                           'group_sum_identity_rel_maxpos': st['maxrel']}
    print(f"geometry L{LI}: own Me norm {donor_geom[str(LI)]['own_Me_norm']['mean']:.2f} "
          f"donor {donor_geom[str(LI)]['donor_Me_norm']['mean']:.2f} "
          f"cos(own,donor) {donor_geom[str(LI)]['cos_own_donor']['mean']:.3f}", flush=True)
    own_me_norms[str(LI)] = own_me.norm(dim=-1).clone()
    don_me_vecs[str(LI)] = don_me.clone()

    print(f"===== LAYER {LI}: FORCE (boundary mlp-earlier -> not-due targets, dose 1) =====",
          flush=True)
    spec = build_spec(TGT_FORCE, DON_FORCE, ME_ALL, GI_ME, 1.0)
    del ME_ALL, st
    Pf, Cf, gate_f = grids_tr(spec, LI)
    rep = config_report(f'FORCE L{LI}', Pf, Cf, gate_f, EM_FORCE)
    assert gate_f['max_abs_logit_diff_before'] < 1e-6, \
        f"causal zero-collateral gate FAILED (before-positions, layer {LI})"
    layers_out[str(LI)] = {'force': rep}
    del spec, Pf, Cf

# ---------------- section-94 reference (embedded for side-by-side) ----------------
ref94 = json.load(open(f'{QK}/qk_stream_transplant.json'))
sec94 = {'note': ('section-94 layer-17 results at ITS 500 force targets (this script re-runs '
                  'L17 at the same 300 targets used mid-stack, so the depth comparison is '
                  'apples-to-apples; the numbers below are the enshrined references)'),
         'force_Ae_dP': ref94['specificity_summary']['force_boundary_donor_dP'],
         'mlp_earlier_donor_dP': ref94['specificity_summary']['mlp_earlier_donor_dP'],
         'suppress_Ae_dP': ref94['suppress_test']['dP_capital_edited']['mean']}

OUTJ = {
 'meta': {'model': 'bilin18', 'layers': LAYERS, 'held_slice': 'FW[448:600,:128]', 'batch': B0,
          'base_ce': round(base_ce, 4),
          'edit': 'x_mlp_in(p) = x_pre(p) + (Me_donor - Me_own(p)) at target positions only; '
                  'blk.mlp at layer LI recomputes; the edited mlp output enters the residual '
                  'and PROPAGATES through the remaining layers (unlike section 94\'s L17 where '
                  'propagation is empty by construction)',
          'targets': 'mid-sentence capital-not-due (next token lowercase word/subword, '
                     'distance-since-newline >= 8), n=300, donors = genuine boundary positions '
                     '(sentence-end punct or newline), same sequence preferred; SAME target/'
                     'donor POSITIONS at every depth; donor VECTORS collected per layer '
                     '(the groups are defined per layer)',
          'split': 'BEFORE = non-edited positions with no edit strictly before them in the '
                   'sequence (causal gate: logits must be bit-identical); AFTER = non-edited '
                   'positions with an edit strictly before (propagation cost); distance = '
                   'position minus nearest strictly-preceding edited position',
          'P_capital': 'softmax mass on VOCAB_CLASS==capital (arc_caps metric)',
          'machinery': 'VERBATIM qk_stream_transplant.py (transplant, metrics, splits, gates) '
                       '+ qk_allterm_census.py (five-group accumulators, LI parameterized)',
          'gates': {'base_ce': round(base_ce, 5),
                    'base_P_capital_sentend': round(p_se, 5),
                    'base_P_capital_newline': round(p_nl, 5)}},
 'counts': {'valid': int(valid.sum()), 'boundary': int(M_boundary.sum()),
            'target_notdue_pool': int(M_target_notdue.sum()),
            'n_targets': int(EM_FORCE.sum()), 'n_before': int(M_before.sum()),
            'n_after': int(M_after.sum()), 'n_sequences_edited': n_seq_edited,
            'frac_same_sequence_donor': round(frac_same_force, 4)},
 'donor_geometry_by_layer': donor_geom,
 'layers': layers_out,
 'section94_reference': sec94,
}
json.dump(OUTJ, open(OUT, 'w'), indent=1)
torch.save({'TGT_FORCE': TGT_FORCE, 'DON_FORCE': DON_FORCE, 'EM_FORCE': EM_FORCE,
            'DIST': DIST, 'M_before': M_before, 'M_after': M_after,
            'M_before_all': M_before_all, 'M_after_all': M_after_all,
            'Pnat': Pnat, 'Cnat': Cnat, 'valid': valid,
            'own_me_norms': own_me_norms, 'don_me_vecs': don_me_vecs},
           STATE)
print("\nSaved qk_transplant_depth.json (partial) + state. QK TRANSPLANT DEPTH PART 1 DONE",
      flush=True)
