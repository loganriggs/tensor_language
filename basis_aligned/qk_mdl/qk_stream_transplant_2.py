"""CONTEXT TRANSPLANTS at layer 17's feed-forward input -- part 2 of 2: CONTROLS + DOSE.

Loads the part-1 state (qk_stream_transplant_state.pt: attention-earlier / mlp-earlier
group grids, target/donor assignments, natural grids) and runs, at the SAME force
targets (mid-sentence capital-not-due):
  CONTROL a: donor = RANDOM non-boundary position's attention-earlier vector
             (specificity: is it the BOUNDARY signal, or any foreign context vector?)
  CONTROL b: transplant the MLP-EARLIER group instead, same boundary donor positions
             (is the conditioning in attention-earlier specifically, or any history group?)
  CONTROL c: norm-matched RANDOM vector in the attention-earlier slot (placebo)
  DOSE:      x' = x_pre + t*(Ae_donor - Ae_own), t in {0.25, 0.5, 0.75} (t=1 from part 1)
Machinery identical to part 1 (forward VERBATIM from qk_allterm_census.py /
qk_L17_mixer.py / qk_edit_terms.py). Zero-collateral gate per config.
Merges everything into qk_stream_transplant.json.
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
OUT = f'{QK}/qk_stream_transplant.json'
STATE = f'{QK}/qk_stream_transplant_state.pt'

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
FW = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))
SEQL = 128
HELD = FW[448:600, :SEQL].to(DEV); B0 = 6
S_, T_ = HELD.shape
NHELD = S_
LI = 17
GI_AE = 1; GI_ME = 3
print(f"bilin18 NL={NL} D={D} NH={NH} held {S_}x{T_}", flush=True)

SD = torch.load(STATE, map_location='cpu', weights_only=False)
AE_ALL, ME_ALL = SD['AE_ALL'], SD['ME_ALL']
TGT_FORCE, DON_FORCE = SD['TGT_FORCE'], SD['DON_FORCE']
EM_FORCE = SD['EM_FORCE']
Pnat, Cnat, Pf, Cf = SD['Pnat'], SD['Cnat'], SD['Pf'], SD['Cf']
M_nonboundary, valid = SD['M_nonboundary'], SD['valid']

# capital-class mask (VERBATIM lex1 from part 1 / qk_arc_caps.py)
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

@torch.no_grad()
def fwd(idx, mode=None, EM=None, DON=None, gi=None, dose=1.0):
    """Forward verbatim from part 1 (qk_allterm_census.py lineage)."""
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
            if mode == 'transplant':
                x_in = x + EM.unsqueeze(-1) * (dose * (DON - groups[gi]))
                mo = blk.mlp(F.rms_norm(x_in, (D,)))
            del groups
        x = x + mo
        if track and li < LI:
            SA = SA + aout; SM = SM + MR; MR = mo
    return 30*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30)

def se(x):
    x=np.asarray(x,float); n=len(x)
    return float(x.std(ddof=1)/math.sqrt(n)) if n>1 else float('nan')
def stat(vals):
    vals=np.asarray(vals,float)
    return {'mean':round(float(vals.mean()),5) if len(vals) else None,
            'SE':round(se(vals),5) if len(vals)>1 else None, 'n':int(len(vals))}
def paired(a_grid, b_grid, mask):
    d=(a_grid[mask]-b_grid[mask]); return stat(d)

@torch.no_grad()
def grids_tr(spec):
    Pcap=np.full((NHELD,SEQL),np.nan,np.float32); CE=np.full((NHELD,SEQL),np.nan,np.float32)
    max_ne = 0.0; ne_zero = 0; ne_tot = 0; max_ed = 0.0
    for i in range(0,NHELD,B0):
        idx=HELD[i:i+B0]; b=idx.shape[0]
        EM = torch.from_numpy(spec['EMg'][i:i+b]).to(DEV)
        DON = spec['DONg'][i:i+b].to(DEV)
        lg = fwd(idx, mode='transplant', EM=EM.float(), DON=DON,
                 gi=spec['gi'], dose=spec['dose']).float()
        lgb = fwd(idx).float()
        dmax = (lg-lgb).abs().amax(-1)
        ne = ~EM
        if ne.any():
            max_ne = max(max_ne, float(dmax[ne].max()))
            ne_zero += int((dmax[ne]==0).sum()); ne_tot += int(ne.sum())
        if EM.any(): max_ed = max(max_ed, float(dmax[EM].max()))
        del lgb, dmax, EM, DON
        sm=torch.softmax(lg,-1)
        pc=sm[...,CAP_MASK].sum(-1)
        lp=F.log_softmax(lg[:,:-1],-1); tgt=idx[:,1:]
        nll=-lp.gather(-1,tgt.unsqueeze(-1)).squeeze(-1)
        Pcap[i:i+b]=pc.cpu().numpy(); CE[i:i+b,:-1]=nll.cpu().numpy()
        del lg,sm,pc,lp,nll
    return Pcap, CE, {'max_abs_logit_diff_nonedited': max_ne,
                      'frac_nonedited_exactly_zero': round(ne_zero/max(ne_tot,1), 6),
                      'max_abs_logit_diff_edited': max_ed}

def build_spec(targets, donors, src_all, gi, dose):
    EMg = np.zeros((S_, T_), bool)
    DONg = torch.zeros(S_, T_, D)
    for (s, t), (ds, dt) in zip(targets, donors):
        EMg[s, t] = True
        DONg[s, t] = src_all[ds, dt]
    return {'gi': gi, 'EMg': EMg, 'DONg': DONg, 'dose': dose}

def config_report(name, Pg, Cg, gate, EMg):
    tgt = EMg; rest = valid & ~tgt
    rep = {'n_edited': int(tgt.sum()),
           'P_capital_before': stat(Pnat[tgt]), 'P_capital_after': stat(Pg[tgt]),
           'dP_capital_edited': paired(Pg, Pnat, tgt),
           'dCE_edited': paired(Cg, Cnat, tgt),
           'dCE_nonedited_valid': paired(Cg, Cnat, rest),
           'global_dCE_all_predict_positions':
               round(float(np.mean(Cg[:, :T_-1] - Cnat[:, :T_-1])), 6),
           'zero_collateral_gate': gate}
    print(f"[{name}] P(cap) {rep['P_capital_before']['mean']:.4f} -> {rep['P_capital_after']['mean']:.4f} "
          f"(dP {rep['dP_capital_edited']['mean']:+.4f} +- {rep['dP_capital_edited']['SE']:.4f}); "
          f"dCE@edited {rep['dCE_edited']['mean']:+.4f} +- {rep['dCE_edited']['SE']:.4f}; "
          f"dCE@nonedited {rep['dCE_nonedited_valid']['mean']:+.6f}; "
          f"gate max|dlogit|nonedited {gate['max_abs_logit_diff_nonedited']:.2e} "
          f"(frac exactly zero {gate['frac_nonedited_exactly_zero']:.4f})", flush=True)
    return rep

rng = np.random.default_rng(1)   # fresh stream; part-1 assignments are loaded, not resampled

# ---------------- CONTROL a: random non-boundary donor positions (Ae slot) ----------------
print("\n===== CONTROL a: random-position (non-boundary) attention-earlier donors =====", flush=True)
glob_nb = np.argwhere(M_nonboundary)
DON_RAND = glob_nb[rng.integers(len(glob_nb), size=len(TGT_FORCE))]
# avoid donor == target
for r in range(len(DON_RAND)):
    while (DON_RAND[r] == TGT_FORCE[r]).all():
        DON_RAND[r] = glob_nb[int(rng.integers(len(glob_nb)))]
Pa, Ca, ga = grids_tr(build_spec(TGT_FORCE, DON_RAND, AE_ALL, GI_AE, 1.0))
rep_ctrl_rand = config_report('CTRL random-pos', Pa, Ca, ga, EM_FORCE)
assert ga['max_abs_logit_diff_nonedited'] < 1e-4, "zero-collateral gate FAILED (ctrl a)"

# ---------------- CONTROL b: transplant MLP-EARLIER group, same boundary donors ----------------
print("\n===== CONTROL b: mlp-earlier group transplant (same boundary donors) =====", flush=True)
Pb, Cb, gb = grids_tr(build_spec(TGT_FORCE, DON_FORCE, ME_ALL, GI_ME, 1.0))
rep_ctrl_me = config_report('CTRL mlp-earlier', Pb, Cb, gb, EM_FORCE)
assert gb['max_abs_logit_diff_nonedited'] < 1e-4, "zero-collateral gate FAILED (ctrl b)"

# ---------------- CONTROL b2: mlp-earlier transplant from RANDOM non-boundary donors ----------------
# (disambiguates control b: boundary-specific signal in the Me group vs generic foreign-Me disruption)
print("\n===== CONTROL b2: mlp-earlier group transplant (random non-boundary donors) =====", flush=True)
Pb2, Cb2, gb2 = grids_tr(build_spec(TGT_FORCE, DON_RAND, ME_ALL, GI_ME, 1.0))
rep_ctrl_me_rand = config_report('CTRL mlp-earlier random-pos', Pb2, Cb2, gb2, EM_FORCE)
assert gb2['max_abs_logit_diff_nonedited'] < 1e-4, "zero-collateral gate FAILED (ctrl b2)"

# ---------------- CONTROL c2: norm-matched random vector in the Me slot ----------------
print("\n===== CONTROL c2: norm-matched random vector placebo in the mlp-earlier slot =====", flush=True)
gtor2 = torch.Generator().manual_seed(1)
RV2 = torch.randn(len(TGT_FORCE), D, generator=gtor2)
own_me_norm = ME_ALL[TGT_FORCE[:,0], TGT_FORCE[:,1]].norm(dim=-1, keepdim=True)
RV2 = RV2 * own_me_norm / RV2.norm(dim=-1, keepdim=True)
spec_c2 = {'gi': GI_ME, 'EMg': EM_FORCE, 'DONg': torch.zeros(S_, T_, D), 'dose': 1.0}
for r, (s, t) in enumerate(TGT_FORCE):
    spec_c2['DONg'][s, t] = RV2[r]
Pc2, Cc2, gc2 = grids_tr(spec_c2)
rep_ctrl_placebo_me = config_report('CTRL placebo Me', Pc2, Cc2, gc2, EM_FORCE)
assert gc2['max_abs_logit_diff_nonedited'] < 1e-4, "zero-collateral gate FAILED (ctrl c2)"

# ---------------- CONTROL c: norm-matched random vector in the Ae slot ----------------
print("\n===== CONTROL c: norm-matched random vector placebo =====", flush=True)
gtor = torch.Generator().manual_seed(0)
RAND_VEC = torch.randn(len(TGT_FORCE), D, generator=gtor)
own_norm = AE_ALL[TGT_FORCE[:,0], TGT_FORCE[:,1]].norm(dim=-1, keepdim=True)
RAND_VEC = RAND_VEC * own_norm / RAND_VEC.norm(dim=-1, keepdim=True)
spec_c = {'gi': GI_AE, 'EMg': EM_FORCE, 'DONg': torch.zeros(S_, T_, D), 'dose': 1.0}
for r, (s, t) in enumerate(TGT_FORCE):
    spec_c['DONg'][s, t] = RAND_VEC[r]
Pc, Cc, gc = grids_tr(spec_c)
rep_ctrl_placebo = config_report('CTRL placebo', Pc, Cc, gc, EM_FORCE)
assert gc['max_abs_logit_diff_nonedited'] < 1e-4, "zero-collateral gate FAILED (ctrl c)"

# ---------------- DOSE: interpolated force transplant ----------------
print("\n===== DOSE: x' = x_pre + t*(Ae_donor - Ae_own), t in {0.25,0.5,0.75,1} =====", flush=True)
dose_curve = {}
for t in (0.25, 0.5, 0.75):
    Pd, Cd, gd = grids_tr(build_spec(TGT_FORCE, DON_FORCE, AE_ALL, GI_AE, t))
    dose_curve[str(t)] = config_report(f'DOSE t={t}', Pd, Cd, gd, EM_FORCE)
    assert gd['max_abs_logit_diff_nonedited'] < 1e-4, f"zero-collateral gate FAILED (dose {t})"
# t=1 from part 1 (the force test itself)
dose_curve['1.0'] = {'from_part1_force_test': True,
                     'P_capital_after': stat(Pf[EM_FORCE]),
                     'dP_capital_edited': paired(Pf, Pnat, EM_FORCE),
                     'dCE_edited': paired(Cf, Cnat, EM_FORCE)}
ts = [0.25, 0.5, 0.75, 1.0]
gains = [dose_curve[str(t)]['dP_capital_edited']['mean'] for t in ts]
def spearman(a,b):
    a=np.asarray(a,float); b=np.asarray(b,float)
    ra=np.argsort(np.argsort(a)); rb=np.argsort(np.argsort(b))
    if ra.std()==0 or rb.std()==0: return 0.0
    return float(np.corrcoef(ra,rb)[0,1])
dose_summary = {'t': ts, 'dP_capital_edited': gains,
                'spearman_t_vs_gain': round(spearman(ts, gains), 4)}
print(f"dose curve dP(capital): {list(zip(ts, [round(g,4) for g in gains]))} "
      f"spearman {dose_summary['spearman_t_vs_gain']}", flush=True)

# ---------------- merge into the part-1 JSON ----------------
OUTJ = json.load(open(OUT))
OUTJ['controls'] = {
    'random_position_donor_Ae': rep_ctrl_rand,
    'mlp_earlier_group_boundary_donor': rep_ctrl_me,
    'mlp_earlier_group_random_donor': rep_ctrl_me_rand,
    'norm_matched_random_placebo_Ae': rep_ctrl_placebo,
    'norm_matched_random_placebo_Me': rep_ctrl_placebo_me,
    'note': ('all three at the SAME 500 force targets; random-position donors drawn from valid '
             'non-boundary positions (fresh seeded stream); mlp-earlier control uses the SAME '
             'donor positions as the force test but transplants the Me group; placebo replaces '
             'the attention-earlier vector with a random direction at the target\'s own norm')}
OUTJ['dose'] = {'curve': dose_curve, 'summary': dose_summary}
force_gain = OUTJ['force_test']['dP_capital_edited']['mean']
OUTJ['specificity_summary'] = {
    'force_boundary_donor_dP': force_gain,
    'random_position_donor_dP': rep_ctrl_rand['dP_capital_edited']['mean'],
    'mlp_earlier_donor_dP': rep_ctrl_me['dP_capital_edited']['mean'],
    'mlp_earlier_random_donor_dP': rep_ctrl_me_rand['dP_capital_edited']['mean'],
    'norm_matched_placebo_Ae_dP': rep_ctrl_placebo['dP_capital_edited']['mean'],
    'norm_matched_placebo_Me_dP': rep_ctrl_placebo_me['dP_capital_edited']['mean'],
    'boundary_over_random_ratio_Ae': round(force_gain / rep_ctrl_rand['dP_capital_edited']['mean'], 3)
        if abs(rep_ctrl_rand['dP_capital_edited']['mean']) > 1e-9 else None,
    'boundary_over_random_ratio_Me': round(rep_ctrl_me['dP_capital_edited']['mean']
                                           / rep_ctrl_me_rand['dP_capital_edited']['mean'], 3)
        if abs(rep_ctrl_me_rand['dP_capital_edited']['mean']) > 1e-9 else None,
    'boundary_over_mlp_earlier_ratio': round(force_gain / rep_ctrl_me['dP_capital_edited']['mean'], 3)
        if abs(rep_ctrl_me['dP_capital_edited']['mean']) > 1e-9 else None}
json.dump(OUTJ, open(OUT, 'w'), indent=1)
print("\nSaved qk_stream_transplant.json (complete). QK STREAM TRANSPLANT PART 2 DONE", flush=True)
