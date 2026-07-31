"""BOUNDING THE SECTION-94 EDITING LAW: mid-stack context transplants -- part 2 of 2:
PLACEBO CONTROLS per depth + WASHOUT / SURGERY-SCORE analysis.

Loads the part-1 state (qk_transplant_depth_state.pt: target/donor assignments, natural
grids, before/after split, per-layer own-Me norms) and runs, at the SAME 300 force targets
and at each depth L8/L12/L15/L17:
  CONTROL: norm-matched RANDOM vector in the mlp-earlier slot (placebo) -- does an
  arbitrary same-norm perturbation produce the same target movement / the same
  after-position collateral, or is the boundary-context signal specific?
Then the analysis the spec asks for:
  (a) target-gain-by-depth table (force vs placebo, L8/L12/L15/L17);
  (b) collateral: before-positions gate summary, after-positions cost + distance profile;
  (c) surgery score by depth (target gain / after-position collateral; L17 = infinite);
  (d) washout: per-layer attenuation of the target gain with depth-distance-to-readout.
Machinery identical to part 1 (forward VERBATIM from qk_allterm_census.py /
qk_stream_transplant.py, LI parameterized). Merges everything into qk_transplant_depth.json.
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
tok = AutoTokenizer.from_pretrained('gpt2')
FW = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))
SEQL = 128
HELD = FW[448:600, :SEQL].to(DEV); B0 = 6
S_, T_ = HELD.shape
NHELD = S_
LAYERS = [8, 12, 15, 17]
GI_ME = 3
print(f"bilin18 NL={NL} D={D} NH={NH} held {S_}x{T_} layers {LAYERS}", flush=True)

SD = torch.load(STATE, map_location='cpu', weights_only=False)
TGT_FORCE, DON_FORCE, EM_FORCE = SD['TGT_FORCE'], SD['DON_FORCE'], SD['EM_FORCE']
DIST, M_before, M_after = SD['DIST'], SD['M_before'], SD['M_after']
M_before_all, M_after_all = SD['M_before_all'], SD['M_after_all']
Pnat, Cnat, valid = SD['Pnat'], SD['Cnat'], SD['valid']
own_me_norms = SD['own_me_norms']
DIST_BINS = [(1,2),(3,4),(5,8),(9,16),(17,32),(33,64),(65,127)]

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
def fwd(idx, LI=17, mode=None, EM=None, DON=None, gi=None, dose=1.0, stats=None):
    """Forward verbatim from part 1 (qk_allterm_census.py lineage), LI parameterized."""
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
    return {'mean':round(float(vals.mean()),6) if len(vals) else None,
            'SE':round(se(vals),6) if len(vals)>1 else None, 'n':int(len(vals))}
def paired(a_grid, b_grid, mask):
    d=(a_grid[mask]-b_grid[mask]); return stat(d)

@torch.no_grad()
def grids_tr(spec, LI):
    Pcap=np.full((NHELD,SEQL),np.nan,np.float32); CE=np.full((NHELD,SEQL),np.nan,np.float32)
    max_bf = 0.0; bf_zero = 0; bf_tot = 0
    max_af = 0.0; af_zero = 0; af_tot = 0; max_ed = 0.0
    for i in range(0,NHELD,B0):
        idx=HELD[i:i+B0]; b=idx.shape[0]
        EM = torch.from_numpy(spec['EMg'][i:i+b]).to(DEV)
        DON = spec['DONg'][i:i+b].to(DEV)
        lg = fwd(idx, LI=LI, mode='transplant', EM=EM.float(), DON=DON,
                 gi=spec['gi'], dose=spec['dose']).float()
        lgb = fwd(idx).float()
        dmax = (lg-lgb).abs().amax(-1)
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
    return Pcap, CE, {
        'max_abs_logit_diff_before': max_bf,
        'frac_before_exactly_zero': round(bf_zero/max(bf_tot,1), 6),
        'max_abs_logit_diff_after': max_af,
        'frac_after_exactly_zero': round(af_zero/max(af_tot,1), 6),
        'max_abs_logit_diff_edited': max_ed}

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
          f"(gate max|dlogit| {gate['max_abs_logit_diff_before']:.2e}); "
          f"dCE@after {rep['dCE_after_positions']['mean']:+.6f} "
          f"+- {rep['dCE_after_positions']['SE']:.6f}; "
          f"global dCE {rep['global_dCE_all_predict_positions']:+.6f}", flush=True)
    return rep

# ---------------- PLACEBO per depth: norm-matched random vector in the Me slot ----------------
# (VERBATIM control-c2 machinery from qk_stream_transplant_2.py, per layer)
placebo_out = {}
for LI in LAYERS:
    print(f"\n===== LAYER {LI}: PLACEBO (norm-matched random vector, mlp-earlier slot) =====",
          flush=True)
    gtor = torch.Generator().manual_seed(1000 + LI)
    RV = torch.randn(len(TGT_FORCE), D, generator=gtor)
    own_norm = own_me_norms[str(LI)].unsqueeze(-1)
    RV = RV * own_norm / RV.norm(dim=-1, keepdim=True)
    spec = {'gi': GI_ME, 'EMg': EM_FORCE, 'DONg': torch.zeros(S_, T_, D), 'dose': 1.0}
    for r, (s, t) in enumerate(TGT_FORCE):
        spec['DONg'][s, t] = RV[r]
    Pp, Cp, gp = grids_tr(spec, LI)
    rep = config_report(f'PLACEBO L{LI}', Pp, Cp, gp, EM_FORCE)
    assert gp['max_abs_logit_diff_before'] < 1e-6, \
        f"causal zero-collateral gate FAILED (placebo before-positions, layer {LI})"
    placebo_out[str(LI)] = rep
    del spec, Pp, Cp

# ---------------- merge + washout / surgery-score analysis ----------------
OUTJ = json.load(open(OUT))
for LI in LAYERS:
    OUTJ['layers'][str(LI)]['placebo'] = placebo_out[str(LI)]

gains = {LI: OUTJ['layers'][str(LI)]['force']['dP_capital_edited']['mean'] for LI in LAYERS}
colls = {LI: OUTJ['layers'][str(LI)]['force']['dCE_after_positions']['mean'] for LI in LAYERS}
pgains = {LI: placebo_out[str(LI)]['dP_capital_edited']['mean'] for LI in LAYERS}
pcolls = {LI: placebo_out[str(LI)]['dCE_after_positions']['mean'] for LI in LAYERS}

# washout: per-layer attenuation of the force target gain relative to the L17 reference
g17 = gains[17]
washout = {}
for LI in LAYERS[:-1]:
    dl = 17 - LI
    ratio = gains[LI] / g17 if abs(g17) > 1e-9 else None
    washout[str(LI)] = {
        'depth_distance_to_readout': dl,
        'gain': gains[LI],
        'retention_vs_L17': round(ratio, 4) if ratio is not None else None,
        'per_layer_attenuation_factor':
            round(ratio ** (1.0/dl), 4) if (ratio is not None and ratio > 0) else None}

surgery = {str(LI): {
    'target_gain_dP_capital': gains[LI],
    'after_collateral_dCE': colls[LI],
    'surgery_score': OUTJ['layers'][str(LI)]['force']['surgery_score_gain_over_after_collateral'],
    'collateral_per_unit_gain': OUTJ['layers'][str(LI)]['force']['collateral_per_unit_gain'],
    'placebo_gain': pgains[LI], 'placebo_after_collateral': pcolls[LI]} for LI in LAYERS}

OUTJ['analysis'] = {
 'target_gain_by_depth': {str(LI): gains[LI] for LI in LAYERS},
 'after_collateral_by_depth': {str(LI): colls[LI] for LI in LAYERS},
 'placebo_gain_by_depth': {str(LI): pgains[LI] for LI in LAYERS},
 'washout_vs_L17': washout,
 'surgery_score_table': surgery,
 'note': ('surgery score = mean capital-probability gain at edited positions divided by mean '
          'delta cross-entropy at after-positions; L17 after-collateral is exactly zero by '
          'construction (score infinite); washout retention = mid-stack gain / same-target '
          'L17 gain; per-layer attenuation factor = retention^(1/depth-distance)')}
json.dump(OUTJ, open(OUT, 'w'), indent=1)
print("\nwashout:", json.dumps(washout), flush=True)
print("surgery:", json.dumps(surgery), flush=True)
print("\nSaved qk_transplant_depth.json (complete). QK TRANSPLANT DEPTH PART 2 DONE", flush=True)
