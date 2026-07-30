"""OPTION-2 ALGORITHMIC ARC on the LATE FEED-FORWARD CLASS-INTEGRATORS (§68).

Components (§68 verified class-pushers; qk_unsup_classpush.json):
  mlp.L17.d1 -> CAPITAL pusher (firing capital-summed dlogit 20004, tdCE 0.255, z 4.5)
  mlp.L17.d3 -> CAPITAL pusher (firing 7355, tdCE 0.167, z 3.9)
  mlp.L17.d2 -> WORD    pusher (firing 28658, tdCE 0.367, z 5.2)
  mlp.L16.d2 -> WORD    pusher (firing 13014, tdCE 0.178, z 4.2)
  §61 pre-pass: L17 trio {d1,d2,d3} joint dCE 0.911 vs sum-of-solos 0.481 (ratio 1.89) -> score JOINTLY.

DECISIVE question (the §66 distinction that deflated the capitalization circuit): is the class
push a genuine CONTEXT-CONDITIONED class-selection algorithm (push the class WHEN the context calls
for it -- next token actually IS that class -- and NOT push where a different class is due), or a
STATIC always-on class-frequency prior (push equally everywhere regardless of what the next token is)?
§66 failed this test with a boundary-over-proper-noun specificity ratio of 1.0.

ARC:
 1. VERIFY: reproduce §68 firing-position mean-ablation delta-cross-entropy + class-summed delta-logit
    for each direction and the L17 trio jointly; paired standard errors.
 2. DECISIVE red-team: measure the pushed-class class-summed delta-logit (and the direction's own
    activation) SEPARATELY at positions where the true next token IS the pushed class vs IS NOT.
    Context-conditioned => concentrates push where the class is due; static prior => flat.
 3. SIGN-correctness: delta-cross-entropy split by whether the true next token is in the pushed class
    (does ablation HURT where the pushed class is correct, and HELP where it is not?).
 4. MINIMAL circuit + joint-vs-solo structure (one integrated computation or separable ones?).
 5. Content vs position of the trigger (reads context via upstream token/class identity, or fires flat?).

FORWARD + mean-ablation copied VERBATIM from qk_unsup_classpush.py (which copies qk_census_difficulty.py).
Held-back FW[448:600,:128]. Batch<=8, footprint<4GB, GPU guard.
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

# ---------------- GPU GUARD (verbatim from classpush) ----------------
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
print(f"bilin18 NL={NL} NH={NH} HD={HD} D={D} V={V}", flush=True)

FINEWEB = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))
SEQL = 128
TRAIN = FINEWEB[0:256, :SEQL].to(DEV)        # ONLY to recompute MLP dirs (path defn, as discovery)
HELD = FINEWEB[448:600, :SEQL].to(DEV)       # held-back verification slice
NHELD = HELD.shape[0]
BATCH = 4
KCAUSAL = 200                                # top-K firing positions per path (matches census/§68)

_special = {tok.eos_token_id}
for _t in range(V):
    if '�' in tok.decode([_t]): _special.add(_t)
SPECIAL = np.array(sorted(_special))
print(f"{len(SPECIAL)} special token ids masked", flush=True)

# ===================== LEXICAL CLASS LIBRARY -- VERBATIM (lex1 / VOCAB_CLASS) =====================
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
CAP_BOOL  = (VOCAB_CLASS == 'capital')
WORD_BOOL = (VOCAB_CLASS == 'word')
cap_row  = torch.from_numpy(CAP_BOOL.astype(np.float32)).to(DEV)    # (V,)
word_row = torch.from_numpy(WORD_BOOL.astype(np.float32)).to(DEV)   # (V,)
print(f"class sizes: capital={int(CAP_BOOL.sum())} word={int(WORD_BOOL.sum())}", flush=True)

# ===================== MLP directions: top-4 SVD dirs per block from TRAIN gram (VERBATIM) =====================
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
    _evals, _evecs = torch.linalg.eigh(gram[li]); mlp_dirs[li] = _evecs[:, -N_SVD:].T.flip(0)
del gram
print("MLP directions ready.", flush=True)

# ===================== Core forward (VERBATIM from classpush) with multi-component mean-ablation =====================
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
        if li in ab_heads:
            yh4 = yh4.clone()
            for h in ab_heads[li]: yh4[:, :, h] = yhmeans[li][:, h].unsqueeze(0)
        x = x + a.c_proj(yh4.reshape(B, T, -1))
        mo = blk.mlp(F.rms_norm(x, (D,)))
        if collect:
            pr = torch.einsum('btd,nd->btn', mo, mlp_dirs[li])   # (B,T,N_SVD)
            out[('mproj', li)] = pr.cpu().numpy()
        if li in ab_mlps:
            for kk in ab_mlps[li]:
                pr = torch.einsum('btd,d->bt', mo, mlp_dirs[li, kk])
                mo = mo - (pr - projmeans[li][:, kk].unsqueeze(0)).unsqueeze(-1) * mlp_dirs[li, kk]
        x = x + mo
    logits = 30*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30)
    return (logits, out) if collect else logits

# ===================== PASS A: per-position means + signed activations for target dirs =====================
# need per-position mean of MLP projection for ablation, and signed activations for L16.d2, L17.d1/d2/d3
YH_SUM = {li: torch.zeros(SEQL, NH, HD, device=DEV) for li in range(NL)}  # (heads unused here but kept for API)
PROJ_SUM = {li: torch.zeros(SEQL, N_SVD, device=DEV) for li in range(NL)}
TARGETS = {'mlp.L17.d1': (17, 1), 'mlp.L17.d2': (17, 2), 'mlp.L17.d3': (17, 3), 'mlp.L16.d2': (16, 2)}
act = {name: np.zeros((NHELD, SEQL), np.float32) for name in TARGETS}
print("PASS A: per-position means + activations ...", flush=True)
for i in range(0, NHELD, BATCH):
    idx = HELD[i:i+BATCH]; b = idx.shape[0]
    _, out = forward(idx, collect=True)
    for li in range(NL): PROJ_SUM[li] += torch.from_numpy(out[('mproj', li)].sum(0)).to(DEV)
    for name, (li, kk) in TARGETS.items(): act[name][i:i+b] = out[('mproj', li)][:, :, kk]
YHMEAN = {li: YH_SUM[li] / NHELD for li in range(NL)}   # zeros (no head ablation) -- kept for forward API
PROJMEAN = {li: PROJ_SUM[li] / NHELD for li in range(NL)}
del YH_SUM, PROJ_SUM
print("PASS A done.", flush=True)

# ===================== position masks =====================
held_np = HELD.cpu().numpy()
pos_t = np.tile(np.arange(SEQL), NHELD).reshape(NHELD, SEQL)
is_special = np.isin(held_np, SPECIAL)
valid_next = pos_t < (SEQL - 1)
bad_trigger = (pos_t == 0) | is_special | ~valid_next     # firing-selection mask (VERBATIM §68)

nxt = np.zeros_like(held_np); nxt[:, :-1] = held_np[:, 1:]
next_special = np.zeros_like(is_special); next_special[:, :-1] = is_special[:, 1:]
next_class = np.empty_like(held_np, dtype=object); next_class[:] = 'special'
next_class[:, :-1] = VOCAB_CLASS[held_np[:, 1:]]
cur_class = VOCAB_CLASS[held_np]
# analysis-valid: interior, not special, next not special (well-defined next token)
valid = (pos_t > 0) & (pos_t < SEQL - 1) & ~is_special & ~next_special
next_is_cap  = (next_class == 'capital')
next_is_word = (next_class == 'word')

# firing masks (top-KCAUSAL |activation| among firing-eligible) per single direction
def firing_mask(name):
    a = np.abs(act[name]).copy().reshape(-1)
    a[bad_trigger.reshape(-1)] = -1e30
    tk = np.argpartition(a, -KCAUSAL)[-KCAUSAL:]
    mk = np.zeros(NHELD*SEQL, bool); mk[tk] = True
    return mk.reshape(NHELD, SEQL)
fire = {name: firing_mask(name) for name in TARGETS}

# ===================== grids: for an ablation config, per-position capital/word class-summed logit + CE =====================
tgt_all = torch.from_numpy(held_np).to(DEV)
@torch.no_grad()
def grids(ablations):
    capg = np.zeros((NHELD, SEQL), np.float32); wordg = np.zeros((NHELD, SEQL), np.float32)
    ceg = np.full((NHELD, SEQL), np.nan, np.float32)
    for i in range(0, NHELD, BATCH):
        idx = HELD[i:i+BATCH]; b = idx.shape[0]
        lg = forward(idx, ablations=ablations, yhmeans=YHMEAN, projmeans=PROJMEAN).float()
        capg[i:i+b] = torch.einsum('btv,v->bt', lg, cap_row).cpu().numpy()
        wordg[i:i+b] = torch.einsum('btv,v->bt', lg, word_row).cpu().numpy()
        lp = F.log_softmax(lg[:, :SEQL-1], -1)
        nll = -lp.gather(-1, idx[:, 1:].unsqueeze(-1)).squeeze(-1)
        ceg[i:i+b, :SEQL-1] = nll.cpu().numpy()
        del lg, lp, nll
    return capg, wordg, ceg

CONFIGS = {
    'base':        [],
    'mlp.L17.d1':  [('mlp', 17, 1)],
    'mlp.L17.d2':  [('mlp', 17, 2)],
    'mlp.L17.d3':  [('mlp', 17, 3)],
    'mlp.L16.d2':  [('mlp', 16, 2)],
    'L17.d1+d3':   [('mlp', 17, 1), ('mlp', 17, 3)],
    'L17.trio':    [('mlp', 17, 1), ('mlp', 17, 2), ('mlp', 17, 3)],
}
print("Computing grids for", list(CONFIGS), flush=True)
G = {}
for name, abl in CONFIGS.items():
    t0 = time.time(); G[name] = grids(abl)
    print(f"  grid {name} done in {time.time()-t0:.0f}s", flush=True)
CAPb, WORDb, CEb = G['base']

# ===================== helpers =====================
def msd(v):
    v = np.asarray(v, float); n = len(v)
    if n == 0: return {'mean': None, 'SE': None, 'n': 0}
    return {'mean': round(float(v.mean()), 4),
            'SE': round(float(v.std(ddof=1)/math.sqrt(n)), 4) if n > 1 else None, 'n': int(n)}

# pushed-class class-summed delta-logit grid (base - ablated) for the pushed class of a config
def push_delta_grid(config, pushed):
    capg, wordg, _ = G[config]
    if pushed == 'capital': return CAPb - capg
    else: return WORDb - wordg
def dce_grid(config):
    _, _, ceg = G[config]; return (ceg - CEb)[:, :SEQL-1]   # ablated - base (>0 = ablation hurt), width SEQL-1

# ===================== STEP 1: VERIFY -- reproduce §68 firing dCE + class-summed dlogit =====================
print("\n===== STEP 1: VERIFY (reproduce §68) =====", flush=True)
PUSHED = {'mlp.L17.d1': 'capital', 'mlp.L17.d3': 'capital', 'mlp.L17.d2': 'word', 'mlp.L16.d2': 'word'}
step1 = {}
for name, pushed in PUSHED.items():
    fm = fire[name]
    dpush = push_delta_grid(name, pushed)          # base - ablated class-summed dlogit
    dce = dce_grid(name)
    fm_ce = fm[:, :SEQL-1]
    rec = {
        'pushed_class': pushed,
        'firing_class_summed_dlogit': msd(dpush[fm]),       # reproduce §68 (mean over 200 firing)
        'firing_dCE': msd(dce[fm_ce]),                       # reproduce §68 trigger dCE
    }
    rec['firing_dCE_z'] = round(rec['firing_dCE']['mean']/rec['firing_dCE']['SE'], 2) if rec['firing_dCE']['SE'] else None
    step1[name] = rec
    print(f"  {name:12s} push={pushed:7s} firing class-sum dlogit {rec['firing_class_summed_dlogit']['mean']:.1f}"
          f" +-{rec['firing_class_summed_dlogit']['SE']}  firing dCE {rec['firing_dCE']['mean']:.4f}"
          f" +-{rec['firing_dCE']['SE']} (z {rec['firing_dCE_z']})", flush=True)

# L17 trio joint vs solos at UNION of firing positions (reproduce §61 pre-pass)
union_fire = fire['mlp.L17.d1'] | fire['mlp.L17.d2'] | fire['mlp.L17.d3']
uf_ce = union_fire[:, :SEQL-1]
solo_dce = {c: msd(dce_grid(c)[uf_ce]) for c in ['mlp.L17.d1', 'mlp.L17.d2', 'mlp.L17.d3']}
trio_dce = msd(dce_grid('L17.trio')[uf_ce])
sum_solo = sum(solo_dce[c]['mean'] for c in solo_dce)
step1['L17_trio_joint'] = {
    'n_union_firing_positions': int(union_fire.sum()),
    'solos_dCE_at_union': solo_dce,
    'sum_solo_dCE': round(sum_solo, 4),
    'joint_dCE': trio_dce,
    'redundancy_ratio_joint_over_sumsolo': round(trio_dce['mean']/sum_solo, 3) if sum_solo else None,
}
print(f"  L17 trio: joint dCE {trio_dce['mean']:.4f} vs sum-solo {sum_solo:.4f} "
      f"ratio {step1['L17_trio_joint']['redundancy_ratio_joint_over_sumsolo']}", flush=True)

# ===================== STEP 2: DECISIVE red-team -- context-conditioned vs static prior =====================
# For each pusher: pushed-class class-summed delta-logit split by whether true next token IS pushed class.
# Also the direction's own signed activation split the same way. Measured over ALL VALID positions and at
# the direction's own FIRING positions.
print("\n===== STEP 2: DECISIVE (context-conditioned vs static prior) =====", flush=True)
step2 = {}
for name, pushed in PUSHED.items():
    dpush = push_delta_grid(name, pushed)
    nextin = next_is_cap if pushed == 'capital' else next_is_word
    a = act[name]
    fm = fire[name]
    blocks = {}
    for dom_label, dom in [('all_valid', valid), ('at_firing', fm & valid)]:
        m_in  = dom & nextin
        m_out = dom & ~nextin
        push_in  = msd(dpush[m_in]); push_out = msd(dpush[m_out])
        ratio = (push_in['mean']/push_out['mean']) if (push_out['mean'] and abs(push_out['mean']) > 1e-9) else None
        blocks[dom_label] = {
            'push_where_next_IS_class': push_in,
            'push_where_next_NOT_class': push_out,
            'specificity_ratio_IS_over_NOT': round(ratio, 3) if ratio is not None else None,
            'abs_activation_where_next_IS_class': msd(np.abs(a[m_in])),
            'abs_activation_where_next_NOT_class': msd(np.abs(a[m_out])),
        }
    step2[name] = {'pushed_class': pushed, **blocks}
    b = blocks['all_valid']
    print(f"  {name:12s} push={pushed:7s} ALL-VALID: IS-class {b['push_where_next_IS_class']['mean']:.1f}"
          f" (n{b['push_where_next_IS_class']['n']}) vs NOT {b['push_where_next_NOT_class']['mean']:.1f}"
          f" (n{b['push_where_next_NOT_class']['n']}) ratio {b['specificity_ratio_IS_over_NOT']}", flush=True)

# ===================== STEP 3: SIGN-correctness -- dCE split by whether true next in pushed class =====================
print("\n===== STEP 3: SIGN-correctness (dCE split by true next class) =====", flush=True)
step3 = {}
for name, pushed in PUSHED.items():
    dce = dce_grid(name)                # ablated - base (>0 = ablation HURT)
    nextin = (next_is_cap if pushed == 'capital' else next_is_word)
    fm = fire[name]
    blocks = {}
    for dom_label, dom in [('all_valid', valid), ('at_firing', fm & valid)]:
        dom_ce = dom[:, :SEQL-1]; nin = nextin[:, :SEQL-1]
        m_in = dom_ce & nin; m_out = dom_ce & ~nin
        blocks[dom_label] = {
            'dCE_where_next_IS_class': msd(dce[m_in]),
            'dCE_where_next_NOT_class': msd(dce[m_out]),
        }
    step3[name] = {'pushed_class': pushed, **blocks}
    b = blocks['all_valid']
    print(f"  {name:12s} push={pushed:7s} ALL-VALID dCE: IS-class {b['dCE_where_next_IS_class']['mean']:+.4f}"
          f" (n{b['dCE_where_next_IS_class']['n']})  NOT {b['dCE_where_next_NOT_class']['mean']:+.4f}"
          f" (n{b['dCE_where_next_NOT_class']['n']})", flush=True)

# ===================== STEP 4: MINIMAL circuit + joint structure =====================
# CAPITAL pair {d1,d3}: joint vs solo on capital push + dCE at union firing. Do they add or overlap?
print("\n===== STEP 4: MINIMAL circuit + joint structure =====", flush=True)
uf_cap = fire['mlp.L17.d1'] | fire['mlp.L17.d3']
ufc_ce = uf_cap[:, :SEQL-1]
cap_pair = {
    'n_union_firing': int(uf_cap.sum()),
    'solo_d1_capital_push': msd(push_delta_grid('mlp.L17.d1', 'capital')[uf_cap]),
    'solo_d3_capital_push': msd(push_delta_grid('mlp.L17.d3', 'capital')[uf_cap]),
    'joint_d1d3_capital_push': msd(push_delta_grid('L17.d1+d3', 'capital')[uf_cap]),
    'solo_d1_dCE': msd(dce_grid('mlp.L17.d1')[ufc_ce]),
    'solo_d3_dCE': msd(dce_grid('mlp.L17.d3')[ufc_ce]),
    'joint_d1d3_dCE': msd(dce_grid('L17.d1+d3')[ufc_ce]),
}
sp = cap_pair['solo_d1_capital_push']['mean'] + cap_pair['solo_d3_capital_push']['mean']
cap_pair['push_sum_solo'] = round(sp, 1)
cap_pair['push_ratio_joint_over_sumsolo'] = round(cap_pair['joint_d1d3_capital_push']['mean']/sp, 3) if sp else None
sd = cap_pair['solo_d1_dCE']['mean'] + cap_pair['solo_d3_dCE']['mean']
cap_pair['dCE_sum_solo'] = round(sd, 4)
cap_pair['dCE_ratio_joint_over_sumsolo'] = round(cap_pair['joint_d1d3_dCE']['mean']/sd, 3) if sd else None
step4 = {'capital_pair_L17_d1_d3': cap_pair, 'L17_trio_joint_dCE': step1['L17_trio_joint']}
print(f"  cap pair d1+d3: push joint {cap_pair['joint_d1d3_capital_push']['mean']:.1f} vs sum-solo {sp:.1f}"
      f" (ratio {cap_pair['push_ratio_joint_over_sumsolo']}); dCE joint {cap_pair['joint_d1d3_dCE']['mean']:.4f}"
      f" vs sum-solo {sd:.4f} (ratio {cap_pair['dCE_ratio_joint_over_sumsolo']})", flush=True)

# ===================== STEP 5: content vs position of the trigger =====================
# does each direction's activation key on upstream token/class identity (content) or fire flat / positionally?
print("\n===== STEP 5: content vs position of trigger =====", flush=True)
vflat = valid.reshape(-1)
CLASSES_REPORT = ['punct','newline','word','subword','capital','digit','determiner','pronoun','coordinator','quote','other']
def act_by_current_class(name):
    a = np.abs(act[name]).reshape(-1); cc = cur_class.reshape(-1); out = {}
    for c in CLASSES_REPORT:
        mm = vflat & (cc == c)
        if mm.sum() >= 20: out[c] = msd(a[mm])
    return out
# distance-since-newline (positional signal) within a fixed current-class
def dsn_grid_fn(tokens):
    NL_BOOL = np.array([('\n' in tok.decode([t])) for t in range(V)])
    N, T = tokens.shape; nl = NL_BOOL[tokens]
    seg = np.zeros((N, T), np.int64); cur = np.full(N, -1)
    for t in range(T):
        seg[:, t] = np.where(cur >= 0, cur, 0); cur = np.where(nl[:, t], t, cur)
    return np.arange(T)[None, :] - seg
HE_DSN = dsn_grid_fn(held_np)
def act_by_dsn(name, curbool):
    a = np.abs(act[name]); out = {}
    for lab, lo, hi in [('dsn1', 1, 1), ('dsn2-3', 2, 3), ('dsn4-7', 4, 7), ('dsn8-15', 8, 15), ('dsn16+', 16, 999)]:
        mm = valid & curbool & (HE_DSN >= lo) & (HE_DSN <= hi)
        if mm.sum() >= 20: out[lab] = msd(a[mm])
    return out
cur_word_b = (cur_class == 'word'); cur_cap_b = (cur_class == 'capital')
step5 = {}
for name in TARGETS:
    step5[name] = {
        'abs_activation_by_current_class': act_by_current_class(name),
        'abs_activation_by_dsn_within_word': act_by_dsn(name, cur_word_b),
    }
    bc = step5[name]['abs_activation_by_current_class']
    top = sorted(bc.items(), key=lambda kv: -(kv[1]['mean'] or 0))[:4]
    print(f"  {name:12s} top-activating current classes: "
          + ", ".join(f"{c}={v['mean']:.3f}" for c, v in top), flush=True)

# ===================== SAVE =====================
OUT = {
    'meta': {
        'model': 'bilin18', 'tokenizer': 'gpt2 (model exposes none) -- FLAGGED',
        'held_slice': 'FW[448:600,:128]', 'train_slice_for_mlp_dirs': 'FW[0:256,:128]',
        'components_pushed_class': PUSHED,
        'KCAUSAL_firing': KCAUSAL, 'batch': BATCH,
        'forward': 'VERBATIM from qk_unsup_classpush.py (bilin18 two-branch (s1*s2) masked UNNORMALISED, '
                   'per-head QK rms_norm THEN RoPE, v-lerp a.lamb block-0 cache, 30*tanh logits).',
        'ablation': 'mean-ablation: MLP direction projected out to per-position held mean projection.',
        'class_summed_dlogit': 'sum of (base-ablated) logit over all tokens of the pushed CLASS, per position '
                               '(same measure as §68). Positive = the direction BOOSTS that class.',
        'dCE': 'ablated - base cross-entropy at the true next token (positive = ablation HURT prediction).',
        'decisive_test': 'push and activation split by whether the TRUE next token IS the pushed class vs IS NOT. '
                         'Context-conditioned => concentrates push where the class is due (ratio>>1); static prior '
                         '=> flat (ratio~1, cf §66 capitalization circuit ratio 1.0).',
    },
    'step1_verify_reproduce_s68': step1,
    'step2_decisive_context_vs_static': step2,
    'step3_sign_correctness_dCE_by_true_class': step3,
    'step4_minimal_circuit_joint_structure': step4,
    'step5_content_vs_position': step5,
}
json.dump(OUT, open(f'{QK}/qk_arc_integrator.json', 'w'), indent=2)
print("\nSaved qk_arc_integrator.json", flush=True)
print("QK ARC INTEGRATOR DONE", flush=True)
