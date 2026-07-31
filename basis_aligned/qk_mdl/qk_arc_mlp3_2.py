"""ALGORITHM ARC #3, part 2: WHERE DOES mlp.L3's CONTRIBUTION FLOW -- is block 3 the TERMINAL
stage of the iterated-squaring pipeline (H1), and how much of the category code does it own (H2)?

(A) RESTORE-AT-BLOCK-K MEDIATION PROFILE (machinery VERBATIM qk_arc_mlp0_2.py, adapted LI=3).
Mean-ablate mlp.L3 (mo3 -> per-position decomposition mean MEANF from part 1, the census floor
intervention) but keep the true deviation dev3 = mo3 - MEANF, and RESTORE it into the residual
stream at the entry of block k with its exact linear propagation coefficient
prod_{j=4..k-1} lambda0_j (the deviation's coefficient in a clean run -- residual recurrence
x <- lambda0*x + lambda1*x0 is linear, x0 skip carries no mlp3 content).
  restore_4     == no ablation (exactness gate: delta cross-entropy ~0)
  restore_k     == only blocks 4..k-1 saw the ablated stream => damage attributable to consumers
                   among blocks 4..k-1
  readout_only  == all mediated damage, direct path kept
  remove_direct == clean run minus the direct linear path at readout => direct-only damage
  ablate        == never restore (must reproduce the census floor 0.6163)

(B) FREEZE-PATCH ROUTING (machinery VERBATIM qk_arc_square_2.py PART C, ablating the WHOLE mlp3
deviation instead of one term): ablate dev3; freeze all later components to clean cached values
(direct_only); unfreeze one later MLP at a time (direct_plus_mlp{k}); unfreeze all later attention
(direct_plus_all_attn); freeze ONLY mlp4 clean (the section-96 discriminator: if block 4's square
is THE consumer, this alone removes most damage). Plus direct-readout and total class signatures
at block 3's top-200 firing positions.

(C) CATEGORY PROBE (adapted VERBATIM from qk_arc_mlp0_2.py / qk_category_engine.py): ridge probe
for the NEXT token's 6-way category (subword/punct/capital/digit/funcword/other) on the residual
after block 3 and block 4, intact vs EACH of mlp.L0 / L1 / L2 / L3 individually mean-ablated
(per-position held-set means from part 1) -- block 3's share of the category-code gain vs the
other engine members (section-97 block-0 reference: blk4 0.611 -> 0.542, embed baseline 0.527).

Held FW[448:600,:128], paired standard errors, batch 6, GPU guard.
Appends 'H1_mediation', 'H1_freeze_patch_routing', 'H2_category_probe' to qk_arc_mlp3.json."""
import json, os, sys, time, subprocess, math
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot
torch.manual_seed(0)
DEV = 'cuda'; QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
OUT = f'{QK}/qk_arc_mlp3.json'

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
FW = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))
HELD = FW[448:600, :128].to(DEV); B0 = 6
S_, T_ = HELD.shape
LI = 3; KCAUSAL = 200
MST = torch.load(f'{QK}/qk_arc_mlp3_means.pt', map_location=DEV)
MEANF = MST['MEANF']                                   # (T,D) per-position decomposition mean of mo3
MO_MEAN = {int(k): v.to(DEV) for k, v in MST['MO_MEAN'].items()}   # per-position means of mo0..mo3
LAM0 = [float(b.lambdas[0]) for b in m.transformer.h]
# deviation coefficient at entry of block k (before block k's lambda mix) = prod_{j=LI+1..k-1} lam0_j
CREST = {k: float(np.prod(LAM0[LI+1:k])) for k in range(LI+1, NL+1)}   # CREST[NL] = readout coefficient
print(f"bilin18 held {S_}x{T_}; LI={LI}; direct-path coefficient to readout = {CREST[NL]:.3e}", flush=True)

# =====================================================================================
# (A) RESTORE-AT-BLOCK-K -- forward VERBATIM qk_arc_mlp0_2.py fwd, adapted LI=3
# =====================================================================================
@torch.no_grad()
def fwd(idx, mode='full', k=None):
    """mode: 'full' | 'ablate' | 'restore' (at entry of block k, LI+1<=k<=NL-1) | 'readout_only'
    | 'remove_direct'. Skeleton verbatim from qk_allterm_census.py."""
    B, T = idx.shape
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    dev3 = None
    for li in range(NL):
        blk = m.transformer.h[li]
        if mode == 'restore' and li == k and dev3 is not None:
            x = x + CREST[k]*dev3
        x = blk.lambdas[0]*x + blk.lambdas[1]*x0; a = blk.attn; hcur = F.rms_norm(x, (D,))
        def qk(l): z = F.rms_norm(l(hcur).view(B, T, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
        v = a.c_v(hcur).view(B, T, NH, HD)
        if v1 is None: v1 = v
        v = (1-a.lamb)*v + a.lamb*v1.view_as(v)
        q, kk_, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, kk_)/HD; s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD
        pat = (s1*s2).masked_fill(~mask, 0.0); yh = torch.einsum('bhqk,bkhd->bqhd', pat, v)
        x = x + a.c_proj(yh.reshape(B, T, -1))
        mo = blk.mlp(F.rms_norm(x, (D,)))
        if li == LI and mode != 'full':
            dev3 = mo - MEANF.unsqueeze(0)
            if mode in ('ablate', 'restore', 'readout_only'):
                mo = MEANF.unsqueeze(0).expand(B, -1, -1).to(x.dtype)
        x = x + mo
    if mode == 'readout_only': x = x + CREST[NL]*dev3
    if mode == 'remove_direct': x = x - CREST[NL]*dev3
    logits = 30*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30)
    return F.cross_entropy(logits[:, :-1].reshape(-1, V).float(),
                           idx[:, 1:].reshape(-1), reduction='none').view(B, T-1)

print("BASE ...", flush=True)
base = torch.cat([fwd(HELD[i:i+B0]).cpu() for i in range(0, S_, B0)], 0)
print(f"base CE {float(base.mean()):.4f}", flush=True)

def dstat(ce):
    d = (ce - base).flatten().double(); return float(d.mean()), float(d.std()/np.sqrt(d.numel()))

def run(mode, k=None):
    return torch.cat([fwd(HELD[i:i+B0], mode=mode, k=k).cpu() for i in range(0, S_, B0)], 0)

flow = {}
cfgs = [('ablate', 'ablate', None), ('remove_direct', 'remove_direct', None),
        ('readout_only', 'readout_only', None)]
cfgs += [(f'restore_at_block_{k}', 'restore', k) for k in (4, 5, 6, 7, 8, 9, 10, 12, 15)]
for name, mode, k in cfgs:
    mn, se = dstat(run(mode, k))
    flow[name] = {'dCE': round(mn, 4), 'SE': round(se, 5)}
    print(f"  {name:20s} dCE {mn:+.4f} +- {se:.5f}", flush=True)

CENSUS_FLOOR = 0.6163
assert abs(flow['restore_at_block_4']['dCE']) < 0.003, "exactness gate FAILED: restore_4 not ~0"
assert abs(flow['ablate']['dCE'] - CENSUS_FLOOR) < 0.03, "floor gate FAILED vs census 0.6163"
print("gates pass: restore_4 ~ 0, ablate ~ census floor", flush=True)

res = json.load(open(OUT))
res['H1_mediation'] = {
 'method': 'mean-ablate mlp.L3 to the per-position decomposition mean; restore the true deviation '
           'at the entry of block k with its exact linear propagation coefficient '
           'prod lambda0_(4..k-1); restore_4 == no ablation (exactness gate), readout_only == all '
           'mediated damage with direct path kept, remove_direct == direct-only damage, '
           'ablate == census floor',
 'direct_path_coefficient_to_readout': CREST[NL],
 'lambda0_per_block': [round(l, 4) for l in LAM0],
 'flow': flow,
 'census_floor_ref': CENSUS_FLOOR}
json.dump(res, open(OUT, 'w'), indent=1)
print("Saved H1_mediation", flush=True)

# =====================================================================================
# (B) FREEZE-PATCH ROUTING -- machinery VERBATIM qk_arc_square_2.py PART C, ablating the
# whole mlp3 deviation dev3 = mo3 - MEANF. cache: li -> (aout, mo) clean per batch.
# =====================================================================================
# lexical class library (VERBATIM lex1/VOCAB_CLASS)
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
for t in range(V): CMAT[CIDX[VOCAB_CLASS[t]], t] = 1.0
_special = {tok.eos_token_id}
for _t in range(V):
    if '�' in tok.decode([_t]): _special.add(_t)
SPECIAL = np.array(sorted(_special))
held_np = HELD.cpu().numpy()
pos_t = np.tile(np.arange(T_), S_).reshape(S_, T_)
bad_trigger = (pos_t == 0) | np.isin(held_np, SPECIAL) | (pos_t >= T_-1)

def topmask(act):
    a = act.copy().reshape(-1); a[bad_trigger.reshape(-1)] = -1e30
    tk = np.argpartition(a, -KCAUSAL)[-KCAUSAL:]
    mk = np.zeros(S_*T_, bool); mk[tk] = True
    return mk.reshape(S_, T_)

@torch.no_grad()
def fwd_route(idx, config=None, cache=None, fill_cache=False, want_logits=False, dev_norm_out=None):
    """config: dict(ablate=bool, freeze_attn=set, freeze_mlp=set). Freeze = replace component
    output with the clean cached value. Ablate = mo3 -> MEANF (deviation removed at source)."""
    B, T = idx.shape
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    cfg_ = config or {}
    fa = cfg_.get('freeze_attn', ()); fm = cfg_.get('freeze_mlp', ())
    for li in range(NL):
        blk = m.transformer.h[li]; x = blk.lambdas[0]*x + blk.lambdas[1]*x0; a = blk.attn; hcur = F.rms_norm(x, (D,))
        def qk(l): z = F.rms_norm(l(hcur).view(B, T, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
        v = a.c_v(hcur).view(B, T, NH, HD)
        if v1 is None: v1 = v
        v = (1-a.lamb)*v + a.lamb*v1.view_as(v)
        q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k)/HD; s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD
        pat = (s1*s2).masked_fill(~mask, 0.0); yh = torch.einsum('bhqk,bkhd->bqhd', pat, v)
        aout = a.c_proj(yh.reshape(B, T, -1))
        if li in fa: aout = cache[li][0]
        x = x + aout
        mo = blk.mlp(F.rms_norm(x, (D,)))
        if li in fm: mo = cache[li][1]
        if li == LI:
            if dev_norm_out is not None:
                dev_norm_out.append((mo - MEANF[None]).norm(dim=-1).cpu())
            if cfg_.get('ablate'):
                mo = MEANF.unsqueeze(0).expand(B, -1, -1).to(x.dtype)
        if fill_cache: cache[li] = (aout, mo)
        x = x + mo
    logits = 30*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30)
    ce = F.cross_entropy(logits[:, :-1].reshape(-1, V).float(), idx[:, 1:].reshape(-1), reduction='none').view(B, T-1)
    return (ce, logits) if want_logits else (ce, None)

print("PRE-PASS: block-3 deviation firing magnitudes ...", flush=True)
dno = []
for i in range(0, S_, B0):
    fwd_route(HELD[i:i+B0], dev_norm_out=dno)
dev_act = torch.cat(dno, 0).numpy()                    # (S,T)
dev_mask = topmask(dev_act)

LATER = list(range(LI+1, NL))
ALL_FA = set(LATER); ALL_FM = set(LATER)
route_cfgs = [('sanity_freeze_all_no_ablate', {'ablate': False, 'freeze_attn': ALL_FA, 'freeze_mlp': ALL_FM}),
              ('total_damage', {'ablate': True}),
              ('direct_only', {'ablate': True, 'freeze_attn': ALL_FA, 'freeze_mlp': ALL_FM}),
              ('direct_plus_all_attn', {'ablate': True, 'freeze_attn': set(), 'freeze_mlp': ALL_FM}),
              ('freeze_only_mlp4', {'ablate': True, 'freeze_attn': set(), 'freeze_mlp': {4}})]
for li in LATER:
    route_cfgs.append((f'direct_plus_mlp{li}', {'ablate': True, 'freeze_attn': ALL_FA,
                                                'freeze_mlp': ALL_FM - {li}}))
print(f"PART B: {len(route_cfgs)} routing configs ...", flush=True)
acc = {name: [] for name, _ in route_cfgs}
classsum_direct = torch.zeros(len(CLASS_LIST), device=DEV)
classsum_total = torch.zeros(len(CLASS_LIST), device=DEV)
dlog_direct = torch.zeros(V, device=DEV); dlog_total = torch.zeros(V, device=DEV); nfire_t = 0
t0 = time.time()
for i in range(0, S_, B0):
    idx = HELD[i:i+B0]
    cache = {}
    bce, blog = fwd_route(idx, fill_cache=True, cache=cache, want_logits=True)
    mk = torch.from_numpy(dev_mask[i:i+B0]).to(DEV)
    for name, c in route_cfgs:
        want = name in ('direct_only', 'total_damage') and bool(mk.any())
        ce, lg = fwd_route(idx, config=c, cache=cache, want_logits=want)
        acc[name].append(ce.cpu())
        if want:
            dl = (blog.float() - lg.float())[mk]
            if name == 'direct_only':
                classsum_direct += CMAT @ dl.sum(0); dlog_direct += dl.sum(0)
            else:
                classsum_total += CMAT @ dl.sum(0); dlog_total += dl.sum(0); nfire_t += int(dl.shape[0])
            del dl
        del lg
    del cache, blog
print(f"PART B forwards done ({time.time()-t0:.0f}s)", flush=True)
routing = {}
for name, _ in route_cfgs:
    mn, se = dstat(torch.cat(acc[name], 0))
    routing[name] = {'dCE': round(mn, 4), 'SE': round(se, 5)}
    print(f"  {name:26s} dCE {mn:+.4f} +- {se:.5f}", flush=True)
direct = routing['direct_only']['dCE']; total = routing['total_damage']['dCE']
med = {}
for li in LATER:
    med[li] = round(routing[f'direct_plus_mlp{li}']['dCE'] - direct, 4)
routing['mediation_via_single_mlp_minus_direct'] = med
routing['direct_frac_of_total'] = round(direct/total, 4) if total else None
cs_d = (classsum_direct/max(nfire_t, 1)).cpu().numpy()
cs_t = (classsum_total/max(nfire_t, 1)).cpu().numpy()
dm_d = (dlog_direct/max(nfire_t, 1)).cpu().numpy(); dm_t = (dlog_total/max(nfire_t, 1)).cpu().numpy()
def sig(cs, dm):
    co = np.argsort(-np.abs(cs))
    return {'class_summed_top6': {CLASS_LIST[c]: round(float(cs[c]), 4) for c in co[:6]},
            'top_boosted_tokens': [(dec(t), round(float(dm[t]), 4)) for t in np.argsort(-dm)[:6]],
            'top_suppressed_tokens': [(dec(t), round(float(dm[t]), 4)) for t in np.argsort(dm)[:6]]}
routing['direct_readout_signature_at_dev_firing'] = sig(cs_d, dm_d)
routing['total_signature_at_dev_firing'] = sig(cs_t, dm_t)
res = json.load(open(OUT))
res['H1_freeze_patch_routing'] = routing
json.dump(res, open(OUT, 'w'), indent=1)
print("Saved H1_freeze_patch_routing", flush=True)

# =====================================================================================
# (C) CATEGORY PROBE (adapted verbatim from qk_arc_mlp0_2.py / qk_category_engine.py;
# ablate = one of blocks {0,1,2,3}, per-position held-set mean MO_MEAN[li])
# =====================================================================================
import string as _string
_P = set(_string.punctuation)
FUNC = {'the','of','and','to','a','in','is','that','it','for','was','as','with','on','be','at','by','this','are','from','or','an','but','not','which'}
CAT = torch.full((V,), 5, dtype=torch.long)
for i in range(50257):
    s = tok.convert_ids_to_tokens(i)
    if s is None: continue
    core = s.replace('Ġ', ''); lead = s.startswith('Ġ')
    if len(core) and all(c in _P for c in core): CAT[i] = 1
    elif len(core) and all(c.isdigit() for c in core): CAT[i] = 3
    elif core.lower() in FUNC: CAT[i] = 4
    elif not lead and len(core) and core[0].isalpha() and core[0].islower(): CAT[i] = 0
    elif lead and len(core) and core[0].isupper(): CAT[i] = 2
CAT = CAT.to(DEV)
DEPTHS = ['embed', 'blk3', 'blk4']

@torch.no_grad()
def collect(idx, ablate_li=None):
    B, T = idx.shape; x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool)); out = {'embed': x.reshape(-1, D)}
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
        if ablate_li is not None and li == ablate_li:
            mo = MO_MEAN[li][:T].unsqueeze(0).expand(B, -1, -1).to(x.dtype)
        x = x + mo
        if f'blk{li}' in DEPTHS: out[f'blk{li}'] = x.reshape(-1, D)
        if li >= 4: break
    return out

def gather(rng, ablate_li=None):
    R = {d: [] for d in DEPTHS}; Y = []
    for i in rng:
        idx = FW[i:i+4, :128].to(DEV)
        res_ = collect(idx[:, :-1], ablate_li)
        lab = CAT[idx[:, 1:].reshape(-1)]
        for d in DEPTHS: R[d].append(res_[d])
        Y.append(lab)
    return {d: torch.cat(R[d]) for d in DEPTHS}, torch.cat(Y)

print("CATEGORY PROBE: gathering residuals (intact + each of mlp.L0-L3 ablated) ...", flush=True)
Rtr, Ytr = gather(range(0, 240, 4))
Rte, Yte = gather(range(300, 400, 4))
Yoh = F.one_hot(Ytr, 6).float()
maj = torch.bincount(Yte, minlength=6).max().item() / Yte.numel()

def probe_acc(Xtr, Xte):
    Xtr = torch.cat([Xtr, torch.ones(Xtr.shape[0], 1, device=DEV)], 1).double()
    Xte = torch.cat([Xte, torch.ones(Xte.shape[0], 1, device=DEV)], 1).double()
    Wp = torch.linalg.solve(Xtr.T @ Xtr + 50*torch.eye(Xtr.shape[1], device=DEV, dtype=torch.double), Xtr.T @ Yoh.double())
    pred = (Xte @ Wp).argmax(1)
    return float((pred == Yte).float().mean())

probe = {'majority': round(maj, 4)}
for d in DEPTHS:
    probe[f'{d}_intact'] = round(probe_acc(Rtr[d], Rte[d]), 4)
    print(f"  probe {d}: intact {probe[f'{d}_intact']}", flush=True)
for abl in (0, 1, 2, 3):
    Rtr_a, _ = gather(range(0, 240, 4), ablate_li=abl)
    Rte_a, _ = gather(range(300, 400, 4), ablate_li=abl)
    for d in ('blk3', 'blk4'):
        probe[f'{d}_mlp{abl}_ablated'] = round(probe_acc(Rtr_a[d], Rte_a[d]), 4)
    print(f"  ablate mlp.L{abl}: blk3 {probe[f'blk3_mlp{abl}_ablated']} "
          f"blk4 {probe[f'blk4_mlp{abl}_ablated']}", flush=True)
    del Rtr_a, Rte_a

res = json.load(open(OUT))
res['H2_category_probe'] = dict(probe,
    ref='qk_arc_mlp0.json H3_downstream category_probe (block-0 numbers: blk3 0.601->0.546, '
        'blk4 0.611->0.542, embed 0.527); machinery source qk_category_engine.py',
    labels='6-way next-token category: subword/punct/capital/digit/funcword/other',
    train='FW[0:240:4] x4 seqs', test='FW[300:400:4] x4 seqs',
    note='ablation = ONE block at a time, per-position held-set mean MO_MEAN[li] from part 1')
json.dump(res, open(OUT, 'w'), indent=1)
print("Saved H2_category_probe to qk_arc_mlp3.json", flush=True)
print("QK ARC MLP3 PART2 DONE", flush=True)
