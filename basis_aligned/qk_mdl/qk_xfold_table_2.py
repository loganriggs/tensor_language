"""Refinement of qk_xfold_table.py part (c): the raw singular-vector unembed readouts were
format-token-heavy and hard to narrate. Here the per-token effective linear maps are read out the
way the program's conventions favor: apply token t's effective map M_t to the ACTUAL contextual
rest vectors r at held positions carrying token t, average the resulting linear contribution,
and summarize it as (i) lexical-CLASS logit movement through the unembedding (lex1 classes,
VERBATIM from qk_arc_h70.py / qk_unsup_classpush.py) and (ii) top boosted/suppressed vocabulary
items. Also: exact map cosines for INTERPRETABLE pairs (casing variants, determiner vs punctuation,
etc.). This gives the "after token X, layer 1 becomes a map that does Y" narrative with the map
evaluated on real contexts. Mediation caveat unchanged (block 1 feeds later squares; readout is an
interpretive summary). Table from TRAIN FW[0:256]; positions from HELD FW[448:600]. Batch 4."""
import json, sys, time, subprocess
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot
from transformers import AutoTokenizer
torch.manual_seed(0)
DEV = 'cuda'; QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
OUT = f'{QK}/qk_xfold_table.json'

def gpu_guard(min_free=4500, tries=45, sleep=20):
    for _ in range(tries):
        free = int(subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits']
        ).decode().split('\n')[0].strip())
        if free >= min_free:
            print(f"GPU guard: {free} MiB free -- proceeding.", flush=True); return
        print(f"GPU guard: only {free} MiB free (<{min_free}); sleeping {sleep}s ...", flush=True)
        time.sleep(sleep)
    raise RuntimeError("GPU guard timed out")
gpu_guard()

m, cfg = load_elriggs('bilin18')
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
V = cfg['vocab_size']; NL = len(m.transformer.h)
FW = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))
TRAIN = FW[0:256, :128].to(DEV); HELD = FW[448:600, :128].to(DEV)
B0 = 4; STR, _ = TRAIN.shape; S_, T_ = HELD.shape
tok = AutoTokenizer.from_pretrained('gpt2')
b0, b1 = m.transformer.h[0], m.transformer.h[1]
L1 = b1.mlp.Left.weight.detach().float(); R1 = b1.mlp.Right.weight.detach().float()
D1w = b1.mlp.Down.weight.detach().float(); bias1 = b1.mlp.Down_bias.detach().float()
lamE = (b1.lambdas[0] * (b0.lambdas[0] + b0.lambdas[1]) + b1.lambdas[1]).item()
lam0 = b1.lambdas[0].item()

# ---- lexical class library (VERBATIM lex1) ----
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
Wu = m.lm_head.weight.detach().float()

@torch.no_grad()
def fwd_to1(idx):
    B, T = idx.shape
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    out = {}
    for li in range(2):
        blk = m.transformer.h[li]; x = blk.lambdas[0]*x + blk.lambdas[1]*x0; a = blk.attn; hcur = F.rms_norm(x, (D,))
        def qk(l): z = F.rms_norm(l(hcur).view(B, T, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
        v = a.c_v(hcur).view(B, T, NH, HD)
        if v1 is None: v1 = v
        v = (1-a.lamb)*v + a.lamb*v1.view_as(v)
        q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k)/HD; s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD
        pat = (s1*s2).masked_fill(~mask, 0.0); yh = torch.einsum('bhqk,bkhd->bqhd', pat, v)
        aout = a.c_proj(yh.reshape(B, T, -1)); x = x + aout
        mo = blk.mlp(F.rms_norm(x, (D,)))
        if li == 0: out['a0c'] = aout; out['mo0'] = mo
        if li == 1: out['aout1'] = aout; out['x_pre1'] = x
        x = x + mo
    return out

# ---- rebuild token table on TRAIN (verbatim parent) ----
print("train table ...", flush=True)
tok_sum = torch.zeros(V, D, device=DEV); tok_cnt = torch.zeros(V, device=DEV)
for i in range(0, STR, B0):
    o = fwd_to1(TRAIN[i:i+B0])
    fl = TRAIN[i:i+B0].reshape(-1)
    tok_sum.index_add_(0, fl, o['mo0'].reshape(-1, D).float())
    tok_cnt.index_add_(0, fl, torch.ones_like(fl, dtype=torch.float))
GLOB = tok_sum.sum(0)/tok_cnt.sum()
seen = tok_cnt > 0
TT = torch.where(seen[:, None], tok_sum/tok_cnt.clamp_min(1)[:, None], GLOB[None])
WTE_N = F.rms_norm(m.transformer.wte.weight.detach().float(), (D,))
S_ALL = lamE*WTE_N + lam0*TT
del tok_sum

def make_map(t):
    st = S_ALL[int(t)]
    a = L1 @ st; b = R1 @ st
    return D1w @ (a[:, None]*R1 + b[:, None]*L1)

def mapcos(t1, t2):
    M1_ = make_map(t1).reshape(-1); M2_ = make_map(t2).reshape(-1)
    return float((M1_ @ M2_)/(M1_.norm()*M2_.norm()))

def tid(s):
    ids = tok.encode(s)
    assert len(ids) == 1, (s, ids)
    return ids[0]

# ---- interpretable pair cosines ----
pairs = [(' the', ' The'), (' The', 'The'), (' the', 'the'), ('.', ','), ('.', ' the'),
         (' the', ' a'), (' and', ' or'), ('.', '\n'), (' D', ' the'), (' where', ' D'),
         (' of', ' in'), (' is', ' was'), ('.', ' D')]
paircos = {}
for s1, s2 in pairs:
    try:
        c = mapcos(tid(s1), tid(s2))
        paircos[f'{s1!r} vs {s2!r}'] = round(c, 4)
        print(f"  map cosine {s1!r:10s} vs {s2!r:10s}: {c:+.4f}", flush=True)
    except AssertionError:
        pass

# ---- class-level readout of the linear-in-context part on ACTUAL held contexts ----
CHOSEN = [' the', '.', ',', ' The', ' and', ' of', ' D', '\n', ' is', ' where']
chosen_ids = {}
for s in CHOSEN:
    try: chosen_ids[s] = tid(s)
    except AssertionError: pass
held_np = HELD.cpu().numpy()
print("collect linear contributions at held positions ...", flush=True)
accum = {s: {'lin_sum': torch.zeros(D, device=DEV), 'n': 0} for s in chosen_ids}
for i in range(0, S_, B0):
    idx = HELD[i:i+B0]
    o = fwd_to1(idx)
    x = o['x_pre1'].float()
    rho2 = x.pow(2).sum(-1, keepdim=True)/D
    r = (lam0*o['a0c'] + o['aout1']).float()
    s_vec = S_ALL[idx]
    lin = (((s_vec @ L1.T)*(r @ R1.T) + (r @ L1.T)*(s_vec @ R1.T)) @ D1w.T)/rho2
    for s, t in chosen_ids.items():
        msk = (idx == t)
        if msk.any():
            accum[s]['lin_sum'] += lin[msk].sum(0)
            accum[s]['n'] += int(msk.sum())

readouts = {}
for s, t in chosen_ids.items():
    n = accum[s]['n']
    if n == 0: continue
    mlin = accum[s]['lin_sum']/n
    lg = Wu @ mlin                                     # (V,) first-order logit movement
    cls = (CMAT @ lg)                                  # summed logit movement per class
    cls_d = {CLASS_LIST[i]: round(float(cls[i]), 1) for i in range(len(CLASS_LIST))}
    cls_sorted = sorted(cls_d.items(), key=lambda z: -abs(z[1]))[:6]
    topv = torch.argsort(lg, descending=True)[:10]; botv = torch.argsort(lg)[:10]
    readouts[repr(s)] = {
        'n_held_positions': n,
        'mean_linear_contribution_norm': round(float(mlin.norm()), 2),
        'class_logit_movement_top': cls_sorted,
        'boosts': [repr(tok.decode([int(v)])) for v in topv],
        'suppresses': [repr(tok.decode([int(v)])) for v in botv]}
    print(f"\nafter {s!r} (n={n}): classes {cls_sorted}", flush=True)
    print("  boosts", readouts[repr(s)]['boosts'], flush=True)
    print("  suppresses", readouts[repr(s)]['suppresses'], flush=True)

res = json.load(open(OUT))
res['effective_maps']['interpretable_pair_cosines'] = paircos
res['effective_maps']['class_readout_on_actual_contexts'] = {
    'method': 'mean over held positions with token t of the token-indexed LINEAR part '
              '(B(s_t,r)+B(r,s_t))/rho2 applied to the actual rest vector r; read through the '
              'unembedding as summed logit movement per lex1 class; first-order, mediation caveat applies',
    'per_token': readouts}
json.dump(res, open(OUT, 'w'), indent=1)
print("QK XFOLD TABLE 2 DONE", flush=True)
