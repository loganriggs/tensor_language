"""ARC #4 script 2: CONCRETE EXAMPLES, one per region of the feed-forward flow map measured in
script 1 (qk_cascade_end.json): cascade region k=4 (next-square fraction 0.51, direct 0.00),
distributed region k=8 (0.34 / 0.26), readout region k=14 (0.24 / 0.91).

For each representative block: top-200 deviation-firing positions (|mo_k - per-position mean|
norm, VERBATIM topmask/bad_trigger from qk_arc_mlp3_2.py), top-4 held-back snippets with
context, and the class-summed delta-logit signature of the damage at firing positions for
  total_damage  (ablate, everything downstream responds)  and
  direct_only   (ablate, all later components frozen clean -- the block's own direct write)
-- the section-96/98 discriminator: a cascade block's damage signature is all mediated (direct
signature ~ 0), a readout block's direct signature ~= its total signature.
Machinery VERBATIM qk_arc_mlp3_2.py part B (lex1/VOCAB_CLASS, sig, freeze-patch fwd_route).
Held FW[448:600,:128], batch 6. Appends 'examples' to qk_cascade_end.json."""
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import json, sys, time, subprocess
import numpy as np
import torch
import torch.nn.functional as F
from collections import Counter
from transformers import AutoTokenizer
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot
torch.manual_seed(0)
DEV = 'cuda'; QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
OUT = f'{QK}/qk_cascade_end.json'

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
REPS = {'cascade': 4, 'distributed': 8, 'readout': 14}
KS = sorted(REPS.values()); KCAUSAL = 200
print(f"bilin18 held {S_}x{T_}; region representatives {REPS}", flush=True)

# ---------------- lexical class library (VERBATIM lex1/VOCAB_CLASS) ----------------
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

# ---------------- means pass (same as script 1 PASS 0) ----------------
@torch.no_grad()
def fwd_collect_means(idx, sums):
    B, T = idx.shape
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    for li in range(NL):
        blk = m.transformer.h[li]; x = blk.lambdas[0]*x + blk.lambdas[1]*x0; a = blk.attn; hcur = F.rms_norm(x, (D,))
        def qk(l): z = F.rms_norm(l(hcur).view(B, T, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
        v = a.c_v(hcur).view(B, T, NH, HD)
        if v1 is None: v1 = v
        v = (1-a.lamb)*v + a.lamb*v1.view_as(v)
        q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k)/HD; s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD
        pat = (s1*s2).masked_fill(~mask, 0.0); yh = torch.einsum('bhqk,bkhd->bqhd', pat, v)
        x = x + a.c_proj(yh.reshape(B, T, -1))
        mo = blk.mlp(F.rms_norm(x, (D,)))
        if li in sums: sums[li] += mo.float().sum(0)
        x = x + mo

print("PASS 0: per-position mlp-output means ...", flush=True)
sums = {k: torch.zeros(T_, D, device=DEV) for k in KS}
for i in range(0, S_, B0):
    fwd_collect_means(HELD[i:i+B0], sums)
MO_MEAN = {k: (sums[k]/S_) for k in KS}
del sums

# ---------------- freeze-patch forward (VERBATIM qk_arc_mlp3_2.py fwd_route) ----------------
@torch.no_grad()
def fwd_route(idx, config=None, cache=None, fill_cache=False, want_logits=False, dev_norm_out=None):
    B, T = idx.shape
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    cfg_ = config or {}
    LI = cfg_.get('k'); fa = cfg_.get('freeze_attn', ()); fm = cfg_.get('freeze_mlp', ())
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
        if LI is not None and li == LI:
            if dev_norm_out is not None:
                dev_norm_out.append((mo - MO_MEAN[LI][None]).norm(dim=-1).cpu())
            if cfg_.get('ablate'):
                mo = MO_MEAN[LI].unsqueeze(0).expand(B, -1, -1).to(x.dtype)
        if fill_cache: cache[li] = (aout, mo)
        x = x + mo
    logits = 30*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30)
    ce = F.cross_entropy(logits[:, :-1].reshape(-1, V).float(), idx[:, 1:].reshape(-1), reduction='none').view(B, T-1)
    return (ce, logits) if want_logits else (ce, None)

# ---------------- PRE-PASS: deviation firing magnitudes per representative block ----------------
print("PRE-PASS: deviation firing magnitudes ...", flush=True)
dev_act = {}
for k in KS:
    dno = []
    for i in range(0, S_, B0):
        fwd_route(HELD[i:i+B0], config={'k': k}, dev_norm_out=dno)
    dev_act[k] = torch.cat(dno, 0).numpy()
dev_mask = {k: topmask(dev_act[k]) for k in KS}

def sig(cs, dm):
    co = np.argsort(-np.abs(cs))
    return {'class_summed_top6': {CLASS_LIST[c]: round(float(cs[c]), 4) for c in co[:6]},
            'top_boosted_tokens': [(dec(t), round(float(dm[t]), 4)) for t in np.argsort(-dm)[:6]],
            'top_suppressed_tokens': [(dec(t), round(float(dm[t]), 4)) for t in np.argsort(dm)[:6]]}

# ---------------- signatures + snippets per representative block ----------------
examples = {}
for region, k in REPS.items():
    LATER = list(range(k+1, NL)); AF, AM = set(LATER), set(LATER)
    classsum = {n: torch.zeros(len(CLASS_LIST), device=DEV) for n in ('total', 'direct')}
    dlog = {n: torch.zeros(V, device=DEV) for n in ('total', 'direct')}
    nfire = 0
    t0 = time.time()
    for i in range(0, S_, B0):
        idx = HELD[i:i+B0]
        mk = torch.from_numpy(dev_mask[k][i:i+B0]).to(DEV)
        if not bool(mk.any()): continue
        cache = {}
        bce, blog = fwd_route(idx, fill_cache=True, cache=cache, want_logits=True)
        for name, c in [('total', {'k': k, 'ablate': True}),
                        ('direct', {'k': k, 'ablate': True, 'freeze_attn': AF, 'freeze_mlp': AM})]:
            ce, lg = fwd_route(idx, config=c, cache=cache, want_logits=True)
            dl = (blog.float() - lg.float())[mk]
            classsum[name] += CMAT @ dl.sum(0); dlog[name] += dl.sum(0)
            if name == 'total': nfire += int(dl.shape[0])
            del dl, lg
        del cache, blog
    # trigger-class census + snippets at top firing positions
    fs, fp = np.where(dev_mask[k])
    cur = held_np[fs, fp]
    trig_classes = Counter(VOCAB_CLASS[cur].tolist()).most_common(6)
    trig_tokens = Counter([dec(t) for t in cur]).most_common(8)
    flat = dev_act[k].copy(); flat[bad_trigger] = -1e30
    order = np.argsort(-flat.reshape(-1))[:4]
    snips = []
    for o in order:
        s, p = divmod(int(o), T_)
        lo = max(0, p-11)
        snips.append({'seq': int(s), 'pos': int(p),
                      'context_ending_at_pos': tok.decode(held_np[s, lo:p+1]),
                      'current_token': dec(held_np[s, p]),
                      'dev_norm': round(float(dev_act[k][s, p]), 2)})
    cs_t = (classsum['total']/max(nfire, 1)).cpu().numpy()
    cs_d = (classsum['direct']/max(nfire, 1)).cpu().numpy()
    dm_t = (dlog['total']/max(nfire, 1)).cpu().numpy()
    dm_d = (dlog['direct']/max(nfire, 1)).cpu().numpy()
    # magnitude ratio: how much of the total delta-logit push is present in the direct write
    ratio = float(np.linalg.norm(dm_d)/max(np.linalg.norm(dm_t), 1e-9))
    examples[region] = {
        'block': k, 'n_firing_positions': nfire,
        'trigger_classes_top6': [(c, int(n)) for c, n in trig_classes],
        'trigger_tokens_top8': trig_tokens,
        'snippets_top4': snips,
        'total_signature_at_dev_firing': sig(cs_t, dm_t),
        'direct_readout_signature_at_dev_firing': sig(cs_d, dm_d),
        'direct_to_total_delta_logit_norm_ratio': round(ratio, 4)}
    print(f"{region} (k={k}): {nfire} firing pos, direct/total delta-logit norm ratio {ratio:.3f} "
          f"({time.time()-t0:.0f}s)", flush=True)
    print(f"  triggers {trig_classes[:4]} tokens {trig_tokens[:5]}", flush=True)
    print(f"  total push {examples[region]['total_signature_at_dev_firing']['class_summed_top6']}", flush=True)
    print(f"  direct push {examples[region]['direct_readout_signature_at_dev_firing']['class_summed_top6']}", flush=True)

res = json.load(open(OUT))
res['examples'] = {
 'method': 'top-200 deviation-firing positions per representative block (|mo_k - per-position '
           'mean| norm, VERBATIM topmask); class-summed delta-logit of clean minus intervened '
           'logits at firing positions, for total damage vs direct-only (all later components '
           'frozen clean); machinery VERBATIM qk_arc_mlp3_2.py part B',
 'regions': examples}
json.dump(res, open(OUT, 'w'), indent=1)
print(f"Saved examples to {OUT}", flush=True)
print("QK CASCADE END (script 2) DONE", flush=True)
