"""Two 'we understand this' properties Logan asked for:
(A) MINIMALITY: is the 45-component induction circuit locally minimal? For each kept component,
    remove it -> does retention drop below 90%? Count essential vs slack; then iteratively prune
    slack until locally minimal (no single removal keeps >=90%).
(B) PREDICTING GENERALIZATION: does the FineWeb-fit importance map predict knockout effects on a
    DIFFERENT corpus (Pile)? Spearman correlation between FineWeb importance and Pile importance
    over 40 components (top-20 + random-20) for subword and induction.
"""
import json, sys, ast
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot, build_eval_tokens
torch.manual_seed(0)
DEV = 'cuda'; QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
m, cfg = load_elriggs('bilin18')
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
V = cfg['vocab_size']; NL = len(m.transformer.h)
FINEWEB = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))
ALL = [('h', li, h) for li in range(NL) for h in range(NH)] + [('m', li) for li in range(NL)]
tok = AutoTokenizer.from_pretrained('gpt2')
SUB = torch.zeros(V, dtype=torch.bool)
for i in range(50257):
    s = tok.convert_ids_to_tokens(i)
    if s and not s.startswith('Ġ') and len(s.replace('Ġ','')) and s[0].isalpha() and s[0].islower(): SUB[i] = True
SUB = SUB.to(DEV)


@torch.no_grad()
def run(EV, keep, MEAN, collect=False):
    idx = EV[:, :-1]; B, T = idx.shape
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool)); means = {}
    for li in range(NL):
        blk = m.transformer.h[li]; x = blk.lambdas[0]*x + blk.lambdas[1]*x0; a = blk.attn; hcur = F.rms_norm(x, (D,))
        def qk(lin): z = F.rms_norm(lin(hcur).view(B, T, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
        v = a.c_v(hcur).view(B, T, NH, HD)
        if v1 is None: v1 = v
        v = (1-a.lamb)*v + a.lamb*v1.view_as(v)
        q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k)/HD; s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD
        pat = (s1*s2).masked_fill(~mask, 0.0); yh4 = torch.einsum('bhqk,bkhd->bqhd', pat, v)
        if collect: means[('h', li)] = yh4.mean((0, 1))
        if keep is not None:
            for h in range(NH):
                if ('h', li, h) not in keep: yh4[:, :, h, :] = MEAN[('h', li)][h]
        x = x + a.c_proj(yh4.reshape(B, T, -1)); mo = blk.mlp(F.rms_norm(x, (D,)))
        if collect: means[('m', li)] = mo.mean((0, 1))
        if keep is not None and ('m', li) not in keep: mo = MEAN[('m', li)].expand_as(mo)
        x = x + mo
    return 30*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30).float(), means

# ===== (A) minimality of the induction minimal circuit =====
P = 64; NSEQ = 48
pref = FINEWEB[:NSEQ, 1:1+P]; EVi = torch.cat([pref, pref], 1).to(DEV)
SEC = torch.arange(P, 2*P-1, device=DEV); FIR = torch.arange(1, P-1, device=DEV)
def adv_of(lg):
    tgt = EVi[:, 1:]; ce = F.cross_entropy(lg.reshape(-1, V), tgt.reshape(-1), reduction='none').view(NSEQ, -1)
    return ce[:, FIR].mean().item() - ce[:, SEC].mean().item()
_, MEANi = run(EVi, None, None, True)
adv_full = adv_of(run(EVi, None, None)[0]); adv_none = adv_of(run(EVi, set(), MEANi)[0])
TH = adv_none + 0.90*(adv_full - adv_none)
KEEP = set(ast.literal_eval(c) for c in json.load(open(f'{QK}/qk_induction_minimal.json'))['minimal_components'])
ess, slack = [], []
for c in sorted(KEEP):
    a = adv_of(run(EVi, KEEP - {c}, MEANi)[0])
    (ess if a < TH else slack).append((str(c), round(a, 3)))
print(f"(A) minimality: |circuit|=45; essential (removal breaks 90%): {len(ess)}; slack: {len(slack)}", flush=True)
# iterative prune of slack until locally minimal
cur = set(KEEP); rounds = 0
while rounds < 12:
    best = None
    for c in sorted(cur):
        a = adv_of(run(EVi, cur - {c}, MEANi)[0])
        if a >= TH and (best is None or a > best[1]): best = (c, a)
    if best is None: break
    cur.remove(best[0]); rounds += 1
adv_min = adv_of(run(EVi, cur, MEANi)[0])
print(f"(A) locally-minimal size after iterative prune: {len(cur)} (adv {adv_min:+.3f}, {(adv_min-adv_none)/(adv_full-adv_none):.1%})", flush=True)
resA = {'circuit_size': len(KEEP), 'essential': len(ess), 'slack': len(slack),
        'locally_minimal_size': len(cur), 'locally_minimal_adv': round(adv_min, 4),
        'locally_minimal_components': [str(c) for c in sorted(cur)]}

# ===== (B) generalization: FineWeb importance map -> Pile knockouts =====
atlas = json.load(open(f'{QK}/qk_circuit_atlas.json'))['importance_matrix']
def spear(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])
print("(B) building Pile eval...", flush=True)
PILE = build_eval_tokens(n_chunks=40, seq_len=129)  # (40,129) pile-10k, different corpus
EVp = PILE[:, :128].to(DEV)
prefp = PILE[:40, 1:1+P]; EVpi = torch.cat([prefp, prefp], 1).to(DEV)
def adv_of_p(lg):
    tgt = EVpi[:, 1:]; ce = F.cross_entropy(lg.reshape(-1, V), tgt.reshape(-1), reduction='none').view(40, -1)
    return ce[:, FIR].mean().item() - ce[:, SEC].mean().item()
_, MEANp = run(EVp, None, None, True); _, MEANpi = run(EVpi, None, None, True)
def sub_ce(lg, EV):
    tgt = EV[:, 1:]; ce = F.cross_entropy(lg.reshape(-1, V), tgt.reshape(-1), reduction='none').view(EV.shape[0], -1)
    return ce[SUB[tgt]].mean().item()
base_sub = sub_ce(run(EVp, None, None)[0], EVp); base_adv = adv_of_p(run(EVpi, None, None)[0])
rng = np.random.RandomState(3)
resB = {}
for task, evset in [('subword', 'n'), ('induction', 'i')]:
    fw = {ast.literal_eval(k): v for k, v in atlas[task].items()}
    ranked = sorted(ALL, key=lambda c: -fw[c])
    picks = ranked[:20] + [ALL[i] for i in rng.choice(len(ALL), 25, replace=False) if ALL[i] not in ranked[:20]][:20]
    fw_imp, pile_imp = [], []
    for c in picks:
        keep = set(ALL) - {c}
        if evset == 'n':
            d = sub_ce(run(EVp, keep, MEANp)[0], EVp) - base_sub
        else:
            d = base_adv - adv_of_p(run(EVpi, keep, MEANpi)[0])
        fw_imp.append(fw[c]); pile_imp.append(d)
    rho = spear(np.array(fw_imp), np.array(pile_imp))
    resB[task] = {'spearman_fw_to_pile': round(rho, 3), 'n_comps': len(picks)}
    print(f"(B) {task}: Spearman(FineWeb importance -> Pile importance) = {rho:.3f} over {len(picks)} comps", flush=True)

json.dump({'minimality': resA, 'generalization': resB}, open(f'{QK}/qk_understanding_props.json', 'w'), indent=2)
print("QK UNDERSTANDING PROPS DONE", flush=True)
