"""Positive control for the known-recovered h.L8.3 date-successor head: does the
number-boost reproduce in its NATURAL habitat (constructed date sequences)?
Held-out naturalistic FineWeb had only 281 day-number-attending positions (underpowered).
Constructed date prompts; mean-ablate h.L8.3 at the final query position; measure logit
drop on number tokens vs control. Forward copied verbatim from qk_unsup_verify2.py."""
import json, sys, math
import numpy as np, torch, torch.nn.functional as F
from transformers import AutoTokenizer
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot
torch.manual_seed(0)
DEV = 'cuda'
m, cfg = load_elriggs('bilin18')
NH, HD, D = cfg['n_head'], cfg['n_embd']//cfg['n_head'], cfg['n_embd']
V = cfg['vocab_size']; NL = len(m.transformer.h)
tok = AutoTokenizer.from_pretrained('gpt2')
LI, HH = 8, 3

MONTHS = ['January','February','March','April','May','June','July','August','September','October','November','December']
def tid(s):
    t = tok(s)['input_ids']; return t[0] if len(t) == 1 else None
NUMS = [tid(' %d' % d) for d in range(1, 29)]        # day-number tokens (all single-token)
NUMS = [x for x in NUMS if x is not None]

# constructed date sequences: "<Mon> <d1>, <Mon> <d2>, <Mon> <d3>, <Mon>" -> predict a day number
rng = np.random.RandomState(0)
prompts = []
for _ in range(40):
    ms = rng.choice(MONTHS, 4, replace=False)
    ds = rng.choice(range(1, 27), 4, replace=False)
    p = f"{ms[0]} {ds[0]}, {ms[1]} {ds[1]}, {ms[2]} {ds[2]}, {ms[3]}"
    prompts.append(p)
IDS = [tok(p, return_tensors='pt')['input_ids'][0] for p in prompts]
maxlen = max(len(t) for t in IDS)
# left-pad? keep same length by choosing prompts of equal token length
# group by length to batch cleanly
from collections import defaultdict
bylen = defaultdict(list)
for t in IDS: bylen[len(t)].append(t)


@torch.no_grad()
def forward(idx, ablate=False, mean_yh=None, qpos_only=True):
    B, Tt = idx.shape
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(Tt, HD, DEV, x.dtype, 'bf16')
    cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(Tt, Tt, device=DEV, dtype=torch.bool))
    capsrc = None
    for li in range(NL):
        blk = m.transformer.h[li]; x = blk.lambdas[0]*x + blk.lambdas[1]*x0
        a = blk.attn; hc = F.rms_norm(x, (D,))
        def qk(l): z = F.rms_norm(l(hc).view(B, Tt, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
        v = a.c_v(hc).view(B, Tt, NH, HD)
        if v1 is None: v1 = v
        v = (1-a.lamb)*v + a.lamb*v1.view_as(v)
        q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k)/HD
        s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD
        pat = (s1*s2).masked_fill(~mask, 0.0)
        yh = torch.einsum('bhqk,bkhd->bqhd', pat, v)
        if li == LI:
            capsrc = pat[:, HH, -1].abs().argmax(-1)          # (B,) attended key at final pos
            if ablate:
                yh = yh.clone()
                yh[:, -1, HH] = mean_yh.to(yh.dtype)          # mean-ablate head at final query pos
        x = x + a.c_proj(yh.reshape(B, Tt, -1))
        x = x + blk.mlp(F.rms_norm(x, (D,)))
    logits = 30*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30)
    return logits[:, -1].float(), capsrc


# mean yh for head over all final positions (in-distribution mean)
sy = torch.zeros(HD, device=DEV); n = 0
for L, lst in bylen.items():
    idx = torch.stack(lst).to(DEV)
    _, _ = forward(idx)  # warm
for L, lst in bylen.items():
    idx = torch.stack(lst).to(DEV); B, Tt = idx.shape
    # collect yh at head at all positions
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(Tt, HD, DEV, x.dtype, 'bf16')
    cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(Tt, Tt, device=DEV, dtype=torch.bool))
    for li in range(LI+1):
        blk = m.transformer.h[li]; x = blk.lambdas[0]*x + blk.lambdas[1]*x0
        a = blk.attn; hc = F.rms_norm(x, (D,))
        def qk(l): z = F.rms_norm(l(hc).view(B, Tt, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
        v = a.c_v(hc).view(B, Tt, NH, HD)
        if v1 is None: v1 = v
        v = (1-a.lamb)*v + a.lamb*v1.view_as(v)
        q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k)/HD
        s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD
        pat = (s1*s2).masked_fill(~mask, 0.0)
        yh = torch.einsum('bhqk,bkhd->bqhd', pat, v)
        if li == LI:
            sy += yh[:, :, HH].reshape(-1, HD).sum(0); n += B*Tt; break
        x = x + a.c_proj(yh.reshape(B, Tt, -1)); x = x + blk.mlp(F.rms_norm(x, (D,)))
mean_yh = sy / n

# baseline vs ablated at final position
Lb, La, attended = [], [], []
rng2 = np.random.RandomState(1)
CTRL = np.array([t for t in rng2.choice(V, 300, replace=False)][:200])
for Ln, lst in bylen.items():
    idx = torch.stack(lst).to(DEV)
    lb, cs = forward(idx, ablate=False)
    la, _ = forward(idx, ablate=True, mean_yh=mean_yh)
    Lb.append(lb.cpu().numpy()); La.append(la.cpu().numpy())
    attended.append(idx.cpu().numpy()[np.arange(idx.shape[0]), cs.cpu().numpy()])
Lb = np.concatenate(Lb); La = np.concatenate(La); attended = np.concatenate(attended)
dLog = La - Lb
# does the head attend the day-number at final position?
frac_attends_num = float(np.isin(attended, NUMS).mean())
dnum = dLog[:, NUMS].mean(1); dctrl = dLog[:, CTRL].mean(1)
def pse(v): return float(v.std(ddof=1)/math.sqrt(len(v)))
diff = dnum - dctrl
# top-boosted number under baseline, and argmax next token
base_num_rank = [(tok.decode([NUMS[i]]), round(float(Lb[:, NUMS[i]].mean()), 2)) for i in np.argsort(-Lb[:, NUMS].mean(0))[:6]]
out = {
 'n_prompts': len(prompts),
 'head_attends_day_number_frac': round(frac_attends_num, 3),
 'baseline_pred_is_number_frac': round(float(np.isin(Lb.argmax(1), NUMS).mean()), 3),
 'top_baseline_number_logits': base_num_rank,
 'dLogit_numbers_mean': round(float(dnum.mean()), 4), 'dLogit_numbers_se': round(pse(dnum), 4),
 'dLogit_control_mean': round(float(dctrl.mean()), 4), 'dLogit_control_se': round(pse(dctrl), 4),
 'numbers_minus_control_mean': round(float(diff.mean()), 4),
 'numbers_minus_control_z': round(float(diff.mean()/(pse(diff)+1e-12)), 2),
}
print(json.dumps(out, indent=2))
json.dump(out, open('/workspace/tensor_language/basis_aligned/qk_mdl/qk_unsup_verify2_datectrl.json', 'w'), indent=2)
print('DATECTRL DONE')
