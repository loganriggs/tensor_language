# temp_drift: is the family-constant sampling-drift ratio (~0.70, §1199) temperature-
# controlled? Centered retention + rep4 across T in {0.6, 0.8, 1.0} plus greedy (T->0),
# full-context generation only, vs true-text ceiling.
# Registered: (a) retention rises monotonically as T drops 1.0 -> 0.8 -> 0.6;
# (b) T=0.6 recovers >= half the gap to true (retention >= base08 + 0.5*(true - base08));
# (c) greedy DISSOCIATES the instruments: retention >= T0.6 retention while rep4 >= 0.2
# (repetition blow-up — §1135's swiglu loop phenotype appearing under argmax decoding).
#
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'temp_drift_results.json'
NPROMPT = 12; PLEN = 64; GLEN = 128; TOPK = 50; CAPL = 10
are = sys.modules[type(m.transformer.h[0].attn).__module__].apply_rotary_emb


@torch.no_grad()
def forward_masked(idx, win):
    Tn = idx.shape[1]
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    ar = torch.arange(Tn, device=DEV)
    full = torch.tril(torch.ones(Tn, Tn, device=DEV, dtype=torch.bool))
    mask = full if win is None else (full & (((ar[:, None] - ar[None, :]) < win) | (ar[None, :] == 0)))
    B = idx.shape[0]
    for L, blk in enumerate(m.transformer.h):
        at = blk.attn
        xm = blk.lambdas[0] * x + blk.lambdas[1] * x0
        xin = F.rms_norm(xm, (D,))
        cos, sin = at.rotary(at.c_q(xin).view(B, Tn, 9, 128))
        q = are(F.rms_norm(at.c_q(xin).view(B, Tn, 9, 128), (128,)), cos, sin)
        k = are(F.rms_norm(at.c_k(xin).view(B, Tn, 9, 128), (128,)), cos, sin)
        q2 = are(F.rms_norm(at.c_q2(xin).view(B, Tn, 9, 128), (128,)), cos, sin)
        k2 = are(F.rms_norm(at.c_k2(xin).view(B, Tn, 9, 128), (128,)), cos, sin)
        pat = (torch.einsum('bqhd,bkhd->bhqk', q.float(), k.float()) / 128.0) \
            * (torch.einsum('bqhd,bkhd->bhqk', q2.float(), k2.float()) / 128.0)
        pat = pat.masked_fill(~mask, 0.0)
        v = at.c_v(xin).view(B, Tn, 9, 128)
        if v1 is None:
            v1 = v
        vv = (1 - at.lamb) * v + at.lamb * v1.view_as(v)
        y = torch.einsum('bhqk,bkhd->bqhd', pat.to(vv.dtype), vv).reshape(B, Tn, D)
        x = xm + at.c_proj(y)
        x = x + blk.mlp(F.rms_norm(x, (D,)))
    return 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)


@torch.no_grad()
def generate(prompts, temp, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    idx = prompts.clone()
    for _ in range(GLEN):
        lg = forward_masked(idx, None)[:, -1].float()
        if temp is None:
            nxt = lg.argmax(-1, keepdim=True)
        else:
            lg = lg / temp
            v, _ = torch.topk(lg, TOPK)
            lg[lg < v[:, -1:]] = -float('inf')
            nxt = torch.multinomial(F.softmax(lg, -1), 1, generator=g)
        idx = torch.cat([idx, nxt], 1)
    return idx


def rep4_of(gen_tok):
    B = gen_tok.shape[0]; rep = 0.0
    for b in range(B):
        s = gen_tok[b].tolist()
        g4 = [tuple(s[i:i + 4]) for i in range(len(s) - 3)]
        rep += (len(g4) - len(set(g4))) / max(len(g4), 1)
    return round(rep / B, 4)


@torch.no_grad()
def resid10(idx):
    """Residual after block CAPL under the FULL base model."""
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for L, blk in enumerate(m.transformer.h):
        x, v1 = blk(x, v1, x0)
        if L == CAPL:
            return x.detach().float()
    return None


def pools(full_seqs):
    r = resid10(full_seqs)
    return r[:, :PLEN].mean(1), r[:, -64:].mean(1)


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    rows = cl.fineweb_rows(NPROMPT)
    prompts = rows[:, :PLEN].to(DEV).contiguous()
    true_full = rows[:, :PLEN + GLEN].to(DEV).contiguous()

    PV = {}; REP = {}
    PV['true'] = pools(true_full); REP['true'] = rep4_of(true_full[:, PLEN:])
    gens = {}
    for name, temp in (('T1.0', 1.0), ('T0.8', 0.8), ('T0.6', 0.6), ('greedy', None)):
        gens[name] = generate(prompts, temp, seed=7)
        PV[name] = pools(gens[name]); REP[name] = rep4_of(gens[name][:, PLEN:])
    perm = torch.roll(torch.arange(NPROMPT), 1)
    mixed = torch.cat([prompts, gens['T0.8'][perm][:, PLEN:]], 1)
    PV['floor'] = pools(mixed)
    # center by the grand mean of ALL pooled vectors (constant-baseline removal)
    allv = torch.cat([torch.cat(v, 0) for v in PV.values()], 0)
    mu = allv.mean(0, keepdim=True)
    res = {}
    for k, (p, c) in PV.items():
        res[k] = F.cosine_similarity(p - mu, c - mu, dim=-1)
        print(f"{k:>8}: centered retention {float(res[k].mean()):.4f}", flush=True)
    R = {k: round(float(v.mean()), 4) for k, v in res.items()}
    gap = R['true'] - R['T0.8']
    out = {'n_prompts': NPROMPT, 'cap_layer': CAPL, 'retention': R, 'rep4': REP,
           'pred_a_monotone_in_T': bool(R['T0.6'] > R['T0.8'] > R['T1.0']),
           'pred_b_T06_recovers_half': bool(R['T0.6'] >= R['T0.8'] + 0.5 * gap),
           'pred_c_greedy_dissociates': bool(R['greedy'] >= R['T0.6'] - 0.02 and REP['greedy'] >= 0.2),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"retention {R}")
    print(f"rep4 {REP}")
    print(f"pred_a monotone {out['pred_a_monotone_in_T']} | pred_b T0.6 half {out['pred_b_T06_recovers_half']} | pred_c greedy dissociates {out['pred_c_greedy_dissociates']}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
