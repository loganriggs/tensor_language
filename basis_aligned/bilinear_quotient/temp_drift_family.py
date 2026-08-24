# temp_drift_family: §1201's inverted-U on swiglu18.
# Registered: (a) inverted-U replicates (retention T0.8 > T1.0 and T0.8 > T0.6);
# (b) greedy collapses (rep4 >= 0.3 and retention <= floor + 0.05);
# (c) peak/true in [0.6, 0.8] (the 0.70 family constant).
#
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs
import census_lib as cl
DEV = 'cuda'
m, _cfg = load_elriggs('swiglu18'); m = m.to(DEV).eval()

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'temp_drift_family_results.json'
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
        sc = torch.einsum('bqhd,bkhd->bhqk', q.float(), k.float()) / (128 ** 0.5)
        sc = sc.masked_fill(~mask, float('-inf'))
        pat = F.softmax(sc, dim=-1)
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
           'pred_a_inverted_U': bool(R['T0.8'] > R['T1.0'] and R['T0.8'] > R['T0.6']),
           'pred_b_greedy_collapse': bool(REP['greedy'] >= 0.3 and R['greedy'] <= R['floor'] + 0.05),
           'pred_c_peak_ratio': bool(0.6 <= R['T0.8'] / max(R['true'], 1e-6) <= 0.8),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"retention {R}")
    print(f"rep4 {REP}")
    print(f"pred_a U {out['pred_a_inverted_U']} | pred_b greedy-collapse {out['pred_b_greedy_collapse']} | pred_c peak-ratio {out['pred_c_peak_ratio']}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
