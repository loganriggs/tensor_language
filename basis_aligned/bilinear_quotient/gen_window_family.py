# gen_window_family: §1198's centered-retention generation test on swiglu18 (family twin).
# Softmax attention -> masking = -inf before softmax. Registered: (a) instrument sees
# (W16 <= base_gen − 0.15, floor low); (b) W128 within 0.07 of base_gen; (c) ordering
# floor < W16 < W64 <= W128 + eps, base_gen <= true + 0.02.
#
# gen_window3: gen_window2's retention instrument SATURATED (floor 0.973 vs ceiling 0.984)
# because pooled residuals are dominated by the stream's constant baseline (§688-691/§1089 —
# the program's own finding, not applied; disclosed). Fix: CENTER pooled vectors by the grand
# mean of all pools in the run before the cosine — the §1055 deviation move at pool level.
# Registered (re-registered from §1196-97):
#   pred_a INSTRUMENT SEES: centered floor <= 0.35 AND retention(W16) <= retention(base_gen) − 0.15.
#   pred_b W128 CLEAN: |W128 − base_gen| <= 0.07 (only claimable if pred_a holds).
#   pred_c ORDERED: floor < W16 < W64 <= W128 + eps and base_gen <= true + 0.02.
#
# gen_window2: the RIGHT instrument for window-generation damage — TOPIC RETENTION.
#
# gen_window's §1135 metrics failed their own positive control (W16, teacher-forced cost
# 0.59 nats, generates NORMAL per-token statistics — pred_c FALSE, so nothing was certified).
# Reading the texts shows why: W16's damage is TOPIC-HOPPING (commercial → president →
# acupuncture in 400 chars) — invisible to cw-rate/rep4. Instrument here: pooled mid-stream
# residual similarity. For each continuation, run the BASE model over prompt+continuation,
# capture the residual after block 10, mean-pool (i) the prompt positions and (ii) the LAST
# 64 continuation positions, and take the cosine — topic retention over ~128 generated tokens.
# Calibrators: CEILING = true FineWeb continuations of the same prompts; FLOOR = shuffled
# pairing (prompt k vs continuation k+1's text).
#
# Conditions: true (ceiling), base-gen, W128, W64, W16 (positive control), floor.
# Same generations protocol as gen_window (seed 7, temp 0.8, top-k 50, 12 prompts × 128).
#
# Registered predictions:
#   pred_a INSTRUMENT SEES THE DAMAGE: retention(W16) <= retention(base_gen) − 0.15 — the
#          positive control this thread now requires before any claim (§1148 law).
#   pred_b W128 CLEAN: |retention(W128) − retention(base_gen)| <= 0.05.
#   pred_c ORDERED: floor < W16 < W64 <= W128, and base_gen <= true (sampled text drifts
#          more than real text even unwindowed).
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs
import census_lib as cl
DEV = 'cuda'
m, _cfg = load_elriggs('swiglu18'); m = m.to(DEV).eval()

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'gen_window_family_results.json'
NPROMPT = 12; PLEN = 64; GLEN = 128; TEMP = 0.8; TOPK = 50; CAPL = 10
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
def generate(prompts, win, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    idx = prompts.clone()
    for _ in range(GLEN):
        lg = forward_masked(idx, win)[:, -1].float() / TEMP
        v, _ = torch.topk(lg, TOPK)
        lg[lg < v[:, -1:]] = -float('inf')
        nxt = torch.multinomial(F.softmax(lg, -1), 1, generator=g)
        idx = torch.cat([idx, nxt], 1)
    return idx


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

    PV = {}
    PV['true'] = pools(true_full)
    gens = {}
    for name, win in (('base_gen', None), ('W128', 128), ('W64', 64), ('W16', 16)):
        gens[name] = generate(prompts, win, seed=7)
        PV[name] = pools(gens[name])
    perm = torch.roll(torch.arange(NPROMPT), 1)
    mixed = torch.cat([prompts, gens['base_gen'][perm][:, PLEN:]], 1)
    PV['floor'] = pools(mixed)
    # center by the grand mean of ALL pooled vectors (constant-baseline removal)
    allv = torch.cat([torch.cat(v, 0) for v in PV.values()], 0)
    mu = allv.mean(0, keepdim=True)
    res = {}
    for k, (p, c) in PV.items():
        res[k] = F.cosine_similarity(p - mu, c - mu, dim=-1)
        print(f"{k:>8}: centered retention {float(res[k].mean()):.4f}", flush=True)
    R = {k: round(float(v.mean()), 4) for k, v in res.items()}
    out = {'n_prompts': NPROMPT, 'cap_layer': CAPL, 'retention': R,
           'pred_a_instrument_sees': bool(R['floor'] <= 0.35 and R['W16'] <= R['base_gen'] - 0.15),
           'pred_b_W128_clean': bool(abs(R['W128'] - R['base_gen']) <= 0.07),
           'pred_c_ordered': bool(R['floor'] < R['W16'] < R['W64'] <= R['W128'] + 1e-6 and
                                  R['base_gen'] <= R['true'] + 0.02),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"retention {R}")
    print(f"pred_a sees {out['pred_a_instrument_sees']} | pred_b W128 clean {out['pred_b_W128_clean']} | pred_c ordered {out['pred_c_ordered']}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
