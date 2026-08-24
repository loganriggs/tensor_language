# gen_window: does the locality certificate survive FREE-RUNNING generation?
#
# All window certificates so far are teacher-forced (§1166-95). Autoregressive generation
# compounds errors: a 0.014-0.08-nat CE gap could still snowball into a phenotype shift over
# 128 generated tokens — or not. This runs the READ-MASKED model (attention masked beyond W,
# position 0 visible — the §1186 harness, O(1) overhead per step) as a generator and scores
# the §1135 phenotype metrics against the base model's own generations.
#
# Setup: 12 FineWeb prompts (64 tokens), generate 128 tokens, temperature 0.8 top-k 50,
# same seed per condition. Conditions: base (full attention), W128, W64, W16 (expected to
# degrade — the §1135-38 metrics should SEE it if they can). Metrics: content-word rate
# (top-128-frequent-token split, §1151 convention), rep-4gram rate, distinct-2. Texts saved.
#
# Registered predictions:
#   pred_a W128 IS PHENOTYPE-CLEAN: |cw-rate − base| <= 0.05 and |rep4 − base| <= 0.03.
#   pred_b W64 MILD: cw-rate >= 0.40 and rep4 <= base + 0.10 (no soup, no loop collapse).
#   pred_c W16 VISIBLY DEGRADES on at least one metric (|Δcw| > 0.05 or Δrep4 > 0.10) —
#          the instrument's positive control: if W16 looks clean too, the metrics are blind
#          and preds a-b certify nothing (§1148 lesson, built in).
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'gen_window_results.json'
NPROMPT = 12; PLEN = 64; GLEN = 128; TEMP = 0.8; TOPK = 50


@torch.no_grad()
def forward_masked(idx, win):
    Tn = idx.shape[1]
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    ar = torch.arange(Tn, device=DEV)
    full = torch.tril(torch.ones(Tn, Tn, device=DEV, dtype=torch.bool))
    if win is None:
        mask = full
    else:
        mask = full & (((ar[:, None] - ar[None, :]) < win) | (ar[None, :] == 0))
    are = sys.modules[type(m.transformer.h[0].attn).__module__].apply_rotary_emb
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
def generate(prompts, win, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    idx = prompts.clone()
    for _ in range(GLEN):
        lg = forward_masked(idx, win)[:, -1].float() / TEMP
        v, _ = torch.topk(lg, TOPK)
        lg[lg < v[:, -1:]] = -float('inf')
        probs = F.softmax(lg, -1)
        nxt = torch.multinomial(probs, 1, generator=g)
        idx = torch.cat([idx, nxt], 1)
    return idx[:, PLEN:]


def metrics(gen_tok, is_func):
    B, L = gen_tok.shape
    cw = float((~is_func[gen_tok.cpu()]).float().mean())
    rep = 0.0; dist2 = 0.0
    for b in range(B):
        s = gen_tok[b].tolist()
        g4 = [tuple(s[i:i + 4]) for i in range(len(s) - 3)]
        rep += (len(g4) - len(set(g4))) / max(len(g4), 1)
        g2 = [tuple(s[i:i + 2]) for i in range(len(s) - 1)]
        dist2 += len(set(g2)) / max(len(g2), 1)
    return round(cw, 4), round(rep / B, 4), round(dist2 / B, 4)


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    rows = cl.fineweb_rows(NPROMPT)
    prompts = rows[:, :PLEN].to(DEV).contiguous()
    V = int(m.lm_head.weight.shape[0])
    flat = rows[:, :256].reshape(-1)
    cnts = torch.bincount(flat, minlength=V)
    is_func = torch.zeros(V, dtype=torch.bool)
    is_func[torch.topk(cnts, 128).indices.cpu()] = True

    import tiktoken; enc = tiktoken.get_encoding('gpt2')
    CONDS = [('base', None), ('W128', 128), ('W64', 64), ('W16', 16)]
    res = {}; texts = {}
    for name, win in CONDS:
        gen = generate(prompts, win, seed=7)
        cw, rep, d2 = metrics(gen, is_func)
        res[name] = {'cw_rate': cw, 'rep4': rep, 'distinct2': d2}
        texts[name] = [enc.decode([t for t in gen[b].tolist() if t < 50257]) for b in range(3)]
        print(f"{name:>5}: cw {cw} | rep4 {rep} | distinct2 {d2}", flush=True)
    b = res['base']
    out = {'n_prompts': NPROMPT, 'glen': GLEN, 'metrics': res, 'sample_texts': texts,
           'pred_a_W128_clean': bool(abs(res['W128']['cw_rate'] - b['cw_rate']) <= 0.05 and
                                     abs(res['W128']['rep4'] - b['rep4']) <= 0.03),
           'pred_b_W64_mild': bool(res['W64']['cw_rate'] >= 0.40 and
                                   res['W64']['rep4'] <= b['rep4'] + 0.10),
           'pred_c_W16_degrades': bool(abs(res['W16']['cw_rate'] - b['cw_rate']) > 0.05 or
                                       res['W16']['rep4'] - b['rep4'] > 0.10),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"pred_a W128 clean {out['pred_a_W128_clean']} | pred_b W64 mild {out['pred_b_W64_mild']} | pred_c W16 degrades {out['pred_c_W16_degrades']}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
