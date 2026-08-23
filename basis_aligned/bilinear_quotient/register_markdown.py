"""OOD register #4 (registered in §1110): MARKDOWN — the repo's .md files, a MIXED register (prose sentences
inside structural markup). Three-point test of §1110's two reliance axes: markdown should INTERPOLATE between
prose and code/json on predictability-driven metrics, and the grammar/content generality RATIO should again be
~2 (candidate law: 2.05 prose<->code, 2.06 prose<->markdown).

REGISTERED PREDICTIONS:
  (0) SANITY: base CE between prose (3.33) and json (2.75) or above prose (mixed register, some OOD cost); null ~0.056.
  (a) RATIO LAW: prose<->markdown front/deep overlap ratio in [1.7, 2.4] (third register pair at ~2x);
  (b) INTERPOLATION: markdown's value-residual cost-frac between prose's 1.01 and json's 1.89; content-band
      frac between json's 0.34 and prose's 0.69 (mixed register, intermediate reliance re-weighting);
  (c) if markdown breaks the ratio or lands outside the brackets, the two-axis story needs a third axis
      (report which metric broke)."""
import json, time, sys, glob, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl
import tiktoken

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'register_markdown_results.json'
NSEQ = 150; SEQ = 256; K = 64
H = m.transformer.h
CONTENT_BAND = list(range(5, 15)); GRAMMAR_BAND = [0, 1]
FRONT = [0, 1, 2]; DEEP = list(range(6, 15))
SUB = {'mlp_mean': {}, 'active': set()}
CAP = {}


def fwd(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in H: x, v1 = blk(x, v1, x0)
    return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


def load_md_blocks(nseq, seq):
    enc = tiktoken.get_encoding('gpt2'); toks = []
    for fp in sorted(glob.glob('/workspace/tensor_language/**/*.md', recursive=True)):
        try:
            txt = open(fp).read()
            if len(txt) < 200: continue
            toks.extend(enc.encode(txt))
        except Exception:
            continue
        if len(toks) >= nseq*seq: break
    return torch.tensor(toks[:nseq*seq], dtype=torch.long).view(nseq, seq)


# ---- (1) representation: per-band deviation subspaces ----
@torch.no_grad()
def dev_subspaces(blocks, layers):
    for L in layers: CAP[L] = []
    hs = []
    for L in layers:
        def mk(L):
            def h(mo, i_, o_): CAP[L].append((i_[0] if isinstance(i_, tuple) else i_).detach().float().reshape(-1, D))
            return h
        hs.append(H[L].mlp.register_forward_hook(mk(L)))
    idsL = []
    for i in range(0, blocks.shape[0], 8):
        idx = blocks[i:i+8].to(DEV)[:, :-1].contiguous(); idsL.append(idx.reshape(-1)); fwd(idx)
    for h in hs: h.remove()
    tok = torch.cat(idsL, 0); V = int(m.lm_head.weight.shape[0]); U = {}
    for L in layers:
        X = torch.cat(CAP[L], 0); CAP[L] = []
        xb = torch.zeros(V, D, device=DEV); cn = torch.zeros(V, device=DEV)
        xb.index_add_(0, tok, X); cn.index_add_(0, tok, torch.ones_like(tok, dtype=torch.float))
        dev = X - (xb/cn.clamp_min(1).unsqueeze(1))[tok]; dev = dev - dev.mean(0)
        _, _, Vt = torch.linalg.svd(dev, full_matrices=False); U[L] = Vt[:K].T.contiguous()
        del X, dev
    return U


def ov(A, B): return round(float((A.T @ B).pow(2).sum()/K), 4)


# ---- (2) reliance ----
def mlp_mean_hook(L):
    def h(mo, i_, o_):
        if L not in SUB['active']: return None
        return SUB['mlp_mean'][L].view(1, 1, D).expand_as(o_).to(o_.dtype)
    return h


@torch.no_grad()
def compute_mlp_means(blocks, layers):
    caps = {L: [] for L in layers}; hs = []
    for L in layers:
        def mk(L):
            def h(mo, i_, o_): caps[L].append(o_.detach().float().reshape(-1, D))
            return h
        hs.append(H[L].mlp.register_forward_hook(mk(L)))
    SUB['active'] = set()
    for i in range(0, blocks.shape[0], 8): fwd(blocks[i:i+8].to(DEV)[:, :-1].contiguous())
    for h in hs: h.remove()
    for L in layers: SUB['mlp_mean'][L] = torch.cat(caps[L], 0).mean(0)


@torch.no_grad()
def ce(blocks):
    tot = 0.0; n = 0
    for i in range(0, blocks.shape[0], 8):
        bb = blocks[i:i+8].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].reshape(-1)
        lp = F.log_softmax(fwd(idx).float(), -1)
        tot += float(-lp.reshape(-1, lp.shape[-1])[torch.arange(tgt.shape[0], device=DEV), tgt].sum()); n += tgt.shape[0]
    return tot/n


@torch.no_grad()
def run_reliance(blocks, hooks_registered):
    orig_lam = [blk.lambdas.data.clone() for blk in H]; orig_vl = [blk.attn.lamb.data.clone() for blk in H]
    def restore():
        for blk, l, vl in zip(H, orig_lam, orig_vl): blk.lambdas.data.copy_(l); blk.attn.lamb.data.copy_(vl)
    compute_mlp_means(blocks, CONTENT_BAND + GRAMMAR_BAND)
    SUB['active'] = set(); base = ce(blocks)
    res = {'base_ce': round(base, 4)}
    SUB['active'] = set(CONTENT_BAND); res['content_band'] = round(ce(blocks)-base, 4); SUB['active'] = set()
    SUB['active'] = set(GRAMMAR_BAND); res['grammar_band'] = round(ce(blocks)-base, 4); SUB['active'] = set()
    for blk in H: blk.attn.lamb.data.fill_(0.0)
    res['value_residual'] = round(ce(blocks)-base, 4); restore()
    for blk in H: blk.lambdas.data[1] = 0.0
    res['x0_reinject'] = round(ce(blocks)-base, 4); restore()
    for k2 in ('content_band', 'grammar_band', 'value_residual', 'x0_reinject'):
        res[k2+'_frac'] = round(res[k2]/max(base, 1e-6), 4)
    return res


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    prose = cl.fineweb_rows(NSEQ*2)[:, :SEQ].contiguous()
    prA, prB = prose[:NSEQ], prose[NSEQ:]
    js = load_md_blocks(NSEQ, SEQ)
    print(f"markdown blocks: {js.shape}", flush=True)

    LAYERS = FRONT + [4, 5] + DEEP
    Up = dev_subspaces(prA, LAYERS); Up2 = dev_subspaces(prB, LAYERS); Uj = dev_subspaces(js, LAYERS)
    g = torch.Generator(device=DEV).manual_seed(0)
    Rnd = torch.linalg.qr(torch.randn(D, K, generator=g, device=DEV))[0]
    def band(ls): return round(sum(ov(Up[L], Uj[L]) for L in ls)/len(ls), 4)
    def bandc(ls): return round(sum(ov(Up[L], Up2[L]) for L in ls)/len(ls), 4)
    rep = {'front_L0_2': band(FRONT), 'deep_L6_14': band(DEEP),
           'front_ceiling': bandc(FRONT), 'deep_ceiling': bandc(DEEP),
           'random_null': ov(Up[6], Rnd)}
    print(f"REPRESENTATION prose<->markdown: front {rep['front_L0_2']} (ceil {rep['front_ceiling']}) | deep {rep['deep_L6_14']} (ceil {rep['deep_ceiling']}) | null {rep['random_null']}", flush=True)

    hooks = [H[L].mlp.register_forward_hook(mlp_mean_hook(L)) for L in CONTENT_BAND + GRAMMAR_BAND]
    rel = {'prose': run_reliance(prA, hooks), 'markdown': run_reliance(js, hooks)}
    for h in hooks: h.remove()
    for reg in ('prose', 'markdown'):
        r = rel[reg]
        print(f"RELIANCE {reg}: base {r['base_ce']} | content-frac {r['content_band_frac']} | grammar-frac {r['grammar_band_frac']} | vresid-frac {r['value_residual_frac']} | x0-frac {r['x0_reinject_frac']}", flush=True)

    out = {'representation': rep, 'reliance': rel,
           'code_reference_1081': {'content_frac': 0.28, 'grammar_frac': 0.65, 'vresid_frac': 1.10, 'x0_frac': 0.30}}
    j = rel['markdown']
    ratio = rep['front_L0_2']/max(rep['deep_L6_14'], 1e-4)
    out['front_deep_ratio'] = round(ratio, 2)
    out['pred_a_ratio_law'] = bool(1.7 <= ratio <= 2.4)
    out['pred_b_interpolation'] = bool(1.01 <= j['value_residual_frac'] <= 1.89
                                       and 0.34 <= j['content_band_frac'] <= 0.69)
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"ratio {out['front_deep_ratio']} | pred_a ratio-law {out['pred_a_ratio_law']} | pred_b interpolation {out['pred_b_interpolation']} | wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
