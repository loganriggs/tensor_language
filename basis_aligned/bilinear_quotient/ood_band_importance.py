"""Causal OOD companion to §1080. §1080 (subspace overlap) showed grammar is register-general, content register-specific.
Causal test: on OOD CODE vs in-distribution PROSE, how much does the model RELY on each machine? Mean-ablate (replace
output with this-data's mean) the CONTENT band (deep-middle MLPs L5-14), the GRAMMAR band (front MLPs L0-1), the value-
residual (content channel, §1075: lamb=0), and x0 re-injection (grammar channel: lambdas[1]=0); measure CE cost on
prose vs code. If content is register-specific/less useful on code, the content-side ablations should cost RELATIVELY
LESS on code than prose; grammar-side ablations should cost relatively similar or more -> on OOD code the model leans on
the register-general grammar machine.

REGISTERED PREDICTIONS:
  (0) SANITY: base CE much higher on code (OOD); restoring params recovers base.
  (a) CONTENT LEANS OUT ON CODE: content-band and value-residual ablation cost as a FRACTION of base is LOWER on code
      than prose (the model relies less on the register-specific content there);
  (b) GRAMMAR HOLDS ON CODE: grammar-band and x0 ablation cost-fraction on code is >= its prose value (grammar still
      needed for code syntax). Report raw cost and cost/base for each ablation on prose vs code."""
import json, time, sys, glob, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl
import tiktoken

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'ood_band_importance_results.json'
NSEQ = 150; SEQ = 256
H = m.transformer.h
CONTENT_BAND = list(range(5, 15)); GRAMMAR_BAND = [0, 1]
SUB = {'mlp_mean': {}, 'active': set()}


def fwd(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in H: x, v1 = blk(x, v1, x0)
    return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


def mlp_mean_hook(L):
    def h(mo, i_, o_):
        if L not in SUB['active']: return None
        o = o_[0] if isinstance(o_, tuple) else o_; B, T, _ = o.shape
        ny = SUB['mlp_mean'][L].view(1, 1, D).expand(B, T, D)
        return ny.to(o.dtype)
    return h


@torch.no_grad()
def compute_mlp_means(blocks, layers):
    caps = {L: [] for L in layers}; hs = []
    for L in layers:
        def mk(L):
            def h(mo, i_, o_): caps[L].append((o_[0] if isinstance(o_, tuple) else o_).detach().float().reshape(-1, D))
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
        bb = blocks[i:i+8].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        lp = F.log_softmax(fwd(idx).float(), -1); tf = tgt.reshape(-1)
        tot += float(-lp.reshape(-1, lp.shape[-1])[torch.arange(tf.shape[0], device=DEV), tf].sum()); n += tf.shape[0]
    return tot / n


def load_code(nseq, seq):
    enc = tiktoken.get_encoding('gpt2'); toks = []
    for fp in sorted(glob.glob('/workspace/tensor_language/**/*.py', recursive=True)):
        try: toks.extend(enc.encode(open(fp).read()))
        except Exception: continue
        if len(toks) >= nseq*seq: break
    return torch.tensor(toks[:nseq*seq], dtype=torch.long).view(nseq, seq)


@torch.no_grad()
def run_on(blocks, hooks_mlp):
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
    for k in ('content_band', 'grammar_band', 'value_residual', 'x0_reinject'):
        res[k+'_frac'] = round(res[k]/max(base, 1e-6), 4)
    return res


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    prose = cl.fineweb_rows(NSEQ)[:, :SEQ].contiguous(); code = load_code(NSEQ, SEQ).contiguous()
    hooks = [H[L].mlp.register_forward_hook(mlp_mean_hook(L)) for L in CONTENT_BAND + GRAMMAR_BAND]
    out = {'prose': run_on(prose, hooks), 'code': run_on(code, hooks)}
    for h in hooks: h.remove()
    pr, co = out['prose'], out['code']
    out['content_leans_out_on_code'] = bool(co['content_band_frac'] < pr['content_band_frac'] and co['value_residual_frac'] < pr['value_residual_frac'])
    out['grammar_holds_on_code'] = bool(co['grammar_band_frac'] >= pr['grammar_band_frac'] and co['x0_reinject_frac'] >= pr['x0_reinject_frac'])
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"PROSE base {pr['base_ce']} | content {pr['content_band']} ({pr['content_band_frac']}) grammar {pr['grammar_band']} ({pr['grammar_band_frac']}) vresid {pr['value_residual']} ({pr['value_residual_frac']}) x0 {pr['x0_reinject']} ({pr['x0_reinject_frac']})", flush=True)
    print(f"CODE  base {co['base_ce']} | content {co['content_band']} ({co['content_band_frac']}) grammar {co['grammar_band']} ({co['grammar_band_frac']}) vresid {co['value_residual']} ({co['value_residual_frac']}) x0 {co['x0_reinject']} ({co['x0_reinject_frac']})", flush=True)
    print(f"content-leans-out-on-code {out['content_leans_out_on_code']} | grammar-holds {out['grammar_holds_on_code']} | wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
