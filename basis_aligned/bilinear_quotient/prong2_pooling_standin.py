"""PRONG 2: the content-pooling components (attn3/4/5, the L3-5 gatherers; §998/§1007) scored NEGATIVE in the
per-component benchmark (§1035: attn5 = -4.84) because I scored a POOLING operation with a PER-POSITION stand-in
(token/topic/prev). That is the wrong stand-in for a bag-of-words pool. Score them with their ACTUAL mechanism -- a
BAG-OF-WORDS stand-in (a ridge map from the causal running-mean-of-embeddings to the component output) -- and see if
the "anti-understood" -4.84 turns POSITIVE. Compare bag stand-in vs per-token-table stand-in, both held-out, via
loss-recovery when that ONE component is replaced.

For each L in {3,4,5}: recovery = (CE[mean-ablate attn_L] - CE[stand-in attn_L]) / (CE[mean-ablate attn_L] - CE_full).

REGISTERED PREDICTIONS:
  (0) SANITY: per-token-table stand-in reproduces the §1035-style negative/near-zero for attn5 (wrong stand-in for a
      pool); mean-ablate = recovery 0, full = 1.
  (a) POOLING STAND-IN UNDERSTANDS THE POOLER: the BAG-OF-WORDS stand-in gives POSITIVE, substantial loss-recovery
      for attn5 (and attn3/4), FAR above the per-token-table stand-in -> the content-gatherers ARE understood, as
      bag-of-words poolers; the -4.84 was a wrong-stand-in artifact, not a real limit;
  (b) report bag vs token recovery for attn3/4/5."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl
import torch.nn.functional as F

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'prong2_pooling_standin_results.json'
NEVAL = 220; SEQ = 256; LAYERS = [3, 4, 5]; RIDGE = 1e3
SUB = {'L': None, 'kind': None, 'Mbag': {}, 'tok': {}, 'bagfeat': None, 'tokids': None}


def bagfeat(idx):
    E = F.rms_norm(m.transformer.wte(idx), (D,)).float(); T = E.shape[1]
    num = torch.cumsum(E, 1); den = torch.arange(1, T+1, device=DEV, dtype=E.dtype).view(1, T, 1)
    return num/den  # (B,T,D) causal running mean of embeddings


def forward_logits(idx):
    SUB['bagfeat'] = bagfeat(idx); SUB['tokids'] = idx
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
    return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


def sub_hook_factory(L):
    def h(mo, i_, o_):
        if SUB['L'] != L or SUB['kind'] is None: return None
        y = o_[0] if isinstance(o_, tuple) else o_; B, T, _ = y.shape
        if SUB['kind'] == 'mean':
            ny = SUB['tok'][L]['gmean'].view(1, 1, D).expand(B, T, D)
        elif SUB['kind'] == 'bag':
            bf = SUB['bagfeat'].reshape(-1, D)
            ny = (torch.cat([bf, torch.ones(bf.shape[0], 1, device=DEV)], 1) @ SUB['Mbag'][L]).reshape(B, T, D)
        else:  # token table
            ny = SUB['tok'][L]['table'][SUB['tokids'].reshape(-1)].reshape(B, T, D)
        return (ny.to(y.dtype),) + tuple(o_[1:]) if isinstance(o_, tuple) else ny.to(y.dtype)
    return h


@torch.no_grad()
def capture(blocks, layers):
    caps = {L: [] for L in layers}; bags = []; toks = []; hs = []
    for L in layers:
        a = m.transformer.h[L].attn
        def mk(L):
            def h(mo, i_, o_):
                y = o_[0] if isinstance(o_, tuple) else o_
                caps[L].append(y.detach().float().reshape(-1, D).cpu())
            return h
        hs.append(a.register_forward_hook(mk(L)))
    SUB['L'] = None
    for i in range(0, blocks.shape[0], 4):
        idx = blocks[i:i+4].to(DEV)[:, :-1].contiguous()
        bags.append(bagfeat(idx).reshape(-1, D).cpu()); toks.append(idx.reshape(-1).cpu())
        forward_logits(idx)
    for h in hs: h.remove()
    return {L: torch.cat(caps[L], 0) for L in layers}, torch.cat(bags, 0), torch.cat(toks, 0)


@torch.no_grad()
def ce(blocks):
    tot = 0.0; n = 0
    for i in range(0, blocks.shape[0], 8):
        bb = blocks[i:i+8].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        lp = F.log_softmax(forward_logits(idx).float(), -1); tf = tgt.reshape(-1)
        tot += float(-lp.reshape(-1, lp.shape[-1])[torch.arange(tf.shape[0], device=DEV), tf].sum()); n += tf.shape[0]
    return tot / n


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NEVAL)
    nb = rows.shape[0]; ntr = int(0.6*nb); tr = rows[:ntr, :SEQ].contiguous(); te = rows[ntr:, :SEQ].contiguous()
    Y, bag, tok = capture(tr, LAYERS)
    V = int(m.lm_head.weight.shape[0])
    bag = bag.to(DEV); tok = tok.to(DEV)
    bag1 = torch.cat([bag, torch.ones(bag.shape[0], 1, device=DEV)], 1)
    A = bag1.T @ bag1 + RIDGE*torch.eye(D+1, device=DEV)
    for L in LAYERS:
        Yl = Y[L].to(DEV)
        SUB['Mbag'][L] = torch.linalg.solve(A, bag1.T @ Yl)
        gmean = Yl.mean(0)
        table = gmean.view(1, D).repeat(V, 1)
        sums = torch.zeros(V, D, device=DEV); cnts = torch.zeros(V, device=DEV)
        sums.index_add_(0, tok, Yl); cnts.index_add_(0, tok, torch.ones_like(tok, dtype=torch.float))
        seen = cnts >= 5; table[seen] = sums[seen]/cnts[seen].unsqueeze(1)
        SUB['tok'][L] = {'gmean': gmean, 'table': table.half()}
        del Yl
    hooks = [m.transformer.h[L].attn.register_forward_hook(sub_hook_factory(L)) for L in LAYERS]
    SUB['L'] = None; ce_full = ce(te); print(f"ce_full {ce_full:.4f}", flush=True)
    out = {'ce_full': round(ce_full, 4), 'layers': {}}
    for L in LAYERS:
        SUB['L'] = L
        SUB['kind'] = 'mean'; ce_ma = ce(te); denom = max(ce_ma - ce_full, 1e-6)
        SUB['kind'] = 'bag'; ce_bag = ce(te)
        SUB['kind'] = 'token'; ce_tok = ce(te)
        SUB['L'] = None
        rb = round(float((ce_ma - ce_bag)/denom), 3); rt = round(float((ce_ma - ce_tok)/denom), 3)
        out['layers'][str(L)] = {'meanabl_cost': round(ce_ma-ce_full,3), 'recovery_bag': rb, 'recovery_token': rt}
        print(f"attn{L}: meanabl {ce_ma-ce_full:.3f} | recovery BAG {rb} | recovery TOKEN {rt}", flush=True)
    for h in hooks: h.remove()
    a5 = out['layers']['5']
    out['pred_a_pooling_understands'] = bool(a5['recovery_bag'] > 0.3 and a5['recovery_bag'] > a5['recovery_token'] + 0.5)
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"attn5: BAG recovery {a5['recovery_bag']} vs TOKEN recovery {a5['recovery_token']} (§1035 per-position was -4.84)", flush=True)
    print(f"pred_a pooling stand-in understands the pooler: {out['pred_a_pooling_understands']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
