"""What IS block-0's value v1 that the value-residual broadcasts to every layer (§1075, and feeds content)? Two tests:
(A) PREDICTABILITY: is block-0's value ~a static function of the token (word content), or context-dependent? R^2
predicting block-0 c_v output from (i) the token embedding [linear], (ii) a per-token table [pure identity], (iii) the
block-0 input residual [context]. (B) FUNCTIONAL: replace the broadcast v1 with its static per-token table (each token's
average block-0 value) for all downstream blocks and measure CE cost -- if small, the value-residual broadcasts ~static
word content; if large, it broadcasts something context-dependent.

REGISTERED PREDICTIONS:
  (0) SANITY: no-override CE == baseline; context R^2 >= token-table R^2 (context is a superset).
  (a) v1 IS ~STATIC WORD CONTENT: block-0 value is largely predictable from the token identity/embedding (token-table
      R^2 high, > ~0.7), and replacing broadcast v1 with the static per-token table costs little CE (< ~0.3 nats) -> the
      value-residual is a mechanism for broadcasting each word's (near-static) content to all layers, feeding the
      content bag;
  (b) OR it is context-dependent: token-table R^2 modest and the static-table override costs a lot. Report R^2 (emb/
      table/context) + static-override CE cost + shuffled null."""
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'blockzero_value_results.json'
NEVAL = 200; SEQ = 256; RIDGE = 1e2
import census_lib as cl
H = m.transformer.h
CAPV = []; CAPIN = []


def fwd_plain(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in H: x, v1 = blk(x, v1, x0)
    return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


@torch.no_grad()
def fwd_override(idx, table):
    """run with block-0's broadcast v1 replaced by a static per-token table for downstream blocks."""
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    B, T = idx.shape
    for li, blk in enumerate(H):
        x, v1 = blk(x, v1, x0)
        if li == 0 and table is not None:
            v1 = table[idx.reshape(-1)].view(B, T, v1.shape[2], v1.shape[3]).to(v1.dtype)  # static per-token value
    return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


@torch.no_grad()
def ce(blocks, table=None):
    tot = 0.0; n = 0
    for i in range(0, blocks.shape[0], 8):
        bb = blocks[i:i+8].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        logits = fwd_override(idx, table) if table is not None else fwd_plain(idx)
        lp = F.log_softmax(logits.float(), -1); tf = tgt.reshape(-1)
        tot += float(-lp.reshape(-1, lp.shape[-1])[torch.arange(tf.shape[0], device=DEV), tf].sum()); n += tf.shape[0]
    return tot / n


def r2(Xtr, Ytr, Xte, Yte):
    X1 = torch.cat([Xtr, torch.ones(Xtr.shape[0], 1, device=DEV)], 1)
    M = torch.linalg.solve(X1.T @ X1 + RIDGE*torch.eye(X1.shape[1], device=DEV), X1.T @ Ytr)
    Xt1 = torch.cat([Xte, torch.ones(Xte.shape[0], 1, device=DEV)], 1); pred = Xt1 @ M
    return float(1 - ((Yte-pred)**2).sum()/((Yte-Yte.mean(0))**2).sum())


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NEVAL)
    blocks = rows[:, :SEQ].contiguous(); V = int(m.lm_head.weight.shape[0])
    # capture block-0 value (c_v output) and block-0 input residual
    hv = H[0].attn.c_v.register_forward_hook(lambda mo, i_, o_: CAPV.append((o_[0] if isinstance(o_, tuple) else o_).detach().float().reshape(-1, D)))
    hi = H[0].attn.register_forward_pre_hook(lambda mo, i_: CAPIN.append((i_[0] if isinstance(i_, tuple) else i_).detach().float().reshape(-1, D)))
    idsL = []; embs = []
    for i in range(0, blocks.shape[0], 8):
        idx = blocks[i:i+8].to(DEV)[:, :-1].contiguous(); idsL.append(idx.reshape(-1))
        embs.append(F.rms_norm(m.transformer.wte(idx), (D,)).float().reshape(-1, D)); fwd_plain(idx)
    hv.remove(); hi.remove()
    tok = torch.cat(idsL, 0); Vval = torch.cat(CAPV, 0); Xin = torch.cat(CAPIN, 0); Emb = torch.cat(embs, 0)
    # per-token table of block-0 value (static)
    table = torch.zeros(V, D, device=DEV); cnts = torch.zeros(V, device=DEV)
    table.index_add_(0, tok, Vval); cnts.index_add_(0, tok, torch.ones_like(tok, dtype=torch.float))
    table = table / cnts.clamp_min(1).unsqueeze(1)
    n = Vval.shape[0]; ntr = int(0.7*n)
    Ytr, Yte = Vval[:ntr], Vval[ntr:]
    out = {'n': n, 'r2': {}}
    out['r2']['from_embedding'] = round(r2(Emb[:ntr], Ytr, Emb[ntr:], Yte), 4)
    out['r2']['from_token_table'] = round(r2(table[tok][:ntr], Ytr, table[tok][ntr:], Yte), 4)  # trivially = identity-explained frac
    out['r2']['from_context_input'] = round(r2(Xin[:ntr], Ytr, Xin[ntr:], Yte), 4)
    # variance explained by token identity directly (1 - within-token var / total var)
    within = ((Vval - table[tok])**2).mean(); total = ((Vval - Vval.mean(0))**2).mean()
    out['token_identity_var_frac'] = round(float(1 - within/total), 4)
    # functional: static-table override
    base = ce(blocks); ce_static = ce(blocks, table=table)
    # shuffled-token-table null (broadcast a random-token value)
    g = torch.Generator(device=DEV).manual_seed(0); table_sh = table[torch.randperm(V, generator=g, device=DEV)]
    ce_sh = ce(blocks, table=table_sh)
    out['base_ce'] = round(base, 4); out['ce_static_v1'] = round(ce_static, 4); out['static_override_cost'] = round(ce_static-base, 4)
    out['ce_shuffled_v1'] = round(ce_sh, 4); out['shuffled_override_cost'] = round(ce_sh-base, 4)
    out['pred_a_v1_is_static_word_content'] = bool(out['token_identity_var_frac'] > 0.7 and out['static_override_cost'] < 0.3)
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"block-0 value R2: emb {out['r2']['from_embedding']} | token-table {out['r2']['from_token_table']} | context {out['r2']['from_context_input']}", flush=True)
    print(f"token-identity var frac {out['token_identity_var_frac']} | static-override cost {out['static_override_cost']} (shuffled {out['shuffled_override_cost']})", flush=True)
    print(f"pred_a v1-is-static-word-content: {out['pred_a_v1_is_static_word_content']} | wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
