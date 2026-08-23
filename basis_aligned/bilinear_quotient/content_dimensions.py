"""Stop CONFIRMING the deep-middle content is high-rank; start NAMING what it tracks. §1049-1054 established the middle
carries one shared, high-rank, load-bearing content object. This asks WHAT its principal directions are: take the
deep-middle content reference subspace (top PCA of pooled L8-12 MLP-input content deviation, §1052/1053), and for the
top directions, decode the CONTEXT SNIPPETS at the most positive / most negative positions. If the top axes read as
interpretable topic/register distinctions, we have moved from 'the frontier is high-rank content' to 'here is what that
content represents' -- understanding the contextual computation itself.

REGISTERED PREDICTIONS:
  (0) SANITY: each PC's max+ and max- snippets are internally coherent (not random) -- a shuffled-projection control
      would give incoherent snippets.
  (a) INTERPRETABLE TOPIC AXES: the top content PCs correspond to broad, human-readable topic/register distinctions
      (e.g. technical/code vs narrative/prose, or specific subject clusters), consistent with the 'high-rank topic
      manifold' claim (§930) -- and being high-rank means MANY such axes, each a real distinction;
  (b) report, per top PC, the current token + preceding context at the top max-activating positions (both signs)."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl
import torch.nn.functional as F
import tiktoken

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'content_dimensions_results.json'
NEVAL = 300; SEQ = 256; REF_LAYERS = [8, 10, 12]; K = 64; NPC = 10; NTOP = 6; CTX = 14
CAP = {}


def forward_logits(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
    return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


def capture(idx, layers):
    hs = []
    for L in layers:
        mlp = m.transformer.h[L].mlp
        def mk(L):
            def h(mo, i_, o_): CAP[L].append((i_[0] if isinstance(i_, tuple) else i_).detach().float())  # (B,T,D)
            return h
        hs.append(mlp.register_forward_hook(mk(L)))
    forward_logits(idx)
    for h in hs: h.remove()


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NEVAL)
    enc = tiktoken.get_encoding('gpt2'); V = int(m.lm_head.weight.shape[0])
    blocks = rows[:, :SEQ].contiguous(); T = blocks.shape[1] - 1
    for L in REF_LAYERS: CAP[L] = []
    idsL = []
    for i in range(0, blocks.shape[0], 8):
        idx = blocks[i:i+8].to(DEV)[:, :-1].contiguous(); idsL.append(idx.cpu()); capture(idx, REF_LAYERS)
    ids = torch.cat(idsL, 0)                                     # (Nseq,T) cpu
    Nseq = ids.shape[0]
    # per-layer content deviation (input minus per-token mean), averaged across ref layers for a shared content signal
    flatids = ids.reshape(-1).to(DEV)
    devsum = torch.zeros(Nseq*T, D, device=DEV)
    for L in REF_LAYERS:
        X = torch.cat(CAP[L], 0).reshape(-1, D)                  # (Nseq*T,D)
        xbar = torch.zeros(V, D, device=DEV); cnts = torch.zeros(V, device=DEV)
        xbar.index_add_(0, flatids, X); cnts.index_add_(0, flatids, torch.ones_like(flatids, dtype=torch.float))
        xbar = xbar / cnts.clamp_min(1).unsqueeze(1)
        devsum += (X - xbar[flatids]); del X
        CAP[L] = []
    dev = devsum / len(REF_LAYERS)                              # (Nseq*T,D) mean content deviation
    devc = dev - dev.mean(0)
    _, _, Vt = torch.linalg.svd(devc, full_matrices=False); U = Vt[:K].T.contiguous()   # (D,K)
    proj = devc @ U                                             # (Nseq*T,K)
    ids_np = ids.numpy()

    def snippet(flatidx):
        s, p = divmod(int(flatidx), T)
        lo = max(0, p - CTX)
        toks = ids_np[s, lo:p+1].tolist()
        try: txt = enc.decode(toks)
        except Exception: txt = '<decode-err>'
        cur = enc.decode([ids_np[s, p]])
        return {'cur_token': cur, 'context': txt.replace('\n', ' ')}

    out = {'K': K, 'ref_layers': REF_LAYERS, 'n_positions': int(proj.shape[0]), 'directions': []}
    for k in range(NPC):
        pk = proj[:, k]
        top_pos = torch.topk(pk, NTOP).indices.tolist()
        top_neg = torch.topk(-pk, NTOP).indices.tolist()
        out['directions'].append({
            'pc': k,
            'pos_snippets': [snippet(i) for i in top_pos],
            'neg_snippets': [snippet(i) for i in top_neg],
        })
    # variance explained per PC
    S = torch.linalg.svdvals(devc)[:NPC]
    tot = float((torch.linalg.svdvals(devc)**2).sum())
    for k in range(NPC):
        out['directions'][k]['var_frac'] = round(float(S[k]**2)/tot, 4)
    out['top10_var_cumfrac'] = round(float((S**2).sum())/tot, 4)
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    for k in range(min(NPC, 6)):
        d = out['directions'][k]
        print(f"PC{k} (var {d['var_frac']}): +[{', '.join(repr(s['cur_token']) for s in d['pos_snippets'][:4])}] | -[{', '.join(repr(s['cur_token']) for s in d['neg_snippets'][:4])}]", flush=True)
    print(f"top-10 PCs explain {out['top10_var_cumfrac']} of content variance | wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
