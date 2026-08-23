"""Positive confirmation that the content tracks DOCUMENT-LEVEL TOPIC. §1064 showed the content axes are not surface
register (only ~2% surface-feature variance); §1055 read them as semantic topic. Positive test: if the content encodes
what a document is ABOUT, its per-position coordinates should be STABLE WITHIN a document and DIFFER ACROSS documents.
Quantify with the fraction of each content coordinate's variance explained by document identity (ANOVA eta^2 =
between-document variance / total variance). High eta^2 => document-topic-coherent. Compare content vs a random subspace
of the residual (control) and a shuffled-document-label null.

REGISTERED PREDICTIONS:
  (0) SANITY: shuffled-document-label eta^2 ~ 0 (a few *1/n_docs).
  (a) DOCUMENT-TOPIC CONTENT: the content coordinates have HIGH document-eta^2 (topic stable within a document), clearly
      above a random-subspace-of-residual control and far above the shuffled-label null -> the content encodes
      document-level topic (positively confirming the semantic-topic reading of §1055/§1064);
  (b) report mean document-eta^2 for content vs random-subspace vs shuffled null."""
import json, time, sys, torch
import torch.nn.functional as F
import numpy as np
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'content_doc_coherence_results.json'
NEVAL = 300; SEQ = 256; REF = [8, 10, 12]; K = 64
CAP = {}


def fwd(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
    return x


def capture(idx):
    hs = []
    for L in REF:
        def mk(L):
            def h(mo, i_, o_): CAP[L].append((i_[0] if isinstance(i_, tuple) else i_).detach().float().reshape(-1, D))
            return h
        hs.append(m.transformer.h[L].mlp.register_forward_hook(mk(L)))
    fwd(idx)
    for h in hs: h.remove()


def eta2(coords, doc):
    """mean over dims of between-doc variance / total variance. coords (N,Kc), doc (N,) long."""
    N, Kc = coords.shape; ndoc = int(doc.max().item()) + 1
    tot = coords.var(0, unbiased=False)                                   # (Kc,)
    sums = torch.zeros(ndoc, Kc, device=DEV); cnt = torch.zeros(ndoc, device=DEV)
    sums.index_add_(0, doc, coords); cnt.index_add_(0, doc, torch.ones(N, device=DEV))
    means = sums / cnt.clamp_min(1).unsqueeze(1)                          # (ndoc,Kc)
    gm = coords.mean(0)
    between = ((means - gm)**2 * cnt.unsqueeze(1)).sum(0) / N             # (Kc,)
    return float((between / tot.clamp_min(1e-8)).mean())


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NEVAL)
    blocks = rows[:, :SEQ].contiguous(); V = int(m.lm_head.weight.shape[0]); T = blocks.shape[1] - 1
    for L in REF: CAP[L] = []
    idsL = []; docL = []
    for i in range(0, blocks.shape[0], 8):
        idx = blocks[i:i+8].to(DEV)[:, :-1].contiguous(); idsL.append(idx.reshape(-1))
        docids = torch.arange(i, i+idx.shape[0], device=DEV).unsqueeze(1).expand(idx.shape[0], T)
        docL.append(docids.reshape(-1)); capture(idx)
    flat = torch.cat(idsL, 0); doc = torch.cat(docL, 0).long()
    devsum = None
    for L in REF:
        X = torch.cat(CAP[L], 0); xbar = torch.zeros(V, D, device=DEV); cnts = torch.zeros(V, device=DEV)
        xbar.index_add_(0, flat, X); cnts.index_add_(0, flat, torch.ones_like(flat, dtype=torch.float))
        xbar = xbar / cnts.clamp_min(1).unsqueeze(1)
        dv = X - xbar[flat]; devsum = dv if devsum is None else devsum + dv; del X; CAP[L] = []
    dev = devsum / len(REF); devc = dev - dev.mean(0)
    _, _, Vt = torch.linalg.svd(devc, full_matrices=False)
    Ccontent = devc @ Vt[:K].T                                            # (N,K) content coords
    g = torch.Generator(device=DEV).manual_seed(0)
    Rnd = torch.linalg.qr(torch.randn(D, K, generator=g, device=DEV))[0]
    Crandom = devc @ Rnd                                                  # random subspace of the SAME residual deviation
    # shuffled-doc-label null
    perm = torch.randperm(doc.shape[0], generator=g, device=DEV); doc_sh = doc[perm]
    out = {'K': K, 'n_positions': int(doc.shape[0]), 'n_docs': int(doc.max().item())+1,
           'content_doc_eta2': round(eta2(Ccontent, doc), 4),
           'random_subspace_doc_eta2': round(eta2(Crandom, doc), 4),
           'shuffled_label_eta2': round(eta2(Ccontent, doc_sh), 4)}
    out['content_over_random_ratio'] = round(out['content_doc_eta2']/max(out['random_subspace_doc_eta2'], 1e-6), 2)
    out['pred_a_document_topic'] = bool(out['content_doc_eta2'] > 0.3 and
                                        out['content_doc_eta2'] > 1.5*out['random_subspace_doc_eta2'] and
                                        out['content_doc_eta2'] > 5*out['shuffled_label_eta2'])
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"content doc-eta2 {out['content_doc_eta2']} | random-subspace {out['random_subspace_doc_eta2']} | shuffled null {out['shuffled_label_eta2']}", flush=True)
    print(f"content/random ratio {out['content_over_random_ratio']} | pred_a document-topic {out['pred_a_document_topic']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
