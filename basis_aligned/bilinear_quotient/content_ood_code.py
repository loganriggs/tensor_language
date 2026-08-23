"""FIRST out-of-distribution test (all prior work is in-distribution FineWeb prose, LESSONS rule 10). Does the content
machine's subspace generalize to CODE (a very different register)? Build the deep-middle content subspace from FineWeb
PROSE (top-64 PCA of pooled L8-12 content deviation), then measure how much of CODE's deep-middle content-deviation
variance that PROSE subspace captures -- vs code's OWN top-64 (upper bound) and a random-64 subspace (lower bound). Code
sequences = the repo's own .py files tokenized with GPT-2 BPE. Also check whether code's content is high-rank too.

REGISTERED PREDICTIONS:
  (0) SANITY: prose content captured by the prose subspace ~= prose's own top-64 fraction (high); random-64 null ~ 64/1152 ~0.056.
  (a) REGISTER-GENERAL CONTENT: the PROSE content subspace captures CODE's deep-middle content variance well above the
      random null and a good fraction (>0.5) of code's OWN top-64 capture -> the content directions generalize to OOD
      code, not prose-specific;
  (b) PARTIAL SPECIFICITY: code retained-fraction < code's own top-64 (some off-manifold code-specific content); code is
      also high-rank (top-10 var fraction small). Report retained fractions (prose-subspace / own / random) for code and
      prose + code high-rank."""
import json, time, sys, glob, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl
import tiktoken

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'content_ood_code_results.json'
NSEQ = 150; SEQ = 256; REF = [8, 10, 12]; K = 64
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


@torch.no_grad()
def content_dev(blocks):
    """pooled L8-12 content deviation (mean-centered) for a set of sequences."""
    for L in REF: CAP[L] = []
    idsL = []
    for i in range(0, blocks.shape[0], 8):
        idx = blocks[i:i+8].to(DEV)[:, :-1].contiguous(); idsL.append(idx.reshape(-1)); capture(idx)
    tok = torch.cat(idsL, 0); V = int(m.lm_head.weight.shape[0]); devsum = None
    for L in REF:
        X = torch.cat(CAP[L], 0); xbar = torch.zeros(V, D, device=DEV); cnts = torch.zeros(V, device=DEV)
        xbar.index_add_(0, tok, X); cnts.index_add_(0, tok, torch.ones_like(tok, dtype=torch.float))
        xbar = xbar / cnts.clamp_min(1).unsqueeze(1)
        dv = X - xbar[tok]; devsum = dv if devsum is None else devsum + dv; del X; CAP[L] = []
    dev = devsum/len(REF); return dev - dev.mean(0)


def load_code_blocks(nseq, seq):
    enc = tiktoken.get_encoding('gpt2'); toks = []
    files = sorted(glob.glob('/workspace/tensor_language/**/*.py', recursive=True))
    for fp in files:
        try:
            txt = open(fp).read()
        except Exception:
            continue
        toks.extend(enc.encode(txt))
        if len(toks) >= nseq*seq + seq: break
    t = torch.tensor(toks[:nseq*seq], dtype=torch.long).view(nseq, seq)
    return t


def retained(dev, U):
    proj = dev @ U; return float((proj**2).sum() / (dev**2).sum())


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    prose = cl.fineweb_rows(NSEQ)[:, :SEQ].contiguous()
    code = load_code_blocks(NSEQ, SEQ).contiguous()
    dev_prose = content_dev(prose); dev_code = content_dev(code)
    # subspaces
    _, Sp, Vtp = torch.linalg.svd(dev_prose, full_matrices=False); U_prose = Vtp[:K].T.contiguous()
    _, Sc, Vtc = torch.linalg.svd(dev_code, full_matrices=False); U_code = Vtc[:K].T.contiguous()
    g = torch.Generator(device=DEV).manual_seed(0); U_rand = torch.linalg.qr(torch.randn(D, K, generator=g, device=DEV))[0]
    Sc2 = Sc**2; totc = float(Sc2.sum()); Sp2 = Sp**2; totp = float(Sp2.sum())
    out = {'K': K, 'ref_layers': REF, 'n_code_seq': int(code.shape[0]), 'n_prose_seq': int(prose.shape[0]),
           'code_content_top10_var_frac': round(float(Sc2[:10].sum())/totc, 4),
           'prose_content_top10_var_frac': round(float(Sp2[:10].sum())/totp, 4),
           'retained': {
               'code_by_prose_subspace': round(retained(dev_code, U_prose), 4),
               'code_by_own_top64': round(retained(dev_code, U_code), 4),
               'code_by_random64': round(retained(dev_code, U_rand), 4),
               'prose_by_prose_subspace': round(retained(dev_prose, U_prose), 4),
               'prose_by_code_subspace': round(retained(dev_prose, U_code), 4)}}
    # subspace overlap prose<->code
    out['prose_code_subspace_overlap'] = round(float((U_prose.T @ U_code).pow(2).sum()/K), 4)
    r = out['retained']
    out['code_frac_of_own_captured_by_prose'] = round(r['code_by_prose_subspace']/max(r['code_by_own_top64'], 1e-6), 3)
    out['pred_a_register_general'] = bool(r['code_by_prose_subspace'] > 5*r['code_by_random64'] and out['code_frac_of_own_captured_by_prose'] > 0.5)
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"code content top10 var {out['code_content_top10_var_frac']} (prose {out['prose_content_top10_var_frac']})", flush=True)
    print(f"retained: code-by-prose {r['code_by_prose_subspace']} | code-own {r['code_by_own_top64']} | code-random {r['code_by_random64']} | prose-own {r['prose_by_prose_subspace']}", flush=True)
    print(f"prose<->code subspace overlap {out['prose_code_subspace_overlap']} | code frac-of-own via prose {out['code_frac_of_own_captured_by_prose']} | pred_a {out['pred_a_register_general']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
