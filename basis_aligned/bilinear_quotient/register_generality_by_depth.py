"""§1079 showed the deep-middle CONTENT subspace is register-specific (prose<->code overlap 0.19). Is the GRAMMAR machine
more register-general? Grammar (syntax / token-class, front layers) should transfer across registers better than topic-
content, since code still has punctuation/keywords/number structure. Measure the prose<->code MLP-input deviation
subspace overlap at EVERY layer (top-64 PCA each), giving a register-generality-by-depth profile. Prediction: overlap is
HIGH at the grammar front (L0-2) and DROPS through the content middle (L6-14, ~§1079's 0.19), tracking the two-machine
split -- grammar register-general, content register-specific.

REGISTERED PREDICTIONS:
  (0) SANITY: random-subspace null overlap ~ K/D ~ 0.056 at every layer; a layer's prose subspace vs its own prose (2nd half) ~ high.
  (a) GRAMMAR REGISTER-GENERAL, CONTENT REGISTER-SPECIFIC: prose<->code subspace overlap is HIGHER at the front (L0-2,
      grammar) than in the deep-middle (L6-14, content); front overlap > ~2x the deep-middle overlap -> grammar
      generalizes across registers, content is register-specific;
  (b) report prose<->code overlap by layer + front/middle/readout band means + random null + a prose-vs-prose(split) ceiling."""
import json, time, sys, glob, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl
import tiktoken

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'register_generality_by_depth_results.json'
NSEQ = 150; SEQ = 256; LAYERS = list(range(18)); K = 64
CAP = {}


def fwd(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
    return x


@torch.no_grad()
def dev_subspaces(blocks):
    """per-layer top-K content-deviation subspace (D,K) for a set of sequences."""
    for L in LAYERS: CAP[L] = []
    hs = []
    for L in LAYERS:
        def mk(L):
            def h(mo, i_, o_): CAP[L].append((i_[0] if isinstance(i_, tuple) else i_).detach().float().reshape(-1, D))
            return h
        hs.append(m.transformer.h[L].mlp.register_forward_hook(mk(L)))
    idsL = []
    for i in range(0, blocks.shape[0], 8):
        idx = blocks[i:i+8].to(DEV)[:, :-1].contiguous(); idsL.append(idx.reshape(-1)); fwd(idx)
    for h in hs: h.remove()
    tok = torch.cat(idsL, 0); V = int(m.lm_head.weight.shape[0]); U = {}
    for L in LAYERS:
        X = torch.cat(CAP[L], 0); xbar = torch.zeros(V, D, device=DEV); cnts = torch.zeros(V, device=DEV)
        xbar.index_add_(0, tok, X); cnts.index_add_(0, tok, torch.ones_like(tok, dtype=torch.float))
        xbar = xbar / cnts.clamp_min(1).unsqueeze(1)
        dev = X - xbar[tok]; dev = dev - dev.mean(0)
        _, _, Vt = torch.linalg.svd(dev, full_matrices=False); U[L] = Vt[:K].T.contiguous()
        CAP[L] = []; del X, dev
    return U


def load_code(nseq, seq):
    enc = tiktoken.get_encoding('gpt2'); toks = []
    for fp in sorted(glob.glob('/workspace/tensor_language/**/*.py', recursive=True)):
        try: toks.extend(enc.encode(open(fp).read()))
        except Exception: continue
        if len(toks) >= nseq*seq: break
    return torch.tensor(toks[:nseq*seq], dtype=torch.long).view(nseq, seq)


def ov(A, B): return round(float((A.T @ B).pow(2).sum()/K), 4)


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    prose = cl.fineweb_rows(NSEQ*2)[:, :SEQ].contiguous()      # split for prose-vs-prose ceiling
    prA = prose[:NSEQ].contiguous(); prB = prose[NSEQ:2*NSEQ].contiguous()
    code = load_code(NSEQ, SEQ).contiguous()
    Up = dev_subspaces(prA); Up2 = dev_subspaces(prB); Uc = dev_subspaces(code)
    g = torch.Generator(device=DEV).manual_seed(0); Rnd = torch.linalg.qr(torch.randn(D, K, generator=g, device=DEV))[0]
    by_layer = {}
    for L in LAYERS:
        by_layer[str(L)] = {'prose_code': ov(Up[L], Uc[L]), 'prose_prose_ceiling': ov(Up[L], Up2[L]), 'random_null': ov(Up[L], Rnd)}
    def band(ls): return round(sum(by_layer[str(L)]['prose_code'] for L in ls)/len(ls), 4)
    out = {'K': K, 'by_layer': by_layer,
           'front_L0_2': band([0, 1, 2]), 'transition_L3_5': band([3, 4, 5]),
           'deepmiddle_L6_14': band(list(range(6, 15))), 'readout_L15_17': band([15, 16, 17]),
           'random_null': by_layer['8']['random_null']}
    out['pred_a_grammar_general_content_specific'] = bool(out['front_L0_2'] > 2*out['deepmiddle_L6_14'])
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    for L in LAYERS:
        b = by_layer[str(L)]; print(f"L{L}: prose-code {b['prose_code']} | prose-prose {b['prose_prose_ceiling']} | null {b['random_null']}", flush=True)
    print(f"BANDS prose-code: front(0-2) {out['front_L0_2']} | transition(3-5) {out['transition_L3_5']} | deepmiddle(6-14) {out['deepmiddle_L6_14']} | readout(15-17) {out['readout_L15_17']}", flush=True)
    print(f"pred_a grammar-general/content-specific: {out['pred_a_grammar_general_content_specific']} | wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
