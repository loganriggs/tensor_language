"""Is the deep-middle high-dimensional interpretable topic/register content manifold UNIVERSAL across the bilinear family,
or specific to bilin18? Replicate the core finding (§1055) on two family models -- bilin12 (12L, D=768) and swiglu18
(18L, D=1152) -- via their own forward + hooks. For each: capture middle MLP-input, form content deviation (input minus
per-token mean), PCA it, report (a) the top-10 PCs' variance fraction (high-rank if small, ~0.12 for bilin18) and (b)
the context snippets at the extremes of the top axes (interpretable topic/register if universal). Same FineWeb corpus
(GPT-2 ids are valid in these models' padded 50304 vocab).

REGISTERED PREDICTIONS:
  (0) SANITY: models load and run; content deviation is nonzero.
  (a) UNIVERSAL HIGH-RANK: both family models show a high-rank middle content -- top-10 content PCs explain a SMALL
      fraction of variance (target < ~0.25), like bilin18's 0.12 -> high-rank content is a general family property, not
      a bilin18 quirk;
  (b) UNIVERSAL INTERPRETABILITY: the top content axes decode to topic/register contrasts (as in §1055) in the family
      models too. Report per-model top-10 variance fraction + top-PC snippets."""
import json, time, sys, torch
import torch.nn.functional as F
import tiktoken
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
sys.path.insert(0, '/workspace/rspd')
from tier2_model import load_elriggs
import census_lib as cl

PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'crossmodel_content_results.json'
DEV = 'cuda'; NEVAL = 300; SEQ = 256; K = 64; NPC = 8; NTOP = 5; CTX = 14
MODELS = {'bilin12': [5, 6, 7], 'swiglu18': [8, 10, 12]}
CAP = {}


@torch.no_grad()
def run_model(name, ref_layers, blocks, enc):
    mdl, cfg = load_elriggs(name, device=DEV, dtype=torch.float32); mdl.eval()
    D = mdl.transformer.wte.weight.shape[1]; V = mdl.transformer.wte.weight.shape[0]
    for L in ref_layers: CAP[(name, L)] = []
    hs = []
    for L in ref_layers:
        mlp = mdl.transformer.h[L].mlp
        def mk(key):
            def h(mo, i_, o_): CAP[key].append((i_[0] if isinstance(i_, tuple) else i_).detach().float().reshape(-1, D))
            return h
        hs.append(mlp.register_forward_hook(mk((name, L))))
    idsL = []
    for i in range(0, blocks.shape[0], 8):
        idx = blocks[i:i+8].to(DEV)[:, :-1].contiguous(); idsL.append(idx.cpu())
        mdl(idx, idx)   # dummy target (forward requires one); we only need the mlp-input hooks
    for h in hs: h.remove()
    ids = torch.cat(idsL, 0); T = ids.shape[1]; flat = ids.reshape(-1).to(DEV)
    devsum = None
    for L in ref_layers:
        X = torch.cat(CAP[(name, L)], 0)
        xbar = torch.zeros(V, D, device=DEV); cnts = torch.zeros(V, device=DEV)
        xbar.index_add_(0, flat, X); cnts.index_add_(0, flat, torch.ones_like(flat, dtype=torch.float))
        xbar = xbar / cnts.clamp_min(1).unsqueeze(1)
        dv = X - xbar[flat]; devsum = dv if devsum is None else devsum + dv; del X, dv; CAP[(name, L)] = []
    dev = devsum / len(ref_layers); devc = dev - dev.mean(0)
    S = torch.linalg.svdvals(devc); S2 = S**2; tot = float(S2.sum())
    _, _, Vt = torch.linalg.svd(devc, full_matrices=False)
    proj = devc @ Vt[:NPC].T
    ids_np = ids.numpy()

    def snip(fi):
        s, p = divmod(int(fi), T); lo = max(0, p - CTX)
        try: txt = enc.decode(ids_np[s, lo:p+1].tolist())
        except Exception: txt = '<err>'
        return txt.replace('\n', ' ')
    dirs = []
    for k in range(NPC):
        pk = proj[:, k]
        dirs.append({'pc': k, 'var_frac': round(float(S2[k])/tot, 4),
                     'pos': [snip(i) for i in torch.topk(pk, NTOP).indices.tolist()],
                     'neg': [snip(i) for i in torch.topk(-pk, NTOP).indices.tolist()]})
    res = {'layers': len(mdl.transformer.h), 'D': D, 'ref_layers': ref_layers,
           'top10_var_frac': round(float(S2[:10].sum())/tot, 4), 'directions': dirs}
    del mdl; torch.cuda.empty_cache()
    return res


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NEVAL)
    blocks = rows[:, :SEQ].contiguous(); enc = tiktoken.get_encoding('gpt2')
    out = {'bilin18_ref_top10_var_frac': 0.116, 'models': {}}
    for name, refl in MODELS.items():
        print(f"=== {name} ===", flush=True)
        r = run_model(name, refl, blocks, enc); out['models'][name] = r
        print(f"{name}: top10 content var frac {r['top10_var_frac']} (bilin18 0.116)", flush=True)
        for k in range(min(4, NPC)):
            d = r['directions'][k]
            print(f"  PC{k} v{d['var_frac']} +{d['pos'][0][-60:]!r} | -{d['neg'][0][-60:]!r}", flush=True)
    out['pred_a_universal_highrank'] = bool(all(out['models'][n]['top10_var_frac'] < 0.25 for n in MODELS))
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"pred_a universal high-rank {out['pred_a_universal_highrank']} | wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
