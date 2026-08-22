"""IS CAUSAL CLASS STEERING UNIVERSAL? (§837 cross-model, robust/keep-only-free). §826 showed the
class DIRECTIONS are universal (naming); §837 showed class STEERING is causal & class-specific in
bilin18. Test whether steering works in GPT-2 and Pythia too: inject a source class B's amplified
class-deviation at the front components and measure whether the prediction moves toward B's typical
continuation p_B (KL drops), vs a matched-norm random-direction null. If yes, class+position is a
causally real representation across architectures — independent of the retracted keep-only metric.

REGISTERED PREDICTIONS:
  (0) SANITY: unsteered KL(avg‖p_B) > 0;
  (a) UNIVERSAL: in GPT-2 and Pythia, class-deviation steering reduces KL to p_B (moves toward the
      injected class) more than the matched random-direction null, for grammatically-clear sources;
  (b) report per-source KL(base→cp→rand) per model."""
import json, time, torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import torch.nn.functional as F

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'cross_model_steering_results.json'
MODELS = ['gpt2', 'EleutherAI/pythia-410m']
SEQ = 128; NBLOCK = 96; MINCOUNT = 5; RTOK = 64; RPOS = 32; ALPHA = 16.0
# frequent function-word tokens as class sources (decoded per-model; ids differ, so pick by string)
SRC_STRINGS = [' the', ' a', ' and', ' of']
ST = {'on': False, 'mode': 'cp', 'delta': {}, 'rand': {}, 'front': None}


def front_layers(mdl):
    if hasattr(mdl, 'transformer'): nL = len(mdl.transformer.h); return mdl.transformer.h, nL, lambda blk: [('attn', blk.attn), ('mlp', blk.mlp)]
    nL = len(mdl.gpt_neox.layers); return mdl.gpt_neox.layers, nL, lambda blk: [('attn', blk.attention), ('mlp', blk.mlp)]


def get_text(nc=2000, npass=60):
    ds = load_dataset('HuggingFaceFW/fineweb', name='sample-10BT', split='train', streaming=True)
    t = []
    for i, ex in enumerate(ds):
        t.append(ex['text'][:nc])
        if i >= npass: break
    return '\n\n'.join(t)


def kl(p, q):
    p = p + 1e-9; q = q + 1e-9; p = p/p.sum(); q = q/q.sum()
    return float((p*(p/q).log()).sum())


def run(model_id):
    tok = AutoTokenizer.from_pretrained(model_id); mdl = AutoModelForCausalLM.from_pretrained(model_id).to(DEV).eval()
    d = mdl.config.hidden_size if hasattr(mdl.config, 'hidden_size') else mdl.config.n_embd
    blocks_mod, nL, comps_of = front_layers(mdl)
    FRONT = list(range(nL // 3))                       # first third
    ids = tok(get_text(), return_tensors='pt')['input_ids'][0]; nb = min(NBLOCK, ids.shape[0]//SEQ)
    blocks = ids[:nb*SEQ].reshape(nb, SEQ)
    # source token ids (first token of each string)
    src = {}
    for s in SRC_STRINGS:
        enc = tok(s, add_special_tokens=False)['input_ids']
        if enc: src[s] = enc[0]

    # hook state maps keyed by (L, which)
    hookmods = {}
    for L in FRONT:
        for wname, mod in comps_of(blocks_mod[L]):
            hookmods[(L, wname)] = mod

    def mk_hook(key):
        def hook(mo, i_, o_):
            if not ST['on']: return o_
            y = o_[0] if isinstance(o_, tuple) else o_; sh = y.shape; v = y.reshape(-1, d).float()
            dd = ST['delta'].get(key) if ST['mode'] == 'cp' else ST['rand'].get(key)
            if dd is None: return o_
            v2 = v + ALPHA * dd; yn = v2.reshape(sh).to(y.dtype)
            return (yn,) + tuple(o_[1:]) if isinstance(o_, tuple) else yn
        return hook

    @torch.no_grad()
    def capture(key):
        mod = hookmods[key]; cap = []; toks = []; pos = []
        def h(mo, i_, o_): cap.append((o_[0] if isinstance(o_, tuple) else o_).detach().float().reshape(-1, d))
        hh = mod.register_forward_hook(h)
        for i in range(0, nb, 4):
            b = blocks[i:i+4].to(DEV); mdl(b); toks.append(b.reshape(-1).cpu().numpy()); pos.append(np.broadcast_to(np.arange(b.shape[1]), b.shape).reshape(-1))
        hh.remove(); return torch.cat(cap, 0), np.concatenate(toks), np.concatenate(pos)

    def mean_subspace(O, labels, r):
        g = O.mean(0, keepdim=True); rows = []; wt = []
        for t in np.unique(labels):
            mk = labels == t
            if mk.sum() < MINCOUNT: continue
            rows.append(O[mk].mean(0) - g[0]); wt.append(np.sqrt(mk.sum()))
        M = torch.stack(rows, 0) * torch.tensor(wt, device=O.device, dtype=O.dtype)[:, None]
        return torch.linalg.svd(M, full_matrices=False)[2][:r].T.contiguous()

    subs = {}; gm = {}; tmean = {}
    g_ = torch.Generator(device=DEV).manual_seed(0)
    for key in hookmods:
        O, toks, pos = capture(key); g = O.mean(0, keepdim=True)
        Ut = mean_subspace(O, toks, RTOK); Up = mean_subspace(O, pos.astype(np.int64), RPOS)
        subs[key] = torch.linalg.svd(torch.cat([Ut, Up], 1), full_matrices=False)[0][:, :RTOK+RPOS].contiguous(); gm[key] = g
        for s, b in src.items():
            mk = toks == b
            if mk.sum() >= MINCOUNT: tmean[(s, key)] = O[mk].mean(0, keepdim=True).to(DEV)

    handles = [hookmods[key].register_forward_hook(mk_hook(key)) for key in hookmods]

    @torch.no_grad()
    def avg_pred(tok_id=None):
        ps = []
        for i in range(0, nb, 4):
            b = blocks[i:i+4].to(DEV); lg = mdl(b).logits.float()[:, :, :]
            p = F.softmax(lg, -1).reshape(-1, lg.shape[-1])
            if tok_id is None: ps.append(p.mean(0).cpu())
            else:
                mk = (b.reshape(-1) == tok_id)
                if mk.any(): ps.append(p[mk].cpu())
        return (torch.cat(ps, 0) if tok_id is not None else torch.stack(ps, 0)).mean(0)

    ST['on'] = False
    res = {}
    for s, b in src.items():
        pB = avg_pred(b); base = kl(avg_pred(), pB)
        for key in hookmods:
            if (s, key) in tmean:
                dev = tmean[(s, key)] - gm[key]; U = subs[key]; dcp = (dev @ U) @ U.T; ST['delta'][key] = dcp
                rd = torch.randn(1, d, generator=g_, device=DEV); ST['rand'][key] = rd/rd.norm()*dcp.norm()
            else: ST['delta'][key] = torch.zeros(1, d, device=DEV); ST['rand'][key] = torch.zeros(1, d, device=DEV)
        ST['on'] = True; ST['mode'] = 'cp'; kcp = kl(avg_pred(), pB)
        ST['mode'] = 'rand'; krand = kl(avg_pred(), pB); ST['on'] = False
        res[s] = {'kl_base': round(base, 4), 'kl_cp': round(kcp, 4), 'kl_rand': round(krand, 4), 'moved_toward': round(base - kcp, 4)}
        print(f'  {model_id} steer->{repr(s)}: base {base:.3f} -> cp {kcp:.3f} (rand {krand:.3f}) moved {base-kcp:+.3f}', flush=True)
    for h in handles: h.remove()
    moves = [res[s]['moved_toward'] for s in res]
    del mdl; torch.cuda.empty_cache()
    return {'d': d, 'front_layers': FRONT, 'per_source': res, 'mean_moved': round(float(np.mean(moves)), 4),
            'cp_beats_rand': bool(np.mean([res[s]['kl_cp'] < res[s]['kl_rand'] for s in res]) >= 0.75)}


def main():
    t0 = time.time(); out = {}
    for mid in MODELS:
        print(f'=== {mid} ===', flush=True); out[mid] = run(mid)
    out['pred_a_universal'] = bool(all(out[m]['mean_moved'] > 0.05 and out[m]['cp_beats_rand'] for m in MODELS))
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\n(a) causal class steering universal (moves toward B, beats random, both models): {out["pred_a_universal"]}', flush=True)
    print(f'wrote {OUT} ({out["runtime_s"]}s)')


if __name__ == '__main__':
    main()
