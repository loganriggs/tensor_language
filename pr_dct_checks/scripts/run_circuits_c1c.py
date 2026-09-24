"""Circuits lane rung C1c (plans/CIRCUITS_PLAN_V4.md): zero-free-parameter decomposition of the interaction forms —
unit read pairs pulled back through the exact linearised transport, weighted by the exact downstream weights.

  python scripts/run_circuits_c1c.py --smoke
  python scripts/run_circuits_c1c.py --device cuda --c1a results/circuits_c1a.json --out results/circuits_c1c.json
"""
import argparse, collections, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn.functional as F
from torch.func import vjp, jacrev, vmap
from torch.nn.attention import SDPBackend, sdpa_kernel

from circuit_checks.tensorgpt import TensorGPTSpans, hidden_features, down_project
from circuit_checks.inrepo_model import load, state_before_block, ROOT
from scripts.run_symmetric_dct_v1 import full_hessian, readers_for
from scripts.run_symmetric_dct_v3 import solve
from scripts.run_circuits_c1b import RankOneDictionary

BARS = dict(identity=1e-4, source_share=0.8, all_units=0.6, stable=0.4, refit_gap=0.15, core=0.5)


def make_runner(model, source, n_layer, d, final_norm=True):
    """run(ctx, theta, hidden_add=None, stop=None): the span with theta added at block `source`; hidden_add = (L, i, delta) adds delta to
    block L's hidden activation at position i; stop = (L, i) returns the normalised MLP input of block L at position i instead."""
    blocks = model.transformer.h
    def run(ctx, theta, hidden_add=None, stop=None):
        values, init, first = ctx; values = values + theta
        for L in range(source, n_layer):
            block = blocks[L]; values = block.lambdas[0] * values + block.lambdas[1] * init
            attn, first = block.attn(F.rms_norm(values, (d,)), first); values = values + attn
            n = F.rms_norm(values, (d,))
            if stop is not None and stop[0] == L:
                return n[0, stop[1]]
            h = hidden_features(block.mlp, n)
            if hidden_add is not None and hidden_add[0] == L:
                h = h.clone(); h[:, hidden_add[1]] = h[:, hidden_add[1]] + hidden_add[2]
            values = values + down_project(block.mlp, h)
        if final_norm:
            values = F.rms_norm(values, (d,))
        return values[:, -3:].mean(1)[0]
    return run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true"); ap.add_argument("--device", default="cuda"); ap.add_argument("--source-layer", type=int, default=8)
    ap.add_argument("--sequence-length", type=int, default=32); ap.add_argument("--contexts", type=int, default=8); ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--c1a", default="results/circuits_c1a.json"); ap.add_argument("--n-tokens", type=int, default=16); ap.add_argument("--ridge", type=float, default=1e-6)
    ap.add_argument("--out", default="results/circuits_c1c.json")
    a = ap.parse_args(); t_start = time.time(); os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    if a.smoke:
        sys.path.insert(0, ROOT); import jacclust.tt_model as TT
        a.device = "cpu"; a.source_layer = 1; a.contexts = 2; a.sequence_length = 8; a.chunk = 8; a.n_tokens = 2
        cfg = TT.GPTConfig(vocab_size=64, n_layer=4, n_head=2, n_embd=16, bilinear=True, bilinear_attn=True, squared_attn=True)
        torch.manual_seed(0); model = TT.GPT(cfg)
        for p in model.parameters():
            if p.dim() == 2: torch.nn.init.normal_(p, std=0.3)
        model = model.eval(); model.requires_grad_(False); readers = F.normalize(torch.randn(cfg.n_embd, 2), dim=0)
        held_rows = torch.randint(0, 64, (2, a.sequence_length), generator=torch.Generator().manual_seed(20)); toks = [3, 5]
        stable = [(1, i) for i in range(8)] + [(3, i) for i in range(8)]; core = set(stable[:4])
    else:
        model, cfg, meta = load("bilinear-attn", a.device); readers = readers_for(model, a.device)
        rows = torch.load(os.path.join(ROOT, "basis_aligned/bilinear_quotient/.rowcache/fineweb_n192_skip7000.pt"), map_location="cpu"); held_rows = rows[96:96 + a.contexts, :a.sequence_length].long()
        c1a = json.load(open(a.c1a)); toks = c1a["tokens"][:a.n_tokens]
        uc = collections.Counter(u for p in c1a["per_direction"] for u in p["stable_units"]); nd = len(c1a["per_direction"])
        stable = [tuple(int(z) for z in u.split(".")) for u in uc]; core = {tuple(int(z) for z in u.split(".")) for u, n in uc.items() if n >= 0.5 * nd}
    dev = a.device; d = cfg.n_embd; n_layer = cfg.n_layer; T = a.sequence_length; positions = [T - 3, T - 2, T - 1]
    W = model.lm_head.weight.float(); tokdirs = F.normalize(W[toks].T, dim=0)
    U = torch.cat([readers.to(dev), tokdirs.to(dev)], 1); names = [f"reader{k}" for k in range(readers.shape[1])] + [f"tok{t}" for t in toks]; K = U.shape[1]
    layers = list(range(a.source_layer, n_layer)); run = make_runner(model, a.source_layer, n_layer, d)
    spans = TensorGPTSpans(model, a.source_layer, n_layer, slice(-3, None), final_norm=True)
    def ctx_of(ids):
        with torch.no_grad():
            st = state_before_block(model, ids.to(dev), a.source_layer)
        return (st["values"].float(), st["initial_values"].float(), st["first_values"].float())
    ctxs = [ctx_of(held_rows[i:i + 1]) for i in range(len(held_rows))]; zero = torch.zeros(d, device=dev)
    with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
        parity = float((run(ctxs[0], zero) - spans.make_span(ctxs[0])(zero, None)[0]).abs().max() / spans.make_span(ctxs[0])(zero, None)[0].abs().max())
    mlpw = {L: (model.transformer.h[L].mlp.Left.weight.float(), model.transformer.h[L].mlp.Right.weight.float()) for L in layers}
    by_block = collections.defaultdict(list)
    for (L, h) in stable: by_block[L].append(h)
    # ---- control (a): closed-form unit Hessian identity on context 0 for two units ---------------------------------------------------------
    ctrl_id = []; ctrl_share = []
    with sdpa_kernel(SDPBackend.MATH):
        ctx = ctxs[0]; test_units = [stable[0], stable[-1]]
        for (L, h) in test_units:
            i = positions[-1]; A_, B_ = mlpw[L]; a_, b_ = A_[h], B_[h]
            nfn = lambda th: run(ctx, th, stop=(L, i))
            with torch.no_grad(): n0 = nfn(zero)
            _, pull = vjp(nfn, zero); ga = pull(a_)[0]; gb = pull(b_)[0]
            Ha = full_hessian(lambda th: (a_ @ nfn(th))[None], d, dev, a.chunk)[0]; Hb = full_hessian(lambda th: (b_ @ nfn(th))[None], d, dev, a.chunk)[0]
            Hh = full_hessian(lambda th: ((a_ @ nfn(th)) * (b_ @ nfn(th)))[None], d, dev, a.chunk)[0]
            cross = torch.outer(ga, gb) + torch.outer(gb, ga); closed = cross + float(a_ @ n0) * Hb + float(b_ @ n0) * Ha
            ctrl_id.append(float((Hh - closed).norm() / Hh.norm())); ctrl_share.append(1 - float((Hh - cross).norm() ** 2 / Hh.norm() ** 2))
    print(f"[controls] runner parity {parity:.2e} | closed-form unit Hessian identity {max(ctrl_id):.2e} | cross-term share of unit Hessian (units {test_units}) {np.round(ctrl_share, 3).tolist()}", flush=True)
    # ---- main loop -------------------------------------------------------------------------------------------------------------------------
    rec = []; t0 = time.time(); unit_energy = collections.defaultdict(list); g_rand = torch.Generator().manual_seed(0)
    all_units_flat = [(L, h) for L in layers for h in range(mlpw[L][0].shape[0])]
    with sdpa_kernel(SDPBackend.MATH):
        for c, ctx in enumerate(ctxs):
            Hc = full_hessian(lambda th: U.T @ run(ctx, th), d, dev, a.chunk)                              # [K, d, d]
            Bc = 0.5 * (Hc + Hc.transpose(1, 2)); nB = Bc.flatten(1).square().sum(1)                       # [K]
            # exact downstream weights w[u, L, i] over all units of block L, and Jacobians J[L, i]
            wts = {}; J = {}; pulls = {}
            for L in layers:
                for i in positions:
                    with torch.no_grad(): h0 = None
                    def fout(delta, L=L, i=i): return U.T @ run(ctx, zero, hidden_add=(L, i, delta))
                    width = mlpw[L][0].shape[0]; _, pullw = vjp(fout, torch.zeros(width, device=dev))
                    wts[(L, i)] = torch.stack([pullw(e)[0] for e in torch.eye(K, device=dev)])              # [K, width]
                    nfn = lambda th, L=L, i=i: run(ctx, th, stop=(L, i)); _, pulls[(L, i)] = vjp(nfn, zero)
                    J[(L, i)] = jacrev(nfn)(zero)                                                            # [d, d]: dn/dtheta
            # all-unit prediction: B_all[u] = sum_{L,i} J^T (L^T diag(w) R + R^T diag(w) L) J
            Ball = torch.zeros(K, d, d, device=dev)
            for (L, i), Jli in J.items():
                A_, B_ = mlpw[L]; AJ = A_ @ Jli; BJ = B_ @ Jli                                                # [width, d]: unit reads pulled back
                for k in range(K):
                    w = wts[(L, i)][k]; Ball[k] += (AJ.T * w) @ BJ + (BJ.T * w) @ AJ
            # stable-set prediction and its forms
            X, Y, labels = [], [], []
            for L, hs in by_block.items():
                A_, B_ = mlpw[L]
                for i in positions:
                    AJ = A_[hs] @ J[(L, i)]; BJ = B_[hs] @ J[(L, i)]
                    for j, h in enumerate(hs):
                        X.append(AJ[j]); Y.append(BJ[j]); labels.append((L, h, i))
            X = torch.stack(X); Y = torch.stack(Y); dic = RankOneDictionary(X.double(), Y.double(), labels); G = dic.gram(); idx = torch.arange(dic.n, device=dev)
            # random control set of the same size
            ridx = torch.randperm(len(all_units_flat), generator=g_rand)[:len(stable)].tolist(); rand_units = [all_units_flat[r] for r in ridx]
            Xr, Yr, wr_lab = [], [], []
            for (L, h) in rand_units:
                A_, B_ = mlpw[L]
                for i in positions:
                    Xr.append(A_[h] @ J[(L, i)]); Yr.append(B_[h] @ J[(L, i)]); wr_lab.append((L, h, i))
            Xr = torch.stack(Xr); Yr = torch.stack(Yr)
            for k in range(K):
                B = Bc[k]; n2 = float(nB[k])
                r2_all = 1 - float((B - Ball[k]).square().sum()) / n2
                coef = torch.tensor([2 * float(wts[(L, i)][k, h]) for (L, h, i) in labels], device=dev, dtype=torch.float64)
                Bst = (X.double().T * coef) @ Y.double(); Bst = 0.5 * (Bst + Bst.T); r2_stable = 1 - float((B.double() - Bst).square().sum()) / n2
                coefr = torch.tensor([2 * float(wts[(L, i)][k, h]) for (L, h, i) in wr_lab], device=dev, dtype=torch.float64)
                Br = (Xr.double().T * coefr) @ Yr.double(); Br = 0.5 * (Br + Br.T); r2_rand = 1 - float((B.double() - Br).square().sum()) / n2
                al, e_fit = solve(G, dic.inner(B.double()), idx, a.ridge); r2_refit = e_fit / n2
                per_unit = collections.Counter()
                for j, (L, h, i) in enumerate(labels):
                    per_unit[f"{L}.{h}"] += float(coef[j]) * float(X[j].double() @ B.double() @ Y[j].double()) / n2
                core_share = sum(v for uu, v in per_unit.items() if tuple(int(z) for z in uu.split(".")) in core) / max(sum(per_unit.values()), 1e-30)
                for uu, v in per_unit.items(): unit_energy[uu].append(v)
                rec.append(dict(context=c, direction=k, name=names[k], norm2=n2, r2_all_units=r2_all, r2_stable=r2_stable, r2_refit=r2_refit, r2_random_set=r2_rand, core_share=core_share,
                                top_units=[u for u, _ in per_unit.most_common(8)], top_units_signed={u: round(v, 4) for u, v in per_unit.most_common(8)}))
            m = lambda key: float(np.median([r[key] for r in rec]))
            print(f"[closed] context {c + 1}/{len(ctxs)} ({time.time() - t0:.0f}s): R2 all-units {m('r2_all_units'):.3f} stable-set {m('r2_stable'):.3f} refit {m('r2_refit'):.3f} random-set {m('r2_random_set'):.3f} core share {m('core_share'):.2f}", flush=True)
            json.dump(dict(partial=True, records=rec), open(a.out + ".partial", "w"), default=float)
    m = lambda key: float(np.median([r[key] for r in rec]))
    preds = dict(pred_a_instrument=max(ctrl_id) <= BARS["identity"] and parity <= 1e-5 and ctrl_share[0] >= BARS["source_share"],
                 pred_b_all_units=m("r2_all_units") >= BARS["all_units"], pred_c_stable_set=m("r2_stable") >= BARS["stable"],
                 pred_d_refit_bound=(m("r2_refit") - m("r2_stable")) <= BARS["refit_gap"], pred_e_core_units=m("core_share") >= BARS["core"])
    ue = {u: float(np.mean(v)) for u, v in unit_energy.items()}
    res = dict(args=vars(a), predictions=preds, controls=dict(parity=parity, closed_form_identity=max(ctrl_id), cross_share=ctrl_share, test_units=[list(t) for t in test_units]),
               summary=dict(r2_all_units=m("r2_all_units"), r2_stable=m("r2_stable"), r2_refit=m("r2_refit"), r2_random_set=m("r2_random_set"), core_share=m("core_share"),
                            r2_all_readers=float(np.median([r["r2_all_units"] for r in rec if r["name"].startswith("reader")])), r2_all_tokens=float(np.median([r["r2_all_units"] for r in rec if r["name"].startswith("tok")])),
                            top_units_mean_energy=sorted(ue.items(), key=lambda kv: -abs(kv[1]))[:25], n_stable=len(stable), n_core=len(core)),
               records=rec, seconds=time.time() - t_start)
    json.dump(res, open(a.out, "w"), indent=1, default=float)
    print(json.dumps(dict(predictions=preds, controls=res["controls"], summary={k: v for k, v in res["summary"].items() if k != "top_units_mean_energy"}), indent=1, default=float), flush=True); print("wrote", a.out)


if __name__ == "__main__":
    main()
