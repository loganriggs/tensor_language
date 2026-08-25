# optimal_ablation: LEARNED-CONSTANT ABLATION ANCHORS (user spec, 2026-08-25; ref
# Li & Janson arXiv 2409.09951). For each component c: freeze all weights, train ONE
# constant vector a (the component's output shape, broadcast over batch/positions) to
# minimize the FULL model's CE on reference rows, init at the mean activation. Report
# Delta_opt = CE(a) - CE(clean) vs Delta_mean = CE(mean) - CE(clean), both HELD OUT
# (train skip=80, eval skip=7000, mask >= 64 to match the ladder). Constants saved to
# opt_ablation_consts.pt for future fidelity scoring:
#   fidelity(repl) = (Delta_opt - Delta_repl) / Delta_opt (opt const = 0, component = 1).
# Components: mlp4 (live thread), mlp1 (token table), mlp16 (register), head 13.8
# (flagship owner; constant = its 128-dim c_proj input slice — spans exactly the head's
# residual contribution through c_proj's columns, registered convention).
# Training: Adam lr 3e-3, 150 steps, batch 8, positions >= 64 in the loss.
#
# Registered predictions:
#   pred_a optimization beats the mean anchor for >= 2 of 4 components by >= 10%
#          (Delta_opt <= .9 x Delta_mean).
#   pred_b for mlp4 the mean is NEAR-optimal: Delta_opt >= .8 x Delta_mean.
#   pred_c re-anchored fidelity of mlp4's lin5 (CE 2.9781 on these eval rows) >= .60.
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'optimal_ablation_results.json'
CONSTS = PT + 'opt_ablation_consts.pt'
NFIT = 480; NEV = 960
STEPS = 150; BS = 8; LR = 3e-3
H = m.transformer.h
COMPS = [('mlp', 4), ('mlp', 1), ('mlp', 16), ('head', 13, 8)]
STATE = {'mode': None, 'vec': None}


def fwd(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in H:
        x, v1 = blk(x, v1, x0)
    return 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)


def hook_mlp(mod, args, output):
    if STATE['mode'] == 'const':
        return STATE['vec'].to(output.dtype).expand_as(output)
    return None


def hook_head(mod, args):
    if STATE['mode'] != 'const':
        return None
    y = args[0].clone()
    h = STATE['head_idx']
    y[..., h * 128:(h + 1) * 128] = STATE['vec'].to(y.dtype)
    return (y,)


def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    for p in m.parameters():
        p.requires_grad_(False)
    FITR = cl.fineweb_rows(NFIT, skip=80)[:, :T + 1].contiguous()
    EVR = cl.fineweb_rows(NEV, skip=7000)[:, :T + 1].contiguous()

    @torch.no_grad()
    def eval_ce():
        s_ = 0.0; n_ = 0
        for i in range(0, NEV, 8):
            bb = EVR[i:i + 8].to(DEV)
            idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].contiguous()
            lo = fwd(idx).float()
            ce = F.cross_entropy(lo.reshape(-1, lo.shape[-1]), tg.reshape(-1),
                                 reduction='none').view(tg.shape)
            mk = torch.ones_like(tg, dtype=torch.bool); mk[:, :64] = False
            s_ += float(ce[mk].sum()); n_ += int(mk.sum())
        return s_ / max(n_, 1)

    STATE['mode'] = None
    clean = eval_ce()
    print(f"clean {clean:.4f}", flush=True)

    results = {}; consts = {}
    g = torch.Generator().manual_seed(5)
    for comp in COMPS:
        if comp[0] == 'mlp':
            L = comp[1]; name = f'mlp{L}'
            mod = H[L].mlp
            hk = mod.register_forward_hook(hook_mlp)
            # mean init
            caps = []
            hcap = mod.register_forward_hook(
                lambda mo, a, o: caps.append(o.detach().float().mean((0, 1)).cpu()))
            STATE['mode'] = None
            with torch.no_grad():
                for i in range(0, 64, 8):
                    fwd(FITR[i:i + 8, :-1].to(DEV).contiguous())
            hcap.remove()
            init = torch.stack(caps).mean(0).to(DEV)
            shape_note = 'D'
        else:
            L, hidx = comp[1], comp[2]; name = f'head{L}.{hidx}'
            mod = H[L].attn.c_proj
            STATE['head_idx'] = hidx
            hk = mod.register_forward_pre_hook(hook_head)
            caps = []
            hcap = mod.register_forward_pre_hook(
                lambda mo, a: caps.append(
                    a[0].detach().float().reshape(-1, 9, 128)[:, hidx].mean(0).cpu()))
            STATE['mode'] = None
            with torch.no_grad():
                for i in range(0, 64, 8):
                    fwd(FITR[i:i + 8, :-1].to(DEV).contiguous())
            hcap.remove()
            init = torch.stack(caps).mean(0).to(DEV)
            shape_note = '128 (c_proj slice)'

        # mean-anchor CE
        STATE['mode'] = 'const'; STATE['vec'] = init.clone()
        with torch.no_grad():
            ce_mean = eval_ce()

        # train
        a = init.clone().requires_grad_(True)
        optim = torch.optim.Adam([a], lr=LR)
        STATE['vec'] = a
        losses = []
        for step in range(STEPS):
            ridx = torch.randint(0, NFIT, (BS,), generator=g)
            bb = FITR[ridx].to(DEV)
            idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].contiguous()
            lo = fwd(idx).float()
            ce = F.cross_entropy(lo.reshape(-1, lo.shape[-1]), tg.reshape(-1),
                                 reduction='none').view(tg.shape)
            loss = ce[:, 64:].mean()
            optim.zero_grad(); loss.backward(); optim.step()
            losses.append(float(loss))
            if (step + 1) % 50 == 0:
                print(f"{name} step {step + 1}: loss {sum(losses[-50:]) / 50:.4f}",
                      flush=True)
        STATE['vec'] = a.detach()
        with torch.no_grad():
            ce_opt = eval_ce()
        STATE['mode'] = None
        hk.remove()
        d_mean = ce_mean - clean; d_opt = ce_opt - clean
        drift = float((a.detach() - init).norm() / init.norm().clamp_min(1e-6))
        results[name] = {'shape': shape_note, 'ce_mean': round(ce_mean, 4),
                         'ce_opt': round(ce_opt, 4), 'delta_mean': round(d_mean, 4),
                         'delta_opt': round(d_opt, 4),
                         'opt_over_mean': round(d_opt / max(d_mean, 1e-6), 4),
                         'rel_drift_from_mean': round(drift, 4)}
        consts[name] = a.detach().cpu()
        print(f"{name}: d_mean {d_mean:.4f} -> d_opt {d_opt:.4f} "
              f"(ratio {results[name]['opt_over_mean']})", flush=True)
        json.dump({'partial': True, 'clean': round(clean, 4), 'results': results},
                  open(OUT, 'w'), indent=1)

    torch.save(consts, CONSTS)
    ratios = {k: v['opt_over_mean'] for k, v in results.items()}
    pa = sum(1 for v in ratios.values() if v <= 0.90) >= 2
    pb = ratios['mlp4'] >= 0.80
    LIN5_CE = 2.9781
    d_lin5 = LIN5_CE - clean
    d_opt4 = results['mlp4']['delta_opt']
    fid_lin5 = (d_opt4 - d_lin5) / max(d_opt4, 1e-6)
    pc = fid_lin5 >= 0.60
    out = {'clean': round(clean, 4), 'results': results,
           'lin5_fidelity_reanchored': round(fid_lin5, 4),
           'consts_file': CONSTS,
           'pred_a_opt_beats_mean_2of4': bool(pa), 'pred_b_mlp4_mean_near_opt': bool(pb),
           'pred_c_lin5_fidelity_60': bool(pc), 'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"ratios {ratios} | lin5 fidelity {fid_lin5:.4f}")
    print(f"pred_a {pa} | pred_b {pb} | pred_c {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
