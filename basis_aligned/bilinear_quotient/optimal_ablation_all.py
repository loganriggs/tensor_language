# optimal_ablation_all: THE FULL OPTIMAL-ABLATION BASELINE (user directive 2026-08-25:
# every single module — this is the standing anchor everything compares against; ref
# Li & Janson arXiv 2409.09951). Components: all 18 MLPs, all 162 heads (128-dim c_proj
# slice), all 18 whole attention layers (D-dim c_proj output). For each: freeze weights,
# train one constant vector against full-model CE (mean-init, Adam 3e-3, 150 steps for
# MLPs/attn layers, 100 for heads, batch 8, positions >= 64), report Delta_mean and
# Delta_opt HELD OUT (train skip=80, eval skip=7000). RESUMABLE: results json and
# opt_ablation_consts_all.pt are checkpointed after every component; already-done
# components are skipped on restart.
#
# Registered predictions (scored at completion over all 198):
#   pred_a the mean anchor is generally good: median opt_over_mean >= 0.90.
#   pred_b where optimization helps a lot (ratio < 0.8), it is mostly ATTENTION
#          components: >= 75% of such components are heads or attn layers.
#   pred_c sanity: no optimal constant beats the clean model (delta_opt > 0 for all).
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'optimal_ablation_all_results.json'
CONSTS = PT + 'opt_ablation_consts_all.pt'
NFIT = 480; NEV = 960
BS = 8; LR = 3e-3
H = m.transformer.h
COMPS = [('mlp', L) for L in range(18)] \
    + [('head', L, h) for L in range(18) for h in range(9)] \
    + [('attn', L) for L in range(18)]
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


def hook_attn(mod, args, output):
    if STATE['mode'] == 'const':
        return STATE['vec'].to(output.dtype).expand_as(output)
    return None


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

    import os
    results = {}; consts = {}
    if os.path.exists(OUT):
        prev = json.load(open(OUT))
        results = prev.get('results', {})
    if os.path.exists(CONSTS):
        consts = torch.load(CONSTS)
    g = torch.Generator().manual_seed(5)
    for comp in COMPS:
        if comp[0] == 'mlp':
            L = comp[1]; name = f'mlp{L}'
        elif comp[0] == 'attn':
            L = comp[1]; name = f'attn{L}'
        else:
            L, hidx = comp[1], comp[2]; name = f'head{L}.{hidx}'
        if name in results:
            continue
        STEPS = 100 if comp[0] == 'head' else 150
        if comp[0] == 'mlp':
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
        elif comp[0] == 'attn':
            mod = H[L].attn.c_proj
            hk = mod.register_forward_hook(hook_attn)
            caps = []
            hcap = mod.register_forward_hook(
                lambda mo, a, o: caps.append(o.detach().float().mean((0, 1)).cpu()))
            STATE['mode'] = None
            with torch.no_grad():
                for i in range(0, 64, 8):
                    fwd(FITR[i:i + 8, :-1].to(DEV).contiguous())
            hcap.remove()
            init = torch.stack(caps).mean(0).to(DEV)
            shape_note = 'D (attn c_proj out)'
        else:
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
                         'rel_drift_from_mean': round(drift, 4),
                         'loss_curve': [round(x, 4) for x in losses]}
        consts[name] = a.detach().cpu()
        print(f"{name}: d_mean {d_mean:.4f} -> d_opt {d_opt:.4f} "
              f"(ratio {results[name]['opt_over_mean']})", flush=True)
        torch.save(consts, CONSTS)
        json.dump({'partial': True, 'clean': round(clean, 4), 'results': results},
                  open(OUT, 'w'), indent=1)

    torch.save(consts, CONSTS)
    ratios = sorted(v['opt_over_mean'] for v in results.values())
    med = ratios[len(ratios) // 2]
    helped = [(k, v) for k, v in results.items() if v['opt_over_mean'] < 0.80]
    attn_helped = sum(1 for k, _ in helped if not k.startswith('mlp'))
    pa = med >= 0.90
    pb = (attn_helped / max(len(helped), 1)) >= 0.75 if helped else True
    pc = all(v['delta_opt'] > 0 for v in results.values())
    out = {'clean': round(clean, 4), 'results': results,
           'data_budget': {'train_rows': NFIT, 'train_skip': 80,
                           'steps_mlp_attn': 150, 'steps_head': 100, 'batch_rows': BS,
                           'positions_per_step': BS * 192,
                           'eval_rows': NEV, 'eval_skip': 7000,
                           'eval_positions': NEV * 192},
           'median_opt_over_mean': round(med, 4),
           'n_helped_lt_080': len(helped),
           'helped': {k: v['opt_over_mean'] for k, v in helped},
           'consts_file': CONSTS,
           'pred_a_median_90': bool(pa), 'pred_b_helped_are_attn': bool(pb),
           'pred_c_sanity_positive': bool(pc), 'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"median {med:.4f} | helped<0.8: {len(helped)}")
    print(f"pred_a {pa} | pred_b {pb} | pred_c {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
