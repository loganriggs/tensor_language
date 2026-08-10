"""Noise-robustness probe on the four trained DGP checkpoints (Logan's question).

Perturbs the token-embedding table W -> W + sigma * rms(W) * N(0,1) (iid over
entries, global-RMS scale, stated in qk_dgp_noise_predictions.json), evaluates
full-fp32 held CE per (variant, arm, sigma, noise seed) on fresh held sequences
(sampler seed 999, disjoint from train/held seeds used in the experiment), and
writes qk_dgp_noise_probe.json.  Positive control: sigma=0 must reproduce each
cell's stored final held CE to within resampling wobble of the fresh split
(reported, not gated -- the fresh split differs from the experiment's held split
by construction).
"""
import json, torch
import qk_dgp_lang as L

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
SIGMAS = [0.0, 0.05, 0.1, 0.2, 0.4, 0.8]
NOISE_SEEDS = [0, 1, 2]
N_HELD = 512

out = {'config': {'sigmas': SIGMAS, 'noise_seeds': NOISE_SEEDS, 'n_held': N_HELD,
                  'held_sampler_seed': 999,
                  'noise': 'W + sigma * rms(W) * randn, iid entries, global RMS'},
       'cells': {}}

for variant in ('identifiable', 'overlap'):
    tabs = L.DGPTables(variant=variant)
    held = L.sample_seqs(tabs, N_HELD, seed=999)
    for arm in ('semi', 'learned'):
        ck = torch.load(f'qk_dgp_{variant}_{arm}.pt', map_location='cpu',
                        weights_only=False)
        model = L.make_arm_model(tabs, arm, device=DEV)
        model.load_state_dict(ck['state_dict'])
        W0 = model.wte.weight.detach().clone()
        rms = float(W0.pow(2).mean().sqrt())
        cell = {'rms_W': rms, 'base_ce_fresh_held': None, 'curve': {}}
        for sig in SIGMAS:
            ces = []
            for ns in ([-1] if sig == 0.0 else NOISE_SEEDS):
                if sig == 0.0:
                    with torch.no_grad():
                        model.wte.weight.copy_(W0)
                else:
                    g = torch.Generator(device='cpu').manual_seed(ns)
                    noise = torch.randn(W0.shape, generator=g) * sig * rms
                    with torch.no_grad():
                        model.wte.weight.copy_(W0 + noise.to(DEV))
                ces.append(L.eval_ce(model, held, DEV))
            m = sum(ces) / len(ces)
            if sig == 0.0:
                cell['base_ce_fresh_held'] = round(m, 5)
            cell['curve'][str(sig)] = {
                'ce_mean': round(m, 5),
                'ce_per_seed': [round(c, 5) for c in ces],
                'dce_mean': round(m - cell['base_ce_fresh_held'], 5)
                            if cell['base_ce_fresh_held'] is not None else 0.0}
        with torch.no_grad():
            model.wte.weight.copy_(W0)
        out['cells'][f'{variant}/{arm}'] = cell
        print(variant, arm, 'base', cell['base_ce_fresh_held'],
              {s: cell['curve'][s]['dce_mean'] for s in cell['curve']}, flush=True)

json.dump(out, open('qk_dgp_noise_probe.json', 'w'), indent=1)
print('written qk_dgp_noise_probe.json')
