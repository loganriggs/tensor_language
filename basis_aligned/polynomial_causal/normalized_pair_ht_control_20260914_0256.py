"""Bounded native-weight, synthetic-corner source-pair estimator control.

Frozen predictions: all 9 expected HT Gram entries replay within 1e-10
relative Frobenius; naive squared HT write has positive bias predicted by
the same-source correction (1e-10 replay); a fixed scalar witness shows
>1% relative bias if its denominator is recomputed from the sampled write.
Enumerate all masks, no Monte Carlo tolerance, fitting, text or GPU work.
"""
import itertools
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from extracted_circuits.three_corner_head17_interaction_v1.ports import (
    EPS, additive, from_projections, project,
)
from retained_contraction_error_control_v1 import components
from three_group_shared_dag_v1 import execute
import head17_output_block_objective_v1 as target_builder

P = Path(__file__).resolve().parent
CHECKPOINT = Path('/home/loganriggs/.local/share/bilin18/hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin')


def source_terms(ports):
    a, b, v = ports[0]
    ac, bc, vc = (x-y for x, y in zip(ports[1], ports[0]))
    ar, br, vr = (x-y for x, y in zip(ports[2], ports[0]))
    abar, bbar, vbar = a+ac+ar, b+bc+br, v+vc+vr
    ea, eb = ports[3][0]-abar, ports[3][1]-bbar
    gates = (ar*bbar+(a+ac)*br, ac*bbar+(a+ar)*bc,
             ea*bbar+abar*eb)
    return torch.stack([g[..., None]*v for g, v in
                        zip(gates, (vc, vr, vbar))], 1)


def enumerate_design(f, p):
    """f: contexts x 3 branches x sources x outputs; fixed before masks."""
    masks = torch.tensor(list(itertools.product((0., 1.), repeat=len(p))),
                         dtype=f.dtype)
    probs = torch.where(masks.bool(), p, 1-p).prod(-1)
    sampled = torch.einsum('ms,nkso->mnko', masks/p, f)
    plugin = torch.einsum('mnko,mnlo->mnkl', sampled, sampled)
    correction = torch.einsum('ms,nkso,nlso->mnkl',
                               masks*(1-p)/p.square(), f, f)
    return masks, probs, plugin, plugin-correction


@torch.no_grad()
def run():
    start = time.perf_counter()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    sd = torch.load(CHECKPOINT, weights_only=True, mmap=True, map_location='cpu')
    def head(layer, name):
        return sd[f'transformer.h.{layer}.attn.{name}.weight'].reshape(9,128,1152)[2].double()
    matrices = tuple(head(17, k) for k in ('c_q','c_k','c_q2','c_k2','c_v'))
    package = torch.load(P/'extracted_circuits/three_corner_head17_interaction_v1/program.pt', weights_only=True)
    writer = package['output_matrix'].double()
    mu = float(sd['transformer.h.17.attn.lamb'])
    assert mu == float(package['mixture'])
    target_builder.CHECKPOINT = CHECKPOINT  # process-local path resolution only
    tensor, ids = target_builder.build()
    # Fixed error direction, not a fitted candidate and not selected on outcomes.
    signs = torch.where(torch.arange(128) % 2 == 0, 1., -1.).double()
    error = .01*tensor*signs
    generator = torch.Generator().manual_seed(17021456)
    def rand(*shape):
        return torch.randn(*shape, generator=generator, dtype=torch.float64)
    n, s, d = 8, 5, 1152
    x = rand(n,s,d)
    dc, dr = .1*rand(n,s,d), .2*rand(n,s,d)
    raw = (x, x+dc, x+dr)
    projections = tuple(project(r, matrices) for r in raw)
    norms = tuple(r.square().mean(-1)+EPS for r in raw)
    added_norm = norms[1]+norms[2]-norms[0]+2*(dc*dr).mean(-1)
    first_raw = .6*x+.8*rand(n,s,d)
    first = (first_raw/head_rms(first_raw)) @ head(0,'c_v').T
    ports = tuple(from_projections(a,b,first,mu) for a,b in zip(projections,norms))
    ports += (from_projections(additive(*projections),added_norm,first,mu),)
    writes = source_terms(ports)
    retained = execute(*ports)
    z = .7*x[:,-1]+.3*first_raw[:,-1]
    state = z+retained@writer.T
    final_rms = head_rms(z).squeeze(-1)  # supplied conditional readout port
    denom = (state.square().mean(-1)+EPS)*final_rms
    reader = torch.einsum('oih,ni->noh', error, z)
    f = torch.einsum('noh,nksh->nkso',reader,writes)/denom[:,None,None,None]
    totals = f.sum(2)
    exact = torch.einsum('nko,nlo->nkl',totals,totals)
    p = torch.tensor([.25,.4,.55,.7,.85],dtype=torch.float64)
    masks, probs, plugin, ht = enumerate_design(f,p)
    expected = torch.einsum('m,mnkl->nkl',probs,ht)
    expected_plugin = torch.einsum('m,mnkl->nkl',probs,plugin)
    known_bias = torch.einsum('s,nkso,nlso->nkl',(1-p)/p,f,f)
    rel = lambda a,b: float((a-b).norm()/b.norm())
    sampled_write = torch.einsum('ms,nksh->mnh',masks/p,writes)
    sampled_state = z[None]+sampled_write@writer.T
    sampled_denom = (sampled_state.square().mean(-1)+EPS)*final_rms[None]
    wrong = ht*(denom[None]/sampled_denom).square()[...,None,None]
    wrong_mean = torch.einsum('m,mnkl->nkl',probs,wrong)
    energy = float(exact.mean(0).sum())
    draws = ht.sum((-1,-2))
    # Independent masks per context: variance of their equally weighted mean.
    var_per_context = torch.einsum('m,mn->n',probs,
        (draws-exact.sum((-1,-2))[None]).square())
    cv = float((var_per_context.sum()/n**2).sqrt()/energy)
    # Fixed rational scalar witness; same inclusion correction, changed denominator.
    wf = torch.zeros(1,3,2,1,dtype=torch.float64)
    wf[0,0,:,0] = torch.tensor([1.,2.])/((1+3.)**2+EPS)
    wp = torch.tensor([.5,.5],dtype=torch.float64)
    wm, wprob, _, wh = enumerate_design(wf,wp)
    wd = (1+(wm/wp)@torch.tensor([1.,2.],dtype=torch.float64)).square()+EPS
    wt = float(wf.sum(2).sum().square())
    wb = float((wprob*wh.sum((1,2,3))*(((1+3.)**2+EPS)/wd).square()).sum())
    result = dict(
        utc=datetime.now(timezone.utc).isoformat(), seed=17021456,
        dimensions=dict(contexts=n,sources=s,residual=d,head=128,outputs=12),
        checkpoint=str(CHECKPOINT), checkpoint_blob=CHECKPOINT.resolve().name,
        target_token_ids=ids, target_norm=float(tensor.norm()), mixture=mu,
        inclusion_probabilities=p.tolist(), masks_enumerated=len(masks),
        producer_replay_relative_error=rel(writes.sum(2).sum(1),retained),
        existing_components_replay_relative_error=rel(writes.sum(2),components(*ports)),
        expected_complete_gram_relative_error=rel(expected,exact),
        expected_complete_gram_max_entry_absolute_error=float((expected-exact).abs().max()),
        naive_bias_formula_relative_error=rel(expected_plugin-exact,known_bias),
        exact_mean_gram=exact.mean(0).tolist(), expected_ht_mean_gram=expected.mean(0).tolist(),
        exact_energy=energy, naive_expected_energy=float(expected_plugin.mean(0).sum()),
        naive_relative_energy_bias=float((expected_plugin-exact).mean(0).sum()/energy),
        sampled_denominator_relative_gram_bias=rel(wrong_mean,exact),
        independent_context_mask_energy_cv=cv,
        masks_per_context_for_one_percent_relative_se_heuristic=int((cv/.01)**2)+1,
        negative_energy_probability_per_context=(probs[:,None]*(draws<0)).sum(0).tolist(),
        scalar_normalization_witness=dict(exact_energy=wt,biased_expectation=wb,
                                          relative_bias=abs(wb-wt)/wt),
        scope='Actual producer and downstream weights; synthetic shared raw corners and correlated inherited inputs/background. Final RMS supplied from z. No native trajectories, fitting, fidelity, compression or speed claim.',
        threads=torch.get_num_threads(),cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],
        seconds=time.perf_counter()-start,
    )
    result['pred_a'] = result['expected_complete_gram_relative_error'] <= 1e-10
    result['pred_b'] = result['naive_bias_formula_relative_error'] <= 1e-10 and result['naive_relative_energy_bias'] > 0
    result['pred_c'] = result['scalar_normalization_witness']['relative_bias'] > .01
    return result


def head_rms(x):
    return (x.square().mean(-1,keepdim=True)+EPS).sqrt()


if __name__ == '__main__':
    result = run()
    with (P/'NORMALIZED_PAIR_HT_CONTROL_20260914_0256_RESULT.json').open('x') as handle:
        json.dump(result,handle,indent=2)
        handle.write('\n')
    print(json.dumps(result,indent=2))
