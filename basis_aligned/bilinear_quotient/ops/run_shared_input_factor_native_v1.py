#!/usr/bin/env python3
"""A identities; B8starts converge; Ccenteredcapture>=.008/stability>=.99; Dpartner16>=90%."""
# BQGATE: 0forwards0seq; shared-input factor and exact local removal program.
import os,sys,json,time,signal,hashlib
from pathlib import Path
RUNNER=Path(__file__).resolve();P=RUNNER.parents[3]/'basis_aligned/polynomial_causal';sys.path.insert(0,str(P))
import torch
from shared_input_factor_v1 import optimize,value_gradient,native_partner
from congruence_block_operator_v1 import sandwich
CK=Path('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin')


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    binding=json.loads((P/'SHARED_INPUT_FACTOR_NATIVE_V1_BINDING.json').read_text());assert all(digest(p)==h for p,h in binding.items())
    assert json.loads((P/'SHARED_INPUT_FACTOR_V1_CONTROL.json').read_text())['instrument_passed']
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps(dict(gpu_accessed=False,body_forwards=0,corpus_access=False,
                             objectives=['full','centered'],starts_each=4,steps_each=4000)))
        return
    out=P/'SHARED_INPUT_FACTOR_NATIVE_V1_RESULT.json';assert not out.exists();signal.alarm(1200)
    start=time.perf_counter();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    sd=torch.load(CK,weights_only=True,map_location='cpu',mmap=True);u=sd['lm_head.weight'].double().cuda();uc=u-u.mean(0)
    l,r,down=[sd[f'transformer.h.17.mlp.{key}.weight'].double().cuda() for key in ['Left','Right','Down']]
    eye=torch.eye(1152,dtype=u.dtype,device=u.device);results={};saved={};errors=[]
    prior=json.loads((P/'SHARED_INPUT_FACTOR_V1_PRIOR_BOUND.json').read_text())
    for name,output,reference in [('full',u,99245061353.473),('centered',uc,92104252412.19983)]:
        k=down.T@(output.T@output)@down
        second=sandwich(l,r,k,eye);second=(second+second.T)/2
        eigen,vectors=torch.linalg.eigh(second);total=float(second.trace())
        bound=2*float(eigen[-1])/total
        errors.extend([abs(total/reference-1),abs(bound/prior[name]['one_shared_reader_capture_upper_bound']-1)])
        fits=[];readers=[]
        for seed in [None,1103,1109,1117]:
            if seed is None:initial=vectors[:,-1]
            else:
                torch.manual_seed(seed);initial=torch.randn(1152,dtype=u.dtype,device=u.device)
            a,fit=optimize(initial,l,r,k,second,max_steps=4000,tolerance=1e-8)
            fits.append(dict(seed=seed,**fit));readers.append(a)
            print(json.dumps(dict(objective=name,**{kk:vv for kk,vv in fits[-1].items() if kk!='history'})),flush=True)
        best=max(range(4),key=lambda j:fits[j]['capture']);a=readers[best];partner=native_partner(a,l,r,down)
        jroot=eye+(2**.5-1)*(a[:,None]*a[None,:])
        root=torch.linalg.cholesky(output.T@output);weighted=root.T@partner@jroot
        energy=float(weighted.square().sum()/2)
        errors.append(abs(energy/total-fits[best]['capture']))
        singular=torch.linalg.svdvals(weighted);squares=singular.square()
        full_weighted=torch.linalg.cholesky(u.T@u).T@partner@jroot
        full_energy=float(full_weighted.square().sum()/2)
        common=float(len(u)*((u.mean(0)@partner@jroot).square().sum())/2)
        torch.manual_seed(1123);x=torch.randn(64,1152,dtype=u.dtype,device=u.device)
        def native(xx):return ((xx@l.T)*(xx@r.T))@down.T
        original=native(x);removed=x-(x@a)[:,None]*a
        predicted=(x@a)[:,None]*(x@partner.T)
        deletion=float((original-native(removed)-predicted).norm()/original.norm());errors.append(deletion)
        b=torch.randn(1152,dtype=u.dtype,device=u.device);b-=a*(a@b);b/=b.norm()
        bpartner=native_partner(b,l,r,down)
        interaction=down@((l@a)*(r@b)+(l@b)*(r@a))
        joint_removed=removed-(x@b)[:,None]*b
        composed=predicted+(x@b)[:,None]*(x@bpartner.T)-(x@a)[:,None]*(x@b)[:,None]*interaction
        composition=float((original-native(joint_removed)-composed).norm()/original.norm());errors.append(composition)
        cosines=[float(abs(a@other)) for other in readers]
        results[name]=dict(total_energy=total,analytic_one_reader_upper_bound=bound,fits=fits,best=best,
                          best_capture=fits[best]['capture'],absolute_reader_cosines=cosines,
                          partner_rank16_capture=float(squares[:16].sum()/squares.sum()),
                          partner_rank90=int(torch.searchsorted(squares.cumsum(0),.9*squares.sum()))+1,
                          selected_component_full_energy=full_energy,selected_component_common_energy=common,
                          common_fraction_of_component=common/full_energy,
                          deletion_relative_error=deletion,composition_relative_error=composition)
        saved[name]=dict(readers=torch.stack(readers).cpu(),best=best,partner=partner.cpu())
    valid=max(errors)<=1e-8;converged=all(fit['converged'] for row in results.values() for fit in row['fits'])
    center=results['centered'];cache=Path('/dev/shm/bilin18_shared_input_factor_native_v1.pt');assert not cache.exists();torch.save(saved,cache)
    result=dict(predictions={'pred_a_instrument':valid,'pred_b_all_starts_converged':valid and converged,
        'pred_c_centered_coverage_and_stability':valid and converged and center['best_capture']>=.008 and min(center['absolute_reader_cosines'])>=.99,
        'pred_d_simple_partner':valid and converged and center['partner_rank16_capture']>=.90},
        objectives=results,maximum_instrument_error=max(errors),
        full_centered_reader_cosine=float(abs(saved['full']['readers'][saved['full']['best']]@saved['centered']['readers'][saved['centered']['best']])),
        cache=dict(path=str(cache),sha256=digest(cache),bytes=cache.stat().st_size,ephemeral=True),
        price=dict(reader_numbers=1152,full_partner_numbers=1152**2,rank16_reader_and_partner_numbers=1152+2*1152*16,
                   native_background_required=True,body_forwards=0,corpus_access=False),wall_seconds=time.perf_counter()-start,
        scope='One shared scalar input and native-compatible linear partner, exact postnorm removal and cross-term composition. Coefficient coverage/stability only; not semantic selectivity or full-model circuit adoption.')
    with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({kk:vv for kk,vv in result.items() if kk!='objectives'}),flush=True)


if __name__=='__main__':main()
