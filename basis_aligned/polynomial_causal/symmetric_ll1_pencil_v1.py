"""Two-mixture initialization for exact symmetric LL1 with independent input blocks.

Requires total input rank <= d, full column rank concatenated input bases,
invertible within-block cores, and distinct ratios of two nonzero output mixes.
This is not a general noisy/overcomplete LL1 recovery algorithm.
"""
import json,itertools,time
from pathlib import Path
import torch
from symmetric_ll1_objective_v1 import cp
from chunked_bilinear_coefficient_v1 import dense


def initialize(forms,ranks,seed=2001):
    outputs,d,_=forms.shape;rank=sum(ranks)
    if rank>d:raise ValueError('Total prescribed input rank exceeds dimension')
    second=torch.einsum('oij,ojk->ik',forms,forms)
    _,vectors=torch.linalg.eigh(second);z=vectors[:,-rank:]
    reduced=torch.einsum('ia,oij,jb->oab',z,forms,z)
    generator=torch.Generator(device=forms.device).manual_seed(seed)
    mixtures=torch.randn(2,outputs,device=forms.device,dtype=forms.dtype,generator=generator)
    first,second_mix=torch.einsum('ko,oij->kij',mixtures,reduced)
    operator=torch.linalg.solve(first.T,second_mix.T).T
    values,vectors=torch.linalg.eig(operator)
    order=values.real.argsort();values=values[order];vectors=vectors[:,order]
    bases=[];centers=[];widths=[];start=0
    for r in ranks:
        group=vectors[:,start:start+r]
        # Complex-conjugate vectors may span a real invariant block. Retain
        # its real span, rather than dropping imaginary parts and losing rank.
        real=torch.cat([group.real,group.imag],dim=1)
        basis,sv,_=torch.linalg.svd(real,full_matrices=False)
        bases.append(basis[:,:r]);eigen=values[start:start+r]
        center=eigen.mean();centers.append(center)
        widths.append(float((eigen-center).abs().max()))
        start+=r
    full_basis=torch.cat(bases,dim=1)
    inverse=torch.linalg.inv(full_basis)
    cores=torch.einsum('ai,oij,bj->oab',inverse,reduced,inverse)
    groups=[];core_rank1=[];offset=0
    for basis,r in zip(bases,ranks):
        block=cores[:,offset:offset+r,offset:offset+r]
        u,s,vh=torch.linalg.svd(block.reshape(outputs,r*r),full_matrices=False)
        c=u[:,0]*s[0];h=vh[0].reshape(r,r);h=(h+h.T)/2
        eigen,rotation=torch.linalg.eigh(h)
        a=(z@basis@rotation).T
        groups.append((a,eigen,c));core_rank1.append(float(s[0].square()/s.square().sum()))
        offset+=r
    centers=torch.stack(centers)
    gap=min(float(abs(centers[i]-centers[j])) for i in range(len(ranks)) for j in range(i)) if len(ranks)>1 else float('inf')
    reconstructed=sum(dense(*cp(a[None],s[None],c[None])) for a,s,c in groups)
    return groups,dict(first_matrix_condition=float(torch.linalg.cond(first)),basis_condition=float(torch.linalg.cond(full_basis)),
                       eigen_cluster_centers=centers.real.tolist(),maximum_cluster_radius=max(widths),
                       minimum_cluster_gap=gap,maximum_eigen_imaginary=float(values.imag.abs().max()),
                       core_rank1_capture=core_rank1,
                       relative_reconstruction_error=float((forms-reconstructed).norm()/forms.norm()))


def control():
    start=time.perf_counter();torch.set_num_threads(2);torch.set_default_dtype(torch.float64)
    saved=torch.load('/dev/shm/bilin18_ll1_projected_recovery_v1.pt',weights_only=True,map_location='cpu')
    a,s,c=saved['planted'];truth=dense(*saved['target'])
    native=[dense(*cp(a[j:j+1],s[j:j+1],c[j:j+1])) for j in range(3)]
    rows=[]
    for noise in [0.,1e-5]:
        torch.manual_seed(2002);perturb=torch.randn_like(truth);perturb=(perturb+perturb.transpose(1,2))/2
        observed=truth+noise*truth.norm()*perturb/perturb.norm()
        groups,diagnostics=initialize(observed,[3,3,3])
        forms=[dense(*cp(aa[None],ss[None],cc[None])) for aa,ss,cc in groups]
        cosine=max(min(float((forms[j]*native[p[j]]).sum()/(forms[j].norm()*native[p[j]].norm())) for j in range(3)) for p in itertools.permutations(range(3)))
        rows.append(dict(noise_relative_frobenius=noise,clean_relative_frobenius=float((truth-sum(forms)).norm()/truth.norm()),
                         minimum_matched_group_cosine=cosine,**diagnostics))
    # Equal output loadings make two blocks indistinguishable by the pencil;
    # this must be visible as a collapsed between-group eigenvalue gap.
    tied=c.clone();tied[1]=tied[0]
    _,degenerate=initialize(dense(*cp(a,s,tied)),[3,3,3])
    result=dict(predictions=dict(pred_a_exact=rows[0]['clean_relative_frobenius']<=1e-8 and rows[0]['minimum_matched_group_cosine']>=.999,
                                 pred_b_conditions=rows[0]['minimum_cluster_gap']>=1e-6 and min(rows[0]['core_rank1_capture'])>=1-1e-10 and degenerate['minimum_cluster_gap']<=1e-6,
                                 pred_c_noise=rows[1]['clean_relative_frobenius']<=1e-3),
                rows=rows,collapsed_output_control=degenerate,seconds=time.perf_counter()-start,
                scope='Exact independent-input-block symmetric LL1 restriction; one small-noise control. Conditions not established for native model; no overcomplete/global noisy guarantee.')
    Path(__file__).with_name('SYMMETRIC_LL1_PENCIL_V1_CONTROL.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':control()
