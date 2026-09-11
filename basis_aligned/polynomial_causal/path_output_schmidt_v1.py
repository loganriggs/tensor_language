"""Frozen path output/function Schmidt screen; no text fitting.
A Gram/reconstruction/spectral-tail identities<=1e-8.
B >=8 one-to-one top16 centered complete components cosine>=.95.
C centered leading8 groups total function cosine>=.95.
Other ranks, spectra and subspaces descriptive. No circuit promotion.
"""
import json
from pathlib import Path
import torch
from scipy.optimize import linear_sum_assignment
from sparse_path_coefficient_gram_v1 import gram
from sparse_path_stability_atlas_v1 import digest


@torch.no_grad()
def main():
    torch.set_num_threads(2);torch.set_default_dtype(torch.float64)
    p=Path(__file__).parent;out=p/'PATH_OUTPUT_SCHMIDT_V1.json';assert not out.exists()
    ap=p/'COUPLED_SPARSE_PATH_CONTINUE_V1_PROGRAMS.pt'
    receipt=json.loads((p/'COUPLED_SPARSE_PATH_CONTINUE_V1_RESULT.json').read_text())
    assert receipt['pred_a'] and receipt['pred_b'] and digest(ap)==receipt['artifact_sha256']
    programs=[a for a in torch.load(ap,weights_only=True,map_location='cpu')['programs'] if a['mode']=='joint']
    binding=json.loads((p/'COUPLED_SPARSE_PATH_CONTINUE_V1_BINDING.json').read_text())['files']
    ck=next(k for k in binding if k.endswith('/pytorch_model.bin'));assert digest(ck)==binding[ck]
    state=torch.load(ck,weights_only=True,mmap=True,map_location='cpu');u=state['lm_head.weight'].double()
    full=u.T@u;mean=u.mean(0);centered=full-len(u)*torch.outer(mean,mean)
    errors=[];reports=[];saved=[]
    for name,metric in [('full',full),('centered',centered)]:
        j=torch.linalg.cholesky(metric).T
        sides=[]
        for a in programs:
            _,h,_=gram(a,a,metric);errors.append(float((h-torch.eye(len(h))).norm()))
            c=j@a['physical_writer'];left,s,vh=torch.linalg.svd(c,full_matrices=False)
            errors.append(float((c-(left*s)@vh).norm()/c.norm()))
            w=torch.linalg.solve_triangular(j,left*s,upper=True)
            errors.append(float((w@vh-a['physical_writer']).norm()/a['physical_writer'].norm()))
            sides.append((left,s,vh.T,w,c))
            saved.append(dict(metric=name,seed=a['seed'],program=a,group_writer=w,edge_to_group=vh.T,singular_values=s))
        left0,s0,v0,w0,c0=sides[0];left1,s1,v1,w1,c1=sides[1]
        _,h01,_=gram(*programs,metric)
        output_cross=left0.T@left1;input_cross=v0.T@h01@v1
        cosine=output_cross*input_cross
        errors.append(max(0.,float(cosine.abs().max())-1))
        ii,jj=linear_sum_assignment(-cosine[:16,:16].numpy())
        ranks=[]
        for k in [1,2,4,8,16,32,96]:
            n0=s0[:k].square().sum();n1=s1[:k].square().sum()
            cross=(s0[:k,None]*cosine[:k,:k]*s1[None,:k]).sum()
            cos=float(cross/(n0*n1).sqrt())
            for c,left,s,v in [(c0,left0,s0,v0),(c1,left1,s1,v1)]:
                actual=(c-(left[:,:k]*s[:k])@v[:,:k].T).square().sum()
                errors.append(float((actual-s[k:].square().sum()).abs()/c.square().sum()))
            ranks.append(dict(rank=k,energy_fraction=[float(n0/s0.square().sum()),float(n1/s1.square().sum())],
                function_cosine=cos,symmetric_relative_coefficient_rms=float(((n0+n1-2*cross).clamp_min(0)/((n0+n1)/2)).sqrt()),
                min_output_principal_cosine=float(torch.linalg.svdvals(output_cross[:k,:k]).min()),
                min_input_function_principal_cosine=float(torch.linalg.svdvals(input_cross[:k,:k]).min()),
                relative_boundary_gaps=[float((s[k-1]-s[k])/s[0]) if k<len(s) else None for s in (s0,s1)],
                fitted_floats=36864+1152*k+96*k,products=96,group_accumulation_multiply_adds=96*k,
                scope='Shared input features and native background retained; storage includes dense group mixing.'))
        reports.append(dict(metric=name,singular_values=[s0.tolist(),s1.tolist()],top16_matching=list(zip(ii.tolist(),jj.tolist())),
            top16_matched_cosines=cosine[ii,jj].tolist(),matches_ge_point95=int((cosine[ii,jj]>=.95).sum()),ranks=ranks))
    artifact=p/'PATH_OUTPUT_SCHMIDT_V1.pt';assert not artifact.exists();torch.save(dict(groups=saved),artifact)
    centered_report=reports[1]
    result=dict(pred_a=max(errors)<=1e-8,pred_b=centered_report['matches_ge_point95']>=8,
        pred_c=next(r for r in centered_report['ranks'] if r['rank']==8)['function_cosine']>=.95,
        maximum_identity_error=max(errors),reports=reports,artifact_sha256=digest(artifact),source_sha256=digest(ap),
        checkpoint_sha256=binding[ck],script_sha256=digest(__file__),
        scope='Exact output/function SVD of frozen96edge joint path approximations. Does not refit the native tensor or improve its capture. Weight structure is not native circuit replication.')
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='reports'},indent=2))
    print(json.dumps([dict(metric=r['metric'],matches=r['matches_ge_point95'],ranks=r['ranks']) for r in reports],indent=2))
    assert result['pred_a']


if __name__=='__main__':main()
