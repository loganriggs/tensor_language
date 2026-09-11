"""A dense execution B edge deletion C node union D two-node inclusion-exclusion <=1e-10.
Synthetic generic output metric and nonunit source scale; not behavioral validation.
"""
import json
from pathlib import Path
import torch
from sparse_path_program_v1 import run,edges
from coupled_sparse_path_v1 import coefficients
from check_coupled_sparse_path_v1 import features


def relative(a,b):return float((a-b).norm()/b.norm().clamp_min(1e-30))


def main():
    torch.set_default_dtype(torch.float64);torch.set_num_threads(2);torch.manual_seed(11301)
    p=Path(__file__).parent;out=p/'SPARSE_PATH_PROGRAM_V1_CONTROL.json';assert not out.exists();reports=[]
    left,right=torch.randn(2,11,5),torch.randn(2,11,5);d=torch.randn(4,11)
    u=torch.randn(9,4);root=torch.linalg.cholesky(u.T@u).T
    r,h=torch.randn(13,5),torch.randn(13,5);scale=2.7
    raw=torch.cat([r,scale*h],1);denominator=torch.rand(13)+.5
    for mode,count in [('joint',2),('independent',4)]:
        bank=torch.linalg.qr(torch.randn(count,5,2)).Q
        c=coefficients(left,right,root@d,bank,mode)
        # Keep every edge to make shared-node overlap a deliberately live control.
        support=torch.arange(c.shape[1]);writer=torch.linalg.solve_triangular(root,c,upper=True)
        program=dict(mode=mode,bank=bank,support=support,physical_writer=writer,source_scale=scale)
        result=run(program,r,h,denominator);feat=features(bank,mode)
        tensor=torch.einsum('oe,eij->oij',writer,feat)
        expected=torch.einsum('ni,oij,nj->no',raw,tensor,raw)
        a=max(relative(result['numerator'],expected),relative(result['write'],expected/denominator[:,None]))
        pair=edges(program);cross=torch.nonzero(pair[0]!=pair[1])[0].item();i,j=pair[:,cross].tolist()
        removed=run(program,r,h,denominator,zero_edges=(cross,))['write']
        contribution=result['amplitudes'][:,cross,None]*writer[:,cross]/denominator[:,None]
        b=relative(result['write']-removed,contribution)
        ni=run(program,r,h,denominator,zero_nodes=(i,))['write'];nj=run(program,r,h,denominator,zero_nodes=(j,))['write']
        union=run(program,r,h,denominator,zero_nodes=(i,j))['write']
        incident=((pair==i).any(0)|(pair==j).any(0));edge_union=run(program,r,h,denominator,zero_edges=tuple(torch.where(incident)[0].tolist()))['write']
        cc=relative(union,edge_union)
        overlap=((pair==i).any(0)&(pair==j).any(0));shared=result['amplitudes'][:,overlap]@writer[:,overlap].T/denominator[:,None]
        dd=relative(union-result['write'],ni+nj-2*result['write']+shared)
        assert shared.norm()>1e-4,'Dead common-edge tripwire'
        # A scaled source can be absorbed into its corresponding reader maps, exactly.
        adjusted=bank.clone()
        adjusted[[1] if mode=='joint' else [2,3]]*=scale
        alternative=dict(program,bank=adjusted,source_scale=1.)
        scale_error=relative(run(alternative,r,h,denominator)['write'],result['write']);a=max(a,scale_error)
        reports.append(dict(mode=mode,execution_error=a,edge_removal_error=b,node_union_error=cc,composition_error=dd,common_edge_write_norm=float(shared.norm())))
    result=dict(pred_a=max(r['execution_error'] for r in reports)<=1e-10,pred_b=max(r['edge_removal_error'] for r in reports)<=1e-10,
        pred_c=max(r['node_union_error'] for r in reports)<=1e-10,pred_d=max(r['composition_error'] for r in reports)<=1e-10,reports=reports,
        scope='Synthetic exact sparse path execution and internal node/edge edit semantics. Shared denominator externally supplied. No native behavioral result or standalone extraction.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));assert all(result[k] for k in ('pred_a','pred_b','pred_c','pred_d'))

if __name__=='__main__':main()
