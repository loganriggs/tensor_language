"""V1 rank-deficient private-basis repair; same registered experiment and price."""
import hashlib
import json
from pathlib import Path
import torch
from ll1_joint_parent_graph_v2 import factors
from global_reader_rebase_v1 import inspect, matrix_inner


def private_basis(a,u):
    residual=a-(a@u)[:,None]*u[None]
    vectors,singular,_=torch.linalg.svd(residual.T,full_matrices=False)
    basis=vectors[:,singular>singular.max()*1e-10]
    while basis.shape[1]<len(a):
        index=int((u.square()+basis.square().sum(1)).argmin())
        v=-u*u[index]-basis@basis[index]
        v[index]+=1
        # Reorthogonalize to remove accumulated rounding error.
        v-=u*(u@v)+basis@(basis.T@v)
        basis=torch.cat((basis,(v/v.norm())[:,None]),dim=1)
    assert float((basis.T@u).norm())<=1e-9
    return basis.T


@torch.no_grad()
def rebase(original,reader):
    a,s,c=factors(original);u=reader/reader.norm();groups=[]
    for aa,ss,cc in zip(a,s,c):
        private=private_basis(aa,u)
        ap=aa@private.T
        core=(ap.T*ss)@ap
        lam,rotation=torch.linalg.eigh(core)
        private=rotation.T@private
        au,av=aa@u,aa@private.T
        groups.append(dict(parent_ids=torch.tensor([0]),pair_ids=torch.tensor([0]),
            shared_coeff=(au.square()*ss).sum().reshape(1),private=private,lam=lam,
            cross=(2*((au*ss)@av)).reshape(1,-1),writer=cc))
    return dict(readers=u[None],pairs=torch.tensor([[0,0]]),groups=groups,
                output_whitener=original['output_whitener'],bias=original['bias'])


@torch.no_grad()
def main():
    torch.set_default_dtype(torch.float64);torch.set_num_threads(2)
    root=Path(__file__).parent;output=root/'GLOBAL_READER_REBASE_V2_RESULT.json'
    assert not output.exists()
    originals=[];sources={}
    for label in ('SPECTRAL','NATIVE'):
        path=root/f'SHARED_READER_JOINT_FIT_V1_{label}_GRAPH.pt'
        sources[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        originals.append(torch.load(path,weights_only=True,map_location='cpu'))
    rows=[]
    for parent in (1,9):
        u=originals[0]['readers'][parent];u=u/u.norm()
        pairs=[]
        for old in originals:
            graph=rebase(old,u)
            stats,matrix=inspect(old,graph)
            stats['maximum_private_shared_overlap']=max(float((g['private']@u).norm()) for g in graph['groups'])
            pairs.append((stats,matrix))
        a,b=[v[1] for v in pairs]
        aa,bb,ab=matrix_inner(u,a,u,a),matrix_inner(u,b,u,b),matrix_inner(u,a,u,b)
        rows.append(dict(spectral_reader=parent,scope_results=[v[0] for v in pairs],
            common_interface_function_cosine=float(ab/(aa*bb).sqrt()),
            relative_function_difference=float(((aa+bb-2*ab)/aa).clamp_min(0).sqrt())))
    valid=all(s['maximum_replay']<=1e-9 and s['maximum_private_shared_overlap']<=1e-9 for r in rows for s in r['scope_results'])
    result=dict(pred_a=valid,pred_b=valid and all(r['common_interface_function_cosine']>=.8 for r in rows),
        pred_c=valid and all(s['effective_consumers']>=2 for r in rows for s in r['scope_results']),
        rows=rows,sources=sources,
        scope='V1 orthogonal-completion repair. Exact consumer rewrite at fixed common readers; '
              'no fitting or price improvement. Agreement is not semantic identification.')
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
    for r in rows:
        print(json.dumps(dict(parent=r['spectral_reader'],cosine=r['common_interface_function_cosine'],
            relative_function_difference=r['relative_function_difference'],
            effective_consumers=[s['effective_consumers'] for s in r['scope_results']],
            maximum_replay=max(s['maximum_replay'] for s in r['scope_results']),
            maximum_orthogonality_error=max(s['maximum_private_shared_overlap'] for s in r['scope_results'])),indent=2))


if __name__=='__main__':main()
