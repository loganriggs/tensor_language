"""Exact global interface for nonorthogonal shared readers.

A coefficients/executor/duals/oblique interventions/commutation/pairs <=1e-9.
B at least one orthogonal-vs-node removal differs by >=1% in each fitted graph.
C joint-removal cross-start cosine >=.8 (not semantic specificity).
"""
import hashlib
import json
from pathlib import Path
import torch
from ll1_joint_parent_graph_v2 import factors,execute,price
from shared_parent_intervention_v1 import banks,disable,concatenate,subset,value
from structured_branch_amplitudes_v1 import inner


def private_basis(a,frame):
    residual=a-(a@frame)@frame.T
    vectors,singular,_=torch.linalg.svd(residual.T,full_matrices=False)
    basis=vectors[:,singular>singular.max()*1e-10]
    while basis.shape[1]<len(a):
        index=int((frame.square().sum(1)+basis.square().sum(1)).argmin())
        v=-frame@frame[index]-basis@basis[index];v[index]+=1
        v-=frame@(frame.T@v)+basis@(basis.T@v)
        basis=torch.cat((basis,(v/v.norm())[:,None]),dim=1)
    assert float((basis.T@frame).norm())<=1e-9
    return basis.T


@torch.no_grad()
def rebase(original,readers):
    a,s,c=factors(original)
    u=readers.T
    frame=torch.linalg.qr(u,mode='reduced').Q
    dual=torch.linalg.solve(u.T@u,u.T).T
    k=len(readers);pairs=[(i,j) for i in range(k) for j in range(i,k)]
    groups=[]
    for aa,ss,cc in zip(a,s,c):
        private=private_basis(aa,frame);ap=aa@private.T
        lam,rotation=torch.linalg.eigh((ap.T*ss)@ap)
        private=rotation.T@private
        shared_a,private_a=aa@dual,aa@private.T
        shared=(shared_a.T*ss)@shared_a
        cross=2*(shared_a.T*ss)@private_a
        groups.append(dict(parent_ids=torch.arange(k),pair_ids=torch.arange(len(pairs)),
            shared_coeff=torch.stack([shared[i,j]*(1 if i==j else 2) for i,j in pairs]),
            private=private,lam=lam,cross=cross,writer=cc))
    return dict(readers=readers,pairs=torch.tensor(pairs),groups=groups,
                output_whitener=original['output_whitener'],bias=original['bias']),dual


@torch.no_grad()
def inspect(original,graph,dual):
    old,new=factors(original),factors(graph)
    checks=[]
    for aa,ss,bb,tt in zip(old[0],old[1],new[0],new[1]):
        qr=torch.linalg.qr(torch.cat((aa,bb)).T,mode='reduced').R
        difference=(qr*torch.cat((ss,-tt)))@qr.T
        norm=((aa@aa.T).square()*ss[:,None]*ss[None]).sum().sqrt()
        checks.append(float(difference.norm()/norm))
    readers=graph['readers'];k=len(readers)
    checks.append(float((readers@dual-torch.eye(k)).norm()))
    torch.manual_seed(4601);x=torch.randn(13,readers.shape[1])
    before,base=execute(original,x),execute(graph,x)
    checks.append(float((before-base).norm()/before.norm()))
    def transform(x,ids):return x-(x@readers[ids].T)@dual[:,ids].T
    effects=[];scope_differences=[]
    mixed,pair,nodes=banks(graph)
    for i in range(k):
        node_effect=base-disable(graph,x,[i])
        input_effect=base-execute(graph,transform(x,[i]))
        checks.extend([float((node_effect-input_effect).norm()/node_effect.norm()),
                       float((node_effect-value(nodes[i],x)).norm()/node_effect.norm())])
        orth=x-(x@readers[i])[:,None]*readers[i][None]
        orth_effect=base-execute(graph,orth)
        scope_differences.append(float((orth_effect-node_effect).norm()/node_effect.norm()))
        effects.append(node_effect)
    first=transform(transform(x,[0]),[1]);second=transform(transform(x,[1]),[0])
    joint_input=transform(x,[0,1])
    checks.extend([float((first-second).norm()/x.norm()),float((first-joint_input).norm()/x.norm())])
    actual=base-disable(graph,x,[0,1])
    checks.append(float((actual-(base-execute(graph,joint_input))).norm()/actual.norm()))
    offdiag=subset(pair,graph['pairs'][:,0]!=graph['pairs'][:,1])
    joint=concatenate(nodes[0],nodes[1],(offdiag[0],offdiag[1],-offdiag[2]))
    checks.append(float((actual-value(joint,x)).norm()/actual.norm()))
    naive=effects[0]+effects[1]
    return dict(maximum_replay=max(checks),single_orthogonal_scope_differences=scope_differences,
        naive_joint_relative_error=float((actual-naive).norm()/actual.norm()),
        pair_correction_over_joint_energy=float(inner(offdiag,offdiag)/inner(joint,joint)),
        price=price(graph,old)),joint


@torch.no_grad()
def main():
    torch.set_default_dtype(torch.float64);torch.set_num_threads(2)
    root=Path(__file__).parent;output=root/'GLOBAL_MULTI_PARENT_REBASE_V1_RESULT.json'
    assert not output.exists()
    paths=[root/f'SHARED_READER_JOINT_FIT_V1_{label}_GRAPH.pt' for label in ('SPECTRAL','NATIVE')]
    originals=[torch.load(p,weights_only=True,map_location='cpu') for p in paths]
    readers=originals[0]['readers'][[1,9]];readers/=readers.norm(dim=1,keepdim=True)
    rows=[];nodes=[]
    for old in originals:
        graph,dual=rebase(old,readers)
        row,node=inspect(old,graph,dual);rows.append(row);nodes.append(node)
    cosine=float(inner(*nodes)/(inner(nodes[0],nodes[0])*inner(nodes[1],nodes[1])).sqrt())
    valid=all(r['maximum_replay']<=1e-9 for r in rows)
    result=dict(pred_a=valid,pred_b=valid and all(max(r['single_orthogonal_scope_differences'])>=.01 for r in rows),
        pred_c=valid and cosine>=.8,reader_cosine=float(readers[0]@readers[1]),
        reader_gram_condition=float(torch.linalg.cond(readers@readers.T)),joint_function_cosine=cosine,
        rows=rows,sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        scope='Exact local quadratic multi-parent interface; node deletion is oblique input removal. '
              'Generic consistency, not semantic identification, native behavior, or cheaper execution.')
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
