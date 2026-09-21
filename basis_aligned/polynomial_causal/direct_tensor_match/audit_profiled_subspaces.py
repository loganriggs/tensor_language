"""Test whether unstable product atoms span a stable joint tensor subspace.
Primary: >=80% of principal cosines >=.9 in every comparison. CPU only.
This allows arbitrary mixing (potentially expensive), not an arithmetic rewrite.
"""
from pathlib import Path
import json
import torch
P=Path(__file__).resolve().parent
torch.set_num_threads(2)
torch.set_grad_enabled(False)

def gram(x,y):
    L,R,W=x; A,B,V=y
    return (W.T@V)*.5*((L.T@A)*(R.T@B)+(L.T@B)*(R.T@A))

def whiten(K):
    e,V=torch.linalg.eigh((K+K.T)/2)
    keep=e>e.max()*1e-12
    return V[:,keep]/e[keep].sqrt(), int(keep.sum())

def compare_grams(K,H,C):
    X,n=whiten(K); Y,m=whiten(H)
    singular=torch.linalg.svdvals(X.T@C@Y).clamp(0,1)
    # X,Y are coefficient maps of orthonormal bases; YY^T=H pseudoinverse.
    projected=C@Y
    retained=projected.square().sum(1)
    atom_fraction=retained.sum()/K.diag().sum()
    ones=torch.ones(K.shape[0],dtype=K.dtype)
    energy=ones@K@ones
    represented=(ones@projected).square().sum()
    return dict(rank_reference=n,rank_candidate=m,
        principal_cosines=singular.tolist(),
        median_principal_cosine=float(singular.median()),
        fraction_principal_above_090=float((singular>=.9).double().sum()/max(n,m)),
        atom_energy_fraction_in_other_span=float(atom_fraction),
        total_function_projection_error=float(((energy-represented).clamp_min(0)/energy).sqrt()))

def compare(x,y):
    return compare_grams(gram(x,x),gram(y,y),gram(x,y))

def main():
    g=torch.Generator().manual_seed(741)
    A=torch.randn(80,12,dtype=torch.double,generator=g)
    mixing=torch.eye(12,dtype=torch.double)+.1*torch.randn(12,12,dtype=torch.double,generator=g)
    B=A@mixing
    control=compare_grams(A.T@A,B.T@B,A.T@B)
    assert min(control['principal_cosines'])>1-1e-10
    d=torch.load(P/'SHARED_PRODUCT_NATIVE_INPUTS_V1.pt',weights_only=True)
    root=torch.linalg.inv(d['inverse_root'])
    programs=torch.load(P/'PROFILED_PARTIAL_GRAPH_PROGRAMS_V1.pt',weights_only=True)
    winner=json.loads((P/'PROFILED_PARTIAL_GRAPH_FIT_V1.json').read_text())['winner']
    def factors(p):
        p=p['shared_mixed']
        return root@p['left_reader'],root@p['right_reader'],p['product_weights'].T/d['scales'][:4,None]
    reference=factors(programs[winner])
    records=[]
    for name,p in programs.items():
        if name==winner: continue
        row=dict(candidate=name,**compare(reference,factors(p)))
        records.append(row)
    L,R,W=reference
    permutation=torch.randperm(L.shape[0],generator=g)
    signs=torch.where(torch.rand(L.shape[0],generator=g)>.5,1.,-1.).double()
    null=compare(reference,(L[permutation]*signs[:,None],R[permutation]*signs[:,None],W))
    out=dict(reference=winner,positive_control=control,coordinate_rotation_null=null,records=records,
       predictions=dict(pred_a_control=min(control['principal_cosines'])>1-1e-10,
          pred_b_stable_span=all(r['fraction_principal_above_090']>=.8 for r in records),
          pred_c_null_separation=null['fraction_principal_above_090']<.2),
       scope='Coefficient-space spans of joint output-weighted quadratic atoms. Allows arbitrary linear combinations, not necessarily cheap products. Near-start/rate comparisons only; no semantic or causal identification.')
    (P/'PROFILED_SUBSPACE_AUDIT_V1.json').write_text(json.dumps(out,indent=2)+'\n')
    for name,rows in [('fits',records),('null',[null])]:
        print(name,[{k:v for k,v in r.items() if k!='principal_cosines'} for r in rows])
    print(out['predictions'])
if __name__=='__main__': main()
