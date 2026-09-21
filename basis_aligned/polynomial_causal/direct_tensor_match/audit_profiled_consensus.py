"""CPU consequence of unstable full spans: find common combinations.
Select eigenvectors of mean alternate projection with mean retention >=.81.
Screen: retain >=95% fitted-function energy, <=16 directions, and >=.81
retention in EACH alternate for every retained eigenvector. Not semantic units.
"""
import json
from pathlib import Path
import torch
from audit_profiled_subspaces import gram,whiten
P=Path(__file__).resolve().parent

def main():
    d=torch.load(P/'SHARED_PRODUCT_NATIVE_INPUTS_V1.pt',weights_only=True)
    root=torch.linalg.inv(d['inverse_root'])
    ps=torch.load(P/'PROFILED_PARTIAL_GRAPH_PROGRAMS_V1.pt',weights_only=True)
    winner=json.loads((P/'PROFILED_PARTIAL_GRAPH_FIT_V1.json').read_text())['winner']
    def f(p):
        p=p['shared_mixed'];return root@p['left_reader'],root@p['right_reader'],p['product_weights'].T/d['scales'][:4,None]
    ref=f(ps[winner]);K=gram(ref,ref);X,_=whiten(K)
    projects=[]
    for key,p in ps.items():
        if key==winner:continue
        other=f(p);Y,_=whiten(gram(other,other));cross=X.T@gram(ref,other)@Y
        projects.append(cross@cross.T)
    mean=torch.stack(projects).mean(0)
    eig,V=torch.linalg.eigh(mean);order=torch.arange(len(eig)-1,-1,-1);eig=eig[order];V=V[:,order]
    take=eig>=.81;S=V[:,take]
    coords=X.T@K@torch.ones(K.shape[0],dtype=K.dtype)
    energy=coords.square().sum();captured=(S.T@coords).square().sum()/energy
    min_each=[float(torch.diag(S.T@A@S).min()) if S.shape[1] else 0. for A in projects]
    # Stronger all-linear-combinations control: smallest restricted projection eigenvalue.
    minimum_subspace=[float(torch.linalg.eigvalsh(S.T@A@S).min()) if S.shape[1] else 0. for A in projects]
    coeff=X@S@(S.T@coords)
    # A fixed combination of original joint atoms remains executable at same product count.
    out=dict(reference=winner,retained_directions=int(take.sum()),top_mean_retention=eig[:20].tolist(),function_energy_fraction=float(captured),
         leading_direction_function_energy_fractions=((V.T@coords).square()/energy)[:20].tolist(),
         nonnegligible_atom_coefficients=int((coeff.abs()>1e-8).sum()),
         per_alternate_minimum_direction_retention=min_each,per_alternate_minimum_subspace_retention=minimum_subspace,
         reconstruction_error=float((1-captured).clamp_min(0).sqrt()),
         predictions=dict(pred_a_small=int(take.sum())<=16,pred_b_function=float(captured)>=.95,pred_c_each=min(min_each)>=.81),
         scope='Consensus of already fitted joint tensors. Stable coefficient groups are not cheap scalar product nodes; all256 original products may still be required. No held-out semantics or original-target fidelity claim.')
    torch.save(dict(atom_coefficients=coeff,reference=winner),P/'PROFILED_CONSENSUS_V1.pt')
    (P/'PROFILED_CONSENSUS_V1.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))
if __name__=='__main__':main()
