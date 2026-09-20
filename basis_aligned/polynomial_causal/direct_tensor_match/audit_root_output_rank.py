"""Optimal output-rank relaxation of each EXPORTED student, not the native teacher."""
import json
from pathlib import Path
import torch
from shared_quadratic_bank import bank_gram,bank_entries
P=Path(__file__).resolve().parent

def main():
 torch.set_num_threads(1);torch.set_default_dtype(torch.float64)
 source=torch.load(P/'NATIVE_LEARNED_SHARED_BANK_V1.pt',weights_only=True)['students'];rows=[]
 for key,s in source.items():
  U,V,C=[s[n].double() for n in ['U','V','C']];G=bank_gram(U,V);ev,B=torch.linalg.eigh((G+G.T)/2);assert float(ev.min())>0
  root=B*ev.sqrt();matrix=C@root;W,sigma,Z=torch.linalg.svd(matrix,full_matrices=False);energy=sigma.square();retained=float(energy[:8].sum()/energy.sum());writer=(W[:,:8]*sigma[:8])@Z[:8]@torch.linalg.inv(root);delta=C-writer;direct_error=float((((delta.T@delta)*G).sum()/((C.T@C)*G).sum()).sqrt());assert abs(direct_error-(1-retained)**.5)<1e-10
  rows.append(dict(optimizer=key[0],lr=key[1],seed=key[2],retained_student_energy=retained,relative_student_error=direct_error,squared_singular_values=energy.tolist(),root_gram_condition=float(torch.linalg.cond(G)),reduced_float_parameters=46080+80,root_interactions=10,additional_mixing_scalars=80))
 out=dict(records=rows,scope='Exact output-rank8 approximation of each exported fullbank student in coefficient norm. Ten root interactions retained and mixed; not an eight-root program, not native objective refit.46,160reduced scalars vs46,080CP8.')
 (P/'ROOT_OUTPUT_RANK_V1.json').write_text(json.dumps(out,indent=2)+'\n')
 for r in rows:print(r['optimizer'],r['lr'],r['seed'],r['retained_student_energy'],r['root_gram_condition'])
if __name__=='__main__':main()
