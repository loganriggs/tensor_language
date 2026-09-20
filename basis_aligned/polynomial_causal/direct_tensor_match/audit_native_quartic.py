import itertools,json,sys
from pathlib import Path
import torch,numpy as np
from core import Model,normrows,terms,coefficients_from_dense
p=Path(__file__).resolve().parent;sys.path.insert(0,str(p.parent))
from quartic_pair_tree import factor,reconstruct,symmetrize

def matrix(packed,d):
 out=packed.new_zeros(len(packed),d,d)
 for k,(i,j) in enumerate(terms(d,2)):
  out[:,i,j]=packed[:,k]/(1 if i==j else 2)
  out[:,j,i]=out[:,i,j]
 return out

def main():
 torch.set_num_threads(1);r=json.loads((p/'NATIVE_QUARTIC_PILOT_V1.json').read_text());saved=torch.load(p/'NATIVE_QUARTIC_PILOT_V1.pt',map_location='cpu',weights_only=True)
 source=torch.load(p.parent.parent/'bilinear_quotient/circuits/followups/native_two_mlp_quartic_ht_v1r1.pt',map_location='cpu',weights_only=True)[0]['hessian_not_applicable_quartic'][0].double()/saved['normalizer'];target=symmetrize(source.numpy());reports=[]
 for kind in ['tree','dag']:
  for width in [1,2,4,8]:
   i=min((i for i,x in enumerate(r['records']) if x['kind']==kind and x['width']==width and x['metric']=='frobenius'),key=lambda i:r['records'][i]['relative_error']);row=r['records'][i];s=saved['students'][i]
   if kind=='tree':
    left=matrix(normrows(s['left']),5);right=matrix(normrows(s['right']),5);H=torch.einsum('oab,aij,bkl->oijkl',s['weight'].reshape(4,width,width),left,right)
   else:
    bank=matrix(normrows(s['bank']),5);ij=list(terms(width,2));H=sum(torch.einsum('o,ij,kl->oijkl',s['weight'][:,k],bank[a],bank[b]) for k,(a,b) in enumerate(ij))
   err=np.linalg.norm(symmetrize(H.numpy())-target)/np.linalg.norm(target);assert abs(err-row['relative_error'])<1e-10
   model=Model(5,4,kind,width,4);model.load_state_dict(s);assert float((coefficients_from_dense(H)-model()).abs().max())<1e-10
   reports.append(dict(kind=kind,width=width,values=row['parameter_values'],optimized_frobenius_error=err,optimizer=row['optimizer'],seed=row['seed']))
 baseline=[]
 for representative,t in [('raw',source.numpy()),('symmetric',target)]:
  for rank in [1,2,4,8]:
   h=reconstruct(factor(t,rank));err=np.linalg.norm(symmetrize(h)-target)/np.linalg.norm(target);baseline.append(dict(representative=representative,rank=rank,old_dense_values=2*25*rank+4*rank*rank,error=err))
 result=dict(optimized=reports,same_target_spectral=baseline,canonical_values=280,scope='One native quartic context; independent dense tensor symmetrization confirms coefficient loss. Native generators/norms excluded; not population or causal evidence.',correction='Pilot grid is96fits (2families*4widths*2optimizers*3seeds*2metrics), not192 stated in initial prereg prose.')
 out=p/'NATIVE_QUARTIC_AUDIT_V1.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
