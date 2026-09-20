import json,time
from pathlib import Path
import torch
P=Path(__file__).resolve().parent

def entries(Q,R,ix):
 i,j,k,l=ix.T
 return (Q[i,j]*R[k,l]+R[i,j]*Q[k,l]+Q[i,k]*R[j,l]+R[i,k]*Q[j,l]+Q[i,l]*R[j,k]+R[i,l]*Q[j,k])/6

def main():
 torch.set_num_threads(4);torch.set_default_dtype(torch.float64);out=P/'NATIVE_ROOT_SPECTRAL_V1.json';assert not out.exists();start=time.perf_counter();data=torch.load(P/'NATIVE_HIERARCHICAL_ROOT_V1.pt',weights_only=True);A=data['A'].double();B=data['B'].double();s=data['students'][8];writer=s['writer'].double();d=A.shape[1];matrices=[];spectra=[];eigenvectors=[];descriptions=[]
 for side in ['left','right']:
  for root,weights in enumerate(s[side].double()):
   raw=A.T@(weights[:,None]*B);Q=(raw+raw.T)/2;ev,V=torch.linalg.eigh(Q);matrices.append(Q);spectra.append(ev);eigenvectors.append(V);thresholds={}
   for eps in [1e-10,1e-6,1e-4,1e-2]:
    cut=eps*ev.abs().max();p=int((ev>cut).sum());n=int((ev< -cut).sum());thresholds[str(eps)]=dict(positive=p,negative=n,width_bound=max(p,n))
   descriptions.append(dict(side=side,root=root,thresholds=thresholds));print(side,root,thresholds['1e-06'],flush=True)
 torch.manual_seed(1732);ix=torch.randint(d,(8192,4));x=torch.randn(256,d);m=(x@A.T)*(x@B.T);source=((m@s['left'].double().T)*(m@s['right'].double().T))@writer.T
 def evaluate(mats):
  roots=torch.stack([entries(mats[k],mats[k+8],ix) for k in range(8)],1);values=torch.stack([((x@mats[k])*x).sum(1)*((x@mats[k+8])*x).sum(1) for k in range(8)],1);return roots@writer.T,values@writer.T
 coefficient,full=evaluate(matrices);replay=float((source-full).norm()/source.norm());assert replay<1e-6
 records=[]
 for k in [16,32,64,128,256,512]:
  approx=[];errors=[]
  for Q,ev,V in zip(matrices,spectra,eigenvectors):
   pos=torch.where(ev>0)[0].flip(0)[:k];neg=torch.where(ev<0)[0][:k];keep=torch.cat([pos,neg]);S=(V[:,keep]*ev[keep])@V[:,keep].T;approx.append(S);errors.append(float((S-Q).norm()/Q.norm()))
  c,f=evaluate(approx);row=dict(bilinear_width_per_quadratic=k,quadratic_errors=errors,coefficient_error_vs_exported_program=float((c-coefficient).norm()/coefficient.norm()),gaussian_error_vs_exported_program=float((f-full).norm()/full.norm()),standalone_independent_parameters=4*8*k*d+8*d,shared_bank_parameters=2*A.numel()+2*8*A.shape[0]+8*d);records.append(row);print(json.dumps({key:value for key,value in row.items() if key!='quadratic_errors'}),flush=True)
 out.write_text(json.dumps(dict(quadratics=descriptions,records=records,full_matrix_replay_error=replay,seconds=time.perf_counter()-start,scope='CPU spectral compression of exported8root approximation, not original native teacher. Finite coefficient/function diagnostics; independently factored quadratic price sacrifices sharedbank reuse. Inertia counts are threshold-dependent numerical counts.'),indent=2)+'\n')
if __name__=='__main__':main()
