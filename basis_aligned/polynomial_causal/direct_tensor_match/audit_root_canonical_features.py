"""Compare primitive products with canonical output-shared feature combinations."""
import itertools,json
from pathlib import Path
import torch
from scipy.optimize import linear_sum_assignment
from root_product_fit import coefficients
P=Path(__file__).resolve().parent

def main():
 torch.set_num_threads(1);fits=torch.load(P/'ROOT_IDENTITY_REPLICATION_V1.pt',weights_only=True)['fits'];metric=torch.load(P/'ROOT_ARCHIVE_METRIC_V1.pt',weights_only=True);G=metric['covariance'];S=metric['sqrt_covariance'];canonical={};spectra={};worst=None
 for seed,f in fits.items():
  H=coefficients(f['a'],f['b']);T=f['c']@H;U,sv,_=torch.linalg.svd(T@S,full_matrices=False);U=U[:,:4];canonical[seed]=(U,U.T@T);spectra[seed]=sv[:4].square().tolist()
 pairs=[]
 for i,j in itertools.combinations(range(8),2):
  f,g=fits[i],fits[j];H,J=coefficients(f['a'],f['b']),coefficients(g['a'],g['b']);normH=((H@G)*H).sum(1).sqrt();normJ=((J@G)*J).sum(1).sqrt();cos=(H@G@J.T)/(normH[:,None]*normJ[None,:]);ii,jj=linear_sum_assignment(-cos.abs().numpy());z=int(cos[ii,jj].abs().argmin());a,b=int(ii[z]),int(jj[z]);value=float(cos[a,b].abs());Tf=f['c']@H;Tg=g['c']@J
  if worst is None or value<worst['feature_cosine']:
   worst=dict(first_seed=i,second_seed=j,first_feature=a,second_feature=b,feature_cosine=value,first_component_energy_fraction=float(f['c'][:,a].square().sum()*normH[a]**2/((Tf@G)*Tf).sum()),second_component_energy_fraction=float(g['c'][:,b].square().sum()*normJ[b]**2/((Tg@G)*Tg).sum()))
  U,A=canonical[i];V,B=canonical[j];feature=(A@G@B.T)/(((A@G)*A).sum(1).sqrt()[:,None]*((B@G)*B).sum(1).sqrt()[None,:]);output=U.T@V;ii,jj=linear_sum_assignment(-output.abs().numpy());pairs.append(dict(first=i,second=j,canonical_output_cosines=output[ii,jj].abs().tolist(),canonical_feature_cosines=feature[ii,jj].abs().tolist(),output_subspace_minimum_cosine=float(torch.linalg.svdvals(U.T@V).min())))
 out=dict(worst_primitive=worst,output_mode_energies=spectra,pairs=pairs,minimum_canonical_output_cosine=min(min(r['canonical_output_cosines']) for r in pairs),minimum_canonical_feature_cosine=min(min(r['canonical_feature_cosines']) for r in pairs),scope='Post-hoc spectral canonicalization of centered output-shared combinations, exact for each fitted function. Each canonical feature is generally a sum of root products, not a newly unique primitive. Basis depends on Gaussian metric and output eigenvalue gaps; no semantic claim.');(P/'ROOT_CANONICAL_FEATURE_AUDIT_V1.json').write_text(json.dumps(out,indent=2)+'\n');print({k:v for k,v in out.items() if k not in ['pairs','output_mode_energies']})
if __name__=='__main__':main()
