"""Exact Gaussian root metric for the rounded, exported six-product bank."""
import json
from pathlib import Path
import torch
from root_function_moments import root_moments
P=Path(__file__).resolve().parent

def main():
 torch.set_num_threads(1);s={k:v.double() for k,v in torch.load(P/'BANK_WIDTH_FRONTIER_V1.pt',weights_only=True)['programs'][6].items()};panel=torch.load(P/'NATIVE_QUARTIC_COVARIANCE_V1.pt',weights_only=True)['panels'][0];L=torch.linalg.cholesky(panel['covariance'].double());a=s['A']@L;b=s['B']@L;basis,_=torch.linalg.qr(torch.cat([a,b]).T,mode='reduced');A=a@basis;B=b@basis;Q=torch.einsum('vk,kij->vij',s['bank_writer'],.5*(A[:,:,None]*B[:,None,:]+B[:,:,None]*A[:,None,:]));m=basis.T@torch.linalg.solve_triangular(L,panel['mean'].double()[:,None],upper=False).flatten();mean,G=root_moments(Q,m);ev,E=torch.linalg.eigh(G);assert ev.min()>0;sqrtG=(E*ev.sqrt())@E.T;output_factor=torch.linalg.cholesky(s['W'].T@s['W']).T;T=output_factor@s['Z'];sv=torch.linalg.svdvals(T@sqrtG).square();reference=torch.load(P/'ROOT_FUNCTION_METRIC_V1.pt',weights_only=True);drift=float((G-reference['covariance']).norm()/reference['covariance'].norm());out=dict(coordinate_dimension=len(m),archive_covariance_relative_drift=drift,archive_mean_relative_drift=float((mean-reference['mean']).norm()/reference['mean'].norm()),output_frame_orthogonality_error=float((s['W'].T@s['W']-torch.eye(8)).norm()),centered_output_rank_retention_bounds=(sv.cumsum(0)/sv.sum()).tolist(),minimum_covariance_eigenvalue=float(ev.min()),scope='Metric computed from actual FP32 archived bank in exact12dim Gaussian subspace. Output frame Gram explicitly retained, not assumed identity. Pre-export comparison is diagnostic.');assert drift<1e-5;torch.save(dict(Q=Q,m=m,mean=mean,covariance=G,sqrt_covariance=sqrtG,output_factor=output_factor,weighted_root_writer=T),P/'ROOT_ARCHIVE_METRIC_V1.pt');(P/'ROOT_ARCHIVE_METRIC_V1.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
if __name__=='__main__':main()
