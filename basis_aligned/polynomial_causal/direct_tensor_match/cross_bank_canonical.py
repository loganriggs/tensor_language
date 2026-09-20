"""Registered canonical-feature replication across both hierarchy levels."""
import itertools,json,time
from pathlib import Path
import torch
from scipy.optimize import linear_sum_assignment
from root_function_moments import root_moments
from root_product_fit import fit,coefficients
P=Path(__file__).resolve().parent

def modes(F,G):
 ev,E=torch.linalg.eigh((F@G@F.T+F@G.T@F.T)/2);order=torch.argsort(ev,descending=True)[:4];U=E[:,order];return U,U.T@F,ev[order]

def main():
 torch.set_num_threads(1);torch.set_default_dtype(torch.float64);start=time.perf_counter();bank=torch.load(P/'BANK_WIDTH_FRONTIER_V1.pt',weights_only=True);result=json.loads((P/'BANK_WIDTH_FRONTIER_V1.json').read_text());winner=next(r for r in result['winners'] if r['width']==6);assert winner['optimizer']=='muon' and winner['lr']==.03;other=1-winner['seed'];f=bank['allfits'][(6,'muon',.03,other)];core=torch.load(P/'QUARTIC_BANK_CORE_V1.pt',weights_only=True);base={k:v.double() for k,v in bank['programs'][6].items()};alt=dict(A=(f['a']@core['input_mapback']).float().double(),B=(f['b']@core['input_mapback']).float().double(),bank_writer=torch.linalg.solve(core['sqrt_root_sensitivity'],f['c']).float().double());panel=torch.load(P/'NATIVE_QUARTIC_COVARIANCE_V1.pt',weights_only=True)['panels'][0];L=torch.linalg.cholesky(panel['covariance'].double());basis,_=torch.linalg.qr(torch.cat([s[k]@L for s in [base,alt] for k in ['A','B']]).T,mode='reduced');m=basis.T@torch.linalg.solve_triangular(L,panel['mean'].double()[:,None],upper=False).flatten();Qs=[]
 for s in [base,alt]:
  a=s['A']@L@basis;b=s['B']@L@basis;Qs.append(torch.einsum('vk,kij->vij',s['bank_writer'],.5*(a[:,:,None]*b[:,None,:]+b[:,:,None]*a[:,None,:])))
 pairs=list(itertools.combinations_with_replacement(range(4),2))+list(itertools.combinations_with_replacement(range(4,8),2));means,G=root_moments(torch.cat(Qs),m,pairs);G0,G1,G01=G[:10,:10],G[10:,10:],G[:10,10:];reference=torch.load(P/'ROOT_ARCHIVE_METRIC_V1.pt',weights_only=True);drift=float((G0-reference['covariance']).norm()/G0.norm());assert drift<1e-8;factor=torch.linalg.cholesky(base['W'].T@base['W']).T;old={k:v.double() for k,v in torch.load(P/'ROOT_PRODUCT_REFACTOR_V1.pt',weights_only=True)['programs'][4].items()};F0=factor@old['root_writer']@coefficients(old['root_left'],old['root_right']);U0,D0,e0=modes(F0,G0);mean0=old['constant']+old['W']@old['root_writer']@coefficients(old['root_left'],old['root_right'])@means[:10];meanenergy=mean0.square().sum();energy0=((F0@G0)*F0).sum();records=[];fits={}
 for seed in range(4):
  info,(a,b,c)=fit(factor@base['Z'],G1,4,'muon',.03,seed,500);F1=c@coefficients(a,b);U1,D1,e1=modes(F1,G1);output=U0.T@U1;i,j=linear_sum_assignment(-output.abs().numpy());feature=(D0@G01@D1.T)/(((D0@G0)*D0).sum(1).sqrt()[:,None]*((D1@G1)*D1).sum(1).sqrt()[None,:]);cross=((F0@G01)*F1).sum();energy1=((F1@G1)*F1).sum();centered=float(cross/(energy0*energy1).sqrt());full=float((cross+meanenergy)/((energy0+meanenergy)*(energy1+meanenergy)).sqrt());row=dict(seed=seed,**info,canonical_feature_cosines=feature[i,j].abs().tolist(),canonical_output_cosines=output[i,j].abs().tolist(),centered_prediction_cosine=centered,complete_prediction_cosine=full,reference_mode_energies=e0.tolist(),alternate_mode_energies=e1.tolist());records.append(row);fits[seed]=dict(a=a,b=b,c=c);print(row,flush=True)
 best=min(r['penalized_objective'] for r in records)
 for r in records:r['relative_objective_gap']=(r['penalized_objective']-best)/abs(best)
 pred=dict(pred_feature=min(min(r['canonical_feature_cosines']) for r in records)>=.95,pred_output=min(min(r['canonical_output_cosines']) for r in records)>=.98,pred_prediction=min(r['complete_prediction_cosine'] for r in records)>=.995);out=dict(selected_bank_seed=winner['seed'],alternate_bank_seed=other,coordinate_dimension=len(m),reference_covariance_replay=drift,joint_covariance_minimum_eigenvalue=float(torch.linalg.eigvalsh(G).min()),records=records,predictions=pred,seconds=time.perf_counter()-start,scope='Exact cross-moments in common Gaussian input frame, actual FP32 archived bank readers. Root fits FP64 and final Gaussian mean matched analytically; selected deployed model unchanged. Complete and centered prediction cosines both reported. Not semantic or native causal identification.');torch.save(dict(alternate_bank=alt,fits=fits,joint_root_covariance=G,means=means,output_factor=factor),P/'CROSS_BANK_CANONICAL_V1.pt');(P/'CROSS_BANK_CANONICAL_V1.json').write_text(json.dumps(out,indent=2)+'\n');print(pred)
if __name__=='__main__':main()
