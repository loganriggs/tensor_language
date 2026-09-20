"""One bounded warm-start diagnostic for the failed alternate-bank root seed."""
import json
from pathlib import Path
import torch
from scipy.optimize import linear_sum_assignment
from root_product_fit import fit,coefficients
from cross_bank_canonical import modes
P=Path(__file__).resolve().parent

def main():
 torch.set_num_threads(1);torch.set_default_dtype(torch.float64);prior=torch.load(P/'CROSS_BANK_CANONICAL_V1.pt',weights_only=True);records=json.loads((P/'CROSS_BANK_CANONICAL_V1.json').read_text())['records'];G=prior['joint_root_covariance'];G0,G1,G01=G[:10,:10],G[10:,10:],G[:10,10:];factor=prior['output_factor'];s={k:v.double() for k,v in torch.load(P/'BANK_WIDTH_FRONTIER_V1.pt',weights_only=True)['programs'][6].items()};old={k:v.double() for k,v in torch.load(P/'ROOT_PRODUCT_REFACTOR_V1.pt',weights_only=True)['programs'][4].items()};F0=factor@old['root_writer']@coefficients(old['root_left'],old['root_right']);U0,D0,_=modes(F0,G0);f=prior['fits'][0];row,(a,b,c)=fit(factor@s['Z'],G1,4,'muon',.005,0,1500,(f['a'],f['b']));F1=c@coefficients(a,b);U1,D1,_=modes(F1,G1);output=U0.T@U1;i,j=linear_sum_assignment(-output.abs().numpy());feature=(D0@G01@D1.T)/(((D0@G0)*D0).sum(1).sqrt()[:,None]*((D1@G1)*D1).sum(1).sqrt()[None,:]);best=min(r['penalized_objective'] for r in records);start=next(r for r in records if r['seed']==0)['penalized_objective'];gap=row['penalized_objective']-best;cross=((F0@G01)*F1).sum();e0=((F0@G0)*F0).sum();e1=((F1@G1)*F1).sum();out=dict(fit=row,original_objective_gap=start-best,final_objective_gap=gap,canonical_feature_cosines=feature[i,j].abs().tolist(),canonical_output_cosines=output[i,j].abs().tolist(),centered_prediction_cosine=float(cross/(e0*e1).sqrt()),predictions=dict(pred_gap=gap<=.25*(start-best),pred_feature=float(feature[i,j].abs().min())>=.95,pred_output=float(output[i,j].abs().min())>=.98),scope='One preregistered warm-start refinement with reset optimizer. Original cross-bank all-start result remains failed; selected model unchanged.');torch.save(dict(a=a,b=b,c=c),P/'CROSS_BANK_OPTIMIZATION_CONTROL_V1.pt');(P/'CROSS_BANK_OPTIMIZATION_CONTROL_V1.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
if __name__=='__main__':main()
