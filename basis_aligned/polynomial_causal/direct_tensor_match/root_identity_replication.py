"""Registered root4 identity replication; exported model remains frozen."""
import itertools,json,time
from pathlib import Path
import torch
from root_product_fit import fit
from audit_root_identity import compare
P=Path(__file__).resolve().parent

def main():
 torch.set_num_threads(1);torch.set_default_dtype(torch.float64);start=time.perf_counter();metric=torch.load(P/'ROOT_ARCHIVE_METRIC_V1.pt',weights_only=True);prior=torch.load(P/'ROOT_PRODUCT_REFACTOR_V1.pt',weights_only=True)['allfits'];old=json.loads((P/'ROOT_PRODUCT_REFACTOR_V1.json').read_text())['records'];fits={i:prior[(4,'muon',.03,i)] for i in [0,1]};records=[dict(next(r for r in old if r['width']==4 and r['optimizer']=='muon' and r['lr']==.03 and r['seed']==i)) for i in [0,1]]
 for seed in range(2,8):
  row,(a,b,c)=fit(metric['weighted_root_writer'],metric['covariance'],4,'muon',.03,seed,500);row.update(seed=seed,width=4,optimizer='muon',lr=.03);records.append(row);fits[seed]=dict(a=a,b=b,c=c);print(row,flush=True)
 pairs=[dict(first=i,second=j,**compare(fits[i],fits[j],metric['covariance'])) for i,j in itertools.combinations(range(8),2)];minimum=min(p['minimum_feature_cosine'] for p in pairs);median=float(torch.tensor([p['minimum_feature_cosine'] for p in pairs]).quantile(.5));outmin=min(p['matched_output_direction_minimum_cosine'] for p in pairs);function=min(p['centered_function_cosine'] for p in pairs);best=min(r['penalized_objective'] for r in records)
 for r in records:r['relative_objective_gap']=(r['penalized_objective']-best)/abs(best)
 predictions=dict(pred_function=function>=.995,pred_median_feature=median>=.85,pred_worst_feature=minimum>=.75,pred_worst_output=outmin>=.75);result=dict(records=records,pairs=pairs,minimum_function_cosine=function,median_minimum_feature_cosine=median,minimum_feature_cosine=minimum,minimum_output_direction_cosine=outmin,predictions=predictions,seconds=time.perf_counter()-start,scope='Identity replication only; selected/exported seed0 model unchanged. Eight root starts on one frozen bank, exact centered Gaussian metric, no empirical fitting or semantics.');torch.save(dict(fits=fits),P/'ROOT_IDENTITY_REPLICATION_V1.pt');(P/'ROOT_IDENTITY_REPLICATION_V1.json').write_text(json.dumps(result,indent=2)+'\n');print({k:v for k,v in result.items() if k not in ['records','pairs']})
if __name__=='__main__':main()
