"""Input coverage and empirical restart-function stability, no teacher targets."""
import itertools,json,time
from pathlib import Path
import torch
P=Path(__file__).resolve().parent

def main():
 torch.set_num_threads(4);torch.set_default_dtype(torch.float64);start=time.perf_counter();models=torch.load(P/'NATIVE_WEIGHTED_BANK_V1.pt',weights_only=True)['students'];panels=torch.load(P/'NATIVE_QUARTIC_COVARIANCE_V1.pt',weights_only=True)['panels'];mu=panels[0]['mean'].double();x=panels[1]['rows'].double();i,j=torch.triu_indices(4,4);rows=[];predictions={}
 for key,s in models.items():
  U,V,C=[s[k].double() for k in ['U','V','C']];vectors=torch.cat([U.flatten(0,1),V.flatten(0,1)]);_,singular,frame=torch.linalg.svd(vectors,full_matrices=False);rank=int((singular>singular[0]*1e-6).sum());basis=frame[:rank].T;projection=float((basis.T@mu).square().sum()/mu.square().sum());q=((x@U.flatten(0,1).T)*(x@V.flatten(0,1).T)).reshape(len(x),4,4).sum(2);pred=(q[:,i]*q[:,j])@C.T;predictions[key]=pred;center=pred-pred.mean(0);rows.append(dict(metric=key[0],optimizer=key[1],seed=key[2],reader_rank_at_relative_1e6=rank,mean_input_energy_captured=projection,prediction_mean_energy_fraction=float(len(x)*pred.mean(0).square().sum()/pred.square().sum()),prediction_centered_norm=float(center.norm()),prediction_total_norm=float(pred.norm())))
 pairs=[]
 for metric in sorted({k[0] for k in models}):
  keys=[k for k in models if k[0]==metric]
  for a,b in itertools.combinations(keys,2):
   u,v=predictions[a],predictions[b];uc=u-u.mean(0);vc=v-v.mean(0);pairs.append(dict(metric=metric,first=list(a[1:]),second=list(b[1:]),empirical_function_cosine=float((u*v).sum()/u.norm()/v.norm()),centered_function_cosine=float((uc*vc).sum()/uc.norm()/vc.norm())))
 out=dict(records=rows,restart_pairs=pairs,seconds=time.perf_counter()-start,scope='Diagnostic calibration-mean reader coverage and secondpanel prediction stability. No teacher targets; highcosine doesnot imply correctvariation or featureidentity. Rankcutoff explicit.')
 (P/'WEIGHTED_READER_AUDIT_V1.json').write_text(json.dumps(out,indent=2)+'\n')
 for metric in sorted({r['metric'] for r in rows}):
  rs=[r for r in rows if r['metric']==metric];ps=[r for r in pairs if r['metric']==metric];print(metric,'mean coverage',[r['mean_input_energy_captured'] for r in rs],'prediction mean fraction',[r['prediction_mean_energy_fraction'] for r in rs],'centeredcos range',min(r['centered_function_cosine'] for r in ps),max(r['centered_function_cosine'] for r in ps))
if __name__=='__main__':main()
