import itertools,json,time,math
from pathlib import Path
import torch
from quadratic_student_fit import normalize,gram,cross
from centered_feature_stability import objects,compare
P=Path(__file__).resolve().parent

def main():
 start=time.perf_counter();torch.set_num_threads(1);torch.set_default_dtype(torch.float64);core=torch.load(P/'CENTERED_QUADRATIC_CAPACITY_V1.pt',weights_only=True);C,A,B=[core[k] for k in ['C','A','B']];norm=float(core['core'].square().sum());source=torch.load(P/'CENTERED_FEATURE_STABILITY_V1.pt',weights_only=True)['fits'];records=[];fits={};checks=[]
 for penalty,(key,initial) in itertools.product([1e-4,.001,.01],source.items()):
  a=torch.nn.Parameter(initial['a'].clone());b=torch.nn.Parameter(initial['b'].clone());opt=torch.optim.Adam([a,b],lr=.005) if key[0]=='adam' else torch.optim.Muon([a,b],lr=.005,weight_decay=0.,adjust_lr_fn='match_rms_adamw');best=float('inf');saved=None
  for step in range(601):
   opt.zero_grad();u,v=normalize(a,b);G=gram(u,v);X=cross(C,A,B,u,v);writer=torch.linalg.solve(G+penalty*torch.eye(4),X.T).T;unreg=((writer@G)*writer).sum()-2*(writer*X).sum();loss=unreg+penalty*writer.square().sum()
   if float(loss.detach())<best:best=float(loss.detach());saved=(u.detach().clone(),v.detach().clone(),writer.detach().clone());beststep=step
   if step==600:break
   (loss/norm).backward();opt.step()
   for group in opt.param_groups:group['lr']=.005*(.01+.99*.5*(1+math.cos(math.pi*(step+1)/600)))
  u,v,w=saved;q,t,f=objects(u,v,w);G=gram(u,v);X=cross(C,A,B,u,v);unreg=float(((w@G)*w).sum()-2*(w*X).sum());normal=float((w@(G+penalty*torch.eye(4))-X).norm()/X.norm());checks.append(max(normal,float((q.norm(dim=1)-1).abs().max())));row=dict(penalty=penalty,optimizer=key[0],seed=key[1],retained_energy=-unreg/norm,cancellation_ratio=float(t.square().sum()/f.square().sum()),gram_condition=float(torch.linalg.cond(G)),normal_residual=normal,selected_step=beststep,relative_penalized_error=1+best/norm);records.append(row);fits[(penalty,*key)]=dict(a=u,b=v,c=w);print(row,flush=True)
 groups=[]
 for penalty in [1e-4,.001,.01]:
  rs=[r for r in records if r['penalty']==penalty];ps=[objects(s['a'],s['b'],s['c']) for key,s in fits.items() if key[0]==penalty];pairs=[compare(a,b) for a,b in itertools.combinations(ps,2)];med=lambda values:float(torch.tensor(values).quantile(.5));groups.append(dict(penalty=penalty,median_retention=med([r['retained_energy'] for r in rs]),minimum_retention=min(r['retained_energy'] for r in rs),median_cancellation=med([r['cancellation_ratio'] for r in rs]),median_component_cosine=med([r['mean_component_cosine'] for r in pairs]),worst_feature_cosine=min(r['minimum_feature_cosine'] for r in pairs),minimum_function_cosine=min(r['function_cosine'] for r in pairs)))
 eligible=[r for r in groups if r['median_retention']>=.95 and r['median_cancellation']<=10];chosen=min(eligible,key=lambda r:r['penalty']) if eligible else None;pred=dict(pred_a=max(checks)<1e-9,pred_b=bool(eligible),pred_c=chosen is not None and chosen['median_component_cosine']>=.85 and chosen['worst_feature_cosine']>=.8);result=dict(records=records,groups=groups,selected_penalty_group=chosen,predictions=pred,seconds=time.perf_counter()-start,scope='Warmstartedfour-productnormalizedwriterpenalty, all8starts retained. Noempiricalselection or semanticclaim.');(P/'CENTERED_REGULARIZED_REFACTOR_V1.json').write_text(json.dumps(result,indent=2)+'\n');torch.save(dict(fits=fits,core_source='CENTERED_QUADRATIC_CAPACITY_V1.pt'),P/'CENTERED_REGULARIZED_REFACTOR_V1.pt');print('GROUPS',groups,'PREDICTIONS',pred)
if __name__=='__main__':main()
