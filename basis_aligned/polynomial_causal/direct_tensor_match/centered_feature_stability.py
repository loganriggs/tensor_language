import itertools,json,time
from pathlib import Path
import torch
from quadratic_student_fit import fit
P=Path(__file__).resolve().parent

def objects(a,b,c):
 q=.5*(a[:,:,None]*b[:,None,:]+b[:,:,None]*a[:,None,:]);q=q.flatten(1);components=c.T[:,:,None]*q[:,None,:];return q,components.flatten(1),components.sum(0).flatten()

def compare(first,second):
 q,t,f=first;r,u,g=second;q=q/q.norm(dim=1,keepdim=True);r=r/r.norm(dim=1,keepdim=True);t=t/t.norm(dim=1,keepdim=True);u=u/u.norm(dim=1,keepdim=True);cos=q@r.T;component=t@u.T;n=len(q);permutations=list(itertools.permutations(range(n)));best=max(permutations,key=lambda p:sum(abs(float(cos[i,j])) for i,j in enumerate(p)));cbest=max(permutations,key=lambda p:sum(float(component[i,j]) for i,j in enumerate(p)));matched=[abs(float(cos[i,j])) for i,j in enumerate(best)];qspan=torch.linalg.qr(q.T).Q;rspan=torch.linalg.qr(r.T).Q
 return dict(function_cosine=float(f@g/(f.norm()*g.norm())),mean_feature_cosine=sum(matched)/n,minimum_feature_cosine=min(matched),matched_feature_cosines=matched,mean_component_cosine=sum(float(component[i,j]) for i,j in enumerate(cbest))/n,minimum_subspace_cosine=float(torch.linalg.svdvals(qspan.T@rspan).min()))

def main():
 start=time.perf_counter();torch.set_num_threads(1);torch.set_default_dtype(torch.float64);core=torch.load(P/'CENTERED_QUADRATIC_CAPACITY_V1.pt',weights_only=True);C,A,B=[core[k] for k in ['C','A','B']];norm=float(core['core'].square().sum());previous=json.load(open(P/'CENTERED_CONTINUOUS_REFACTOR_V1.json'))['records'];fits={};rows=[];repeat=[]
 for opt,seed in itertools.product(['adam','muon'],range(4)):
  row,(a,b,c)=fit(C,A,B,4,torch.eye(16),opt,.03,seed,1000);row.update(optimizer=opt,seed=seed,retained_energy=row['gain']/norm);rows.append(row);fits[(opt,seed)]=dict(a=a,b=b,c=c)
  if seed<2:old=next(r for r in previous if r['width']==4 and r['optimizer']==opt and r['seed']==seed and r['lr']==.03);repeat.append(abs(row['retained_energy']-old['retained_quadratic_energy']))
  print(row,flush=True)
 first=next(iter(fits.values()));a,b,c=[first[k] for k in ['a','b','c']];perm=torch.tensor([2,0,3,1]);scale=torch.tensor([-3.,2.,-.5,4.]);control=compare(objects(a,b,c),objects(a[perm]*scale[:,None],b[perm],c[:,perm]/scale[None,:]));assert abs(control['function_cosine']-1)<1e-10 and abs(control['minimum_feature_cosine']-1)<1e-10 and abs(control['mean_component_cosine']-1)<1e-10
 pairs=[]
 for (ka,sa),(kb,sb) in itertools.combinations(fits.items(),2):pairs.append(dict(first=list(ka),second=list(kb),**compare(objects(sa['a'],sa['b'],sa['c']),objects(sb['a'],sb['b'],sb['c']))))
 median=float(torch.tensor([r['mean_feature_cosine'] for r in pairs]).quantile(.5));worst=min(r['minimum_feature_cosine'] for r in pairs);pred=dict(pred_a=max(repeat)<1e-8 and abs(control['function_cosine']-1)<1e-10,pred_b=min(r['function_cosine'] for r in pairs)>=.99,pred_c=median>=.9 and worst>=.8);out=dict(records=rows,pairs=pairs,gauge_control=control,max_repeat_energy_error=max(repeat),median_matched_feature_cosine=median,worst_individual_matched_feature_cosine=worst,predictions=pred,seconds=time.perf_counter()-start,scope='Correlatedrestartcomparisons withinfixednative-student quadraticcore. Featuregaugeaccounted; no semantic/OOD/datasplitclaim.');(P/'CENTERED_FEATURE_STABILITY_V1.json').write_text(json.dumps(out,indent=2)+'\n');torch.save(dict(fits=fits,core_source='CENTERED_QUADRATIC_CAPACITY_V1.pt'),P/'CENTERED_FEATURE_STABILITY_V1.pt');print('SUMMARY',pred,median,worst)
if __name__=='__main__':main()
