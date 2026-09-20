"""Export one training-selected program per registered writer penalty."""
import json
from pathlib import Path
import torch
from audit_centered_compact import evaluate
from export_centered_dag import build
P=Path(__file__).resolve().parent

def main():
 torch.set_num_threads(2);core=torch.load(P/'CENTERED_QUADRATIC_CAPACITY_V1.pt',weights_only=True);source=torch.load(P/'NATIVE_CENTERED_COMPACT_V1.pt',weights_only=True);s={k:v.double() for k,v in source['students'][core['source_key']].items()};fits=torch.load(P/'CENTERED_REGULARIZED_REFACTOR_V1.pt',weights_only=True)['fits'];result=json.load(open(P/'CENTERED_REGULARIZED_REFACTOR_V1.json'));panels=torch.load(P/'NATIVE_QUARTIC_COVARIANCE_V1.pt',weights_only=True)['panels'];targets=torch.load(P/'NATIVE_VARIATION_AUDIT_V1.pt',weights_only=True)['targets'];mean=s['constant']+core['output_frame']@(core['C']@(core['A']*core['B']).sum(1));exports={};rows=[]
 for penalty in [1e-4,.001,.01]:
  winner=min((r for r in result['records'] if r['penalty']==penalty),key=lambda r:r['relative_penalized_error']);key=(penalty,winner['optimizer'],winner['seed']);f=fits[key];a,b,c=f['a'],f['b'],f['c'];model={k:v.clone() for k,v in s.items()};writer=core['output_frame']@c;model.update(quadratic_left=a@core['input_mapback'],quadratic_right=b@core['input_mapback'],quadratic_writer=writer,constant=mean-writer@(a*b).sum(1));model={k:v.float() for k,v in model.items()};d,out=build(model);errors=[];variations=[];replay=[]
  for panel,y in zip(panels,targets):
   x=panel['rows'].double();p=evaluate(model,x);errors.append(float((p-y).norm()/y.norm()));variations.append(float(((p-p.mean(0))-(y-y.mean(0))).norm()/(y-y.mean(0)).norm()));replay.append(float((d.evaluate(out,x[:32])-p[:32]).norm()/p[:32].norm()))
  price=d.cost(out);assert price['stored_coefficients']==34560 and price['products']==4 and max(replay)<1e-10;rows.append(dict(source_key=list(key),retained_quadratic_energy=winner['retained_energy'],cancellation_ratio=winner['cancellation_ratio'],cost=price,full_quartic_errors=errors,centered_variation_errors=variations,archive_graph_replay=max(replay)));exports[penalty]=model;print(rows[-1],flush=True)
 chosen=result['selected_penalty_group']['penalty'] if result['selected_penalty_group'] else None;(P/'REGULARIZED_CENTERED_EXPORT_V1.json').write_text(json.dumps(dict(records=rows,penalty_selected_without_empirical_targets=chosen,scope='Bestpenalizedtrainingobjective perpenalty; priorregisteredgroupselectionpreserved. FP32exportedprograms,FP64replay. Noempiricalselection.'),indent=2)+'\n');torch.save(dict(programs=exports,teacher_scale=source['teacher_scale']),P/'REGULARIZED_CENTERED_EXPORT_V1.pt')
if __name__=='__main__':main()
