"""Calibration-only empirical scalar decoder capacity diagnostic."""
import json
from pathlib import Path
import torch
from extract_scalar_modes import evaluate,build
P=Path(__file__).resolve().parent

def main():
 torch.set_num_threads(3);torch.set_grad_enabled(False)
 source=torch.load(P/'EXTRACTED_SCALAR_MODES_V1.pt',weights_only=True);base={k:v.double() for k,v in source['program'].items()}
 view=torch.load(P/'CANONICAL_ROOT_FEATURES_V1.pt',weights_only=True);u=view['output_directions'].double();mean=view['output_mean'].double()
 cal=torch.load(P/'NATIVE_QUARTIC_COVARIANCE_V1.pt',weights_only=True)['panels'][0];targets=torch.load(P/'NATIVE_VARIATION_AUDIT_V1.pt',weights_only=True);held=torch.load(P/'FROZEN_FRESH_VALIDATION_V1.pt',weights_only=True)
 assert cal['token_sha256']==targets['token_sha256'][0]
 assert abs(float(held['teacher_scale'])/float(targets['teacher_scale'])-1)<1e-6
 def target(y):return (y.double()-mean)@u
 panels=[('calibration',cal['rows'].double(),target(targets['targets'][0]))]+[(f'diagnostic_{p["context"]}',p['rows'].double(),target(p['targets'])) for p in held['panels']]
 x,y=panels[0][1:];ymean=y.mean(0);i,j=torch.triu_indices(6,6);identity=torch.eye(6,dtype=torch.float64)
 models={'frozen':base};exports={};fit_stats=[]
 for family in ['fixed','dense']:
  left,right=(base['root_left'],base['root_right']) if family=='fixed' else (identity[i],identity[j])
  p=(x@base['A'].T)*(x@base['B'].T);h=(p@left.T)*(p@right.T);features=torch.cat([p,h],1);fm=features.mean(0);fs=features.std(0,unbiased=False);assert fs.min()>0
  z=(features-fm)/fs;yc=y-ymean
  for ridge in [0.,1e-4,1e-2]:
   if ridge==0:theta=torch.linalg.lstsq(z,yc,driver='gelsd').solution
   else:theta=torch.linalg.solve(z.T@z/len(z)+ridge*torch.eye(z.shape[1]),z.T@yc/len(z))
   raw=theta/fs[:,None];constant=ymean-fm@raw
   model={k:base[k].clone() for k in ['A','B']};model.update(root_left=left,root_right=right,quadratic_readout=raw[:6].T,quartic_readout=raw[6:].T,constant=constant)
   key=f'{family}_{ridge:g}';archive={k:v.float() for k,v in model.items()};loaded={k:v.double() for k,v in archive.items()};exports[key]=archive;models[key]=loaded
   replay=max(float((evaluate(loaded,xx)-evaluate(model,xx)).norm()/evaluate(model,xx).norm()) for _,xx,_ in panels)
   assert replay<1e-5
   dag,out=build(loaded);fit_stats.append(dict(name=key,export_replay=replay,cost=dag.cost(out),design_singular_values=torch.linalg.svdvals(z).tolist()))
 records=[]
 for name,xx,yy in panels:
  for key,model in models.items():
   pred=evaluate(model,xx);sse=(pred-yy).square().sum(0);energy=yy.square().sum(0);variation=(yy-yy.mean(0)).square().sum(0)
   records.append(dict(panel=name,model=key,mode_sse=sse.tolist(),mode_relative_error=(sse/energy).sqrt().tolist(),mode_error_relative_centered_variation=(sse/variation).sqrt().tolist(),joint_relative_error=float((sse.sum()/energy.sum()).sqrt())))
  sse=(yy-ymean).square().sum(0);records.append(dict(panel=name,model='calibration_mean',mode_sse=sse.tolist(),mode_relative_error=(sse/yy.square().sum(0)).sqrt().tolist()))
 def row(panel,model):return next(r for r in records if r['panel']==panel and r['model']==model)
 instrument=all(row('calibration','dense_0')['mode_sse'][g]<=row('calibration','fixed_0')['mode_sse'][g]+1e-3 and row('calibration','fixed_0')['mode_sse'][g]<=row('calibration','frozen')['mode_sse'][g]+1e-3 for g in range(4))
 diagnostics=[name for name,_,_ in panels[1:]]
 predicates=dict(pred_a_instrument=instrument,pred_b_readout=all(row(n,'fixed_0.0001')['mode_sse'][1]<.9*row(n,'frozen')['mode_sse'][1] for n in diagnostics),pred_c_roots=all(row(n,'dense_0.0001')['mode_sse'][1]<.9*row(n,'fixed_0.0001')['mode_sse'][1] for n in diagnostics))
 result=dict(primary_ridge=1e-4,fit_stats=fit_stats,records=records,predictions=predicates,calibration_token_sha256=cal['token_sha256'],scope='Data-informed fits on original calibration only; disjoint panels reused diagnostics, not fresh confirmation. Dense decoder increases root capacity and product cost; no native intervention claim.')
 torch.save(dict(programs=exports,teacher_scale=targets['teacher_scale'],calibration_token_sha256=cal['token_sha256']),P/'PRIMITIVE_DECODER_PROGRAMS_V1.pt')
 (P/'PRIMITIVE_DECODER_V1.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(predicates));print(json.dumps([{k:r[k] for k in ['panel','model','mode_relative_error']} for r in records if r['model'] in ['frozen','fixed_0.0001','dense_0.0001','calibration_mean']],indent=2));print(json.dumps([{k:f[k] for k in ['name','cost','export_replay']} for f in fit_stats],indent=2))
if __name__=='__main__':main()
