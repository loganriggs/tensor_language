"""Fixed arithmetic graph, calibration-only amplitude and pair-difference fitting."""
import json,hashlib
from pathlib import Path
import torch
from extract_scalar_modes import evaluate,build
P=Path(__file__).resolve().parent;ROOT=P.parents[1]

def pairs(tokens):
 n= len(tokens);t=tokens.shape[1]-1;flat=tokens[:,:t].reshape(-1);doc=torch.arange(n).repeat_interleave(t);pos=torch.arange(t).repeat(n);eligible=pos>=16;same=torch.full_like(doc,-1)
 for token in torch.unique(flat[eligible]):
  ids=torch.where((flat==token)&eligible)[0];distance=(pos[ids,None]-pos[ids][None,:]).abs();distance[doc[ids,None]==doc[ids][None,:]]=10000;best=distance.argmin(1);valid=distance[torch.arange(len(ids)),best]<10000;same[ids[valid]]=ids[best[valid]]
 recipients=torch.where(same>=0)[0];donors=same[recipients];assert torch.all(doc[recipients]!=doc[donors]) and torch.all(flat[recipients]==flat[donors]);aligned=((doc[recipients]+1)%n)*t+pos[recipients]
 return dict(same_token=(recipients,donors),aligned=(recipients,aligned))

def main():
 torch.set_num_threads(3);torch.set_grad_enabled(False)
 base={k:v.double() for k,v in torch.load(P/'SELECTIVE_SCALAR_READOUT_V1.pt',weights_only=True)['program'].items()};view=torch.load(P/'CANONICAL_ROOT_FEATURES_V1.pt',weights_only=True);u=view['output_directions'].double()[:,1];mu=view['output_mean'].double();cal=torch.load(P/'NATIVE_QUARTIC_COVARIANCE_V1.pt',weights_only=True)['panels'][0];targets=torch.load(P/'NATIVE_VARIATION_AUDIT_V1.pt',weights_only=True);held=torch.load(P/'FROZEN_FRESH_VALIDATION_V1.pt',weights_only=True);cache=ROOT/'bilinear_quotient/.rowcache'
 # ROOT is basis_aligned: paths are relative to that common directory.
 ids=torch.load(cache/'fineweb_n96_skip1200.pt',weights_only=True)[:32,:65];tokenhash=hashlib.sha256(ids.numpy().tobytes()).hexdigest();assert tokenhash==cal['token_sha256']==targets['token_sha256'][0]
 panels=[('calibration',cal['rows'].double(),(targets['targets'][0].double()-mu)@u,ids)]
 for p in held['panels']:
  context=int(p['context']);tok=torch.load(cache/'fineweb_n192_skip7000.pt',weights_only=True)[32:64,:context+1];assert len(p['rows'])==32*context;panels.append((f'diagnostic_{context}',p['rows'].double(),(p['targets'].double()-mu)@u,tok))
 maps={n:pairs(tok) for n,x,y,tok in panels};x,y=panels[0][1:3];p=(x@base['A'].T)*(x@base['B'].T);h=(p@base['root_left'].T)*(p@base['root_right'].T);features=torch.cat([p,h],1);fm=features.mean(0);fs=features.std(0,unbiased=False);z=(features-fm)/fs;ym=y.mean();yc=y-ym;G=z.T@z/len(z);K=z.T@yc/len(z);models={'baseline':base};exports={};checks=[];prices=[]
 for family in ['same_token','aligned']:
  a,b=maps['calibration'][family];dz=z[b]-z[a];dy=y[b]-y[a];factor=yc.square().mean()/dy.square().mean();GP=dz.T@dz/len(a)*factor;KP=dz.T@dy/len(a)*factor
  for tau in [0.,.25,1.,4.]:
   coef=torch.linalg.solve((G+tau*GP)/(1+tau)+1e-4*torch.eye(10),(K+tau*KP)/(1+tau))/fs;s={k:v.clone() for k,v in base.items()};s['quadratic_readout'][1]=coef[:6];s['quartic_readout'][1]=coef[6:];s['constant'][1]=ym-fm@coef;key=f'{family}_{tau:g}';archive={k:v.float() for k,v in s.items()};model={k:v.double() for k,v in archive.items()};models[key]=model;exports[key]=archive
   checks.append(float((evaluate(s,x)-evaluate(model,x)).norm()/evaluate(s,x).norm()))
   if tau==0:checks.append(float((evaluate(model,x)-evaluate(base,x)).norm()/evaluate(base,x).norm()))
   dag,out=build(model);cost=dag.cost(out);assert cost['products']==10 and cost['stored_coefficients']==13916;checks.append(float((dag.evaluate(out,x[:8])-evaluate(model,x[:8])).norm()/evaluate(model,x[:8]).norm()));prices.append(dict(model=key,cost=cost))
 records=[]
 for name,xx,yy,tok in panels:
  a,b=maps[name]['same_token'];reference=evaluate(base,xx)
  for key,s in models.items():
   prediction=evaluate(s,xx);assert torch.equal(prediction[:,[0,2,3]],reference[:,[0,2,3]]);delta=prediction[b,1]-prediction[a,1]-(yy[b]-yy[a]);records.append(dict(panel=name,model=key,absolute_mse=float((prediction[:,1]-yy).square().mean()),absolute_relative_error=float((prediction[:,1]-yy).norm()/yy.norm()),pair_mse=float(delta.square().mean()),pair_relative_error=float(delta.norm()/(yy[b]-yy[a]).norm()),pairs=len(a)))
 def mse(n,k):return next(r['pair_mse'] for r in records if r['panel']==n and r['model']==k)
 pred=dict(pred_a_instrument=max(checks)<1e-5,pred_b_pairs=all(mse(n,'same_token_1')<=.9*mse(n,'baseline') for n in ['diagnostic_64','diagnostic_256']),pred_c_specificity=all(mse(n,'same_token_1')<=.9*mse(n,'aligned_1') for n in ['diagnostic_64','diagnostic_256']))
 result=dict(predictions=pred,records=records,prices=prices,replay_max=max(checks),calibration_token_sha256=tokenhash,primary='same_token_1',scope='Calibration-only data-informed paired polynomial metric; reused diagnostics, no native intervention or fresh/OOD claim.')
 torch.save(dict(programs=exports,teacher_scale=targets['teacher_scale'],primary='same_token_1'),P/'PAIRED_READOUT_PROGRAMS_V1.pt');(P/'PAIRED_READOUT_V1.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(pred));print(json.dumps(records,indent=2))
if __name__=='__main__':main()
