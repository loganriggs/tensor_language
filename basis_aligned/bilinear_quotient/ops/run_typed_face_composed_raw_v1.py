#!/usr/bin/env python3
# BQGATE:120bodyforwards;40prefixes;300seconds;no fitting.
"""pred_a write<=1e-4; pred_b readout<=1e-5abs/1e-6rel, anchors<=1e-5.
pred_c each effect<=1e-3, norm>=1e-6.120forwards. Exact algebra, FP32 replay test.
"""
import hashlib,json,os,signal,sys,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal'
sys.path[:0]=[str(Path(__file__).parent),str(P),str(ROOT)]
from regional_endpoint_batching_v1 import group_rows,expand
from regional_paired_write_runtime_v3 import measure
from run_even_value_factorial_native_v1 import setup
from typed_face_write_atoms_v1 import native
D=P/'extracted_circuits/typed_face_composed_raw_v1'
sys.path.insert(0,str(D))
from execute import Program
STEM='TYPED_FACE_COMPOSED_RAW_V1'
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
@torch.no_grad()
def main():
 binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files'];assert all(digest(k)==v for k,v in binding.items())
 rows=json.loads((P/'TYPED_FACE_PROSPECTIVE_V1_ROWS.json').read_text())['rows'];groups,mapping=group_rows(rows)
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
  assert len(groups)==40;print('120bodyforwards;40prefixes;combined three-state head8/head9 write');return
 out=P/(STEM+'_RESULT.json');assert not out.exists();start=time.perf_counter();signal.alarm(300);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
 from fastload import load_model_fast
 model=load_model_fast().cuda().eval();graph,_,_,_=setup('cuda');p={k:v.to('cuda') for k,v in torch.load(P/'TYPED_FACE_KEY_SOURCE_FRESH_V1_PROGRAM.pt',weights_only=True).items()}
 errors=[];fixtures=[];state={}
 used=['q1','q2','k1','k2','key_coordinates','current_value','output','mixture'];p9={k:graph.p[k] for k in used}
 combined=Program(p,p9,model.transformer.h[9].lambdas[0])
 def write(arm,row,donor_row,current,donor,mask):
  city=row['city_position'];state.update(current=current,donor=donor[:,city],rt=row['ids'][city],dt=donor_row['ids'][city],city=city)
  return .5*native.execute(p,current,donor[:,city],row['ids'][city],donor_row['ids'][city],city,mask)
 def odd_write(arm,g,raw,delta,lam,mask,original):
  if arm=='original':return original
  new=combined.execute(state['current'],state['donor'],raw,state['rt'],state['dt'],state['city'],mask);errors.append(float((new-original).norm()/original.norm()))
  fixtures.append({'raw':raw.detach().cpu(),'current':state['current'].cpu(),'donor':state['donor'].cpu(),'rt':state['rt'],'dt':state['dt'],'city':state['city'],'mask':mask.cpu(),'expected':original.cpu()})
  return new
 m=measure(model,graph,groups,['native','original','reduced'],write,odd_write);v=expand(m['values'],mapping,6)
 old=torch.load(P/'TYPED_FACE_PROSPECTIVE_V1_ARTIFACT.pt',weights_only=True)['values'][[0,3]];anchor=float((v[:2]-old).abs().max())
 diff=v[2]-v[1];effect=v[1]-v[0];relative=float(diff.norm()/v[1].norm());effect_errors=(diff.norm(dim=0)/effect.norm(dim=0)).tolist()
 program={'head8':{k:v.detach().cpu().clone() for k,v in p.items()},'head9':{k:v.detach().cpu().clone() for k,v in p9.items()},'lambda90':combined.lambda90.detach().cpu().clone()}
 torch.save(program,D/'program.pt');torch.save(fixtures,P/(STEM+'_FIXTURES.pt'));torch.save({'values':v},P/(STEM+'_ARTIFACT.pt'))
 result={'pred_a':max(errors)<=1e-4 and bool(torch.isfinite(v).all()) and max(m['outside'])==0 and m['body_forwards']==120,'pred_b':float(diff.abs().max())<=1e-5 and relative<=1e-6 and anchor<=1e-5,
 'pred_c':max(effect_errors)<=1e-3 and float(effect.norm(dim=0).min())>=1e-6,'write_errors':errors,'anchor_max_abs':anchor,'readout_max_abs':float(diff.abs().max()),'readout_relative':relative,'effect_errors':effect_errors,
 'program_float_scalars':sum(x.numel() for part in ['head8','head9'] for x in program[part].values() if x.is_floating_point())+1,'program_bytes':(D/'program.pt').stat().st_size,'body_forwards':m['body_forwards'],'seconds':time.perf_counter()-start,'source_shas':binding,
 'scope':'Opened native precision test of complete current8/donor-city8/raw9 -> head9 write. Three native-state arrays; native suffix remains. Isolated CPU replay pending. No fresh or causal composition claim.'}
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='source_shas'},indent=2));signal.alarm(0)
if __name__=='__main__':main()
