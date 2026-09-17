#!/usr/bin/env python3
# BQGATE:48bodyforwards;16prefixes;300seconds;no fitting.
"""pred_a state/write error<=1e-5; pred_b margins<=1e-5abs/1e-6relative.
pred_c each effect error<=1e-3 with live norm>=1e-6. 48 forwards.
Null: moving donor input before MLP7 changes behavior beyond these gates.
Extraction only; no new fresh selectivity/composition claim.
"""
import hashlib,json,os,signal,sys,time
from pathlib import Path
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal'
D=P/'extracted_circuits/odd_attention8h2_mlp7_donor_v1'
sys.path[:0]=[str(D),str(Path(__file__).parent),str(P),str(ROOT)]
import execute,donor,native
from regional_endpoint_batching_v1 import group_rows,expand
from regional_paired_write_runtime_v2 import measure
from run_even_value_factorial_native_v1 import setup
STEM='TYPED_FACE_MLP7_DONOR_V1'
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rel(a,b):return float((a-b).norm()/b.norm().clamp_min(1e-30))
@torch.no_grad()
def main():
    binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files'];assert all(digest(k)==v for k,v in binding.items())
    rows=json.loads((P/'TYPED_FACE_KEY_SOURCE_FRESH_V1_ROWS.json').read_text())['rows'];groups,mapping=group_rows(rows)
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        assert len(groups)==16;print('48bodyforwards;16prefixes;exact MLP7 donor generator');return
    out=P/(STEM+'_RESULT.json');assert not out.exists()
    start=time.perf_counter();signal.alarm(300);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    from fastload import load_model_fast
    model=load_model_fast().cuda().eval();graph,_,_,_=setup('cuda');b7=model.transformer.h[7];b8=model.transformer.h[8]
    head={k:v.to('cuda') for k,v in torch.load(P/'TYPED_FACE_KEY_SOURCE_FRESH_V1_PROGRAM.pt',weights_only=True).items()}
    ids=head['token_ids'];dp={'left':b7.mlp.Left.weight,'right':b7.mlp.Right.weight,'down':b7.mlp.Down.weight,'bias':b7.mlp.Down_bias,'lambda8':b8.lambdas,'token_ids':ids,
      'initial_table':F.rms_norm(model.transformer.wte(ids),(1152,))}
    program={'head':head,'donor':dp};state={};early=[]
    def pre(module,args):
        if len(early)<len(groups):state['u']=module.lambdas[0]*args[0]+module.lambdas[1]*args[2]
    def after(module,args,result):
        if len(early)<len(groups):
            city=groups[len(early)]['city_position'];early.append((state['u']+result[0])[:,city].detach().clone())
    index={(r['context_id'],r['cue']):i for i,r in enumerate(groups)};se=[];we=[];fixtures=[]
    def write(arm,row,donor_row,current,native_donor,mask):
        city=row['city_position'];rt=row['ids'][city];dt=donor_row['ids'][city]
        expected=native.execute(head,current,native_donor[:,city],rt,dt,city,mask)
        if arm=='original':return expected
        i=index[(row['context_id'],row['cue'])];g=early[i^1];initial=dp['initial_table'][ids.tolist().index(dt)].unsqueeze(0)
        generated=donor.generate(dp,g,initial);actual=execute.execute(program,current,g,rt,dt,city,mask)
        se.append(rel(generated,native_donor[:,city]));we.append(rel(actual,expected))
        fixtures.append({'current':current.cpu(),'g7':g.cpu(),'recipient_token':rt,'donor_token':dt,'city':city,'destination':mask.cpu(),'state':native_donor[:,city].cpu(),'write':expected.cpu()})
        return actual
    hooks=[b7.register_forward_pre_hook(pre),b7.attn.register_forward_hook(after)]
    try:m=measure(model,graph,groups,['native','original','generated'],write)
    finally:
        for h in hooks:h.remove()
    v=expand(m['values'],mapping,6);diff=v[2]-v[1];effect=v[1]-v[0]
    errors=[rel(diff[:,j]+effect[:,j],effect[:,j]) for j in range(5)];norms=effect.norm(dim=0).tolist()
    package={part:{k:v.detach().cpu().clone() for k,v in values.items()} for part,values in program.items()}
    torch.save(package,D/'program.pt');torch.save(fixtures,P/(STEM+'_FIXTURES.pt'))
    counts={part:sum(v.numel() for v in values.values() if v.is_floating_point()) for part,values in package.items()}
    result={'pred_a':max(se)<=1e-5 and max(we)<=1e-5,'pred_b':float(diff.abs().max())<=1e-5 and rel(v[2],v[1])<=1e-6,
      'pred_c':max(errors)<=1e-3 and min(norms)>=1e-6,'state_errors':se,'write_errors':we,'readout_max_abs':float(diff.abs().max()),'readout_relative':rel(v[2],v[1]),
      'effect_errors':errors,'effect_norms':norms,'float_scalars':counts,'program_bytes':(D/'program.pt').stat().st_size,'program_sha256':digest(D/'program.pt'),
      'native_state_ports':2,'input_float_scalars_T32':32*1152+1152,'body_forwards':m['body_forwards'],'seconds':time.perf_counter()-start,
      'panel_status':'opened 16-sequence fresh-transfer panel','scope':'Exact earlier donor boundary; same two native-state ports, added explicit MLP7 weights. No new OOD, selective or compositional certification. Isolated CPU replay pending.','source_shas':binding}
    torch.save({'values':v},P/(STEM+'_ARTIFACT.pt'));out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='source_shas'},indent=2));signal.alarm(0)
if __name__=='__main__':main()
