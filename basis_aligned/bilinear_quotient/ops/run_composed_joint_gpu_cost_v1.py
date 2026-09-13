#!/usr/bin/env python3
# BQGATE:2prefixes;4implementations;7timingrepeats;180seconds.
"""pred_a all state replay relative errors<=1e-5.
pred_b sharedbatched>=10%faster than nativebatched at12pairs bothprefixes.
pred_c sharedbatched no slower than nativebatched at1pair bothprefixes.
Null: FP64 generator overhead defeats algebraic savings on GPU.
Price2nativeprefixes,1/4/12pairs,4variants,7timingrepeats,180seconds.
"""
from pathlib import Path
import json,sys,time,signal,os,statistics
from hashlib import sha256
import torch
import torch.nn.functional as F
P=Path(__file__).resolve().parents[2]/'polynomial_causal';sys.path.insert(0,str(P))
sys.path.insert(0,str(P.parent/'bilinear_quotient/ops'))
from fastload import load_model_fast
from composed_mlp10_context_v1 import prepare
from composed_joint_response_v1 import branch
from regional_cue_row_check_v1 import validate


@torch.no_grad()
def main():
    files=json.loads((P/'COMPOSED_JOINT_GPU_COST_V1_BINDING.json').read_text())['files']
    assert all(sha256(Path(k).read_bytes()).hexdigest()==v for k,v in files.items())
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print('2pristineprefixes,1/4/12pairs,4implementations,7timingrepeats;180seconds');return
    assert not (P/'COMPOSED_JOINT_GPU_COST_V1_RESULT.json').exists()
    signal.alarm(180);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;tic=time.perf_counter()
    model=load_model_fast().cuda().eval()
    assert next(model.parameters()).device.type=='cuda'
    rows=json.loads((P/'FIRST_TOKEN_PATH_FRESH_V1_ROWS.json').read_text())['rows']
    row_check=validate(rows)
    rows+=json.loads((P/'FIRST_TOKEN_REMOVAL_NEWLINE_FRESH_V1_ROWS.json').read_text())['rows']
    child=torch.load(P/'CROSSFIRST_STATE_EXECUTOR_V1_ARTIFACT.pt',weights_only=True,map_location='cpu')['fields']
    parent=torch.load(P/'CROSSFIRST_HIERARCHY_V1_ARTIFACT.pt',weights_only=True,map_location='cpu')['parent_fields']
    program=torch.load(P/'MLP9_CROSSFIRST_RESPONSE_V1_PROGRAM.pt',weights_only=True,map_location='cpu')
    child={k:v.cuda() for k,v in child.items()};parent={k:v.cuda() for k,v in parent.items()};program={k:v.cuda() for k,v in program.items()}
    w=program['direction'];b9=model.transformer.h[9];b10=model.transformer.h[10]
    matrices=[getattr(b10.attn,k).weight.double() for k in ('c_q','c_k','c_q2','c_k2','c_v')]
    rel=lambda x,y:float((x-y).norm()/y.norm().clamp_min(1e-30))
    cells=[]
    for index in (0,96):
        ids=torch.tensor([rows[index]['ids']],device='cuda');x=F.rms_norm(model.transformer.wte(ids),(1152,));x0=x;v1=None
        for block in model.transformer.h[:9]:x,v1=block(x,v1,x0)
        raw9=b9.lambdas[0]*x+b9.lambdas[1]*x0
        att9,v1=b9.attn(F.rms_norm(raw9,(1152,)),v1)
        z9=raw9+att9;m9=b9.mlp(F.rms_norm(z9,(1152,)));h9=z9+m9
        raw0=b10.lambdas[0]*h9+b10.lambdas[1]*x0
        z0=raw0+b10.attn(F.rms_norm(raw0,(1152,)),v1)[0]
        a=child[index][...,None];b=(parent[index]-child[index])[...,None]
        arguments=(z9,m9.double()-b9.mlp.Down_bias.double(),raw0,z0,v1.double(),
                   program,float(b10.lambdas[0]),matrices,float(b10.attn.lamb),b10.attn.c_proj.weight.double())
        pairs=[(a*(i+1)/12,b*(13-i)/12) for i in range(12)]
        def post(z):
            z=z.float()
            return z+b10.mlp(F.rms_norm(z,(1152,)))
        def direct(amplitudes):
            za=raw9+(att9-(amplitudes*w).to(att9.dtype))
            ha=za+b9.mlp(F.rms_norm(za,(1152,)))
            raw=b10.lambdas[0]*ha+b10.lambdas[1]*x0
            state=raw+b10.attn(F.rms_norm(raw,(1152,)),v1.expand(raw.shape[0],*v1.shape[1:]))[0]
            return post(state)
        def amplitudes(count):
            return [v for aa,bb in pairs[:count] for v in (aa,bb,aa+bb)]
        def native_serial(count):return torch.cat([direct(v) for v in amplitudes(count)],dim=0)
        def native_batch(count):return direct(torch.cat(amplitudes(count),dim=0))
        def shared_serial(count):
            context=prepare(*arguments)
            return torch.cat([post(branch(v,context)) for v in amplitudes(count)],dim=0)
        def shared_batch(count):
            context=prepare(*arguments);values=torch.cat(amplitudes(count),dim=0)
            context['first_values']=arguments[4].expand(values.shape[0],*arguments[4].shape[1:])
            return post(branch(values,context))
        variants={'native_serial':native_serial,'native_batch':native_batch,'shared_serial':shared_serial,'shared_batch':shared_batch}
        for count in (1,4,12):
            expected=native_serial(count);errors={}
            for name,fn in variants.items():errors[name]=rel(fn(count).double(),expected.double())
            assert max(errors.values())<=1e-5
            samples={name:[] for name in variants};names=list(variants)
            for repeat in range(7):
                for name in names[repeat%4:]+names[:repeat%4]:
                    torch.cuda.synchronize();start=time.perf_counter();result=variants[name](count);torch.cuda.synchronize()
                    samples[name].append(time.perf_counter()-start);del result
            medians={name:statistics.median(v) for name,v in samples.items()}
            cells.append(dict(row=index,tokens=len(rows[index]['ids']),pairs=count,state_errors=errors,
                              median_seconds=medians,samples_seconds=samples,
                              matched_batched_speed_ratio=medians['native_batch']/medians['shared_batch']))
            print(json.dumps({k:v for k,v in cells[-1].items() if k!='samples_seconds'}),flush=True)
    result={'pred_a':all(max(c['state_errors'].values())<=1e-5 for c in cells),
            'pred_b':all(c['matched_batched_speed_ratio']>=1/0.9 for c in cells if c['pairs']==12),
            'pred_c':all(c['matched_batched_speed_ratio']>=1 for c in cells if c['pairs']==1),
            'cells':cells,'seconds':time.perf_counter()-tic,'gpu':torch.cuda.get_device_name(),
            'scope':'Native FP32 MLP9/attention10/MLP10 vs validated FP64 generator followed by FP32 MLP10. Both serial and batched on fixed historical rows0/96,1/4/12amplitude pairs. Includes shared preparation, synchronized wall time. Pristine context and suffix excluded equally; no full-model performance or freshOOD claim.'}
    (P/'COMPOSED_JOINT_GPU_COST_V1_RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='cells'},indent=2))


if __name__=='__main__':main()
