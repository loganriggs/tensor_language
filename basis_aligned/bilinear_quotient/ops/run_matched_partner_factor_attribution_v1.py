#!/usr/bin/env python3
# BQGATE:0bodyforwards,0textsequences;1024cachedendpoints,4swapvariants;300sec.
"""pred_a exact factor split/replay; pred_b norm effect<=20% all progressive;
pred_c partner>=2x parent in6/8; pred_d effect sum error<=10% all cells.
Frozen branch8; null: parent or normalization drives the specialization.
"""
import os,sys,json,time,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal';sys.path.insert(0,str(P))
import torch
import torch.nn.functional as F
from sparse_path_stability_atlas_v1 import digest
STEM='MATCHED_PARTNER_FACTOR_ATTRIBUTION_V1'

@torch.no_grad()
def main():
    binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files'];assert all(digest(k)==v for k,v in binding.items())
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps(dict(body_forwards=0,text_sequences=0,cached_endpoints=1024,swap_variants=4)));return
    signal.alarm(300);tic=time.perf_counter();torch.set_num_threads(2);torch.set_default_dtype(torch.float64);torch.backends.cuda.matmul.allow_tf32=False
    out=P/(STEM+'_RESULT.json');assert not out.exists()
    sd=torch.load(next(k for k in binding if k.endswith('/pytorch_model.bin')),weights_only=True,mmap=True)
    program=torch.load(P/'MATCHED_PARTNER_SUBSPACE_V1_PROGRAM.pt',weights_only=True)
    cache=torch.load(P/'PRODUCER_METRIC_CONTEXT_HOLDOUT_V1_PORTS.pt',weights_only=True,mmap=True);ports=cache['ports']
    rows=json.loads((P/'PRODUCER_METRIC_CONTEXT_HOLDOUT_V1_ROWS.json').read_text())['rows'];assert len(rows)==512
    x=ports['input16'].double().cuda();l,r=[sd[f'transformer.h.16.mlp.{n}.weight'].double().cuda() for n in ('Left','Right')]
    values=((x@l.T)*(x@r.T))@program['compiled_producer_readers'][:,[0,8]].cuda()
    s,t=values.T;inv=(ports['pre'].double().cuda().square().mean(-1)+torch.finfo(torch.float32).eps).reciprocal()
    mid=lambda z:(z[::2]+z[1::2])/2
    delta=lambda z:z[1::2]-z[::2]
    parts={'partner':mid(inv)*mid(s)*delta(t),'parent':mid(inv)*mid(t)*delta(s),'normalization':mid(s*t)*delta(inv)}
    full=delta(s*t*inv);identity=float((sum(parts.values())-full).norm()/full.norm());parts['full']=full
    writer=program['output_writers'][:,8];state=ports['pre']+ports['native_output'];u=sd['lm_head.weight'].float()
    torch.set_default_dtype(torch.float32);effects={}
    for name,change in parts.items():
        changed=state[::2]+(change.cpu()[:,None]*writer[None,:]).float();vals=[]
        for i,row in enumerate(rows):
            readers=u[[row['donor_answer_id'],row['donor_foil_id']]];zz=30*torch.tanh(F.linear(F.rms_norm(torch.stack([state[2*i],changed[i]]),(1152,)),readers)/30)
            vals.append(float((zz[1,0]-zz[1,1])-(zz[0,0]-zz[0,1])))
        effects[name]=torch.tensor(vals,dtype=torch.float64)
    prior=json.loads((P/'MATCHED_PARTNER_BRANCH_SCREEN_V1_RESULT.json').read_text())['reports'][8]['swaps']
    replay=float((effects['full']-torch.tensor(prior,dtype=torch.float64)).abs().max());cells=[]
    for family in sorted({row['family'] for row in rows}):
        ids=torch.tensor([i for i,row in enumerate(rows) if row['family']==family]);e={k:v[ids] for k,v in effects.items()};rms={k:float(v.square().mean().sqrt()) for k,v in e.items()}
        interaction=e['full']-e['partner']-e['parent']-e['normalization']
        cells.append(dict(family=family,rms=rms,means={k:float(v.mean()) for k,v in e.items()},
                          normalization_ratio=rms['normalization']/max(rms['full'],1e-30),partner_parent_ratio=rms['partner']/max(rms['parent'],1e-30),
                          additive_relative_error=float(interaction.norm()/e['full'].norm().clamp_min(1e-30))))
    target=[c for c in cells if c['family'].startswith('progressive:')]
    result={'pred_a':identity<=1e-10 and replay<=1e-5 and all(bool(torch.isfinite(v).all()) for v in effects.values()),
            'pred_b':all(c['normalization_ratio']<=.2 and c['rms']['full']>=1e-4 for c in target),
            'pred_c':sum(c['partner_parent_ratio']>=2 for c in target)>=6,
            'pred_d':all(c['additive_relative_error']<=.1 for c in cells)}
    result.update(cells=cells,effects={k:v.tolist() for k,v in effects.items()},identity=identity,replay=replay,
                  execution_seconds=time.perf_counter()-tic,source_shas=binding,scope='Exact scalar decomposition and conditional native swap diagnostics; not upstream causal identification or OOD.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('cells','effects','source_shas')}),flush=True)

if __name__=='__main__':main()
