#!/usr/bin/env python3
# BQGATE:0bodyforwards,0textsequences;1024cachedendpoints32cells;300sec.
"""pred_a exact replay; pred_b allcells swaps<=.1/sign>=.9/live>=4;
pred_c removals<=.02; pred_d writes<=.05. Frozen node, no fitting.
"""
import os,sys,json,time,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal';sys.path.insert(0,str(P))
import torch
import torch.nn.functional as F
from sparse_path_stability_atlas_v1 import digest
from quadratic_producer_projection_v1 import atom_gram
from sparse_frame_function_inner_v1 import coefficients
from sparse_producer_graph_execute_v1 import execute,incident
from quartic_frozen_native_score_v2 import score
STEM='SPARSE_NODE_CONTEXT_V1'

@torch.no_grad()
def main():
    binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files'];assert all(digest(k)==v for k,v in binding.items())
    cache=torch.load(P/'PRODUCER_METRIC_CONTEXT_HOLDOUT_V1_PORTS.pt',weights_only=True,mmap=True)
    assert cache['rows_sha256']==digest(P/'PRODUCER_METRIC_CONTEXT_HOLDOUT_V1_ROWS.json')
    rows=json.loads((P/'PRODUCER_METRIC_CONTEXT_HOLDOUT_V1_ROWS.json').read_text())['rows'];assert len(rows)==512 and len({r['family'] for r in rows})==32
    old=json.loads((P/'AMORTIZED_SPARSE_FRAME_FIDELITY_V1_RESULT.json').read_text());assert old['node']['write_bar'] and old['node']['effects']['pred_b'] and old['node']['effects']['pred_c']
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps(dict(body_forwards=0,text_sequences=0,cached_endpoints=1024)));return
    signal.alarm(300);tic=time.perf_counter();torch.set_num_threads(2);torch.set_default_dtype(torch.float64);torch.backends.cuda.matmul.allow_tf32=False
    out=P/(STEM+'_RESULT.json');assert not out.exists()
    sd=torch.load(next(k for k in binding if k.endswith('/pytorch_model.bin')),weights_only=True,mmap=True)
    l0,r0,d0,l1,r1,d1=[sd[f'transformer.h.{layer}.mlp.{name}.weight'].double().cuda() for layer in (16,17) for name in ('Left','Right','Down')]
    scale=float(sd['transformer.h.17.lambdas'][0]);h=scale**2*d0@atom_gram(l0,r0)@d0.T
    he,hv=torch.linalg.eigh(h);hs=(hv*he.sqrt())@hv.T;hi=(hv*he.rsqrt())@hv.T
    saved=torch.load(P/'AMORTIZED_SPARSE_FRAME_NATIVE_V1_FRAMES.pt',weights_only=True);arm=old['best_arm'];q=saved['frames'][arm].cuda();e=saved['edges'][arm].cuda();node=old['node']['index'];mask=incident(e,[node]);assert int(mask.sum())==671
    e=e[:,mask];writer=coefficients(q,l1@hs,r1@hs,d1,e);a=q[:,node]
    ports=cache['ports'];x=ports['input16'].double().cuda();den=ports['pre'].double().cuda().square().mean(-1)+torch.finfo(torch.float32).eps
    hidden=(x@l0.T)*(x@r0.T);p=scale*hidden@d0.T;z=hidden@(scale*d0.T@hi@q)
    native=lambda pp:((pp@l1.T)*(pp@r1.T))@d1.T
    ref=(native(p)-native(p-(p@hi@a)[:,None]*(hs@a)[None,:]))/den[:,None]
    candidate=execute(z,e,writer,den);direct=execute(p@hi@q,e,writer,den)
    error=float((candidate-direct).norm()/candidate.norm());torch.set_default_dtype(torch.float32)
    state=ports['pre']+ports['native_output'];u=sd['lm_head.weight'].float()
    effects=score([ref.cpu(),ref.cpu(),candidate.cpu()],state,rows,u)
    families=[]
    for name in sorted({r['family'] for r in rows}):
        ids=torch.tensor([i for i,r in enumerate(rows) if r['family']==name]);ep=(2*ids[:,None]+torch.tensor([0,1])).flatten();capability=[]
        for side in ('base','donor'):
            offset=0 if side=='base' else 1;readers=torch.stack([u[[rows[i][side+'_answer_id'],rows[i][side+'_foil_id']]] for i in ids.tolist()])
            logits=torch.einsum('nd,nkd->nk',F.rms_norm(state[2*ids+offset],(1152,)),readers);capability.append(float((logits[:,0]>logits[:,1]).float().mean()))
        families.append(dict(family=name,write_relative_error=float((candidate[ep]-ref[ep]).norm()/ref[ep].norm()),native_capability=capability))
    replay=max([f['swap_relative_rms'] for f in effects['reports'][0]['families']]+[f['zero_ce_meanabs_disagreement'] for f in effects['reports'][0]['families']])
    result={'pred_a':error<=1e-8 and replay<=1e-5 and effects['pred_a'],'pred_b':effects['pred_b'],'pred_c':effects['pred_c'],'pred_d':all(f['write_relative_error']<=.05 for f in families)}
    result.update(effects=effects,families=families,node=node,arm=arm,compiled_error=error,reference_replay_error=replay,
                  execution_seconds=time.perf_counter()-tic,source_shas=binding,scope='Frozen node on previously inspected context cache; no exclusions/refit/OOD/semantic circuit claim.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('source_shas','effects','families')}),flush=True)

if __name__=='__main__':main()
