#!/usr/bin/env python3
# BQGATE: 456forwards1824seq512tokens; terminal metric redteam, RMS scalar capture.
"""pred_a physical replay and capture; pred_b Fisher KL within20% all3;
pred_c norm-only KL>=80% full KL shared-refit. No fitted changes or test access.
"""
import os,sys,json,time,signal
from pathlib import Path
RUNNER=Path(__file__).resolve();ROOT=RUNNER.parents[3];P=ROOT/'basis_aligned/polynomial_causal';sys.path[:0]=[str(ROOT),str(P)]
import numpy as np
import torch
import torch.nn.functional as F
from compiled_quadratic_layer_v1 import CompiledQuadraticLayer
from terminal_probability_metric_v1 import terminal_scores_and_tangent,kl_quadratic
from prepare_million_token_panel_v1 import digest
from circuit_fast_screen_producer import Bilin18TorchBackend
from circuit_fast_screen_managed_runner import atomic_create_json
BIND=P/'TERMINAL_PROBABILITY_REDTEAM_V1_BINDING.json';OUT=P/'TERMINAL_PROBABILITY_REDTEAM_V1_RESULT.json'
def main():
    binding=json.loads(BIND.read_text());assert all(digest(p)==h for p,h in binding.items())
    assert json.loads((P/'TERMINAL_PROBABILITY_METRIC_V1_CONTROL.json').read_text())['passed']
    spec=torch.load(P/'MILLION_TOKEN_PANEL_V1_ROWS.pt',weights_only=True,map_location='cpu');rows=spec['rows'][:1824];positions=spec['positions'][:1824]
    assert rows.shape==(1824,513) and torch.equal(spec['validation_rows'],torch.arange(1600,1824))
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps(dict(model_loaded=False,gpu_accessed=False,body_forwards=456,sequences=1824,training_scalars=51200,validation_positions=7168,test_access=False)));return
    assert not OUT.exists();signal.alarm(900);tic=time.perf_counter();torch.set_num_threads(2);torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False
    m=Bilin18TorchBackend.load('cuda').model;eps=torch.finfo(torch.float32).eps;u=m.lm_head.weight.double();cache={};counts=[0,0];scales=[];scale_bridge=0.
    modules={k:CompiledQuadraticLayer(v).cuda() for k,v in torch.load(P/'PHYSICAL_QUADRATIC_V1_MODULES.pt',weights_only=True,map_location='cpu').items()}
    stats={k:{z:[] for z in ['full_kl','numerator_only_kl','norm_only_kl','fisher_kl','ce_damage']} for k in modules}
    def capture(_mod,args,out):cache['input']=args[0];cache['output']=out
    def count(_mod,args):counts[0]+=1;counts[1]+=len(args[0]);assert counts[0]<=456
    h1=m.transformer.h[0].attn.register_forward_pre_hook(count);h2=m.transformer.h[17].mlp.register_forward_hook(capture)
    try:
        for start in range(0,1824,4):
            tokens=rows[start:start+4,:512].cuda();x=F.rms_norm(m.transformer.wte(tokens),(1152,));x0=x;v1=None
            for block in m.transformer.h:x,v1=block(x,v1,x0)
            pos=positions[start:start+4].cuda();bi=torch.arange(4,device='cuda')[:,None];h=x[bi,pos];native=cache['output'][bi,pos];inp=cache['input'][bi,pos]
            pre=h-native;radius=(pre.square().mean(-1,keepdim=True)+eps).sqrt();scale_bridge=max(scale_bridge,float((pre/radius-inp).norm()/inp.norm()))
            scales.append(radius.cpu())
            if start<1600:continue
            hd=h.double();r0=(hd.square().mean(-1,keepdim=True)+eps).sqrt();raw=hd@u.T/r0;base=30*torch.tanh(raw/30);logp=base.log_softmax(-1);prob=logp.exp()
            target=rows[start:start+4].cuda()[bi,pos+1];base_ce=-logp.gather(-1,target[...,None]).squeeze(-1)
            def kl(score):return (prob*(logp-score.log_softmax(-1))).sum(-1)
            for name,mod in modules.items():
                delta=(mod(inp)-native).double();changed=hd+delta;r1=(changed.square().mean(-1,keepdim=True)+eps).sqrt()
                numerator=changed@u.T/r0;full=30*torch.tanh(numerator*r0/r1/30)
                _,tangent=terminal_scores_and_tangent(hd,delta,u,eps)
                ce=-full.log_softmax(-1).gather(-1,target[...,None]).squeeze(-1)
                values=dict(full_kl=kl(full),numerator_only_kl=kl(30*torch.tanh(numerator/30)),norm_only_kl=kl(30*torch.tanh(raw*r0/r1/30)),fisher_kl=kl_quadratic(prob,tangent),ce_damage=ce-base_ce)
                for key,value in values.items():
                    assert torch.isfinite(value).all();stats[name][key].extend(value.mean(1).cpu().tolist())
            if start%56==0:print(json.dumps(dict(sequences_completed=start+4,scale_bridge=scale_bridge)),flush=True)
    finally:h1.remove();h2.remove()
    reference=json.loads((P/'PHYSICAL_QUADRATIC_V1_RESULT.json').read_text());valid=counts==[456,1824] and scale_bridge<=1e-5;summaries={}
    for name,values in stats.items():
        assert all(len(v)==224 for v in values.values());means={k:float(np.mean(v)) for k,v in values.items()}
        ceerr=float(np.max(np.abs(np.array(values['ce_damage'])-reference['per_document'][name]['ce_damage'])))
        kerr=abs(means['full_kl']/reference['summaries'][name]['native_to_candidate_kl']['mean']-1)
        ferr=abs(means['fisher_kl']/means['full_kl']-1);valid=valid and ceerr<=5e-4 and kerr<=1e-3
        summaries[name]=dict(**means,physical_ce_damage_max_absolute_error=ceerr,physical_mean_kl_relative_error=kerr,fisher_mean_kl_relative_error=ferr,norm_only_fraction_of_full_kl=means['norm_only_kl']/means['full_kl'])
    ap=P/'TERMINAL_PREMLP_RADIUS_V1.pt';assert not ap.exists();torch.save(dict(radius=torch.cat(scales),source_row_ids=torch.arange(1824),positions=positions,training_rows=spec['train_rows'],validation_rows=spec['validation_rows'],epsilon=eps),ap)
    result=dict(schema='terminal.probability.redteam.v1',predictions={'pred_a_instrument':bool(valid),'pred_b_fisher_predicts_kl':bool(valid and all(s['fisher_mean_kl_relative_error']<=.2 for s in summaries.values())),'pred_c_norm_only_dominates':bool(valid and summaries['shared_refit']['norm_only_fraction_of_full_kl']>=.8)},summaries=summaries,per_document=stats,premlp_normalized_input_replay_max_relative_l2=scale_bridge,artifact_sha256=digest(ap),price=dict(body_forwards=counts[0],sequences=counts[1],training_scalars=51200,validation_positions=7168,radius_artifact_bytes=ap.stat().st_size),wall_seconds=time.perf_counter()-tic,binding_sha256=digest(BIND),runner_sha256=digest(RUNNER),scope='No fitting. Native-context numerator/norm diagnostics and local probability-curvature prediction for frozen replacements. Same exploratory validation panel; no test access or circuit adoption. Radius scalar extends existing input cache for later probability-aware training, with measured subtraction/reconstruction error.')
    atomic_create_json(OUT,result);print(json.dumps({k:v for k,v in result.items() if k!='per_document'}))
if __name__=='__main__':main()
