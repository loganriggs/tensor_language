#!/usr/bin/env python3
# BQGATE: 226forwards904seq512tokens; MLP17 physical replacements, no fitting.
"""pred_a counts/noop/native-factor controls; pred_b shared-refit MAE_CE<=.05
and top1>=.95; pred_c shared-refit full-vocab KL<=.9 shared-original KL.
Full preregistration fixes224validationdocuments/32positions, prices and nulls.
"""
import os,sys,json,time,signal
from pathlib import Path
RUNNER=Path(__file__).resolve();ROOT=RUNNER.parents[3];P=ROOT/'basis_aligned/polynomial_causal';sys.path[:0]=[str(ROOT),str(P)]
import numpy as np
import torch
import torch.nn.functional as F
from compiled_quadratic_layer_v1 import CompiledQuadraticLayer
from prepare_million_token_panel_v1 import digest
from paired_panel_bootstrap_v1 import PairedPanelBootstrap
from circuit_fast_screen_producer import Bilin18TorchBackend
from circuit_fast_screen_managed_runner import atomic_create_json
BIND=P/'PHYSICAL_QUADRATIC_V1_BINDING.json';OUT=P/'PHYSICAL_QUADRATIC_V1_RESULT.json'

class NoOp(torch.nn.Module):
    def __init__(self,module):super().__init__();self.module=module
    def forward(self,x):return self.module(x)

def main():
    binding=json.loads(BIND.read_text());assert all(digest(p)==h for p,h in binding.items())
    control=json.loads((P/'PHYSICAL_QUADRATIC_V1_COMPILE_CONTROL.json').read_text());assert control['passed']
    spec=torch.load(P/'MILLION_TOKEN_PANEL_V1_ROWS.pt',weights_only=True,map_location='cpu');ids=spec['validation_rows'];rows=spec['rows'][ids];pos=spec['positions'][ids]
    assert rows.shape==(224,513) and pos.shape==(224,32)
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps(dict(model_loaded=False,gpu_accessed=False,body_forwards=226,sequences=904,validation_documents=224,scored_positions=7168,test_access=False)));return
    assert not OUT.exists();signal.alarm(900);tic=time.perf_counter();torch.set_num_threads(2);torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False
    m=Bilin18TorchBackend.load('cuda').model;original=m.transformer.h[17].mlp
    descriptions=torch.load(P/'PHYSICAL_QUADRATIC_V1_MODULES.pt',weights_only=True,map_location='cpu');modules={k:CompiledQuadraticLayer(v).cuda() for k,v in descriptions.items()}
    native_desc=dict(kind='product',bank=torch.empty(0,1152,device='cuda'),left=original.Left.weight,right=original.Right.weight,writer=original.Down.weight,bias=original.Down_bias)
    identity=CompiledQuadraticLayer(native_desc).cuda();noop=NoOp(original);counts=[0,0]
    def count(_module,args):counts[0]+=1;counts[1]+=len(args[0]);assert counts[0]<=226
    hook=m.transformer.h[0].attn.register_forward_pre_hook(count)
    def scores(tokens,positions,module):
        m.transformer.h[17].mlp=module
        x=F.rms_norm(m.transformer.wte(tokens),(1152,));x0=x;v1=None
        for block in m.transformer.h:x,v1=block(x,v1,x0)
        x=F.rms_norm(x,(1152,));selected=x[torch.arange(len(tokens),device='cuda')[:,None],positions]
        return (30*torch.tanh(m.lm_head(selected)/30)).float()
    measurements={k:{metric:[] for metric in ['ce_damage','mean_absolute_token_ce_change','native_to_candidate_kl','native_top1_agreement']} for k in modules}
    native_ce=[];instrument={};valid=True
    try:
        for first in range(0,224,4):
            tokens=rows[first:first+4,:512].cuda();positions=pos[first:first+4].cuda();bi=torch.arange(4,device='cuda')[:,None];target=rows[first:first+4].cuda()[bi,positions+1]
            base=scores(tokens,positions,original);bp=F.log_softmax(base,dim=-1);bce=-bp.gather(-1,target[...,None]).squeeze(-1);native_ce.extend(bce.mean(1).cpu().tolist())
            if first==0:
                for label,mod in [('noop',noop),('native_factor',identity)]:
                    check=scores(tokens,positions,mod);cce=-F.log_softmax(check,dim=-1).gather(-1,target[...,None]).squeeze(-1)
                    err=float((check-base).norm()/base.norm());ceerr=float((cce-bce).abs().max());instrument[label]=dict(logit_relative_l2=err,ce_max_absolute_error=ceerr);valid=valid and err<=1e-5 and ceerr<=2e-5
            for label,mod in modules.items():
                candidate=scores(tokens,positions,mod);cp=F.log_softmax(candidate,dim=-1);ce=-cp.gather(-1,target[...,None]).squeeze(-1);delta=ce-bce
                values=dict(ce_damage=delta.mean(1),mean_absolute_token_ce_change=delta.abs().mean(1),native_to_candidate_kl=(bp.exp()*(bp-cp)).sum(-1).mean(1),native_top1_agreement=(candidate.argmax(-1)==base.argmax(-1)).float().mean(1))
                for key,value in values.items():
                    assert torch.isfinite(value).all();measurements[label][key].extend(value.cpu().tolist())
            if first%56==0:print(json.dumps(dict(documents_completed=first+4,counts=counts)),flush=True)
    finally:hook.remove();m.transformer.h[17].mlp=original
    valid=valid and counts==[226,904];boot=PairedPanelBootstrap(224,91162133);summaries={}
    for label,metrics in measurements.items():
        assert all(len(v)==224 for v in metrics.values())
        summaries[label]={k:dict(mean=float(np.mean(v)),paired_document_interval95=boot.mean(v)) for k,v in metrics.items()}
    shared=summaries['shared_refit'];old=summaries['shared_original']
    result=dict(schema='physical.quadratic.screen.v1',predictions={'pred_a_instrument':bool(valid),'pred_b_shared_prediction_preservation':bool(valid and shared['mean_absolute_token_ce_change']['mean']<=.05 and shared['native_top1_agreement']['mean']>=.95),'pred_c_writer_recalibration_kl':bool(valid and shared['native_to_candidate_kl']['mean']<=.9*old['native_to_candidate_kl']['mean'])},instrument=instrument,summaries=summaries,per_document=measurements,native_ce_per_document=native_ce,native_ce_mean=float(np.mean(native_ce)),price=dict(body_forwards=counts[0],sequences=counts[1],scored_positions=7168,compiled_numbers={k:v.numbers() for k,v in modules.items()},native_mlp_numbers=sum(t.numel() for t in original.parameters()),native_background_retained=True),wall_seconds=time.perf_counter()-tic,binding_sha256=digest(BIND),runner_sha256=digest(RUNNER),scope='Physical MLP17 replacements on previously used Pile validation panel. Positive CE damage is worse than native. No test/OOD certificate, selective manipulation, composition or independent circuit input extraction. All earlier modules and full unembedding retained; candidate weights frozen before screen.')
    atomic_create_json(OUT,result);print(json.dumps({k:v for k,v in result.items() if k not in ['per_document','native_ce_per_document']}))

if __name__=='__main__':main()
