#!/usr/bin/env python3
# BQGATE:0bodyforwards;128cachedendpoints;rank128,512,random128;300sec.
"""pred_a algebra/replay<=1e-5; pred_b bothbranches swaps<=.1/sign>=.9;
pred_c shared128 writeerror <=.8 random128; null: no useful common interface.
Price2*1152*k interfacefloats; native producer/background retained.
"""
import sys,os,json,time,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal'
sys.path[:0]=[str(P),str(ROOT)]
import torch
from sparse_path_stability_atlas_v1 import digest
from quadratic_producer_projection_v1 import atom_gram
from attention_source_quartic_v1 import unpack
from shared_producer_interface_v1 import geometry,interface,apply,objective
from packed_quadratic_branch_v1 import execute
from quartic_frozen_native_score_v2 import score
STEM='SHARED_MLP15_INTERFACE_V1'

@torch.no_grad()
def main():
    binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files'];assert all(digest(k)==v for k,v in binding.items())
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps(dict(body_forwards=0,endpoints=128,ranks=[128,512],seconds=300)));return
    out=P/(STEM+'_RESULT.json');ap=P/(STEM+'_PROGRAM.pt');assert not out.exists() and not ap.exists();signal.alarm(300)
    tic=time.perf_counter();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    state=torch.load(next(k for k in binding if k.endswith('pytorch_model.bin')),weights_only=True,mmap=True,map_location='cpu')
    def weight(key):return state[key].double().cuda()
    l=weight('transformer.h.15.mlp.Left.weight');r=weight('transformer.h.15.mlp.Right.weight');d=weight('transformer.h.15.mlp.Down.weight')
    scale=weight('transformer.h.16.lambdas')[0];bias=scale*weight('transformer.h.15.mlp.Down_bias')
    gram=atom_gram(l,r);metric=scale.square()*(d@gram@d.T);del gram,l,r
    packed=torch.load(P/'MATCHED_PARTNER_EXACT_INPUT_FOLD_V1_PROGRAM.pt',weights_only=True)
    forms=unpack(packed,metric);geo=geometry(forms,metric);frames={'shared128':geo['vectors'][:,:128],'shared512':geo['vectors'][:,:512]}
    generator=torch.Generator(device='cuda').manual_seed(73130)
    frames['random128']=torch.linalg.qr(torch.randn(1152,128,device='cuda',dtype=torch.float64,generator=generator)).Q
    programs={name:interface(geo,v) for name,v in frames.items()};objective_reports={};identities=[]
    for name,v in frames.items():
        obj=objective(geo,v);orth=float((v.T@v-torch.eye(v.shape[1],device='cuda')).norm());identities.append(orth)
        objective_reports[name]=dict(objective=obj,orthogonality=orth)
        if name.startswith('shared'):
            tail=float(geo['values'][v.shape[1]:].sum());identities.append(abs(obj-tail));objective_reports[name]['spectral_tail']=tail
    # Freeze all directions before inspecting any native endpoints/effects.
    rounded={name:{k:v.float().double() for k,v in prog.items()} for name,prog in programs.items()}
    source=torch.load(P/'MATCHED_PARTNER_MLP15_SOURCE_V1_PORTS.pt',weights_only=True,map_location='cpu')['sources']
    cache=torch.load(P/'MATCHED_PARTNER_FRESH_CONSTRUCTION_V1_PORTS.pt',weights_only=True,map_location='cpu');ports=cache['ports']
    background=source['background'].double().cuda();producer=source['producer'].double().cuda();raw=background+producer
    n16=raw.square().mean(-1)+torch.finfo(torch.float32).eps;n17=ports['pre'].double().cuda().square().mean(-1)+torch.finfo(torch.float32).eps
    full_producer=apply(interface(geo,geo['vectors']),producer,bias)
    identities.append(float((full_producer-producer).norm()/producer.norm()))
    full=execute(packed,(background+full_producer)/n16.sqrt()[:,None],n17)
    replay=[float((full[b].cpu()-cache['writes'][str(b)]).norm()/cache['writes'][str(b)].norm()) for b in (3,8)]
    writes={name:execute(packed,(background+apply(prog,producer,bias))/n16.sqrt()[:,None],n17) for name,prog in rounded.items()}
    rows=json.loads((P/'MATCHED_PARTNER_FRESH_CONSTRUCTION_V1_ROWS.json').read_text())['rows'];h=ports['pre']+ports['native_output'];u=state['lm_head.weight'].float();effects={};diagnostic={}
    passed=[];excess=[]
    for branch in (3,8):
        reference=cache['writes'][str(branch)]
        e=score([reference,writes['shared128'][branch].cpu(),writes['random128'][branch].cpu()],h,rows,u);effects[str(branch)]=e
        diagnostic[str(branch)]=score([reference,writes['shared512'][branch].cpu(),writes['shared512'][branch].cpu()],h,rows,u)
        passed.extend(f['swap_relative_rms']<=.1 and f['swap_sign_agreement']>=.9 and f['swap_live']>=4 and f['zero_ce_meanabs_disagreement']<=.02 for f in e['reports'][0]['families'] if f['family']!='quoted_control')
        excess.append(e['reports'][0]['physical_error']<=.8*e['reports'][1]['physical_error'])
    artifact={name:{**{k:v.float().cpu() for k,v in prog.items()},'compiled_hidden_read':(prog['read']@(scale*d)).float().cpu()} for name,prog in programs.items() if name.startswith('shared')}
    torch.save(dict(programs=artifact,bias=bias.float().cpu(),scope='Native MLP15 Left/Right, exact normalization/background ports and packed branch program remain external.'),ap)
    result={'pred_a':max(identities)<=1e-10 and max(replay)<=1e-5 and all(e['pred_a'] for e in effects.values()),'pred_b':all(passed),'pred_c':all(excess)}
    result.update(identities=identities,full_replay=replay,objectives=objective_reports,metric_condition=geo['condition'],effects=effects,diagnostic512=diagnostic,
                  artifact_sha=digest(ap),source_shas=binding,execution_seconds=time.perf_counter()-tic)
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('effects','diagnostic512','source_shas')}),flush=True)

if __name__=='__main__':main()
