#!/usr/bin/env python3
# BQGATE:16prefixes;256suffixreadouts;120seconds.
"""pred_a unpruned FP32 replay=0 and FP64 all-node effect norm growth in[2,6].
pred_b FP64 composition residual norm grows>=8 at both endpoints.
pred_c FP64 composition residual norm grows<=2 at both endpoints.
Opposing diagnostics: smooth nonlinear interaction versus numerical floor.
Price16prefixes,256suffixes,512readouts,120seconds; node strength4.
No pruning adoption or repair of original unit-strength criteria.
"""
from pathlib import Path
import json,sys,time,signal,os
from hashlib import sha256
import torch
import torch.nn.functional as F
P=Path(__file__).resolve().parents[2]/'polynomial_causal';sys.path.insert(0,str(P))
from fastload import load_model_fast
from composed_mlp10_context_v1 import prepare
from composed_joint_response_v1 import branch
from diagonal_mlp10_branch_v1 import prepare as prepare_diagonal,evaluate as diagonal_post
from regional_cue_row_check_v1 import validate

@torch.no_grad()
def main():
    files=json.loads((P/'SELF_CONSUMER_STRENGTH4_V1_BINDING.json').read_text())['files']
    assert all(sha256(Path(k).read_bytes()).hexdigest()==v for k,v in files.items())
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print('16prefixes;256suffixreadouts;512readouts;120seconds');return
    assert not (P/'SELF_CONSUMER_STRENGTH4_V1_RESULT.json').exists()
    signal.alarm(120);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;tic=time.perf_counter()
    model=load_model_fast().cuda().eval();readout_weight=model.lm_head.weight.double()
    rows=json.loads((P/'FIRST_TOKEN_PATH_FRESH_V1_ROWS.json').read_text())['rows'];validate(rows)
    rows+=json.loads((P/'FIRST_TOKEN_REMOVAL_NEWLINE_FRESH_V1_ROWS.json').read_text())['rows'];assert len(rows)==160
    child=torch.load(P/'CROSSFIRST_STATE_EXECUTOR_V1_ARTIFACT.pt',weights_only=True)['fields']
    parent=torch.load(P/'CROSSFIRST_HIERARCHY_V1_ARTIFACT.pt',weights_only=True)['parent_fields']
    program=torch.load(P/'MLP9_CROSSFIRST_RESPONSE_V1_PROGRAM.pt',weights_only=True)
    program={k:v.cuda() for k,v in program.items()};w=program['direction']
    writer=torch.load(P/'extracted_circuits/fixed_writer_self_mlp10_v1/program.pt',weights_only=True)['writer'].cuda()
    b9,b10=model.transformer.h[9],model.transformer.h[10]
    matrices=[getattr(b10.attn,k).weight.double() for k in ('c_q','c_k','c_q2','c_k2','c_v')]
    mlp_matrices=[getattr(b10.mlp,k).weight.double() for k in ('Left','Right','Down')]
    cells=[];node_relative_norms=[]
    for index in range(96,112):
        row=rows[index]
        ids=torch.tensor([row['ids']],device='cuda');x=F.rms_norm(model.transformer.wte(ids),(1152,));x0=x;v1=None
        for block in model.transformer.h[:9]:x,v1=block(x,v1,x0)
        raw9=b9.lambdas[0]*x+b9.lambdas[1]*x0;att9,v1=b9.attn(F.rms_norm(raw9,(1152,)),v1)
        z9=raw9+att9;m9=b9.mlp(F.rms_norm(z9,(1152,)));h9=z9+m9
        raw0=b10.lambdas[0]*h9+b10.lambdas[1]*x0;z0=raw0+b10.attn(F.rms_norm(raw0,(1152,)),v1)[0]
        a=child[index].cuda()[...,None];b=(parent[index]-child[index]).cuda()[...,None]
        context=prepare(z9,m9.double()-b9.mlp.Down_bias.double(),raw0,z0,v1.double(),program,float(b10.lambdas[0]),matrices,float(b10.attn.lamb),b10.attn.c_proj.weight.double())
        diagonal=prepare_diagonal(context,*mlp_matrices,b10.mlp.Down_bias.double())
        native=z0+b10.mlp(F.rms_norm(z0,(1152,)));states={k:[native] for k in ('unpruned','all','final','earlier')}
        for amplitude in (a,b,a+b):
            z=branch(amplitude,context).float().double();full=diagonal_post(z,amplitude,diagonal)
            rho=z.square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps;node=4*amplitude.square()/rho*writer
            node_relative_norms.append(float((node.norm(dim=-1)/full.norm(dim=-1)).max()))
            final=torch.zeros_like(node);final[:,-1]=node[:,-1];earlier=node-final
            states['unpruned'].append(full.float());states['all'].append((full-node).float())
            states['final'].append((full-final).float());states['earlier'].append((full-earlier).float())
        scores=[];scores64=[]
        for name in states:
            outcomes=[];outcomes64=[]
            for initial in states[name]:
                state=initial
                for block in model.transformer.h[11:]:state,_=block(state,v1,x0)
                logits=(30*torch.tanh(model.lm_head(F.rms_norm(state[:,-1],(1152,)))/30))[0]
                if index<96:target=logits[row['uk_id']]-logits[row['us_id']];control=logits[row['control_ids'][0]]-logits[row['control_ids'][1]]
                else:target=-logits.log_softmax(-1)[198];control=logits[198]-logits[11]
                outcomes.append([float(target),float(control)])
                last=state[:,-1].double();rho64=(last.square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps).sqrt()
                logits64=(30*torch.tanh((last/rho64)@readout_weight.T/30))[0]
                outcomes64.append([float(-logits64.log_softmax(-1)[198]),float(logits64[198]-logits64[11])])
            scores.append(outcomes);scores64.append(outcomes64)
        scores=torch.tensor(scores,dtype=torch.float64);effects=scores[:,3]-scores[:,1]-scores[:,2]+scores[:,0]
        cells.append(dict(row=index,scores=scores.tolist(),scores64=scores64,interactions=effects.tolist()))
    old=json.loads((P/'FIXED_SELF_CONSUMERS_V1_RESULT.json').read_text())['cells'][96:112]
    scores=torch.tensor([c['scores'] for c in cells],dtype=torch.float64)
    prior=torch.tensor([c['scores'] for c in old],dtype=torch.float64)
    scores64=torch.tensor([c['scores64'] for c in cells],dtype=torch.float64)
    comparisons=[]
    for name,values in [('FP32',scores),('FP64_readout',scores64)]:
        interactions=values[:,:,3]-values[:,:,1]-values[:,:,2]+values[:,:,0]
        changes=interactions[:,0:1]-interactions[:,1:]
        allchange,last,early=changes.unbind(1);residual=last+early-allchange
        comparisons.append(dict(precision=name,composition_error=(residual.norm(dim=0)/allchange.norm(dim=0)).tolist(),all_node_change_norm=allchange.norm(dim=0).tolist(),composition_residual_norm=residual.norm(dim=0).tolist(),max_absolute_composition_residual=residual.abs().amax(0).tolist()))
    first,second=comparisons
    unit=json.loads((P/'SELF_CONSUMER_READOUT64_V1_RESULT.json').read_text())['comparisons'][1]
    growth=[b/a for a,b in zip(unit['composition_residual_norm'],second['composition_residual_norm'])]
    effect_growth=[b/a for a,b in zip(unit['all_node_change_norm'],second['all_node_change_norm'])]
    result={'pred_a':torch.equal(scores[:,0],prior[:,0]) and all(2<=x<=6 for x in effect_growth),'pred_b':min(growth)>=8,
        'pred_c':max(growth)<=2,'composition_residual_growth':growth,'all_node_effect_growth':effect_growth,'max_node_state_relative_norm':max(node_relative_norms),
        'fp32_replay_maxabs':float((scores[:,0]-prior[:,0]).abs().max()),'final_precision_endpoint_maxabs':float((scores64-scores).abs().max()),
        'comparisons':comparisons,'cells':cells,'seconds':time.perf_counter()-tic,
        'scope':'Same16FineWeb0prefixes,FP32transformer withfourfoldnodeintervention,dualreadout. Comparedto unit-strength FP64readout; largerremoval diagnosticnotadoption. Scalingboundsarehypotheses,notuniversalnonlinearityornumericalerrorcertificates.'}
    (P/'SELF_CONSUMER_STRENGTH4_V1_RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='cells'},indent=2))

if __name__=='__main__':main()
