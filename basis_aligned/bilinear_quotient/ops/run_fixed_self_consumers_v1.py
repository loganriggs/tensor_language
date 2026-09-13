#!/usr/bin/env python3
# BQGATE:160prefixes;2560suffixreadouts;240seconds.
"""pred_a prior unpruned/all-node-removal scalar endpoints replay<=1e-4.
pred_b final-position-only predicts all-node joint target change<=10% each regional group.
pred_c sum of separate position-removal changes predicts all-node change<=10%
regional,<=20%FineWeb, both target and control endpoints.
Null: earlier consumers matter or nonlinear suffix defeats additive composition.
Price160pristineprefixes,2560suffixreadouts,240seconds; frozen scalar fields/writer.
Historical panel and supplied context; no fit, freshOOD or wholemodeladoption.
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
    files=json.loads((P/'FIXED_SELF_CONSUMERS_V1_BINDING.json').read_text())['files']
    assert all(sha256(Path(k).read_bytes()).hexdigest()==v for k,v in files.items())
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print('160prefixes;2560suffixreadouts;240seconds');return
    assert not (P/'FIXED_SELF_CONSUMERS_V1_RESULT.json').exists()
    signal.alarm(240);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;tic=time.perf_counter()
    model=load_model_fast().cuda().eval()
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
    cells=[]
    for index,row in enumerate(rows):
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
            rho=z.square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps;node=amplitude.square()/rho*writer
            final=torch.zeros_like(node);final[:,-1]=node[:,-1];earlier=node-final
            states['unpruned'].append(full.float());states['all'].append((full-node).float())
            states['final'].append((full-final).float());states['earlier'].append((full-earlier).float())
        scores=[]
        for name in states:
            outcomes=[]
            for initial in states[name]:
                state=initial
                for block in model.transformer.h[11:]:state,_=block(state,v1,x0)
                logits=(30*torch.tanh(model.lm_head(F.rms_norm(state[:,-1],(1152,)))/30))[0]
                if index<96:target=logits[row['uk_id']]-logits[row['us_id']];control=logits[row['control_ids'][0]]-logits[row['control_ids'][1]]
                else:target=-logits.log_softmax(-1)[198];control=logits[198]-logits[11]
                outcomes.append([float(target),float(control)])
            scores.append(outcomes)
        scores=torch.tensor(scores,dtype=torch.float64);effects=scores[:,3]-scores[:,1]-scores[:,2]+scores[:,0]
        cells.append(dict(row=index,scores=scores.tolist(),interactions=effects.tolist()))
    old=json.loads((P/'DIAGONAL_MLP10_NATIVE_V1_RESULT.json').read_text())['cells'];removed=json.loads((P/'FIXED_SELF_NODE_REMOVAL_V1_RESULT.json').read_text())['cells']
    scores=torch.tensor([c['scores'] for c in cells],dtype=torch.float64)
    replay=[float((scores[:,j]-torch.tensor([c['generated_scores'] for c in reference],dtype=torch.float64)).abs().max()) for j,reference in [(0,old),(1,removed)]]
    effects=torch.tensor([c['interactions'] for c in cells],dtype=torch.float64)
    changes=effects[:,0:1]-effects[:,1:];groups=[]
    for lo,hi,name in [(24*k,24*(k+1),'regional'+str(k)) for k in range(4)]+[(96+16*k,112+16*k,'FineWeb'+str(k)) for k in range(4)]:
        allchange,last,early=changes[lo:hi].unbind(1);den=allchange.norm(dim=0)
        groups.append(dict(name=name,all_node_change_norm=den.tolist(),final_only_error=((last-allchange).norm(dim=0)/den).tolist(),
            earlier_only_error=((early-allchange).norm(dim=0)/den).tolist(),composition_error=((last+early-allchange).norm(dim=0)/den).tolist()))
    result={'pred_a':max(replay)<=1e-4,'pred_b':all(g['final_only_error'][0]<=.1 for g in groups[:4]),
        'pred_c':all(max(g['composition_error'])<=(.1 if g['name'].startswith('regional') else .2) for g in groups),
        'reference_replay_maxabs':replay,'groups':groups,'cells':cells,'seconds':time.perf_counter()-tic,
        'scope':'Fixed selfnode position-consumer removals in generatedbranches; all/final/earlier, native nonlinear suffix. Frozen160historicalprefixes andsuppliedcontext/fields. Additivity tested, notassumed; nofreshOODorindependenttextcircuit.'}
    (P/'FIXED_SELF_CONSUMERS_V1_RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='cells'},indent=2))

if __name__=='__main__':main()
