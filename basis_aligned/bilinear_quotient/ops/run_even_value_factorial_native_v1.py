#!/usr/bin/env python3
# BQGATE:312bodyforwards;104prefixes<=247tokens;120seconds;no fitting.
"""pred_a old/new native anchors<=1e-5 and shared/separate writes<=1e-10.
pred_b full8corner Mobius reconstruction<=1e-10.
pred_c pair-only full-removal effect error<=.01 EACHreadout/family/naturalhalf.
Null: three-way interaction cannot be omitted from this conditional decomposition.
Price312bodyforwards,104prefixes,120seconds; masks0,5,6, all other corners frozen.
"""
import os,sys,json,time,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal'
sys.path[:0]=[str(Path(__file__).parent),str(P),str(ROOT)]
import torch
import torch.nn.functional as F
from sparse_path_stability_atlas_v1 import digest
from regional_cue_row_check_v1 import validate
from key_span_reuse_v1 import KeySpanParent
from even_key_value_native_backend_v1 import FullValueComponents
from even_value_shared_graph_v1 import SharedGraph
from even_value_factorial_v1 import coefficients,reconstruct
STEM='EVEN_VALUE_FACTORIAL_NATIVE_V1'
MASKS=[0,5,6]


def setup(device):
    p={k:v.to(device) for k,v in torch.load(P/'EVEN_VALUE_SHARED_GRAPH_PACKED_V1_PROGRAM.pt',weights_only=True).items()}
    graph=SharedGraph(p)
    separate=FullValueComponents(torch.load(P/'EVEN_KEY_VALUE_BANK_V1_PROGRAM.pt',weights_only=True),device)
    scalar=KeySpanParent({k:v.to(device) for k,v in torch.load(P/'KEY_SPAN_REUSE_V1_PROGRAM.pt',weights_only=True).items()})
    writer=torch.load(P/'SCALAR_PRODUCER_NATIVE_LIFT_V1_ARTIFACT.pt',weights_only=True)['writers'][1].to(device)
    return graph,separate,scalar,writer


@torch.no_grad()
def cpu_control():
    torch.set_num_threads(2);torch.manual_seed(1709141220)
    graph,separate,scalar,writer=setup('cpu')
    current=F.rms_norm(torch.randn(2,17,1152),(1152,))
    first=torch.randn(2,17,128);tokens=torch.zeros(2,17,dtype=torch.long)
    states=graph.state(current,first_values=first)
    even,odd=separate(current,first)
    selected=scalar.scalar(current,tokens,1)[...,None]*writer
    errors=[]
    for mask in [5,6]:
        gains=[int(bool(mask&bit)) for bit in [1,2,4]]
        expected=selected+odd if mask==5 else even-selected+odd
        actual=graph.write(states,gains)
        errors.append(float((actual-expected).norm()/expected.norm()))
    assert max(errors)<=1e-10,errors
    return errors


@torch.no_grad()
def main():
    binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files']
    assert all(digest(k)==v for k,v in binding.items())
    panel=json.loads((P/'SCALAR_JOINT_KEY_CONFIRMATION_V1_ROWS.json').read_text())
    regional,natural=panel['regional'],panel['natural']
    validate(regional[:48]);validate(regional[48:],expected_token_differences=2)
    assert len(regional)==72 and len(natural)==32
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print('312bodyforwards:104prefixes masks0,5,6; CPU shared/separate write errors',cpu_control());return
    out=P/(STEM+'_RESULT.json');artifact=P/(STEM+'_ARTIFACT.pt')
    assert not out.exists() and not artifact.exists()
    start=time.perf_counter();signal.alarm(120);torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    from fastload import load_model_fast
    model=load_model_fast().cuda().eval();graph,separate,scalar,writer=setup('cuda')
    context={};checks=[];count=0
    def hook(module,args,output):
        mask=context['mask']
        if not mask:return output
        current=args[0];first=args[1].reshape(*current.shape[:2],9,128)[:,:,8]
        states=graph.state(current,first_values=first)
        delta=graph.write(states,[int(bool(mask&bit)) for bit in [1,2,4]])
        even,odd=separate(current,first)
        selected=scalar.scalar(current,context['tokens'],1)[...,None]*writer
        reference=selected+odd if mask==5 else even-selected+odd
        checks.append(float((delta-reference).norm()/reference.norm().clamp_min(1e-30)))
        return output[0]-delta.to(output[0].dtype),output[1]
    handle=model.transformer.h[9].attn.register_forward_hook(hook)
    def forward(row,mask):
        nonlocal count
        ids=torch.tensor([row['ids']],device='cuda');context.update(tokens=ids,mask=mask)
        x=F.rms_norm(model.transformer.wte(ids),(1152,));x0=x;v1=None
        for block in model.transformer.h:x,v1=block(x,v1,x0)
        count+=1
        return (30*torch.tanh(model.lm_head(F.rms_norm(x[:,-1],(1152,)))/30))[0]
    captured=torch.zeros(72,3,2,dtype=torch.float64);captured_ce=torch.zeros(32,3,dtype=torch.float64)
    try:
        for i,row in enumerate(regional):
            for j,mask in enumerate(MASKS):
                scores=forward(row,mask)
                captured[i,j,0]=(scores[row['uk_id']]-scores[row['us_id']]).cpu()
                captured[i,j,1]=(scores[row['control_ids'][0]]-scores[row['control_ids'][1]]).cpu()
        for i,row in enumerate(natural):
            for j,mask in enumerate(MASKS):captured_ce[i,j]=-forward(row,mask).log_softmax(-1)[198].cpu()
    finally:handle.remove()
    assert count==312
    old=torch.load(P/'EVEN_KEY_FULL_VALUE_NATIVE_V1_ARTIFACT.pt',weights_only=True)
    rem=torch.load(P/'EVEN_KEY_REMAINDER_NATIVE_V1_ARTIFACT.pt',weights_only=True)
    rel=lambda a,b:float((a-b).norm()/b.norm().clamp_min(1e-30))
    anchors=dict(regional=rel(captured[:,0],old['regional'][:,0]),ce=rel(captured_ce[:,0],old['newline_ce'][:,0]))
    reg=torch.zeros(8,72,2,dtype=torch.float64);ce=torch.zeros(8,32,dtype=torch.float64)
    # Known masks use previous frozen removals, gated by identical native anchors.
    for mask,arm in [(0,0),(1,5),(3,1),(4,2),(7,3)]:
        reg[mask]=old['regional'][:,arm];ce[mask]=old['newline_ce'][:,arm]
    reg[2]=rem['regional'][:,1];ce[2]=rem['newline_ce'][:,1]
    for mask,j in [(5,1),(6,2)]:reg[mask]=captured[:,j];ce[mask]=captured_ce[:,j]
    rg=coefficients(reg);cg=coefficients(ce)
    errors=dict(regional=rel(reconstruct(rg),reg),ce=rel(reconstruct(cg),ce))
    cells=[]
    for family in range(3):
        ix=[i for i,r in enumerate(regional) if r['family']==family]
        target=reg[7,ix]-reg[0,ix];triple=rg[7,ix]
        ratios=(triple.norm(dim=0)/target.norm(dim=0).clamp_min(1e-30)).tolist()
        cells.append(dict(family=family,triple_to_full_effect=ratios,passed=max(ratios)<=.01,
            coefficient_norms=rg[1:,ix].norm(dim=1).tolist(),
            paired_target_coefficient_norms=(rg[1:,ix[::2],0]-rg[1:,ix[1::2],0]).norm(dim=1).tolist()))
    ncells=[]
    for family in range(2):
        ix=[i for i,r in enumerate(natural) if r['family']==family]
        ratio=float(cg[7,ix].norm()/(ce[7,ix]-ce[0,ix]).norm().clamp_min(1e-30))
        ncells.append(dict(family=family,triple_to_full_effect=ratio,passed=ratio<=.01,
                           coefficient_norms=cg[1:,ix].norm(dim=1).tolist()))
    A=max(anchors.values())<=1e-5 and max(checks)<=1e-10
    torch.save(dict(regional_cube=reg,newline_ce_cube=ce,regional_coefficients=rg,newline_coefficients=cg,
                    captured=captured,captured_ce=captured_ce),artifact)
    result={'pred_a':A,'pred_b':max(errors.values())<=1e-10,
            'pred_c':A and all(c['passed'] for c in cells+ncells),
            'anchors':anchors,'max_live_shared_write_error':max(checks),'mobius_replay':errors,
            'regional':cells,'newline':ncells,'coefficient_order':['S','R','SR','O','SO','RO','SRO'],
            'body_forwards':count,'seconds':time.perf_counter()-start,'artifact_sha256':digest(artifact),
            'source_shas':binding,'scope':'Full8corner S/R/O removal decomposition assembled after native-anchor replay. Fixed weights, existing developmentpanel. Pair-only adequacy is conditional response-order compression, not new OOD/circuit naming/model fitting.'}
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='source_shas'}),flush=True)


if __name__=='__main__':main()
