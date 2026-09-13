"""Native intervention-strength transfer of the shared composed response graph."""
import sys,json,time,signal,itertools
from pathlib import Path
import torch
import torch.nn.functional as F
P=Path(__file__).resolve().parent;sys.path.insert(0,str(P.parent/'bilinear_quotient/ops'))
from fastload import load_model_fast
import composed_mlp10_context_v1 as shared
from composed_mlp10_inputs_v1 import execute as legacy_execute

@torch.no_grad()
def main(indices=(0,96), selected_pairs=None, legacy=False, output_name='COMPOSED_STRENGTH_GRID_V1_RESULT.json'):
    signal.alarm(120);torch.set_num_threads(2);start=time.perf_counter();model=load_model_fast().eval();assert next(model.parameters()).device.type=='cpu'
    rows=json.loads((P/'FIRST_TOKEN_PATH_FRESH_V1_ROWS.json').read_text())['rows']+json.loads((P/'FIRST_TOKEN_REMOVAL_NEWLINE_FRESH_V1_ROWS.json').read_text())['rows']
    child=torch.load(P/'CROSSFIRST_STATE_EXECUTOR_V1_ARTIFACT.pt',weights_only=True)['fields'];parent=torch.load(P/'CROSSFIRST_HIERARCHY_V1_ARTIFACT.pt',weights_only=True)['parent_fields']
    program=torch.load(P/'MLP9_CROSSFIRST_RESPONSE_V1_PROGRAM.pt',weights_only=True);w=program['direction'];b9=model.transformer.h[9];b10=model.transformer.h[10]
    mats=[getattr(b10.attn,k).weight.double() for k in ['c_q','c_k','c_q2','c_k2','c_v']];L,R,D=[getattr(b10.mlp,k).weight.double() for k in ['Left','Right','Down']]
    strengths=[0.,.5,1.,1.5];cells=[]
    for index in indices:
        row=rows[index];ids=torch.tensor([row['ids']]);x=F.rms_norm(model.transformer.wte(ids),(1152,));x0=x;v1=None
        for block in model.transformer.h[:9]:x,v1=block(x,v1,x0)
        raw9=b9.lambdas[0]*x+b9.lambdas[1]*x0;att9,v1=b9.attn(F.rms_norm(raw9,(1152,)),v1)
        z9=raw9+att9;m9=b9.mlp(F.rms_norm(z9,(1152,)));h9=z9+m9
        raw0=b10.lambdas[0]*h9+b10.lambdas[1]*x0;z0=raw0+b10.attn(F.rms_norm(raw0,(1152,)),v1)[0]
        h0=(z0+b10.mlp(F.rms_norm(z0,(1152,)))).double();a=child[index][...,None];b=(parent[index]-child[index])[...,None]
        context=shared.prepare(z9,m9.double()-b9.mlp.Down_bias.double(),raw0,z0,v1.double(),program,float(b10.lambdas[0]),mats,float(b10.attn.lamb),b10.attn.c_proj.weight.double())
        native={}
        for label,field in [('c',a),('r',b)]:
            for strength in strengths:
                if strength==0:native[label,strength]=(torch.zeros_like(z0).double(),h0);continue
                za=raw9+(att9-(strength*field*w).to(att9.dtype));ha=za+b9.mlp(F.rms_norm(za,(1152,)))
                raw=b10.lambdas[0]*ha+b10.lambdas[1]*x0;zz=raw+b10.attn(F.rms_norm(raw,(1152,)),v1)[0]
                native[label,strength]=(zz.double()-z0.double(),(zz+b10.mlp(F.rms_norm(zz,(1152,)))).double())
        def read(state):
            state=state.float()
            for block in model.transformer.h[11:]:state,_=block(state,v1,x0)
            logits=(30*torch.tanh(model.lm_head(F.rms_norm(state[:,-1],(1152,)))/30))[0]
            if index<96:return torch.tensor([float(logits[row['uk_id']]-logits[row['us_id']]),float(logits[row['control_ids'][0]]-logits[row['control_ids'][1]])],dtype=torch.float64)
            return torch.tensor([float(-logits.log_softmax(-1)[198]),float(logits[198]-logits[11])],dtype=torch.float64)
        for alpha,beta in (list(itertools.product(strengths,repeat=2)) if selected_pairs is None else selected_pairs):
            c,hc=native['c',alpha];r,hr=native['r',beta];norm=(z0.double()+c+r).square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps
            expected=shared.product(dict(child=c,remainder=r,joint_rho=norm),L,R,D)
            pred=shared.product(shared.evaluate(alpha*a,beta*b,context),L,R,D)
            bar=hc+hr-h0;base=read(bar);reference=read(bar+expected)-base;prediction=read(bar+pred)-base
            cell=dict(row=index,alpha=alpha,beta=beta,product_error=float((pred-expected).norm()/expected.norm().clamp_min(1e-30)),zero_product_maxabs=float(pred.abs().max()) if alpha==0 or beta==0 else None,native_effect=reference.tolist(),predicted_effect=prediction.tolist(),absolute_effect_error=(prediction-reference).abs().tolist())
            if legacy:
                old=legacy_execute(z9,m9.double()-b9.mlp.Down_bias.double(),raw0,z0,v1.double(),alpha*a,beta*b,program,float(b10.lambdas[0]),mats,float(b10.attn.lamb),b10.attn.c_proj.weight.double())
                old_product=shared.product(old,L,R,D);old_effect=read(bar+old_product)-base
                cell.update(legacy_product_replay=float((old_product-pred).norm()/pred.norm().clamp_min(1e-30)),legacy_effect=old_effect.tolist(),legacy_vs_new_effect_maxabs=float((old_effect-prediction).abs().max()))
            cells.append(cell)
        print('completed row',index,flush=True)
    panels=[]
    for index in indices:
        nonzero=[c for c in cells if c['row']==index and c['alpha']!=0 and c['beta']!=0]
        ref=torch.tensor([c['native_effect'] for c in nonzero],dtype=torch.float64);pred=torch.tensor([c['predicted_effect'] for c in nonzero],dtype=torch.float64)
        panels.append(dict(row=index,target_effect_error=float((pred[:,0]-ref[:,0]).norm()/ref[:,0].norm()),control_effect_error=float((pred[:,1]-ref[:,1]).norm()/ref[:,1].norm()),max_absolute_error=float((pred-ref).abs().max()),opposite_signs=(pred*ref<0).sum(0).tolist(),material_target_sign_reversals=int(((pred[:,0]*ref[:,0]<0)&(ref[:,0].abs()>=1e-5)).sum())))
    result=dict(pred_a=all(c['product_error']<=.001 for c in cells if c['zero_product_maxabs'] is None) and all(c['zero_product_maxabs']==0 for c in cells if c['zero_product_maxabs'] is not None),pred_b=all(c['target_effect_error']<=.05 for c in panels if c['row']==0) if 0 in indices else None,pred_c=all(c['material_target_sign_reversals']==0 for c in panels if c['row']==0) if 0 in indices else None,cells=cells,panels=panels,seconds=time.perf_counter()-start,scope='Native strength-grid conditional full-cross prediction; two historical prefixes, changed attention/joint norm generated, native additive background and suffix supplied. No language OOD or full intervention-effect sufficiency claim.')
    (P/output_name).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='cells'},indent=2))
if __name__=='__main__':main()
