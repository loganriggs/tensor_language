"""Input RMS versus quadratic interaction, frozen programs and cached interventions.
A FP64 polarization/accounting relative<=1e-9; prior margin replay<=1e-4 nats.
B normalization correction physical norm<=.25 full interaction each target family/program.
C frozen-denominator margin cross RMS>=.75 actual, sign agreement>=.9 each family/program.
B/C test numerator dominance; null is essential input-normalization coupling.
CPU only: 64 pairs x 4 corners; native three-readout contractions, two-logit tails.
No fit, no full-vocabulary tensor, no standalone replacement or unique attribution.
"""
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from native_relation_split_v1 import evaluate
from native_suffix_mlp16_producer_v1 import digest


def cross(a):
    return a[1]-a[2]-a[3]+a[0]


def rel(a,b):
    return float((a-b).norm()/b.norm().clamp_min(1e-30))


@torch.no_grad()
def main():
    torch.set_num_threads(2)
    p=Path(__file__).parent;out=p/'NATIVE_SUFFIX_INPUT_NORMALIZATION_V1.json';assert not out.exists()
    binding=json.loads((p/'NATIVE_SUFFIX_UPSTREAM_V1_BINDING.json').read_text())['files']
    assert all(digest(k)==v for k,v in binding.items())
    up_result=json.loads((p/'NATIVE_SUFFIX_UPSTREAM_V1_RESULT.json').read_text())
    assert digest(p/'NATIVE_SUFFIX_UPSTREAM_V1_PORTS.pt')==up_result['cache_sha256']
    load=lambda name:torch.load(p/name,weights_only=True,map_location='cpu')
    up=load('NATIVE_SUFFIX_UPSTREAM_V1_PORTS.pt');cache=load('NATIVE_RELATION_OUTPUT_FRESH_V1_ENDPOINTS.pt')
    program=load('NATIVE_RELATION_SPLIT_V1.pt');split=load('NATIVE_RELATION_OUTPUT_SPLIT_V1.pt')
    writer=split['splits']['ambient']['private_writers'].double()
    rows=json.loads((p/'NATIVE_RELATION_OUTPUT_FRESH_V1_ROWS.json').read_text())['rows']
    prior=json.loads((p/'NATIVE_SUFFIX_MLP16_PRODUCER_V1.json').read_text());assert prior['pred_a']
    state=torch.load(next(k for k in binding if k.endswith('/pytorch_model.bin')),weights_only=True,mmap=True,map_location='cpu')
    left,right,down=[state['transformer.h.17.mlp.'+n+'.weight'].double() for n in ('Left','Right','Down')]
    folded=program['readouts'].double()@down
    bias=program['readouts'].double()@state['transformer.h.17.mlp.Down_bias'].double()
    ubias=torch.stack([c['bias'] for c in program['components']]).double()
    def aq(x):
        a,b=evaluate(program,x);return a+b-ubias
    def eq(x):return ((x@left.T)*(x@right.T))@folded.T
    def ac(dp,dg):
        vals=[]
        for c in program['components']:
            v=c['rest_readers'];rad=c['leading_radial']+c['rest_radial']
            vals.append((dp@c['a'])*(dg@c['b'])+(dg@c['a'])*(dp@c['b'])+
                2*((dp@v.T)*(dg@v.T)*c['rest_coefficients']).sum()+2*rad*(dp@dg))
        return torch.stack(vals)
    def ec(dp,dg):return ((dp@left.T)*(dg@right.T)+(dg@left.T)*(dp@right.T))@folded.T
    residual=up['ports']['residual'].double()
    producer=up['block17_lambdas'][0].double()*up['ports']['mlp16_output'].double()
    background=residual-producer;pre=cache['ports']['pre'];h=pre+cache['ports']['native_output']
    eps=torch.finfo(torch.float32).eps;records=[];errors=[];replays=[]
    for i,row in enumerate(rows):
        b,d=2*i,2*i+1;u0=pre[b].double();dp=producer[d]-producer[b];dg=background[d]-background[b]
        corners=torch.stack([u0,u0+dp+dg,u0+dp,u0+dg])
        s=corners.square().mean(-1)+eps
        reader=state['lm_head.weight'][[row['donor_answer_id'],row['donor_foil_id']]].float()
        def margin(scalar):
            edited=h[b]+((scalar-scalar[:1])@writer.T).float()
            z=30*torch.tanh(F.linear(F.rms_norm(edited,(1152,)),reader)/30)
            return (z[:,0]-z[:,1]).double()
        entry=dict(row_id=row['row_id'],family=row['family'],direction=row['direction'],input_denominator_ratios=(s/s[0]).tolist(),programs={})
        for name,qfun,cfun,bb,oldkey in [('approximate',aq,ac,ubias,'approximate_effects'),('exact',eq,ec,bias,'exact_effects')]:
            q=qfun(corners);actual=q/s[:,None]+bb;fixed=q/s[0]+bb
            numerator=cfun(dp,dg)/s[0];errors.append(rel(cross(fixed),numerator))
            correction=cross(q*(1/s[:,None]-1/s[0]));total=cross(actual)
            errors.append(rel(numerator+correction,total))
            ma,mf=margin(actual),margin(fixed)
            old=torch.tensor(prior['records'][i][oldkey],dtype=torch.float64)
            replays.append(float((ma-ma[0]-old).abs().max()))
            entry['programs'][name]=dict(scalar_cross=total.tolist(),numerator_cross=numerator.tolist(),normalization_correction=correction.tolist(),
                physical_cross=(total@writer.T).tolist() if False else None,
                actual_margin_effects=(ma-ma[0]).tolist(),fixed_margin_effects=(mf-mf[0]).tolist(),
                actual_margin_cross=float(cross(ma)),fixed_margin_cross=float(cross(mf)))
            del entry['programs'][name]['physical_cross']
        records.append(entry)
    cells=[]
    for family in sorted(set(r['family'] for r in records)):
        local=[r for r in records if r['family']==family]
        for name in ('approximate','exact'):
            rr=[r['programs'][name] for r in local]
            t=lambda key:torch.tensor([r[key] for r in rr],dtype=torch.float64)
            full=t('scalar_cross')@writer.T;num=t('numerator_cross')@writer.T;corr=t('normalization_correction')@writer.T
            ma,mf=t('actual_margin_cross'),t('fixed_margin_cross')
            cells.append(dict(family=family,program=name,n=len(rr),
                normalization_relative_physical_norm=float(corr.norm()/full.norm()),numerator_relative_physical_norm=float(num.norm()/full.norm()),
                numerator_full_physical_cosine=float((num*full).sum()/num.norm()/full.norm()),
                fixed_actual_margin_cross_rms_ratio=float(mf.norm()/ma.norm()),margin_cross_sign_agreement=float((ma.sign()==mf.sign()).double().mean()),
                fixed_actual_margin_cross_relative_error=rel(mf,ma),actual_margin_cross_meanabs=float(ma.abs().mean()),fixed_margin_cross_meanabs=float(mf.abs().mean())))
    target=[c for c in cells if c['family'] in ('A1','A2')]
    result=dict(pred_a=max(errors)<=1e-9 and max(replays)<=1e-4,
        pred_b=all(c['normalization_relative_physical_norm']<=.25 for c in target),
        pred_c=all(c['fixed_actual_margin_cross_rms_ratio']>=.75 and c['margin_cross_sign_agreement']>=.9 for c in target),
        max_algebra_relative_error=max(errors),max_prior_margin_replay_nats=max(replays),cells=cells,records=records,
        source_producer_sha256=digest(p/'NATIVE_SUFFIX_MLP16_PRODUCER_V1.json'),source_script_sha256=digest(__file__),
        scope='Base input denominator accounting of a conditional component-only intervention; native final RMS/tanh retained. No fitting or whole-module attribution. Norm ratios are not additive variance fractions.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))


if __name__=='__main__':main()
