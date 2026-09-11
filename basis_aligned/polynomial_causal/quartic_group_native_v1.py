"""Developmental native test of frozen2output/16quadratic groups.
A densecoeff/projectednative identity<=1e-8. B compact/reference physical<=.1 and
swaprelative<=.1/sign>=.9 eachseed/family; C compactreplicas physical<=.25 andswap<=.25/sign>=.9.
D compact/reference removalCEmeanabsdiff<=.02 eachseed/family. Swapfloor1e-4, >=4live.
Reference is projectednative quartic group, not fullquartic or wholemodel.
"""
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from quartic_group_program_v1 import run
from sparse_path_stability_atlas_v1 import digest
from stable_path_native_cache_v1 import relative


@torch.no_grad()
def main():
    torch.set_num_threads(2);p=Path(__file__).parent;out=p/'QUARTIC_GROUP_NATIVE_V1.json';assert not out.exists()
    gp=p/'QUARTIC_GROUP_PROGRAM_V1.pt';gr=json.loads((p/'QUARTIC_GROUP_PROGRAM_V1.json').read_text());assert gr['pred_a'] and digest(gp)==gr['artifact_sha256']
    programs=torch.load(gp,weights_only=True,map_location='cpu')['programs']
    port=p/'QUARTIC_GROUP_PORTS_V1_PORTS.pt';pr=json.loads((p/'QUARTIC_GROUP_PORTS_V1_RESULT.json').read_text());assert pr['pred_a'] and pr['pred_b'] and pr['pred_c'] and digest(port)==pr['artifact_sha256']
    cache16=torch.load(port,weights_only=True,map_location='cpu');assert cache16['group_sha256']==digest(gp)
    rowfile=p/'NATIVE_RELATION_OUTPUT_FRESH_V1_ROWS.json';assert digest(rowfile)==cache16['rows_sha256'];rows=json.loads(rowfile.read_text())['rows']
    binding=json.loads((p/'QUARTIC_GROUP_PORTS_V1_BINDING.json').read_text())['files'];assert all(digest(k)==v for k,v in binding.items())
    old=torch.load(p/'NATIVE_RELATION_OUTPUT_FRESH_V1_ENDPOINTS.pt',weights_only=True,map_location='cpu')['ports']
    mr=json.loads((p/'SPARSE_QUARTIC_CEILING_V1_RESULT.json').read_text());mp=p/'SPARSE_QUARTIC_CEILING_V1_MODES.pt';assert mr['pred_a'] and digest(mp)==mr['artifact_sha256']
    modes=[m for m in torch.load(mp,weights_only=True,map_location='cpu')['modes'] if m['metric']=='centered']
    state=torch.load(next(k for k in binding if k.endswith('/pytorch_model.bin')),weights_only=True,mmap=True,map_location='cpu')
    u=state['lm_head.weight'].double();mean=u.mean(0);metric=u.T@u-len(u)*torch.outer(mean,mean);uf=u.float()
    l0,r0,d0,l1,r1,d1=[state[f'transformer.h.{layer}.mlp.{name}.weight'].double() for layer in (16,17) for name in ('Left','Right','Down')]
    scale=float(state['transformer.h.17.lambdas'][0]);x=cache16['input16'].double();den=old['pre'].double().square().mean(-1)+torch.finfo(torch.float32).eps
    def quartic(x):
        producer=((x@l0.T)*(x@r0.T))@d0.T*scale
        return ((producer@l1.T)*(producer@r1.T))@d1.T
    fullwrite=quartic(x)/den[:,None];h=old['pre']+old['native_output']
    targets=torch.tensor([row[side+'_answer_id'] for row in rows for side in ('base','donor')])
    def ce(states):
        values=[]
        for i in range(0,len(states),32):
            logits=30*torch.tanh(F.linear(F.rms_norm(states[i:i+32],(1152,)),uf)/30)
            values.append(F.cross_entropy(logits,targets[i:i+32],reduction='none').double())
        return torch.cat(values)
    baseline=ce(h)
    def effects(w):
        zero=ce(h-w.float())-baseline;swapped=h[::2]+(w[1::2]-w[::2]).float();swaps=[]
        for i,row in enumerate(rows):
            reader=uf[[row['donor_answer_id'],row['donor_foil_id']]]
            logits=30*torch.tanh(F.linear(F.rms_norm(torch.stack([h[2*i],swapped[i]]),(1152,)),reader)/30)
            swaps.append(float((logits[1,0]-logits[1,1])-(logits[0,0]-logits[0,1])))
        return zero,torch.tensor(swaps,dtype=torch.float64)
    def compare(wa,wb,ea,eb):
        za,sa=ea;zb,sb=eb;families=[]
        for family in sorted({r['family'] for r in rows}):
            ids=torch.tensor([i for i,r in enumerate(rows) if r['family']==family]);ep=(2*ids[:,None]+torch.tensor([0,1])).flatten()
            a,b=sa[ids],sb[ids];live=torch.maximum(a.abs(),b.abs())>=1e-4
            families.append(dict(family=family,swap_relative_rms=relative(a,b),swap_live=int(live.sum()),
                swap_sign_agreement=float((a[live].sign()==b[live].sign()).double().mean()) if live.any() else None,
                swap_meanabs=[float(a.abs().mean()),float(b.abs().mean())],zero_ce_meanabs_disagreement=float((za[ep]-zb[ep]).abs().mean()),
                zero_ce_mean=[float(za[ep].mean()),float(zb[ep].mean())]))
        return dict(physical_write_relative_rms=relative(wa,wb),families=families)
    errors=[];reports=[];compact=[];scored=[];raw=[]
    for program in programs:
        m=next(m for m in modes if m['seed']==program['seed']);reads=x@m['bank'];phi=reads[:,m['terms']].prod(1)*m['multiplicity_root']
        reference=(phi@m['scalar_coefficients'][:2].T)@m['physical_writer'][:,:2].T
        direct=quartic(reads@m['bank'].T)@(metric@m['physical_writer'][:,:2])@m['physical_writer'][:,:2].T
        errors.append(float((direct-reference).norm()/reference.norm()))
        reference/=den[:,None];w=run(program,x,den)['write'];ew,er=effects(w),effects(reference);comparison=compare(w,reference,ew,er)
        comparison.update(seed=program['seed'],component_norm_over_full_quartic=float(reference.norm()/fullwrite.norm()),
            component_norm_over_native_mlp17=float(reference.norm()/old['native_output'].norm()))
        reports.append(comparison);compact.append(w);scored.append(ew)
        raw.append(dict(seed=program['seed'],compact_zero_ce=ew[0].tolist(),reference_zero_ce=er[0].tolist(),compact_swaps=ew[1].tolist(),reference_swaps=er[1].tolist()))
    replica=compare(*compact,*scored)
    def swap_pass(r,threshold):return all(f['swap_relative_rms']<=threshold and f['swap_live']>=4 and f['swap_sign_agreement']>=.9 for f in r['families'])
    result=dict(pred_a=max(errors)<=1e-8 and all(torch.isfinite(w).all().item() for w in compact),
        pred_b=all(r['physical_write_relative_rms']<=.1 and swap_pass(r,.1) for r in reports),
        pred_c=replica['physical_write_relative_rms']<=.25 and swap_pass(replica,.25),
        pred_d=all(f['zero_ce_meanabs_disagreement']<=.02 for r in reports for f in r['families']),
        maximum_projected_native_error=max(errors),reports=reports,replica=replica,effects=raw,program_sha256=digest(gp),ports_sha256=digest(port),script_sha256=digest(__file__),
        scope='Frozen partial quartic group on existing developmental rows, native source/RMS/background retained. Approximation fidelity and replica check, not selective semantic/OOD/fullmodel extraction.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='effects'},indent=2));assert result['pred_a']


if __name__=='__main__':main()
