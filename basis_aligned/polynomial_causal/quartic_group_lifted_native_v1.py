"""Fullsource frozenoutputspace nativeeffect replication; projectedreference control.
A finite/physical replay<=1e-5 Bphysical<=.25 Cswaprelative<=.25/sign>=.9 EVERYfamily/live>=4.
DremovalCEreplicameanabsdisagreement<=.02EVERYfamily. Swapfloor1e-4. No fit.
"""
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from sparse_path_stability_atlas_v1 import digest
from stable_path_native_cache_v1 import relative


@torch.no_grad()
def main():
    torch.set_num_threads(2);p=Path(__file__).parent;out=p/'QUARTIC_GROUP_LIFTED_NATIVE_V1.json';assert not out.exists()
    prior=json.loads((p/'QUARTIC_GROUP_BOUNDARY_V1.json').read_text());source=p/'QUARTIC_GROUP_BOUNDARY_V1_WRITES.pt';assert prior['pred_a'] and digest(source)==prior['artifact_sha256']
    writes=torch.load(source,weights_only=True,map_location='cpu');binding=json.loads((p/'QUARTIC_GROUP_PORTS_V1_BINDING.json').read_text())['files'];assert all(digest(k)==v for k,v in binding.items())
    rows=json.loads((p/'NATIVE_RELATION_OUTPUT_FRESH_V1_ROWS.json').read_text())['rows'];old=torch.load(p/'NATIVE_RELATION_OUTPUT_FRESH_V1_ENDPOINTS.pt',weights_only=True,map_location='cpu')['ports'];h=old['pre']+old['native_output']
    state=torch.load(next(k for k in binding if k.endswith('/pytorch_model.bin')),weights_only=True,mmap=True,map_location='cpu');u=state['lm_head.weight'].float()
    targets=torch.tensor([r[s+'_answer_id'] for r in rows for s in ('base','donor')])
    def ce(x):
        values=[]
        for i in range(0,len(x),32):
            logits=30*torch.tanh(F.linear(F.rms_norm(x[i:i+32],(1152,)),u)/30)
            values.append(F.cross_entropy(logits,targets[i:i+32],reduction='none').double())
        return torch.cat(values)
    baseline=ce(h);reports=[];raw=[]
    for name in ['projected','lifted']:
        zero=[];swap=[]
        for w in writes[name]:
            zero.append(ce(h-w.float())-baseline);swapped=h[::2]+(w[1::2]-w[::2]).float();values=[]
            for i,r in enumerate(rows):
                readers=u[[r['donor_answer_id'],r['donor_foil_id']]]
                z=30*torch.tanh(F.linear(F.rms_norm(torch.stack([h[2*i],swapped[i]]),(1152,)),readers)/30)
                values.append(float((z[1,0]-z[1,1])-(z[0,0]-z[0,1])))
            swap.append(torch.tensor(values,dtype=torch.float64))
        families=[]
        for family in sorted({r['family'] for r in rows}):
            ids=torch.tensor([i for i,r in enumerate(rows) if r['family']==family]);ep=(2*ids[:,None]+torch.tensor([0,1])).flatten()
            a,b=swap[0][ids],swap[1][ids];live=torch.maximum(a.abs(),b.abs())>=1e-4
            families.append(dict(family=family,swap_relative_rms=relative(a,b),swap_live=int(live.sum()),
                swap_sign_agreement=float((a[live].sign()==b[live].sign()).double().mean()) if live.any() else None,
                swap_meanabs=[float(a.abs().mean()),float(b.abs().mean())],zero_ce_meanabs_disagreement=float((zero[0][ep]-zero[1][ep]).abs().mean()),
                zero_ce_mean=[float(z[ep].mean()) for z in zero]))
        reports.append(dict(name=name,physical_relative_rms=relative(*writes[name]),families=families))
        raw.append(dict(name=name,zero_ce=[z.tolist() for z in zero],swaps=[s.tolist() for s in swap]))
    lifted=reports[1];error=abs(reports[0]['physical_relative_rms']-prior['projected_replica_relative_rms'])
    result=dict(pred_a=error<=1e-5 and all(torch.isfinite(torch.tensor(v)).all().item() for r in raw for v in (r['zero_ce'],r['swaps'])),
        pred_b=lifted['physical_relative_rms']<=.25,
        pred_c=all(f['swap_relative_rms']<=.25 and f['swap_live']>=4 and f['swap_sign_agreement']>=.9 for f in lifted['families']),
        pred_d=all(f['zero_ce_meanabs_disagreement']<=.02 for f in lifted['families']),reports=reports,effects=raw,
        source_sha256=digest(source),script_sha256=digest(__file__),scope='Exact fullsource liftedgroup changes priorinputprojection boundary and retains nativeweights. Developmentalreplication only, nocompactlift/semanticselectivity/OODclaim.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='effects'},indent=2));assert result['pred_a']


if __name__=='__main__':main()
