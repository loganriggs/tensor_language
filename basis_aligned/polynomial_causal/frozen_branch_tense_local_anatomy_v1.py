"""Native MLP17-only vs branch donor swap; exact two-token tail, no fitting.
A old gap/swap replay<=1e-4 nats. B MLP transfer>=.1 native gap and>=.05 all4cells.
C bank>=.5 MLP transfer in all4cells, each with MLP>=.05. Original screen unchanged.
"""
import hashlib
import json
from pathlib import Path
import torch


@torch.no_grad()
def main():
    torch.set_num_threads(2);torch.set_default_dtype(torch.float64)
    root=Path(__file__).parent;output=root/'FROZEN_BRANCH_TENSE_LOCAL_ANATOMY_V1.json'
    assert not output.exists()
    stem='FROZEN_BRANCH_TENSE_V6'
    paths=[root/(stem+'_ROWS.json'),root/(stem+'_ENDPOINTS.pt'),root/(stem+'_RESULT.json'),
           root/'SHARED_NODE_CANONICAL_BRANCHES_V1_SPECTRAL.pt']
    rows=json.loads(paths[0].read_text())['rows'];cache=torch.load(paths[1],weights_only=True,map_location='cpu')
    previous=json.loads(paths[2].read_text());assert previous['pred_a']
    assert hashlib.sha256(paths[1].read_bytes()).hexdigest()==previous['cache_sha256']
    saved=torch.load(paths[3],weights_only=True,map_location='cpu')
    binding=json.loads((root/(stem+'_BINDING.json')).read_text())['files']
    checkpoint=Path(next(p for p in binding if p.endswith('/pytorch_model.bin')))
    weights=torch.load(checkpoint,weights_only=True,mmap=True,map_location='cpu')
    u=weights['lm_head.weight'].double()
    writers=torch.linalg.solve_triangular(saved['output_whitener'].double(),saved['nodes'][1]['writers'].double(),upper=True)
    amp=cache['amplitudes'].double();pre=cache['ports']['pre'].double();native=cache['ports']['native_output'].double()
    # Preserve recorded FP32 native addition before using a high-precision tail.
    h=(cache['ports']['pre']+cache['ports']['native_output']).double()
    eps=torch.finfo(torch.float32).eps
    def margin(state,readout,radius_reference=None):
        radius=state if radius_reference is None else radius_reference
        scale=(radius.square().mean()+eps).rsqrt()
        logits=30*torch.tanh((readout@state)*scale/30)
        return logits[0]-logits[1]
    records=[];errors=[]
    for i,row in enumerate(rows):
        if row['family'] not in ('A1','A2'):continue
        base,donor=2*i,2*i+1
        readout=u[[row['donor_answer_id'],row['donor_foil_id']]]
        baseline=margin(h[base],readout)
        gap=float(margin(h[donor],readout)-baseline)
        delta=(amp[donor]-amp[base])@writers.T
        bank=h[base]+delta
        local=pre[base]+native[donor]
        bank_effect=float(margin(bank,readout)-baseline)
        local_effect=float(margin(local,readout)-baseline)
        numerator_bank=float(margin(bank,readout,h[base])-baseline)
        numerator_local=float(margin(local,readout,h[base])-baseline)
        old=previous['records'][i]
        assert old['row_id']==row['row_id']
        errors.extend([abs(gap-old['native_gap']),abs(bank_effect-old['margin_shifts'][2])])
        records.append(dict(row_id=row['row_id'],family=row['family'],direction=old['direction'],
            native_gap=gap,bank_effect=bank_effect,mlp_only_effect=local_effect,
            bank_fixed_norm_effect=numerator_bank,mlp_fixed_norm_effect=numerator_local,
            bank_norm_remainder=bank_effect-numerator_bank,mlp_norm_remainder=local_effect-numerator_local))
    cells=[]
    for family in ('A1','A2'):
        for direction in ('present_to_past','past_to_present'):
            local=[r for r in records if r['family']==family and r['direction']==direction]
            means={k:sum(r[k] for r in local)/len(local) for k in records[0] if k not in ('row_id','family','direction')}
            gap,bank,mlp=means['native_gap'],means['bank_effect'],means['mlp_only_effect']
            cells.append(dict(family=family,direction=direction,means=means,
                mlp_to_native=mlp/gap,bank_to_mlp=bank/mlp if mlp!=0 else None,
                pred_b=mlp>=.05 and mlp/gap>=.1,pred_c=mlp>=.05 and bank/mlp>=.5))
    assert all(torch.isfinite(torch.tensor(list(c['means'].values()))).all() for c in cells)
    a=max(errors)<=1e-4
    result=dict(pred_a=a,pred_b=a and all(c['pred_b'] for c in cells),pred_c=a and all(c['pred_c'] for c in cells),
        max_previous_replay_error=max(errors),cells=cells,records=records,
        is_was_writer_loadings=(u[[318,373]]@writers).tolist(),
        sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},checkpoint_sha256=binding[str(checkpoint)],
        scope='Conditional native MLP17-output swap at the final interface, not upstream sufficiency or a new denominator for the old score. '
              'Exact two-token formula retains full state norm and tanh; fixed-norm arm is explanatory, not native execution. '
              'No data-fitted factors or repair of original capability/transfer failures.')
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('records','sources')},indent=2))


if __name__=='__main__':main()
