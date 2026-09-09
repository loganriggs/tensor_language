"""An invariant summary encoding predicts identity for a record-only summary swap."""
import hashlib
import json
from pathlib import Path
import torch

BASE=Path(__file__).resolve().parent
INPUT=BASE/'SUMMARY_LAYOUT_MEDIATION_V1_ROWS.pt'
OUT=BASE/'SUMMARY_INVARIANT_INTERCHANGE_BOUND_V1.json'


def main():
    torch.set_num_threads(2);assert not OUT.exists()
    assert hashlib.sha256(INPUT.read_bytes()).hexdigest()=='3dfdb8db3c3206b3ecfb674468fc0bfe036127a97060308de826940d7224a22e'
    data=torch.load(INPUT,map_location='cpu',weights_only=True);groups={};cases=[]
    for pop,variants in data.items():
        groups[pop]={}
        for name,block in variants.items():
            logits=block['query_logits'];effect=logits['01']-logits['00'];effect-=effect.mean(-1,keepdim=True)
            norms=effect.square().mean(-1).sqrt();groups[pop][name]={}
            for hop in range(4):
                indices=torch.tensor([i for i,m in enumerate(block['metadata']) if m['hop']==hop]);values=norms[indices]
                group_rms=float(effect[indices].square().mean().sqrt());error=group_rms/max(group_rms,1e-6)
                index=int(indices[values.argmax()]);groups[pop][name][str(hop)]={
                    'pairs':len(indices),'native_summary_swap_effect_rms':group_rms,
                    'invariant_zero_prediction_relative_error':error,'one_percent_bar_rejected':error>.01,
                    'max_case':{**block['metadata'][index],'centered_logit_effect':effect[index].tolist(),
                        'native_summary_swap_effect_rms':float(norms[index])}}
            for i,m in enumerate(block['metadata']):
                norm=float(norms[i]);cases.append({'population':pop,'permutation':name,**m,
                    'native_summary_swap_effect_rms':norm,'zero_prediction_relative_error':norm/max(norm,1e-6)})
    result={'scope':'causal-interchange obstruction for summary encodings invariant under the fixed record reorderings, with recipient prefix fixed',
        'formula':'encoded donor summary equals recipient summary implies abstract swap is identity, hence predicted output effect zero',
        'groups':groups,'cases':cases,'all_groups_reject':all(g['one_percent_bar_rejected'] for p in groups.values() for v in p.values() for g in v.values()),
        'total_pairs':len(cases),'native_coefficients_removed':0,
        'limits':'Does not rule out order-sensitive summaries, arbitrary full-model decompositions, or weaker absolute-error requirements; no factual constant-summary replacement evaluated',
        'input_sha256':hashlib.sha256(INPUT.read_bytes()).hexdigest()}
    OUT.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'all_groups_reject':result['all_groups_reject'],'pairs':len(cases),
        'rms_range':[min(c['native_summary_swap_effect_rms'] for c in cases),max(c['native_summary_swap_effect_rms'] for c in cases)],
        'cases_above_floor':sum(c['native_summary_swap_effect_rms']>=1e-6 for c in cases)},indent=2))


if __name__=='__main__':main()
