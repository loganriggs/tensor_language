"""Token-defined before/city/after/self partition for odd-branch sources."""
import json
from datetime import datetime,timezone
from pathlib import Path

import torch
import torch.nn.functional as F
from even_value_shared_graph_v1 import SharedGraph
from inherited_source_positions_v1 import paired_masks
from odd_source_positions_v1 import source_channels,select_sources


def contextual_masks(rows):
    changed=paired_masks(rows);result=[]
    for i,row in enumerate(rows):
        city=torch.nonzero(changed[i],as_tuple=False).flatten()
        if city.numel()!=1:raise ValueError('Exactly one changed city token required')
        position=int(city[0]);tokens=len(row['ids']);index=torch.arange(tokens)
        parts={'pre':index<position,'city':index==position,
               'post':(index>position)&(index<tokens-1),'self':index==tokens-1}
        total=sum(mask.to(torch.int8) for mask in parts.values())
        if not bool((total==1).all()) or not bool(parts['post'].any()):
            raise ValueError('Source partition must be disjoint, exhaustive and have post-city context')
        result.append(parts)
    return result


@torch.no_grad()
def main():
    root=Path(__file__).resolve().parent;out=root/'ODD_CONTEXTUAL_POSITIONS_V1_CPU_CONTROL.json'
    assert not out.exists();torch.set_num_threads(2);torch.manual_seed(14091720)
    rows=json.loads((root/'SRO_ARTICLE_CORRECTION_V1_ROWS.json').read_text())['rows']
    masks=contextual_masks(rows);graph=SharedGraph(torch.load(root/'EVEN_VALUE_SHARED_GRAPH_PACKED_V1_PROGRAM.pt',weights_only=True))
    errors=[];layout=[];counts=[]
    for i in [0,12,24,36]:
        tokens=len(rows[i]['ids']);current=F.rms_norm(torch.randn(1,tokens,1152),(1152,));first=torch.randn(1,tokens,9,128)
        channels=source_channels(graph,current,first);pieces=[select_sources(channels,masks[i][k][None]) for k in ['pre','city','post','self']]
        reference=graph.state(current,first_values=first[:,:,8])[2]
        errors.append(float((sum(pieces)-reference).norm()/reference.norm().clamp_min(1e-30)))
        flat=source_channels(graph,current,first.reshape(1,tokens,1152))
        layout.append(float((flat-channels).norm()/channels.norm().clamp_min(1e-30)))
        counts.append({k:int(v.sum()) for k,v in masks[i].items()})
    result={'utc':datetime.now(timezone.utc).isoformat(),'pred_a':max(errors)<=1e-10 and max(layout)==0,'source_partition_errors':errors,'layout_errors':layout,'representative_counts':counts,'rows':len(rows),'city_positions':sorted(set(int(torch.nonzero(x['city'])[0]) for x in masks)),'additional_model_scalars':0,'scope':'Token-defined exact source partition on synthetic normalized current states. No native causal result; boundaries use only paired token differences and final position.'}
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
