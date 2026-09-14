"""Frozen framing versus quoted-clause partition of odd-branch sources."""
import json
from datetime import datetime,timezone
from pathlib import Path

import torch
import torch.nn.functional as F
from even_value_shared_graph_v1 import SharedGraph
from odd_contextual_positions_v1 import contextual_masks
from odd_source_positions_v1 import source_channels,select_sources

QUOTE_ID=366


def semantic_masks(rows):
    contextual=contextual_masks(rows);result=[]
    for row,base in zip(rows,contextual):
        ids=torch.tensor(row['ids']);quote=torch.nonzero(ids==QUOTE_ID,as_tuple=False).flatten()
        if quote.numel()!=1:raise ValueError('Exactly one frozen quote boundary required')
        q=int(quote[0]);city=int(torch.nonzero(base['city'])[0]);index=torch.arange(len(ids))
        parts={'pre':base['pre'],'city':base['city'],'framing':(index>city)&(index<=q),
               'clause':(index>q)&(index<len(ids)-1),'self':base['self']}
        total=sum(x.to(torch.int8) for x in parts.values())
        if not bool((total==1).all()) or not bool(parts['framing'].any()) or not bool(parts['clause'].any()):
            raise ValueError('Semantic source masks must be disjoint, exhaustive and nonempty')
        if not torch.equal(parts['framing']|parts['clause'],base['post']):
            raise ValueError('Framing and clause must exactly partition prior post-city mask')
        result.append(parts)
    return result


@torch.no_grad()
def main():
    root=Path(__file__).resolve().parent;out=root/'ODD_SEMANTIC_POSITIONS_V1_CPU_CONTROL_V2.json'
    assert not out.exists();torch.set_num_threads(2);torch.manual_seed(14091727)
    rows=json.loads((root/'SRO_ARTICLE_CORRECTION_V1_ROWS.json').read_text())['rows'];masks=semantic_masks(rows)
    graph=SharedGraph(torch.load(root/'EVEN_VALUE_SHARED_GRAPH_PACKED_V1_PROGRAM.pt',weights_only=True))
    errors=[];post_errors=[];counts=[]
    for i in [0,12,24,36]:
        tokens=len(rows[i]['ids']);current=F.rms_norm(torch.randn(1,tokens,1152),(1152,));first=torch.randn(1,tokens,9,128)
        channels=source_channels(graph,current,first);pieces={k:select_sources(channels,v[None]) for k,v in masks[i].items()}
        reference=graph.state(current,first_values=first[:,:,8])[2]
        errors.append(float((sum(pieces.values())-reference).norm()/reference.norm().clamp_min(1e-30)))
        prior=select_sources(channels,(masks[i]['framing']|masks[i]['clause'])[None])
        post_errors.append(float((pieces['framing']+pieces['clause']-prior).norm()/prior.norm().clamp_min(1e-30)))
        counts.append({k:int(v.sum()) for k,v in masks[i].items()})
    result={'utc':datetime.now(timezone.utc).isoformat(),'pred_a':max(errors)<=1e-10 and max(post_errors)<=1e-10,'source_partition_errors':errors,'post_partition_errors':post_errors,'representative_counts':counts,'rows':len(rows),'quote_id':QUOTE_ID,'quote_positions':sorted(set(int(torch.nonzero(torch.tensor(r['ids'])==QUOTE_ID)[0]) for r in rows)),'additional_model_scalars':0,'supersedes_control_only':'ODD_SEMANTIC_POSITIONS_V1_CPU_CONTROL.json required bitwise-zero post addition instead of the registered 1e-10 numerical bar; original failed receipt preserved.','scope':'Frozen token-semantic mask control; exact O source partition on synthetic normalized states. Quote ID and text templates choose boundaries without model outcomes.'}
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
