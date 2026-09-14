"""Outcome-blind descriptive-continuation versus task-instruction source masks."""
import json
from datetime import datetime,timezone
from pathlib import Path

import torch
import torch.nn.functional as F
from even_value_shared_graph_v1 import SharedGraph
from odd_semantic_positions_v1 import semantic_masks
from odd_source_positions_v1 import source_channels,select_sources

PERIOD_ID=13


def role_masks(rows):
    semantic=semantic_masks(rows);result=[]
    for row,base in zip(rows,semantic):
        ids=torch.tensor(row['ids']);city=int(torch.nonzero(base['city'])[0]);quote=int(torch.nonzero(ids==366)[0])
        periods=torch.nonzero((ids==PERIOD_ID)&(torch.arange(len(ids))>city)&(torch.arange(len(ids))<quote)).flatten()
        if periods.numel()!=1:raise ValueError('Exactly one post-city pre-quote period required')
        period=int(periods[0]);index=torch.arange(len(ids))
        description=(index>city)&(index<=period);instruction=(index>period)&(index<=quote)
        if not torch.equal(description|instruction,base['framing']) or bool((description&instruction).any()):
            raise ValueError('Role masks must exactly partition framing')
        if not bool(description.any()) or not bool(instruction.any()):raise ValueError('Both role masks must be live')
        result.append({'description':description,'instruction':instruction,'framing':base['framing']})
    return result


@torch.no_grad()
def main():
    root=Path(__file__).resolve().parent;out=root/'ODD_FRAMING_ROLE_SPLIT_V1_CPU_CONTROL.json';assert not out.exists()
    torch.set_num_threads(2);torch.manual_seed(14091739)
    rows=json.loads((root/'ODD_FRAMING_FRESH_V1_ROWS.json').read_text())['rows'];masks=role_masks(rows)
    graph=SharedGraph(torch.load(root/'EVEN_VALUE_SHARED_GRAPH_PACKED_V1_PROGRAM.pt',weights_only=True));errors=[];counts=[]
    for i in [0,24]:
        tokens=len(rows[i]['ids']);current=F.rms_norm(torch.randn(1,tokens,1152),(1152,));first=torch.randn(1,tokens,9,128)
        channels=source_channels(graph,current,first);pieces={k:select_sources(channels,v[None]) for k,v in masks[i].items()}
        errors.append(float((pieces['description']+pieces['instruction']-pieces['framing']).norm()/pieces['framing'].norm().clamp_min(1e-30)))
        counts.append({k:int(v.sum()) for k,v in masks[i].items()})
    result={'utc':datetime.now(timezone.utc).isoformat(),'pred_a':max(errors)<=1e-10,'framing_partition_errors':errors,'representative_counts':counts,'rows':len(rows),'period_id':PERIOD_ID,'scope':'Outcome-blind token-role mask control. First period after city splits descriptive continuation from explicit instruction; no model scores or fitting.'}
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
