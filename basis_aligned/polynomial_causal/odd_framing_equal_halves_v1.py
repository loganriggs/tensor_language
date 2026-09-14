"""Outcome-blind equal-count early/late partition of framing sources."""
import json
from datetime import datetime,timezone
from pathlib import Path
import torch
import torch.nn.functional as F
from even_value_shared_graph_v1 import SharedGraph
from odd_semantic_positions_v1 import semantic_masks
from odd_source_positions_v1 import source_channels,select_sources


def equal_half_masks(rows):
    result=[]
    for base in semantic_masks(rows):
        positions=torch.nonzero(base['framing']).flatten();n=int(positions.numel())
        if n%2:raise ValueError('Even framing-source count required')
        early=torch.zeros_like(base['framing']);late=torch.zeros_like(base['framing'])
        early[positions[:n//2]]=True;late[positions[n//2:]]=True
        if not torch.equal(early|late,base['framing']) or bool((early&late).any()):raise ValueError('Equal halves must partition framing')
        result.append({'early':early,'late':late,'framing':base['framing']})
    return result


@torch.no_grad()
def main():
    root=Path(__file__).resolve().parent;out=root/'ODD_FRAMING_EQUAL_HALVES_V1_CPU_CONTROL.json';assert not out.exists()
    torch.set_num_threads(2);torch.manual_seed(14091746);rows=json.loads((root/'ODD_FRAMING_FRESH_V1_ROWS.json').read_text())['rows'];masks=equal_half_masks(rows)
    graph=SharedGraph(torch.load(root/'EVEN_VALUE_SHARED_GRAPH_PACKED_V1_PROGRAM.pt',weights_only=True));errors=[];counts=[]
    for i in [0,24]:
        tokens=len(rows[i]['ids']);current=F.rms_norm(torch.randn(1,tokens,1152),(1152,));first=torch.randn(1,tokens,9,128);channels=source_channels(graph,current,first)
        pieces={k:select_sources(channels,v[None]) for k,v in masks[i].items()};errors.append(float((pieces['early']+pieces['late']-pieces['framing']).norm()/pieces['framing'].norm().clamp_min(1e-30)));counts.append({k:int(v.sum()) for k,v in masks[i].items()})
    result={'utc':datetime.now(timezone.utc).isoformat(),'pred_a':max(errors)<=1e-10,'framing_partition_errors':errors,'representative_counts':counts,'rows':len(rows),'scope':'Outcome-blind equal-count early/late framing mask control; no scores, fitting, or learned boundary.'}
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
