"""Model-free mask control for head8.2 city versus other source edges."""
import json
from datetime import datetime,timezone
from pathlib import Path
import torch
from odd_contextual_positions_v1 import contextual_masks
from odd_semantic_positions_v1 import semantic_masks


def edge_masks(rows):
    contextual=contextual_masks(rows);semantic=semantic_masks(rows);out=[]
    for c,s in zip(contextual,semantic):
        city=c['city'];other=~city
        if not bool(city.sum()==1) or not bool(((city.to(torch.int8)+other.to(torch.int8))==1).all()):raise ValueError('City/other must partition sources')
        out.append({'city':city,'other':other,'framing':s['framing']})
    return out


def main():
    root=Path(__file__).resolve().parent;out=root/'ODD_ATTENTION8H2_SOURCE_EDGE_V1_CPU_CONTROL.json';assert not out.exists();torch.manual_seed(14091812)
    rows=json.loads((root/'ODD_FRAMING_FRESH_V1_ROWS.json').read_text())['rows'];masks=edge_masks(rows);errors=[]
    for i in (0,24):
        n=len(rows[i]['ids']);channels=torch.randn(1,n,n,128);city=channels*masks[i]['city'][None,None,:,None];other=channels*masks[i]['other'][None,None,:,None];errors.append(float((city+other-channels).abs().max()))
    paired_framing=all(torch.equal(masks[i]['framing'],masks[i^1]['framing']) for i in range(48));result={'utc':datetime.now(timezone.utc).isoformat(),'pred_a':max(errors)==0 and paired_framing,'partition_max_abs_errors':errors,'paired_framing_masks_equal':paired_framing,'city_counts':sorted(set(int(x['city'].sum()) for x in masks)),'framing_counts':sorted(set(int(x['framing'].sum()) for x in masks)),'rows':len(rows),'scope':'Model-free source/destination mask control for head8.2 edge decomposition; no native scores or causal conclusion.'};out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':main()
