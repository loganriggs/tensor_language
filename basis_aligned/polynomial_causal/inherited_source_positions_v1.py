"""Partition inherited value sources, retaining routing from the full context."""
from pathlib import Path
from datetime import datetime,timezone
import json,torch
import torch.nn.functional as F
from even_value_shared_graph_v1 import SharedGraph
from remainder_value_sources_v1 import split
from regional_city_article_check_v1 import validate_city_articles


def paired_masks(rows):
    validate_city_articles(rows)
    masks=[]
    for i in range(0,len(rows),2):
        a,b=rows[i:i+2];assert len(a['ids'])==len(b['ids'])
        mask=torch.tensor([x!=y for x,y in zip(a['ids'],b['ids'])])
        assert int(mask.sum()) in (1,2)
        masks.extend([mask,mask.clone()])
    return masks


def inherited_at(graph,current,first_values,source_mask):
    mask=torch.as_tensor(source_mask,dtype=first_values.dtype,device=first_values.device)
    assert mask.shape==first_values.shape[:-1]
    return split(graph,current,first_values*mask[...,None])[1]


@torch.no_grad()
def main():
    p=Path(__file__).resolve().parent;out=p/'INHERITED_SOURCE_POSITIONS_V1_CPU_CONTROL.json'
    assert not out.exists();torch.set_num_threads(2);torch.manual_seed(1409261305)
    rows=json.loads((p/'SRO_ARTICLE_CORRECTION_V1_ROWS.json').read_text())['rows']
    masks=paired_masks(rows)
    graph=SharedGraph(torch.load(p/'EVEN_VALUE_SHARED_GRAPH_PACKED_V1_PROGRAM.pt',weights_only=True))
    errors=[];zero=[]
    for i in [0,12,24,36]:
        n=len(rows[i]['ids']);current=F.rms_norm(torch.randn(1,n,1152),(1152,));first=torch.randn(1,n,128)
        mask=masks[i][None]
        cue=inherited_at(graph,current,first,mask);other=inherited_at(graph,current,first,~mask)
        total=split(graph,current,first)[1]
        errors.append(float((cue+other-total).norm()/total.norm().clamp_min(1e-30)))
        zero.append(float(inherited_at(graph,current,first,torch.zeros_like(mask)).abs().max()))
        errors.append(float((inherited_at(graph,current,first,torch.ones_like(mask))-total).norm()/total.norm().clamp_min(1e-30)))
    result=dict(utc=datetime.now(timezone.utc).isoformat(),pred_a=max(errors)<=1e-10 and max(zero)==0,
                recomposition_errors=errors,zero_mask_maxabs=zero,rows=len(rows),mask_counts=[int(m.sum()) for m in masks],
                additional_model_scalars=0,shared_graph_calls_per_source=2,
                scope='Token-defined source partition and synthetic CPU instrument with native weights; Q/K uses the entire supplied context. No native causal conclusion or complete cue-input localization.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='mask_counts'}))


if __name__=='__main__':main()
