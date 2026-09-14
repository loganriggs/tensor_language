"""Diagnostic source split of full even remainder, with original weights.

Uses two existing shared-graph evaluations and one output map per requested
write. This deliberately charges duplicate routing; no speed saving is claimed.
"""
from pathlib import Path
from datetime import datetime,timezone
import json,sys
import torch
import torch.nn.functional as F
from even_value_shared_graph_v1 import SharedGraph
from even_key_value_native_backend_v1 import FullValueComponents


def split(graph,current,first_values):
    full=graph.state(current,first_values=first_values)
    current_only=graph.state(current,first_values=torch.zeros_like(first_values))
    return current_only[1],full[1]-current_only[1]


@torch.no_grad()
def main():
    p=Path(__file__).resolve().parent;out=p/'REMAINDER_VALUE_SOURCES_V1_CPU_CONTROL.json'
    assert not out.exists();torch.set_num_threads(2);torch.manual_seed(1409261254)
    program=torch.load(p/'EVEN_VALUE_SHARED_GRAPH_PACKED_V1_PROGRAM.pt',weights_only=True)
    graph=SharedGraph(program)
    native=FullValueComponents(torch.load(p/'EVEN_KEY_VALUE_BANK_V1_PROGRAM.pt',weights_only=True),'cpu')
    current=F.rms_norm(torch.randn(2,19,1152),(1152,));first=torch.randn(2,19,128);other=torch.randn_like(first)
    a,b=split(graph,current,first);a2,b2=split(graph,current,other)
    am,bm=split(graph,current,.3*first.double()+.7*other.double())
    states=graph.state(current,first_values=first)
    rel=lambda x,y:float((x-y).norm()/y.norm().clamp_min(1e-30))
    even,_=native(current,first);even_current,_=native(current,torch.zeros_like(first))
    scalar=graph.write(states,[1,0,0])
    errors=dict(recompose=rel(a+b,states[1]),
                current_reference=rel(graph.write([a],[1]),even_current-scalar),
                inherited_reference=rel(graph.write([b],[1]),even-even_current),
                current_invariance=float((a-a2).abs().max()),
                inherited_linearity=rel(bm,.3*b+.7*b2))
    result=dict(utc=datetime.now(timezone.utc).isoformat(),
                pred_a=max(errors[k] for k in ['recompose','current_reference','inherited_reference'])<=1e-10,
                pred_b=max(errors['current_invariance'],errors['inherited_linearity'])<=1e-10,
                errors=errors,additional_model_scalars=0,shared_graph_calls_per_split=2,
                scope='CPU instrument only: native weights, synthetic supplied contexts, no body forwards or cue identification. Same original head program and context generators retained.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
