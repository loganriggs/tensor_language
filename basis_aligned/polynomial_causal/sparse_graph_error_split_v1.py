"""Node/complement attribution of frozen full-graph swap errors; no fitting."""
import json
from pathlib import Path
import torch
import torch.nn.functional as F
P=Path(__file__).resolve().parent


def run():
    torch.set_num_threads(2)
    w=torch.load(P/'SPARSE_FRAME_NATIVE_V1_WRITES.pt',weights_only=True)
    ports=torch.load(P/'NATIVE_RELATION_OUTPUT_FRESH_V1_ENDPOINTS.pt',weights_only=True)['ports']
    rows=json.loads((P/'NATIVE_RELATION_OUTPUT_FRESH_V1_ROWS.json').read_text())['rows']
    binding=json.loads((P/'SPARSE_FRAME_NATIVE_V1_BINDING.json').read_text())['files']
    sd=torch.load(next(k for k in binding if k.endswith('/pytorch_model.bin')),weights_only=True,mmap=True)
    readers=torch.stack([sd['lm_head.weight'][[r['donor_answer_id'],r['donor_foil_id']]].float() for r in rows])
    h=ports['pre']+ports['native_output']
    error=w['candidate']-w['reference'];node=w['node_candidate']-w['node_reference'];rest=error-node
    delta=lambda v:v[1::2]-v[::2]
    ref=h[::2]+delta(w['reference']).float();cand=h[::2]+delta(w['candidate']).float()
    def margin(state):
        normalized=F.rms_norm(state,(1152,))
        logits=30*torch.tanh(torch.einsum('nd,nkd->nk',normalized,readers)/30)
        return logits[:,0]-logits[:,1]
    observed=(margin(cand)-margin(ref)).double()
    mid=((ref+cand)/2).detach().requires_grad_(True)
    grad=torch.autograd.grad(margin(mid).sum(),mid)[0].double()
    node_effect=(grad*delta(node)).sum(-1);rest_effect=(grad*delta(rest)).sum(-1)
    predicted=node_effect+rest_effect
    results=[]
    rms=lambda v:float(v.square().mean().sqrt())
    for family in sorted({r['family'] for r in rows}):
        ids=torch.tensor([i for i,r in enumerate(rows) if r['family']==family]);ep=(2*ids[:,None]+torch.tensor([0,1])).flatten()
        n,c,e=node[ep],rest[ep],error[ep]
        results.append(dict(family=family,actual_margin_error_rms=rms(observed[ids]),
                            node_predicted_error_rms=rms(node_effect[ids]),complement_predicted_error_rms=rms(rest_effect[ids]),
                            prediction_relative_error=float((predicted[ids]-observed[ids]).norm()/observed[ids].norm()),
                            margin_cross_mean=float(2*(node_effect[ids]*rest_effect[ids]).mean()),
                            write_node_energy=float(n.square().sum()),write_complement_energy=float(c.square().sum()),
                            write_cross_energy=float(2*(n*c).sum()),write_total_error_energy=float(e.square().sum())))
    return dict(families=results,write_identity_error=float((error-node-rest).norm()/error.norm()),
                scope='Midpoint-gradient error accounting in full-graph swapped states, not isolated node effects; cancellation retained, no fitting.')


if __name__=='__main__':
    r=run();out=P/'SPARSE_GRAPH_ERROR_SPLIT_V1_RESULT.json';assert not out.exists()
    out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
