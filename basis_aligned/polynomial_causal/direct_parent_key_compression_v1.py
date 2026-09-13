"""Direct even-key parent-field screen with existing frozen rank48 fits.

Pred_a original cached scalar replay <=1e-5 relative per context.
Pred_b all three families/two contexts <=10% direct parent scalar error.
This differs from prior upstream-induced scalar-response preservation.
"""
import json,time
from pathlib import Path
import torch
import torch.nn.functional as F
from regional_even_routing_v1 import routing

@torch.no_grad()
def main():
    torch.set_num_threads(2);p=Path(__file__).resolve().parent;start=time.perf_counter()
    cache=torch.load(p/'SELECTIVE_INTERACTION_PORTS_V1_ARTIFACT.pt',weights_only=True,mmap=True)
    native=torch.load(p/'extracted_circuits/regional_even_key_producers_8_2_9_8_v1/program.pt',weights_only=True)
    rows=[];replays=[]
    panel=json.loads((p/'SCALAR_CUE_CHANNEL_SHIFT_V1_ROWS.json').read_text())['rows']
    ends=torch.tensor([len(row['ids'])-1 for row in panel])
    rotations={name:torch.load(p/file,weights_only=True)['basis_rotation'].double() for name,file in
        [('query_product','SHARED_QUERY_PRODUCT_FIT_V1_PROGRAM.pt'),('complete_even','COMPLETE_EVEN_KEY_FIT_V1_PROGRAM.pt')]}
    for context in range(2):
        current=F.rms_norm(cache['raw9'][context].float(),(1152,),eps=torch.finfo(torch.float32).eps)
        values=(current.double()@native['current_value_reader'])[...,None]
        ref=(routing(current,native,1)@values)[...,0]
        old=cache['scalar'][context].double()
        replays.append(float((ref-old).norm()/old.norm()))
        for name,rot in rotations.items():
            candidate=dict(native,key_basis=[native['key_basis'][0],native['key_basis'][1]@rot])
            pred=(routing(current,candidate,1)@values)[...,0]
            for family in range(3):
                x=ref[family*24:(family+1)*24];y=pred[family*24:(family+1)*24]
                end=ends[family*24:(family+1)*24];ix=torch.arange(24)
                rows.append(dict(context=context,method=name,family=family,
                    relative_error=float((x-y).norm()/x.norm()),reference_norm=float(x.norm()),
                    final_position_relative_error=float((x[ix,end]-y[ix,end]).norm()/x[ix,end].norm())))
    result=dict(pred_a=max(replays)<=1e-5,pred_b={name:all(x['relative_error']<=.1 for x in rows if x['method']==name) for name in rotations},
        replays=replays,rows=rows,basis_scalars_before=1152*64,basis_scalars_after=1152*48,
        seconds=time.perf_counter()-start,scope='Correction: final-position diagnostic now gathers each actual sequence endpoint; earlier padded-index NaNs preserved separately. '+__doc__+' Cached contexts, no new fitting or native behavior validation. All full Q/K/value weights and child producer remain required.')
    (p/'DIRECT_PARENT_KEY_COMPRESSION_V1_RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
