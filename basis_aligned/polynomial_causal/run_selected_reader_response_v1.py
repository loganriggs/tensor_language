"""CPU-only trained-weight response compiler and equal-state collision test."""
import hashlib,json,time
from pathlib import Path
import torch
from bilinear_scalar_consumer_v1 import controls,folded_delta
from selected_reader_response_v1 import compile_response,initialize,step


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def relative(a,b):return float((a-b).norm()/b.norm().clamp_min(1e-30))
def column_relative(a,b):return float(((a-b).norm(dim=0)/b.norm(dim=0).clamp_min(1e-30)).max())


def main():
    torch.set_num_threads(2);torch.set_grad_enabled(False);tic=time.perf_counter()
    p=Path(__file__).resolve().parent;binding=json.loads((p/'SELECTED_READER_RESPONSE_V1_BINDING.json').read_text())
    assert all(digest(path)==sha for path,sha in binding.items());tiny=controls()
    checkpoint=next(Path(k) for k in binding if k.endswith('/pytorch_model.bin'))
    weights=torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True)
    l=weights['transformer.h.17.mlp.Left.weight'].double();r=weights['transformer.h.17.mlp.Right.weight'].double();d=weights['transformer.h.17.mlp.Down.weight'].double()
    e=torch.load(p/'GERUND_SHARED_READER_V1_COMPONENT.pt',map_location='cpu',weights_only=True)['e'].double()
    row=json.loads((p/'GERUND_FRESH_TRANSFER_V1_ROWS.json').read_text())['panels']['G'][0]
    g=weights['lm_head.weight'][row['base_answer_id']].double()-weights['lm_head.weight'][row['base_foil_id']].double();g=g/g.norm()
    readers=torch.stack([e,g]);program=compile_response(l,r,d,e,readers)
    basis,_=torch.linalg.qr(torch.stack([e,*program['K']]).T)
    rng=torch.Generator().manual_seed(9111440);z=torch.randn(16,1152,generator=rng,dtype=torch.float64);z=z-(z@basis)@basis.T
    z=z/z.norm(dim=1,keepdim=True)*(1152/4)**.5
    common=torch.randn(16,3,generator=rng,dtype=torch.float64)@basis.T;common=common/common.norm(dim=1,keepdim=True)*(1152/4)**.5
    plus,minus=common+z,common-z;u=torch.cat([plus,minus]);delta=torch.full((32,),.25,dtype=torch.float64)
    native=lambda x:((x@l.T)*(x@r.T))@d.T
    full=native(u+delta[:,None]*e)-native(u);folded=folded_delta(l,r,d,e,u,delta)
    initial=initialize(program,u);predicted,state=step(program,initial,delta)
    response2,state2=step(program,state,-.125)
    total=native(u+.125*e)-native(u)
    invariant=torch.stack([e,*program['K']]);norm=lambda x:(x*x).sum(1)
    checks=dict(fold_relative=relative(folded,full),features_relative=column_relative(plus@invariant.T,minus@invariant.T),
                squared_norm_relative=relative(norm(plus),norm(minus)),selected_response_relative=column_relative(predicted,full@readers.T),
                composition_relative=column_relative(predicted+response2,total@readers.T),state_update_relative=column_relative(state2,initialize(program,u+.125*e)),
                coefficient_identity_relative=relative(program['K']@e,2*program['a']))
    responses_plus,responses_minus=full[:16],full[16:]
    ratios=(responses_plus-responses_minus).norm(dim=1)/(2*(responses_plus.square().sum(1)+responses_minus.square().sum(1))).sqrt()
    a=all(checks[k]<=1e-10 for k in ['fold_relative','features_relative','squared_norm_relative'])
    b=a and all(checks[k]<=1e-10 for k in ['selected_response_relative','composition_relative','state_update_relative','coefficient_identity_relative'])
    c=b and int((ratios>=.1).sum())>=15
    artifact=p/'SELECTED_READER_RESPONSE_V1_PROGRAM.pt';assert not artifact.exists();torch.save(program,artifact)
    result=dict(schema='selected_reader.response.v1',predictions={'pred_a_instrument':a,'pred_b_selected_reader_program':b,'pred_c_full_output_closure_witness':c},checks=checks,
                pair_minimum_relative_errors=ratios.tolist(),qualifying_pairs=int((ratios>=.1).sum()),controls=tiny,
                price=dict(native_forwards=0,cpu_only=True,compiled_coefficients=2306,initial_input_width=1152,response_state_width=2,whole_model_weight_saving=0),
                reader_labels=['fixed_gerund_e','runs_minus_run'],program_sha256=digest(artifact),runner_sha256=digest(__file__),binding_sha256=digest(p/'SELECTED_READER_RESPONSE_V1_BINDING.json'),wall_seconds=time.perf_counter()-tic,
                scope='Exact local selected-reader response and repeated same-direction composition, conditional on initialized context state. Synthetic equal-state normalized-context witnesses are not natural-text OOD or full-model extraction.')
    with (p/'SELECTED_READER_RESPONSE_V1_RESULT.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
