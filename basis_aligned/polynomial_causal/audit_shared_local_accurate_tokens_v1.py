"""Descriptive duplicate/unused-token countercheck of frozen <=10% error rows."""
import argparse,json,time
from pathlib import Path
import torch
from audit_fullu_output_functions_v1 import CK
from joint_quadratic_fit_v1 import product_cross

P=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--g',type=int,default=64,choices=[64,128]);args=parser.parse_args()
    torch.set_num_threads(2);started=time.perf_counter()
    sd=torch.load(CK,map_location='cpu',weights_only=True,mmap=True)
    u=sd['lm_head.weight'].double()
    l,r,d=[sd[f'transformer.h.17.mlp.{key}.weight'].double() for key in ['Left','Right','Down']]
    metric=d@product_cross(l,r,l,r)@d.T;root=torch.linalg.cholesky((metric+metric.T)/2)
    target=u@root
    program=torch.load(P/f'FULLU_SHARED_LOCAL_FIT_V1_G{args.g}_PROGRAM.pt',map_location='cpu',weights_only=True)
    predicted=program['global_codes'].double()@(program['global_reader'].double()@root)
    for k,bank in enumerate(program['local_readers']):
        ix=program['labels']==k
        predicted[ix]+=program['local_codes'][ix].double()@(bank.double()@root)
    predicted+=program['mean'].double()@root
    norms=target.square().sum(1)
    errors=((predicted-target).square().sum(1)/norms).sqrt()
    selected=torch.where(errors<=.1)[0];functions=target[selected]
    normalized=functions/functions.norm(dim=1,keepdim=True)
    cos=(normalized@normalized.T).abs();cos.fill_diagonal_(0)
    closest=(1-cos.max(1).values.square()).clamp_min(0).sqrt()
    energy=torch.linalg.svdvals(functions).square()
    unique,inverse,counts=torch.unique(u[selected],dim=0,return_inverse=True,return_counts=True)
    duplicates=[selected[inverse==j].tolist() for j in range(len(unique)) if int(counts[j])>1]
    # Existing project uses the GPT-2 vocabulary; inspect that encoder directly.
    import tiktoken
    encoding=tiktoken.get_encoding('gpt2')
    examples=[];valid=0;invalid=[]
    for idx in selected.tolist():
        try:
            text=encoding.decode_single_token_bytes(idx).decode('utf-8',errors='replace');valid+=1
        except KeyError:invalid.append(idx);continue
        if len(examples)<24:examples.append(dict(id=idx,text=text,error=float(errors[idx])))
    result=dict(schema='shared.local.accurate.tokens.v1',global_width=args.g,threshold=.1,
        selected_count=len(selected),token_ids=selected.tolist(),valid_tokenizer_ids=valid,
        invalid_tokenizer_ids=invalid,unique_native_U_rows=len(unique),duplicate_groups=duplicates,
        median_closest_proportional_function_error=float(closest.median()),
        fraction_with_neighbor_below_1pct=float((closest<=.01).double().mean()),
        function_energy_capture={str(k):float(energy[:k].sum()/energy.sum()) for k in [1,2,4,8]},
        fraction_of_total_coefficient_energy=float(norms[selected].sum()/norms.sum()),
        group_counts=torch.bincount(program['labels'][selected].long(),minlength=len(program['local_readers'])).tolist(),
        examples=examples,wall_seconds=time.perf_counter()-started,
        scope='Post-fit descriptive subset. Proportional-function concentration, exact duplicate rows and '
        'tokenizer membership are distinct; no inferred usage frequency or semantic circuit.')
    with (P/f'SHARED_LOCAL_ACCURATE_TOKENS_V1_G{args.g}_RESULT.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['token_ids','duplicate_groups','invalid_tokenizer_ids']},indent=2))


if __name__=='__main__':main()
