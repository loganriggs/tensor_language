"""Freeze top8 percenteredmode, sharedpair removal/execution controls<=1e-10. No text selection."""
import json
from pathlib import Path
import torch
from quartic_group_program_v1 import run
from quartic_gram_map_v1 import layout,coefficients
from sparse_path_stability_atlas_v1 import digest


@torch.no_grad()
def main():
    torch.set_num_threads(2);torch.set_default_dtype(torch.float64);torch.manual_seed(11540);p=Path(__file__).parent
    out=p/'QUARTIC_GROUP_PROGRAM_V1.json';ap=p/'QUARTIC_GROUP_PROGRAM_V1.pt';assert not out.exists() and not ap.exists()
    source=p/'NATIVE_QUARTIC_GRAM_NUCLEAR_V2.pt';prior=json.loads((p/'NATIVE_QUARTIC_GRAM_NUCLEAR_V2.json').read_text());assert prior['pred_a'] and prior['pred_c'] and digest(source)==prior['artifact_sha256']
    saved=torch.load(source,weights_only=True,map_location='cpu')['programs'];mapping=layout(16);pair=mapping['pairs'];mult=torch.where(pair[0]==pair[1],1.,2.**.5)
    programs=[];errors=[]
    for seed in [11511,11512]:
        local=sorted([a for a in saved if a['seed']==seed],key=lambda a:a['mode']);qs=[];weights=[];writers=[];refs=[];labels=[]
        for a in local:
            selected=a['signed_weights'].abs().topk(8).indices;q=a['quadratic_coefficients'][:,selected];lam=a['signed_weights'][selected]
            qs.append(q*mult[:,None]);weights.append(lam);writers.append(a['output_writer']);refs.append(coefficients((q*lam)@q.T,mapping));labels.append(selected.tolist())
        program=dict(seed=seed,bank=local[0]['bank'],pairs=pair,quadratic_readers=torch.cat(qs,1),signed_weights=torch.cat(weights),output_writers=torch.stack(writers,1),selected_indices=labels)
        assert torch.equal(local[0]['bank'],local[1]['bank'])
        x=torch.randn(13,1152);den=torch.rand(13)+.1;a=run(program,x,den)
        phi=a['reads'][:,mapping['terms']].prod(1)*mapping['multiplicity_root'];reference=phi@torch.stack(refs).T@program['output_writers'].T
        errors.append(float((reference-a['numerator']).norm()/reference.norm()))
        # Removing two shared quadratic-pair inputs exposes the exact cross term.
        first=run(program,x,den,zero_pairs=(0,));second=run(program,x,den,zero_pairs=(1,));both=run(program,x,den,zero_pairs=(0,1))
        da=a['products'][:,0,None]*program['quadratic_readers'][0];db=a['products'][:,1,None]*program['quadratic_readers'][1]
        cross=(2*da*db*program['signed_weights']).reshape(13,2,8).sum(-1)@program['output_writers'].T
        errors.append(float((both['numerator']-first['numerator']-second['numerator']+a['numerator']-cross).norm()/reference.norm()))
        z0=run(program,x,den,zero_quadratics=(0,));z1=run(program,x,den,zero_quadratics=(1,));z01=run(program,x,den,zero_quadratics=(0,1))
        errors.append(float((z01['numerator']-z0['numerator']-z1['numerator']+a['numerator']).norm()/reference.norm()))
        assert sum(program[k].numel() for k in ('bank','quadratic_readers','signed_weights','output_writers'))==22928
        programs.append(program)
    torch.save(dict(programs=programs),ap)
    result=dict(pred_a=max(errors)<=1e-10,maximum_replay_error=max(errors),fitted_floats_per_seed=22928,shared_pair_products=136,quadratic_square_products=16,
        artifact_sha256=digest(ap),source_sha256=digest(source),scope='Frozen16term/twooutputmode partial quartic numerator, actual normalizedMLP16source andMLP17denominator/background remain required.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));assert result['pred_a']


if __name__=='__main__':main()
