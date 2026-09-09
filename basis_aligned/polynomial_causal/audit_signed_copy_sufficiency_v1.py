"""Registered full-vector test of a96-coefficient signed-copy raw decoder."""
import hashlib
import json
from pathlib import Path
import torch
import field_intervention_metrics as M

BASE=Path(__file__).resolve().parent;SOURCE=BASE/'SIGNED_RAW_READER_AUDIT_V1.json';OUT=BASE/'SIGNED_COPY_SUFFICIENCY_V1.json'


def main():
    torch.set_num_threads(2);assert not OUT.exists();data=json.loads(SOURCE.read_text())
    d=torch.tensor(data['dictionary'],dtype=torch.float64);d=d-d.mean(-1,keepdim=True)
    basis=torch.eye(29,dtype=d.dtype)[:24]-1/29
    alpha=torch.einsum('hec,ec->he',d,basis)/(28/29)
    predicted=alpha[:,:,None]*basis[None];residual=d-predicted
    closure=M.correspondence(predicted+residual,d);orthogonality=float(torch.einsum('hec,ec->he',residual,basis).abs().max())
    rows=[]
    for c in data['cases']:
        gates=torch.tensor([h['signed_gate'] for h in c['heads']],dtype=d.dtype);e=c['source_value']
        target=torch.einsum('h,hc->c',gates,d[:,e]);estimate=torch.einsum('h,hc->c',gates,predicted[:,e])
        err=float((estimate-target).square().mean().sqrt())/max(float(target.square().mean().sqrt()),1e-6)
        rows.append({k:c[k] for k in ('population','world','query','hop','side','primary_side','is_primary','field','source_value')})
        rows[-1].update({'relative_rms':err,'passed':err<=.01,'native_raw_centered_rms':float(target.square().mean().sqrt()),
                        'predicted_effective_copy_coefficient':float((gates*alpha[:,e]).sum())})
    result={'scope':'opened fixed signed-copy structural candidate; full-vector raw-removal prediction, not whole-model adoption',
        'mechanical_passed':closure['passed'] and orthogonality<=1e-9,'closure':closure,'residual_orthogonality_max_abs':orthogonality,
        'candidate_passed':all(r['passed'] for r in rows),'passed_cases':sum(r['passed'] for r in rows),'cases':rows,
        'maximum_case_relative_rms':max(r['relative_rms'] for r in rows),'dictionary_residual_relative_frobenius':float(residual.norm()/d.norm()),
        'alpha':alpha.tolist(),'candidate_coefficients':96,'exact_dictionary_values':2784,'native_background_constants':data['independent_constants'],
        'whole_model_coefficients_removed':0,'subtraction_adapter_still_required':True,'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest()}
    OUT.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('cases','alpha')},indent=2))


if __name__=='__main__':main()
