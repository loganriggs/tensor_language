"""Measure output-space incompatibility of a sparse token-function approximation."""
import hashlib,json,time
from pathlib import Path
import torch
from audit_token_dictionary_debias_v1 import P,CK
from quadratic_token_dictionary_v1 import embed


def main():
    torch.set_num_threads(2);start=time.perf_counter()
    receipt=json.loads((P/'TOKEN_DICTIONARY_DEBIAS_V1_AUDIT.json').read_text())
    cache=Path(receipt['cache']['path']);assert hashlib.sha256(cache.read_bytes()).hexdigest()==receipt['cache']['sha256']
    saved=torch.load(cache,weights_only=True,map_location='cpu');a,b=saved['codes'],saved['dictionary']
    sd=torch.load(CK,weights_only=True,map_location='cpu',mmap=True)
    raw_u=sd['lm_head.weight'].double();u=raw_u-raw_u.mean(0)
    ug=u.T@u;ua=u.T@a
    lift=torch.linalg.solve(ug,ua)
    inside=u@lift;outside=a-inside
    # Gram of functions, not Euclidean coefficient-column norm.
    h=b@b.T
    total=float(((a.T@a)*h).sum());outside_energy=float(((outside.T@outside)*h).sum())
    inside_energy=float(((inside.T@inside)*h).sum())
    orth=float((u.T@outside).norm()/ua.norm())
    closure=abs(total-inside_energy-outside_energy)/total
    meanout=a.mean(0);common_energy=float(len(a)*(meanout@h@meanout))
    l,r,d=[sd[f'transformer.h.17.mlp.{key}.weight'].double() for key in ['Left','Right','Down']]
    _,root,_,scale,mean=embed(raw_u,l,r,d)
    common_coordinate=mean@root/scale
    augmented_b=torch.cat([b,common_coordinate[None]],0)
    augmented_a=torch.cat([a,torch.ones(len(a),1,dtype=a.dtype)],1)
    full_gram=augmented_b@augmented_b.T
    full_lift=torch.linalg.solve(raw_u.T@raw_u,raw_u.T@augmented_a)
    full_outside=augmented_a-raw_u@full_lift
    full_energy=float(((augmented_a.T@augmented_a)*full_gram).sum())
    full_outside_energy=float(((full_outside.T@full_outside)*full_gram).sum())
    full_orth=float((raw_u.T@full_outside).norm()/(raw_u.T@augmented_a).norm())
    result=dict(instrument_passed=max(orth,closure)<1e-8,
                outside_centered_unembedding_fraction=outside_energy/total,
                inside_centered_unembedding_fraction=inside_energy/total,
                induced_common_output_fraction=common_energy/total,
                native_centered_capture_before=receipt['debiased_capture'],
                native_centered_capture_after_projection=receipt['debiased_capture']+outside_energy/len(a),
                full_composition_outside_native_unembedding_fraction=full_outside_energy/full_energy,
                full_composition_projection_orthogonality_error=full_orth,
                full_composition_outside_energy_in_centered_normalized_units=full_outside_energy,
                projection_orthogonality_error=orth,pythagorean_closure_error=closure,
                approximation_energy_in_normalized_target_units=total,
                outside_energy_in_normalized_target_units=outside_energy,
                sparse_links_before=int((a!=0).sum()),dense_projected_writer_numbers=int(inside.numel()),
                wall_seconds=time.perf_counter()-start,
                scope='Centered and full common-restored output-space checks. Projection improves coefficient error but generally destroys token sparsity. Full projection can also change the retained common function. A native residual writer changes norm and needs physical validation.')
    result['instrument_passed']=result['instrument_passed'] and full_orth<1e-8
    (P/'SPARSE_WRITER_INTERFACE_V1_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2));assert result['instrument_passed']


if __name__=='__main__':main()
