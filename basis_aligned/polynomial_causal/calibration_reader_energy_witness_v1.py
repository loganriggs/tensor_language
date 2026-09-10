"""Post-result continuous-input counterexample to scalar energy sufficiency."""
import json
from pathlib import Path
import torch
from calibration_two_readers_v1 import EPS32,scalar
from induction_context_transport_v2 import digest


def main():
    p=Path(__file__).parent;producer=p/'CALIBRATION_TWO_READERS_V2_PRODUCER.pt';gram=p/'CALIBRATION_READER_ENERGY_V1_GRAM.pt'
    q=torch.load(producer,map_location='cpu',weights_only=True);g=torch.load(gram,map_location='cpu',weights_only=True)
    Q=q['Q'].double();K=g['K'].double();n=len(Q)
    values,vectors=torch.linalg.eigh(K);rotated=vectors.T@Q@vectors;off=rotated.clone();off.fill_diagonal_(0)
    index=int(off.abs().argmax());i,j=divmod(index,n);radius2=n/2
    pair=torch.stack([vectors[:,i]+vectors[:,j],vectors[:,i]-vectors[:,j]])*(radius2/2)**.5
    norm=pair.square().sum(-1);energy=((pair@K)*pair).sum(-1);actual=scalar(pair,Q,q['beta']);gap=float(actual[0]-actual[1]);expected_gap=float(2*radius2*rotated[i,j])
    norm_rel=float(abs(norm[0]-norm[1])/radius2);energy_rel=float(abs(energy[0]-energy[1])/energy.abs().max().clamp_min(1e-30))
    gap_rel=abs(gap)/(radius2*float(Q.norm()))
    raw=pair*(2*EPS32)**.5;normalized=raw/(raw.square().mean(-1,keepdim=True)+EPS32).sqrt()
    inverse_error=float((normalized-pair).abs().max());eigen_rel=float((K@vectors-vectors*values).norm()/K.norm())
    assert norm_rel<1e-12 and energy_rel<1e-10 and inverse_error<1e-10 and eigen_rel<1e-10
    assert abs(gap-expected_gap)<=1e-10*max(1.,abs(gap)) and gap_rel>1e-6
    artifact=p/'CALIBRATION_READER_ENERGY_V1_WITNESS.pt'
    with artifact.open('xb') as f:torch.save(dict(normalized_inputs=pair,raw_inputs=raw,indices=(i,j),q=actual,energy=energy),f)
    out=dict(experiment='calibration_reader_energy_witness_v1',model_forwards=0,post_result=True,indices=[i,j],eigenvalues=[float(values[i]),float(values[j])],norm_squared=norm.tolist(),energy=energy.tolist(),q=actual.tolist(),q_difference=gap,analytic_q_difference=expected_gap,q_gap_over_radius2_Qnorm=gap_rel,norm_relative_difference=norm_rel,energy_relative_difference=energy_rel,RMS_inverse_max_abs=inverse_error,eigendecomposition_relative_error=eigen_rel,source_sha256=digest(__file__),producer_sha256=digest(producer),gram_sha256=digest(gram),witness_sha256=digest(artifact),scope='Numerical realization of exact eigenvector sign-flip construction on stored FP64 Gram. Legal continuous RMS inputs, not observed natural text; no full-model or native-manifold impossibility claim. Floating-point equalities are tolerance checks, not exact rational certificates.')
    with (p/'CALIBRATION_READER_ENERGY_WITNESS_V1_RESULT.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
    print(json.dumps(out,indent=2))


if __name__=='__main__':main()
