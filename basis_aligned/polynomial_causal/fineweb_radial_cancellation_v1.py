"""Post-result cached-state explanation of frozen radial validation failure.

No fitting, corpus reads, model body passes or new candidate selection.
The cancellation expectation follows from the already observed energy scores;
it is a mechanism diagnosis, not independent predictive evidence.
"""
import json,hashlib,time
from pathlib import Path
import torch
from native_reader_msp_generalization_v1 import CK
P=Path(__file__).resolve().parent


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    torch.set_num_threads(2);torch.set_default_dtype(torch.float64);started=time.perf_counter()
    receipt=json.loads((P/'FROZEN_RADIAL_FINEWEB_V1_RESULT.json').read_text())
    source=receipt['cache'];assert digest(source['path'])==source['sha256']
    saved=torch.load(source['path'],weights_only=True,map_location='cpu')
    x=saved['ports']['input'].reshape(-1,1152).double()
    native=saved['ports']['native_output'].reshape_as(x).double()
    sd=torch.load(CK,weights_only=True,map_location='cpu',mmap=True)
    l,r,d=[sd[f'transformer.h.17.mlp.{key}.weight'].double() for key in ('Left','Right','Down')]
    bias=sd['transformer.h.17.mlp.Down_bias'].double();m=d@(l*r).sum(1)
    metric=torch.load('/dev/shm/bilin18_native_product_energy_v1.pt',weights_only=True,map_location='cpu')['unembedding_gram'].double()
    # m is the trace; q_radial(x)=m ||x||^2 / d.
    radial=x.square().sum(1,keepdim=True)/1152*m[None,:]
    quadratic=native-bias;traceless=quadratic-radial
    def ip(a,b):return float(((a@metric)*b).sum())
    rr=ip(radial,radial);tt=ip(traceless,traceless);rt=ip(radial,traceless)
    qq=ip(quadratic,quadratic);nn=ip(native,native)
    energy_identity=abs(qq-(rr+tt+2*rt))/max(qq,1e-30)
    # Replay actual FP32 candidate output rounding, as in the frozen executor.
    bias_error=bias.float().double()-native
    radial_error=(bias+radial).float().double()-native
    observed_bias=float(saved['metrics']['bias_only'][:,:,4].sum())
    observed_radial=float(saved['metrics']['radial_only'][:,:,4].sum())
    observed_native=float(saved['metrics']['bias_only'][:,:,5].sum())
    errors=dict(bias=abs(ip(bias_error,bias_error)-observed_bias)/observed_bias,
        radial=abs(ip(radial_error,radial_error)-observed_radial)/observed_radial,
        native=abs(nn-observed_native)/observed_native)
    mean_radial=radial.mean(0);mean_traceless=traceless.mean(0)
    mean_cos=float(mean_radial@metric@mean_traceless/((mean_radial@metric@mean_radial)*(mean_traceless@metric@mean_traceless)).sqrt())
    result=dict(instrument_passed=max(errors.values())<=1e-6 and energy_identity<=1e-10,
        replay_relative_errors=errors,energy_identity_relative_error=energy_identity,
        radial_energy_over_native=rr/nn,traceless_energy_over_native=tt/nn,
        twice_cross_over_native=2*rt/nn,quadratic_energy_over_native=qq/nn,
        traceless_radial_projection=rt/rr,pooled_metric_cosine=rt/(rr*tt)**.5,
        mean_vector_metric_cosine=mean_cos,positions=len(x),
        rms_squared_mean=float(x.square().sum(1).mean()/1152),source=source,seconds=time.perf_counter()-started,
        scope='Post-result diagnosis of cached FineWeb native outputs. No fit, new body passes, fresh validation or independent prediction. Radial/traceless parts need not be orthogonal on natural inputs.')
    with (P/'FINEWEB_RADIAL_CANCELLATION_V1_AUDIT.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2));assert result['instrument_passed']


if __name__=='__main__':main()
