"""Exact radial coefficient and native bias, not an activation-distribution fit."""
import json
from pathlib import Path
import torch
from native_reader_msp_generalization_v1 import P,CK

def main():
    torch.set_default_dtype(torch.float64);torch.set_num_threads(2)
    sd=torch.load(CK,weights_only=True,map_location='cpu',mmap=True)
    l,r,d=[sd[f'transformer.h.17.mlp.{key}.weight'].double() for key in ('Left','Right','Down')]
    bias=sd['transformer.h.17.mlp.Down_bias'].double()
    metric_receipt=json.loads((P/'NATIVE_READER_METRIC_V1_AUDIT.json').read_text())
    metric=torch.load(metric_receipt['cache']['path'],weights_only=True,map_location='cpu')['unembedding_gram']
    traces=(l*r).sum(1);mean=d@traces
    def energy(x):return float(x@metric@x)
    mean_energy,bias_energy=energy(mean),energy(bias)
    cosine=float(mean@metric@bias)/max((mean_energy*bias_energy)**.5,1e-30)
    cancellation_ratio=(energy(mean+bias)/mean_energy)**.5
    generator=torch.Generator().manual_seed(836)
    x=torch.randn(4,1152,generator=generator)
    direct=((x@l.T)*(x@r.T))@d.T+bias
    radial=x.square().sum(1)/1152
    centered_products=(x@l.T)*(x@r.T)-radial[:,None]*traces
    split=centered_products@d.T+radial[:,None]*mean+bias
    replay=float((split-direct).norm()/direct.norm())
    prior=json.loads((P/'FULLU_TRACE_METRIC_V1_AUDIT.json').read_text())
    trace_replay=abs(mean_energy/prior['uniform_sphere_constant_energy']-1)
    harmonic=prior['uniform_sphere_traceless_energy']
    result=dict(predictions=dict(pred_a_split=max(replay,trace_replay)<=1e-10,
        pred_b_anticorrelation=cosine<=-.9,pred_c_cancellation=cancellation_ratio<=.25),
        split_replay=replay,prior_radial_energy_replay=trace_replay,
        radial_readout_energy=mean_energy,bias_readout_energy=bias_energy,
        bias_to_radial_norm_ratio=(bias_energy/mean_energy)**.5,
        radial_bias_readout_cosine=cosine,remaining_radial_bias_norm_ratio=cancellation_ratio,
        uniform_sphere_radial_fraction_before_bias=mean_energy/(mean_energy+harmonic),
        uniform_sphere_radial_fraction_after_bias=energy(mean+bias)/(energy(mean+bias)+harmonic),
        residual_radial_norm=float(mean.norm()),residual_bias_norm=float(bias.norm()),
        scope='Exact polynomial/radial/bias identity; uniform-sphere fractions are a hypothetical distribution, not native activation statistics or removability evidence')
    (P/'NATIVE_BIAS_RADIAL_V1_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))

if __name__=='__main__':main()
