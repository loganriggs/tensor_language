"""Amortize the exact fixed-writer response over many edit amplitudes."""
import torch


def prepare(z, native_biasfree_mlp, program):
    z=z.double();w=program['direction'].to(device=z.device,dtype=torch.float64)
    J=program['mixed_map'].to(device=z.device,dtype=torch.float64)
    width=z.shape[-1];ww=w.square().sum();parallel=(z*w).sum(-1,keepdim=True)/ww.clamp_min(torch.finfo(torch.float64).tiny)
    perpendicular=z-parallel*w
    return dict(direction=w,baseline=native_biasfree_mlp.double(),Jz=z@J.T,Jw=J@w,
                parallel=parallel,perpendicular_rms=perpendicular.square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps,
                writer_rms=ww/width,cross_rms=(z*w).mean(-1,keepdim=True))


def evaluate(amplitude, context):
    """Amplitude ends in singleton dimension; leading strength axes broadcast.

    Positive amplitude removes the writer. Full prefix/background generation is
    external; preparing a context is required again if that background changes.
    """
    a=amplitude.to(context['baseline']);rho=context['perpendicular_rms']+context['writer_rms']*(a-context['parallel']).square()
    scale=a*(2*context['cross_rms']-a*context['writer_rms'])/rho
    return (-a*context['direction']+scale*context['baseline']
            -a/rho*context['Jz']+a.square()/(2*rho)*context['Jw'])
