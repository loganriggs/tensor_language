"""Global fixed projections plus context-specific projections for repeated edits."""
from directional_mlp_response_context_v1 import evaluate
from raw_attention_response_v1 import EPS


def prepare_global(program,matrices,reentry_scale):
    w=program['direction'].double();jw=program['mixed_map'].double()@w
    return dict(writer=tuple((reentry_scale*w)@m.T for m in matrices),
                jw=tuple((reentry_scale*jw)@m.T for m in matrices),scale=reentry_scale)


def prepare(raw,response_context,global_cache,matrices):
    scale=global_cache['scale']
    return dict(raw=raw,response=response_context,global_cache=global_cache,
                baseline=tuple(raw@m.T for m in matrices),
                m0=tuple((scale*response_context['baseline'])@m.T for m in matrices),
                jz=tuple((scale*response_context['Jz'])@m.T for m in matrices))


def changed(amplitude,context):
    r=context['response'];g=context['global_cache'];a=amplitude.to(r['baseline'])
    rho=r['perpendicular_rms']+r['writer_rms']*(a-r['parallel']).square()
    c0=-a;c1=-a/rho;c2=a.square()/(2*rho)
    cm=-2*r['cross_rms']*c1-2*r['writer_rms']*c2
    projections=tuple(p+c0*w+cm*m+c1*j+c2*k for p,w,m,j,k in
        zip(context['baseline'],g['writer'],context['m0'],context['jz'],g['jw']))
    delta=g['scale']*evaluate(a,r)
    norm=(context['raw']+delta).square().mean(-1,keepdim=True)+EPS
    return projections,norm
