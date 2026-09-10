"""Direct skip/reentry embedding producer at native query ports only."""
from contextlib import contextmanager
import torch
import torch.nn.functional as F


def direct_coefficient(lambdas):
    alpha=1.
    for a,b in lambdas:alpha=float(a)*alpha+float(b)
    return alpha


@contextmanager
def install(attn, source, positions, heads=(1,4), identity=False, audit=None):
    handles=[];width=attn.head_dim
    try:
        for kind,projection in enumerate((attn.c_q,attn.c_q2)):
            def hook(module,args,out,kind=kind):
                original=out.detach();changed=out.clone().view(out.shape[0],out.shape[1],-1,width)
                selected_input=torch.stack([args[0][i,int(pos)] for i,pos in enumerate(positions)]) if identity else source.to(args[0])
                rows=torch.cat([module.weight[h*width:(h+1)*width] for h in heads])
                value=F.linear(selected_input,rows.to(selected_input.dtype)).view(len(positions),len(heads),width)
                for i,pos in enumerate(positions):changed[i,int(pos),list(heads)]=value[i]
                changed=changed.reshape_as(out)
                if audit is not None:
                    expected=selected_input.double()@rows.double().T
                    mask=torch.zeros_like(out,dtype=torch.bool).view_as(out.reshape(out.shape[0],out.shape[1],-1,width))
                    for i,pos in enumerate(positions):mask[i,int(pos),list(heads)]=True
                    mask=mask.reshape_as(out)
                    audit.append({'kind':kind,'actual':value.flatten(1).detach(),'expected':expected.detach(),
                                  'unselected_unchanged':torch.equal(changed[~mask],original[~mask])})
                return changed
            handles.append(projection.register_forward_hook(hook))
        yield
    finally:
        for h in handles:h.remove()


def controls():
    from types import SimpleNamespace
    dtype=torch.float64;g=torch.Generator().manual_seed(60922)
    attn=SimpleNamespace(c_q=torch.nn.Linear(5,12,bias=False,dtype=dtype),c_q2=torch.nn.Linear(5,12,bias=False,dtype=dtype),head_dim=4)
    with torch.no_grad():
        for m in (attn.c_q,attn.c_q2):m.weight.copy_(torch.randn(m.weight.shape,generator=g,dtype=dtype))
    x=torch.randn(2,3,5,generator=g,dtype=dtype);e=torch.randn(2,5,generator=g,dtype=dtype);positions=[1,2]
    source=F.rms_norm(e,(5,),eps=.01);audit=[]
    with torch.no_grad():
        original=attn.c_q(x)
        with install(attn,source,positions,(0,2),audit=audit):changed=attn.c_q(x)
        with install(attn,source,positions,(0,2),identity=True):identity=attn.c_q(x)
        try:
            with install(attn,source,positions,(0,2)):raise RuntimeError('fixture')
        except RuntimeError:pass
    # A concrete residual recurrence tracks the original source separately.
    coeffs=[(2.,.5),(.25,-.1)];computed=3.;root=2.;r=root
    r=coeffs[0][0]*r+coeffs[0][1]*root+computed
    r=coeffs[1][0]*r+coeffs[1][1]*root
    checks={'direct_lineage_recurrence':abs(r-(direct_coefficient(coeffs)*root+.25*computed))<1e-12,
        'identity_query_replay':bool(torch.allclose(identity,original,atol=1e-12,rtol=1e-12)),
        'selected_projection_oracle':bool(torch.allclose(audit[0]['actual'],audit[0]['expected'],atol=1e-12,rtol=1e-12)),
        'unselected_query_positions_unchanged':bool(audit[0]['unselected_unchanged']),
        'source_normalization_live':float((source-e).norm())>.01 and float((changed-original).norm())>.01,
        'normal_and_exception_restoration':all(not m._forward_hooks for m in (attn.c_q,attn.c_q2))}
    return {'passed':all(checks.values()),'checks':checks}
