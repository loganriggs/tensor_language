"""Exact mixed-state accounting for a residual update x -> x + write(x).

Quartet order is baseline, first-axis, second-axis, joint. This is an
accounting identity on a fixed quartet, not unique causal attribution.
"""
def split(write,states):
    x0,xs,xt,xst=states
    additive=xs+xt-x0
    incoming=xst-additive
    w0,ws,wt,wst=[write(x) for x in states]
    wa=write(additive)
    generated=wa-ws-wt+w0
    transported=wst-wa
    outgoing=(xst+wst)-(xs+ws)-(xt+wt)+(x0+w0)
    return dict(incoming=incoming,generated=generated,transported=transported,
                outgoing=outgoing,closure=outgoing-incoming-generated-transported)
