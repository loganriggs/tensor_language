"""Capture source Jacobians and output adjoints at each native suffix boundary."""
def capture_ports(model,raw,x0,first,directions,positions,read,pairs):
    import torch
    from native_source_observables import source_observables
    layers=list(range(11,18));outputs=pairs.shape[1];width=directions.shape[1]
    a=torch.zeros(len(raw),width,device=raw.device,dtype=torch.float64)
    capture={};readers=[];full_gradients=[]
    with torch.enable_grad():
        variable=a.clone().requires_grad_();values=source_observables(model,raw,x0,first,directions,positions,read,pairs,variable,capture)
        nodes=tuple(capture['mlp_outputs'][l] for l in layers)+tuple(capture['mlp_inputs'][l] for l in layers)
        for o in range(outputs):
            grads=torch.autograd.grad(values[:,o].sum(),nodes+(variable,),retain_graph=o<outputs-1)
            readers.append([g.detach() for g in grads[:-1]]);full_gradients.append(grads[-1].detach())
    def all_nodes(z):
        local={};source_observables(model,raw,x0,first,directions,positions,read,pairs,z,local)
        return tuple(local['mlp_inputs'][l] for l in layers)+tuple(local['mlp_outputs'][l] for l in layers)
    derivatives=[];replay=[]
    for i in range(width):
        v=torch.zeros_like(a);v[:,i]=1
        values_jvp,tangents=torch.autograd.functional.jvp(all_nodes,a,v,create_graph=False)
        derivatives.append(tangents)
        expected=[capture['mlp_inputs'][l] for l in layers]+[capture['mlp_outputs'][l] for l in layers]
        replay.append(max(float((x-y.detach()).abs().max()) for x,y in zip(values_jvp,expected)))
    ks=[torch.stack([d[j] for d in derivatives],dim=-1) for j in range(14)]
    qs=[torch.stack([r[j] for r in readers],dim=2) for j in range(14)]
    return dict(layers=layers,mlp_inputs=[capture['mlp_inputs'][l].detach() for l in layers],mlp_outputs=[capture['mlp_outputs'][l].detach() for l in layers],mlp_input_directions=ks[:7],mlp_output_directions=ks[7:],mlp_readers=qs[:7],attention_readers=qs[7:],full_gradient=torch.stack(full_gradients,dim=1),max_source_replay=max(replay),calls=dict(full_forward=1,full_source_jvp=width,reader_reverse=outputs))
