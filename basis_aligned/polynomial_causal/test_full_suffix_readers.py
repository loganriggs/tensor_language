import torch
from finite_attention_readers import forward as attention,rms
from full_suffix_readers import secant_readers


def test_dynamic_two_block_secants_close_at_every_boundary_and_limit():
    torch.manual_seed(676);batch,length,d,heads=2,4,8,2;eps=.03
    initial=torch.randn(batch,length,d,dtype=torch.float64)
    changed=initial+torch.randn_like(initial)*.3
    embedding=torch.randn_like(initial);first=torch.randn(batch,length,heads,d//heads,dtype=torch.float64)
    blocks=[]
    for _ in range(2):
        angles=torch.randn(length,d//heads//2,dtype=torch.float64)
        blocks.append(dict(attention={k:torch.randn(d,d,dtype=torch.float64)/d**.5 for k in ['q','k','q2','k2','v','o']},
            left=torch.randn(12,d,dtype=torch.float64)/d**.5,right=torch.randn(12,d,dtype=torch.float64)/d**.5,
            down=torch.randn(d,12,dtype=torch.float64)/12,bias=torch.randn(d,dtype=torch.float64)/10,
            lambdas=torch.tensor([.8,.2],dtype=torch.float64),mixture=torch.tensor(.3,dtype=torch.float64),
            heads=heads,head_eps=.02,cos=angles.cos().bfloat16().double(),sin=angles.sin().bfloat16().double()))
    def trace(x):
        result=dict(states=[x],raw_attention_inputs=[],mlp_inputs=[])
        for b in blocks:
            raw=b['lambdas'][0]*x+b['lambdas'][1]*embedding
            a,_=attention(b['attention'],raw,first,b['mixture'],heads,b['cos'],b['sin'],eps,b['head_eps'])
            h=raw+a;n=rms(h,eps)
            x=h+((n@b['left'].T)*(n@b['right'].T))@b['down'].T+b['bias']
            result['raw_attention_inputs'].append(raw);result['mlp_inputs'].append(h);result['states'].append(x)
        return result
    U=torch.randn(3,2,d,dtype=torch.float64);positions=torch.tensor([2,3])
    def margins(x):
        selected=x[torch.arange(batch),positions]
        logits=30*torch.tanh(torch.einsum('bd,rcd->rbc',rms(selected,eps),U)/30)
        return logits[...,0]-logits[...,1]
    base,edited=trace(initial),trace(changed)
    readers=secant_readers(blocks,base,edited,first,U,positions,eps)
    target=margins(edited['states'][-1])-margins(base['states'][-1])
    for q,x0,x1 in zip(readers,base['states'],edited['states']):
        torch.testing.assert_close((q*(x1-x0)).sum((-1,-2)),target,atol=1e-10,rtol=1e-10)
    x=initial.clone().requires_grad_();same=trace(x)
    q=secant_readers(blocks,same,same,first,U,positions,eps)[0]
    y=margins(same['states'][-1])
    for r in range(len(U)):
        expected=torch.autograd.grad(y[r].sum(),x,retain_graph=True)[0]
        torch.testing.assert_close(q[r],expected,atol=1e-10,rtol=1e-10)

    from selected_suffix_readers import baseline_reader
    from full_suffix_readers import readout_reader
    selected=baseline_reader(blocks,same,first,U,positions,eps,{0,1},{0,1})
    torch.testing.assert_close(selected,q,atol=1e-10,rtol=1e-10)
    carry=baseline_reader(blocks,same,first,U,positions,eps,set(),set())
    expected_carry=readout_reader(same['states'][-1],same['states'][-1],U,positions,eps)
    for b in blocks:expected_carry=expected_carry*b['lambdas'][0]
    torch.testing.assert_close(carry,expected_carry,atol=1e-10,rtol=1e-10)

    from selected_suffix_readers import single_attention_terms
    no_attention,terms=single_attention_terms(blocks,same,first,U,positions,eps)
    reference=baseline_reader(blocks,same,first,U,positions,eps,{0,1},set())
    torch.testing.assert_close(no_attention,reference,atol=1e-10,rtol=1e-10)
    for k in range(2):
        one=baseline_reader(blocks,same,first,U,positions,eps,{0,1},{k})
        torch.testing.assert_close(no_attention+terms[k],one,atol=1e-10,rtol=1e-10)

    from selected_suffix_readers import earliest_attention_reader
    compact=earliest_attention_reader(blocks,same,first,U,positions,eps)
    reference=baseline_reader(blocks,same,first,U,positions,eps,{0,1},{0})
    torch.testing.assert_close(compact,reference,atol=1e-10,rtol=1e-10)

    # Mutate only ports the optimized path claims not to read.
    trimmed={key:[v.detach().clone() for v in vals] for key,vals in same.items()}
    mask=torch.ones(batch,length,dtype=torch.bool)
    mask[torch.arange(batch),positions]=False
    for k in range(1,len(blocks)):
        trimmed['raw_attention_inputs'][k].normal_()
    for h in trimmed['mlp_inputs']:h[mask]=torch.randn_like(h[mask])*100
    trimmed['states'][-1][mask]=torch.randn_like(trimmed['states'][-1][mask])*100
    for h in trimmed['states'][:-1]:h.normal_()
    unchanged=earliest_attention_reader(blocks,trimmed,first,U,positions,eps)
    torch.testing.assert_close(unchanged,compact,atol=1e-10,rtol=1e-10)
