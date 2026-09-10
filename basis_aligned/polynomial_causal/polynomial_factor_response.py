"""First-order responses in actual normalized product factors, not raw inputs."""
import torch


def mlp_response(base_left,base_right,current_left,current_right,down):
    dl=current_left-base_left;dr=current_right-base_right
    return (base_left*dr+dl*base_right)@down.T


def attention(q,k,v,q2,k2):
    d=q.shape[-1];t=q.shape[1]
    dot=lambda a,b:torch.einsum('bthd,bshd->bhts',a,b)
    pattern=dot(q,k)*dot(q2,k2)/d**2
    mask=torch.ones(t,t,device=q.device,dtype=torch.bool).tril()
    return torch.einsum('bhts,bshd->bhtd',pattern*mask,v)


def attention_response(base,current):
    q,k,v,q2,k2=base;dq,dk,dv,dq2,dk2=[b-a for a,b in zip(base,current)]
    dot=lambda a,b:torch.einsum('bthd,bshd->bhts',a,b)
    s1=dot(q,k);s2=dot(q2,k2)
    ds1=dot(dq,k)+dot(q,dk);ds2=dot(dq2,k2)+dot(q2,dk2)
    mask=torch.ones(q.shape[1],q.shape[1],device=q.device,dtype=torch.bool).tril()
    read=lambda p,value:torch.einsum('bhts,bshd->bhtd',p*mask/q.shape[-1]**2,value)
    return read(ds1*s2+s1*ds2,v)+read(s1*s2,dv)


def controls():
    torch.set_num_threads(2);g=torch.Generator().manual_seed(60930)
    rand=lambda *s:torch.randn(*s,generator=g,dtype=torch.float64)
    base=tuple(rand(2,5,2,4) for _ in range(5));delta=tuple(.2*rand(2,5,2,4) for _ in range(5));h=1e-5
    first=attention_response(base,tuple(a+b for a,b in zip(base,delta)))
    numerical=(attention(*(a+h*b for a,b in zip(base,delta)))-attention(*(a-h*b for a,b in zip(base,delta))))/(2*h)
    error=float((first-numerical).abs().max());single=[]
    for i in range(5):
        changed=tuple(a+(delta[j] if j==i else 0) for j,a in enumerate(base))
        single.append(float((attention_response(base,changed)-(attention(*changed)-attention(*base))).abs().max()))
    mixed=float((first-(attention(*(a+b for a,b in zip(base,delta)))-attention(*base))).norm())
    l=rand(3,7);r=rand(3,7);dl=rand(3,7);dr=rand(3,7);w=rand(4,7)
    mlp_error=float((mlp_response(l,r,l+dl,r+dr,w)+(dl*dr)@w.T-((l+dl)*(r+dr)-l*r)@w.T).abs().max())
    # Mixed module-enable dependence exists even for a strictly linear source map.
    a=torch.tensor([[0.,0.],[1.,0.]],dtype=torch.float64);m=a.T;eye=torch.eye(2,dtype=torch.float64);x=torch.tensor([1.,0.],dtype=torch.float64)
    interaction=(eye+m)@(eye+a)@x-(eye+m)@x-(eye+a)@x+x
    checks={'attention_finite_difference':error<1e-8,'all_single_factor_changes_exact':max(single)<1e-11,'joint_factor_nonlinearity_live':mixed>.01,'mlp_cross_change_partition':mlp_error<1e-11,'linear_source_map_has_mixed_module_paths':torch.equal(interaction,m@a@x) and float(interaction.norm())>.1}
    return {'passed':all(checks.values()),'checks':checks,'attention_central_difference_error':error,'single_factor_error':max(single),'mlp_partition_error':mlp_error,'joint_factor_remainder_norm':mixed,'model_forwards':0,'scope':'Product-factor expansion controls. Actual changed RMS/RoPE factors must be computed before use. No native prediction or removed-normalizer claim.'}


if __name__=='__main__':
    import json
    from pathlib import Path
    result=controls();assert result['passed']
    with Path(__file__).with_name('POLYNOMIAL_FACTOR_RESPONSE_V1_CONTROLS.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2))
