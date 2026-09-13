"""Direct nineteen-bank compiler, without a per-context least-squares solve."""
import torch

def prepare(vectors,left,right,down,rho0,beta,gamma):
    """vectors[...,7,d]: w0,w1,p0,p1,q00,q01,q11, including re-entry scale.

    rho0[...,1], beta[...,2], gamma[...,2,2]. Inputs must broadcast to the
    vector batch. Coefficients are ordered b^n, a*b^(n-1), ..., a^n.
    """
    l=vectors@left.T;r=vectors@right.T
    def K(i,j):return l[...,i,:]*r[...,j,:]+l[...,j,:]*r[...,i,:]
    xx=torch.stack([K(1,1),2*K(0,1),K(0,0)],-2)
    def mixed(i,j):
        return torch.stack([-K(j,6),-K(i,6)-2*K(j,5),
                            -2*K(i,5)-K(j,4),-K(i,4)],-2)
    xq=mixed(0,1)
    xp=torch.stack([2*K(1,3),2*(K(0,3)+K(1,2)),2*K(0,2)],-2)
    pp=torch.stack([K(3,3),2*K(2,3),K(2,2)],-2)
    pq=mixed(2,3)
    qq=torch.stack([K(6,6)/4,K(5,6),K(5,5)+K(4,6)/2,
                    K(4,5),K(4,4)/4],-2)
    linear=torch.stack([-2*beta[...,1],-2*beta[...,0]],-1)
    quadratic=torch.stack([gamma[...,1,1],2*gamma[...,0,1],gamma[...,0,0]],-1)
    def convolve(coefficients):
        n=coefficients.shape[-1]
        terms=[]
        for power in range(n+2):
            terms.append(sum(coefficients[...,i,None]*xp[...,power-i,:]
                             for i in range(n) if 0<=power-i<3))
        return torch.stack(terms,-2)
    hidden=torch.cat([xx,xq,pp+rho0.unsqueeze(-1)*xp,
                      pq+convolve(linear),qq+convolve(quadratic)],-2)
    return hidden@down.T
