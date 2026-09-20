"""Exact symmetric quartic coefficient inner product for squared quadratic forms."""
def square_inner(Q,S):
 QS=Q@S
 return (QS.diagonal().sum().square()+2*(QS*QS.T).sum())/3

def symmetric_bilinear(U,V):return (U@V.T+V@U.T)/2

def gaussian_square_inner(Q,S):
 t=Q.diagonal().sum();u=S.diagonal().sum();QQ=Q@Q;SS=S@S;QS=Q@S;a=QQ.diagonal().sum();b=SS.diagonal().sum();c=QS.diagonal().sum()
 return t*t*u*u+2*a*u*u+2*b*t*t+8*c*t*u+16*(QQ*S.T).sum()*u+16*(Q*SS.T).sum()*t+4*a*b+8*c*c+32*(QQ*SS.T).sum()+16*(QS*QS.T).sum()
