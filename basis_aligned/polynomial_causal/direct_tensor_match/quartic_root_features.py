"""Coefficient queries retaining native quadratic-product root features."""

def root_entries(L2,R2,D1,L1,R1,indices):
 def pair(i,j):return .5*((L1[:,i]*R1[:,j]+L1[:,j]*R1[:,i]).T@D1.T)
 def outer(s,t):return .5*((s@L2.T)*(t@R2.T)+(t@L2.T)*(s@R2.T))
 i,j,k,l=indices.T
 return (outer(pair(i,j),pair(k,l))+outer(pair(i,k),pair(j,l))+outer(pair(i,l),pair(j,k)))/3
