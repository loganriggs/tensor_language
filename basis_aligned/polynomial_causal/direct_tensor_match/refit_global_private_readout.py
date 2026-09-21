"""Exact constrained readout refit after shared-to-private graph edits."""
from pathlib import Path
import json,torch
from global_mixed_source_graph import export,score,source_reads
P=Path(__file__).parent;torch.set_num_threads(2);torch.set_grad_enabled(False)

def gram(L,R,A,B):
 return .5*((L.T@A)*(R.T@B)+(L.T@B)*(R.T@A))
def fit(T,L,R,V,M):
 # Shared atoms can feed every output; squares V feed only output5.
 S=gram(L,R,L,R);C=gram(L,R,V,V);K=gram(V,V,V,V)
 rhs=torch.einsum('ir,oij,jr->ro',L,T,R);rp=torch.einsum('ir,oij,jr->ro',V,T,V)
 X=torch.linalg.solve(S,rhs);Z=torch.linalg.solve(S,C)
 projected=K-C.T@Z;v=torch.linalg.solve(projected,(rp-C.T@X)@M[:,5]/M[5,5])
 W=X.clone();W[:,5]-=Z@v
 # Free-coordinate normal equations, with the private-output restriction explicit.
 Es=(S@W+C@(v[:,None]*torch.nn.functional.one_hot(torch.tensor(5),6).to(T)[None])-rhs)@M
 Ep=((C.T@W)@M[:,5]+(K@v)*M[5,5]-rp@M[:,5])
 replay=max(float(Es.norm()/(rhs@M).norm()),float(Ep.norm()/(rp@M[:,5]).norm()))
 return W.T,v,replay
# Small independent dense-autograd check of the constrained normal equations.
g=torch.Generator().manual_seed(8591);T=torch.randn(6,9,9,generator=g,dtype=torch.float64);T=(T+T.transpose(-1,-2))/2
L=torch.randn(9,5,generator=g,dtype=T.dtype);R=torch.randn(9,5,generator=g,dtype=T.dtype);V=torch.randn(9,3,generator=g,dtype=T.dtype);B=torch.randn(6,6,generator=g,dtype=T.dtype);M=B.T@B+torch.eye(6,dtype=T.dtype)
W,v,normal=fit(T,L,R,V,M)
with torch.enable_grad():
 W=W.requires_grad_();v=v.requires_grad_();raw=torch.einsum('ir,or,jr->oij',L,W,R);hat=(raw+raw.transpose(-1,-2))/2
 e5=torch.nn.functional.one_hot(torch.tensor(5),6).to(T);hat=hat+e5[:,None,None]*((V*v)@V.T)[None];E=hat-T;loss=(E*torch.einsum('ab,bij->aij',M,E)).sum();dw,dv=torch.autograd.grad(loss,(W,v));grad=max(float(dw.abs().max()),float(dv.abs().max()));assert grad<1e-9

d=torch.load(P/'SHARED_PRODUCT_NATIVE_INPUTS_V1.pt',weights_only=True);original=torch.load(P/'GLOBAL_PRIVATE_RESIDUAL_PROGRAMS_V1.pt',weights_only=True);root=torch.linalg.inv(d['inverse_root']);T=d['teacher'];F=T.flatten(1);e,U=torch.linalg.eigh(F@F.T);M=(U*e.pow(-.5))@U.T;records=[];programs={}
for k in [8,16,32]:
 old=original[str(k)];L=root@old['left_reader'];R=root@old['right_reader'];V=root@old['square_reader'];W,v,replay=fit(T,L,R,V,M)
 extra=torch.zeros((6,len(v)),dtype=T.dtype);extra[5]=v
 big=export(torch.cat([L,V],1),torch.cat([R,V],1),torch.cat([W,extra],1),d);m=L.shape[1];p={key:val.clone() for key,val in big.items()}
 p['left_reader']=big['left_reader'][:,:m].clone();p['right_reader']=big['right_reader'][:,:m].clone();p['product_weights']=big['product_weights'][:m].clone();p['square_reader']=big['left_reader'][:,m:].clone();p['square_weights']=big['product_weights'][m:,5].clone();p['square_output']=torch.tensor(5)
 z=d['z'][:32];a=source_reads(z,p);a[:,5]+=(z@p['square_reader']).square()@p['square_weights'];b=source_reads(z,big);execution=float((a-b).norm()/b.norm());assert execution<1e-10 and replay<1e-8
 raw=torch.einsum('ir,or,jr->oij',L,W,R);hat=(raw+raw.transpose(-1,-2))/2;hat[5]+=(V*v)@V.T
 error=hat-T;weighted=float(((error*torch.einsum('ab,bij->aij',M,error)).sum()/(T*torch.einsum('ab,bij->aij',M,T)).sum()).sqrt())
 records.append(dict(removed_mixed=k,source_products=383+k,stored_floats=sum(v.numel() for v in p.values() if v.is_floating_point()),weighted_coefficient_error=weighted,original_coefficient_error=float(error.norm()/T.norm()),normal_equation_replay=replay,execution_replay=execution,**score(big,d)));programs[str(k)]=p
out=dict(toy_dense_gradient_maxabs=grad,toy_normal_replay=normal,records=records,primary_removed=16,primary_values_pass=all(a<=.15 and a<=1.1*b for a,b in zip(records[1]['per_mode_errors'],[.03058409729022641,.027558449717507608,.11941807478056159])),scope='Exact output-metric constrained linear refit, fixed directions/topology; no native-row fitting, fresh test, or semantic promotion.')
torch.save(programs,P/'GLOBAL_PRIVATE_READOUT_PROGRAMS_V1.pt');(P/'GLOBAL_PRIVATE_READOUT_REFIT_V1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
