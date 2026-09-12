"""Generic finite, diagonalizable real symmetric pair; explicit numerical checks required.

Not the general canonical-form algorithm: repeated/defective/singular pencils
may require larger blocks and are not repaired here.
"""
import numpy as np
import scipy.linalg as la

def decompose(forms):
    forms=np.asarray(forms,dtype=np.float64)
    norms=np.linalg.norm(forms,axis=(1,2));a,b=forms/norms[:,None,None]
    values,vectors=la.eig(a,b)
    if not np.isfinite(values).all():raise ValueError('Infinite or undefined eigenvalues unsupported')
    columns=[];groups=[];used=set()
    for i,value in enumerate(values):
        if i in used:continue
        start=len(columns)
        if abs(value.imag)<=1e-10*max(1.,abs(value)):
            columns.append(vectors[:,i].real);used.add(i)
        else:
            distances=abs(values-value.conjugate());distances[list(used|{i})]=np.inf
            j=int(distances.argmin())
            if distances[j]>1e-8*max(1.,abs(value)):raise ValueError('Unpaired complex eigenvalue')
            columns.extend([vectors[:,i].real,vectors[:,i].imag]);used.update([i,j])
        groups.append(list(range(start,len(columns))))
    frame=np.column_stack(columns);frame/=np.linalg.norm(frame,axis=0)
    dual=la.inv(frame);blocks=[];blockforms=np.zeros_like(forms)
    transformed=frame.T@forms@frame
    for ids in groups:
        core=transformed[:,ids][:,:,ids];blocks.append(core)
        for m in range(2):blockforms[m][np.ix_(ids,ids)]=core[m]
    rebuilt=dual.T@blockforms@dual
    residual=a@vectors-(b@vectors)*values[None,:]
    denominator=(np.linalg.norm(a)+np.linalg.norm(b)*abs(values))*np.linalg.norm(vectors,axis=0)
    diagnostic=dict(reconstruction_error=float(np.linalg.norm(rebuilt-forms)/np.linalg.norm(forms)),
                    transformed_offblock_error=float(np.linalg.norm(transformed-blockforms)/np.linalg.norm(transformed)),
                    eigen_backward_error=float(np.max(np.linalg.norm(residual,axis=0)/denominator)),
                    frame_condition=float(np.linalg.cond(frame)),scalar_blocks=sum(len(g)==1 for g in groups),pair_blocks=sum(len(g)==2 for g in groups))
    return dual,groups,blocks,diagnostic

def truncate(forms,dual,groups,blocks,budget=32):
    ranked=[]
    for i,(ids,core) in enumerate(zip(groups,blocks)):
        rows=dual[ids];g=rows@rows.T
        energy=sum(np.trace(c@g@c@g) for c in core)
        ranked.append((float(energy)/len(ids),i))
    chosen=[];remaining=budget
    for _,i in sorted(ranked,reverse=True):
        if len(groups[i])<=remaining:chosen.append(i);remaining-=len(groups[i])
    approximation=np.zeros_like(forms)
    for i in chosen:
        rows=dual[groups[i]];approximation+=rows.T@blocks[i]@rows
    return chosen,float(np.linalg.norm(approximation-forms)/np.linalg.norm(forms)),budget-remaining
