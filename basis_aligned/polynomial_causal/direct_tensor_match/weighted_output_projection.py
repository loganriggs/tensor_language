"""Optimal output subspace for a fixed coefficient writer under a PSD feature Gram.
Only compression of a fixed parent, not refitting the native target or its features.
"""
import torch

def project(writer,gram,rank):
    g=(gram+gram.T)/2;e,v=torch.linalg.eigh(g)
    if float(e[-1])<=0:raise ValueError('zero feature metric')
    if float(e[0]) < -1e-10*float(e[-1]):raise ValueError('indefinite feature metric')
    root=v*e.clamp_min(0).sqrt()
    matrix=writer@root
    left,singular,_=torch.linalg.svd(matrix,full_matrices=False)
    if not 1<=rank<=len(singular):raise ValueError('invalid rank')
    basis=left[:,:rank];compressed=basis@(basis.T@writer)
    energy=matrix.square().sum()
    if energy<=0:raise ValueError('zero parent function')
    direct=((writer-compressed)@root).square().sum();tail=singular[rank:].square().sum()
    return basis,compressed,dict(relative_error=float((direct/energy).sqrt()),tail_error=float((tail/energy).sqrt()),certificate=float(abs(direct-tail)/energy),metric_minimum_eigenvalue=float(e[0]),metric_maximum_eigenvalue=float(e[-1]))

def controls():
    records=[]
    for index,family in enumerate(['dense','shared_output','singular_metric','scaled_features','cancellation']):
        torch.manual_seed(9140+index);z=torch.randn(57,10,dtype=torch.float64);c=torch.randn(7,10,dtype=z.dtype)
        if family=='shared_output':c=torch.randn(7,2,dtype=z.dtype)@torch.randn(2,10,dtype=z.dtype)
        if family=='singular_metric':z[:,5:]=z[:,:5]
        if family=='scaled_features':z=z*torch.logspace(-2,2,10,dtype=z.dtype)
        if family=='cancellation':z[:,1]=z[:,0];c[:,1]=-c[:,0]
        prediction=z@c.T;uu,ss,vh=torch.linalg.svd(prediction,full_matrices=False)
        errors=[]
        for rank in [1,3,7]:
            b,compressed,info=project(c,z.T@z,rank);reference=(uu[:,:rank]*ss[:rank])@vh[:rank];error=float((z@compressed.T-reference).norm()/prediction.norm());assert error<1e-8 and info['certificate']<1e-10;errors.append(error)
        records.append(dict(family=family,maximum_direct_prediction_error=max(errors)))
    return records

if __name__=='__main__':
    import json
    from pathlib import Path
    torch.set_num_threads(2);rows=controls();Path(__file__).with_name('WEIGHTED_OUTPUT_PROJECTION_CONTROLS_V1.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows,indent=2))
