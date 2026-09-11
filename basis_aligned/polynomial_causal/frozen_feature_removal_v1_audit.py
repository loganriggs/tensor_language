"""Exact removal computations of 16 frozen shared features; no data fitting.

Reuse sparse feature executor and the fixed-reader spectral metric already
established in shared_input_ranked_partner_v1. A replay<=1e-8;
B >=12/16 rank16 capture>=.8; C >=8/16 rank1 capture>=.5.
"""
import hashlib,json,time
from pathlib import Path
import torch
from native_support_exchange_v1_audit import P
from rectangular_sparse_reader_v1 import RectangularSparseReaderProgram


def main():
    torch.set_grad_enabled(False);torch.set_default_dtype(torch.float64);torch.set_num_threads(2)
    out=P/'FROZEN_FEATURE_REMOVAL_V1_AUDIT.json';assert not out.exists()
    parent=json.loads((P/'NATIVE_SUPPORT_EXCHANGE_V1_AUDIT.json').read_text())
    source=Path(parent['source']['path']);assert hashlib.sha256(source.read_bytes()).hexdigest()==parent['source']['sha256']
    saved=torch.load(source,weights_only=True,map_location='cpu')
    program=RectangularSparseReaderProgram.from_artifact(saved,saved['down'].double())
    basis=program.analysis_basis;codes=program.codes.to_dense();cl,cr=codes.chunk(2)
    l,r=(codes@basis).chunk(2)
    metric=torch.load('/dev/shm/bilin18_native_product_energy_v1.pt',weights_only=True,map_location='cpu')['unembedding_gram'].double()
    wh=torch.linalg.cholesky(metric).T;w=wh@program.down
    x=torch.randn(8,1152,generator=torch.Generator().manual_seed(773))
    features=[i*2304//16 for i in range(16)];rows=[];directions=[];energies=[];started=time.perf_counter()
    y=program(x); maximum_replay=0.
    for i in features:
        a=basis[i];left,right=cl[:,i],cr[:,i];use=(left!=0)|(right!=0)
        m=w[:,use]@(left[use,None]*r[use]+right[use,None]*l[use])
        m-=(w@(left*right))[:,None]*a[None,:]
        direct=(y-program.remove_features(x,[i]))@wh.T
        folded=(x@a)[:,None]*(x@m.T)
        replay=float((direct-folded).norm()/direct.norm().clamp_min(1e-30))
        # For unit a, H^(1/2)=I/sqrt(2)+(1-1/sqrt(2))*a*a^T.
        assert abs(float(a.norm())-1)<=1e-10
        transformed=m/(2**.5)+(1-2**-.5)*(m@a)[:,None]*a[None,:]
        u,s,_=torch.linalg.svd(transformed,full_matrices=False)
        energy=s.square().sum();analytic=.5*(m.square().sum()+(m@a).square().sum())
        energy_replay=float(abs(energy-analytic)/analytic.clamp_min(1e-30))
        captures={str(k):float(s[:k].square().sum()/energy) for k in (1,4,8,16,64,128)}
        rank90=int(torch.searchsorted(s.square().cumsum(0)/energy,torch.tensor(.9)))+1
        row=dict(feature=i,product_uses=int(use.sum()),energy=float(energy),captures=captures,
                 rank90=rank90,executor_replay=replay,energy_replay=energy_replay)
        rows.append(row);directions.append(u[:,0]);energies.append(float(energy))
        maximum_replay=max(maximum_replay,replay,energy_replay)
        print(json.dumps(row),flush=True)
    pairs=[]
    for pos,i in enumerate(features):
        for q in range(pos+1,len(features)):
            j=features[q];a,b=basis[i],basis[j]
            writer=w@(cl[:,i]*cr[:,j]+cl[:,j]*cr[:,i])
            direct=program.disjoint_interaction(x,[i],[j])@wh.T
            folded=((x@a)*(x@b))[:,None]*writer[None,:]
            replay=float((direct-folded).norm()/direct.norm().clamp_min(1e-30))
            energy=float(writer.square().sum()*.5*(a.square().sum()*b.square().sum()+(a@b).square()))
            pairs.append(dict(features=[i,j],energy=energy,
                relative_to_single_geometric_mean=energy/(energies[pos]*energies[q])**.5,
                dominant_output_cosine=float(abs(directions[pos]@directions[q])),executor_replay=replay))
            maximum_replay=max(maximum_replay,replay)
    result=dict(predictions=dict(pred_a_instrument=maximum_replay<=1e-8,
        pred_b_compact=sum(row['captures']['16']>=.8 for row in rows)>=12,
        pred_c_simple=sum(row['captures']['1']>=.5 for row in rows)>=8),
        rows=rows,pairs=pairs,maximum_replay=maximum_replay,compute_seconds=time.perf_counter()-started,
        source=parent['source'],body_forwards=0,corpus_access=False,
        scope='Removal semantics inside a frozen unconverged overcomplete program. '
              'Independent feature edits need not equal changes to physical input x. '
              'Spectral compactness and sampled pair effects do not identify stable semantic circuits.')
    with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','pairs')},indent=2))


if __name__=='__main__':main()
