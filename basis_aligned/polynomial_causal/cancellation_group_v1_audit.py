"""Post-result group cancellation and stability, fixed 16 product indices.

A core/CP/execution<=1e-8; B group/individualenergy<=.01;
C snapshot groupcos>=.95; D deletiondamage<=.005 and individualsum>=.5native.
No semantic group discovery, live-fit mutation or sparse-price reduction claim.
"""
import hashlib,json,time
from pathlib import Path
import torch
from native_support_exchange_v1_audit import P,CK
from folded_sparse_dictionary_v1 import decode
from structured_branch_amplitudes_v1 import inner
from chunked_bilinear_coefficient_v1 import dense


def load(source,wh):
    assert hashlib.sha256(Path(source['path']).read_bytes()).hexdigest()==source['sha256']
    saved=torch.load(source['path'],weights_only=True,map_location='cpu')
    ids=saved['code_indices'].long();values=saved['code_values'].double()
    readers,_,_,_=decode(saved['analysis_basis'].double(),ids,values,torch.ones(len(ids)))
    return (*readers.chunk(2),wh@saved['down'].double())


def main():
    torch.set_grad_enabled(False);torch.set_default_dtype(torch.float64);torch.set_num_threads(2)
    out=P/'CANCELLATION_GROUP_V1_AUDIT.json';assert not out.exists();start=time.perf_counter()
    audit=json.loads((P/'PROJECTED_PRODUCT_CANCELLATION_V1_AUDIT.json').read_text())
    older=json.loads((P/'NATIVE_SUPPORT_EXCHANGE_V1_AUDIT.json').read_text())['source']
    metric=torch.load('/dev/shm/bilin18_native_product_energy_v1.pt',weights_only=True,map_location='cpu')['unembedding_gram'].double()
    wh=torch.linalg.cholesky(metric).T
    current=load(audit['source'],wh);previous=load(older,wh)
    sd=torch.load(CK,weights_only=True,map_location='cpu',mmap=True)
    na,nb,nd=[sd[f'transformer.h.17.mlp.{k}.weight'].double() for k in ('Left','Right','Down')]
    native=(na,nb,wh@nd);total=json.loads((P/'NATIVE_READER_METRIC_V1_AUDIT.json').read_text())['native_total']
    rows=[];worst_replay=0.
    for count in (4,8,16):
        ids=audit['top16_products'][:count]
        a,b,w=current[0][ids],current[1][ids],current[2][:,ids]
        group=(a,b,w);old=(previous[0][ids],previous[1][ids],previous[2][:,ids])
        energy=inner(group,group);oldenergy=inner(old,old)
        cosine=float(inner(group,old)/(energy*oldenergy).sqrt())
        individual=(w.square().sum(0)*.5*(a.square().sum(1)*b.square().sum(1)+(a*b).sum(1).square())).sum()
        deletion=float((energy-2*inner(current,group)+2*inner(native,group))/total)
        # Exact orthonormal input/output contraction; no discarded QR columns.
        input_basis=torch.linalg.qr(torch.cat((a,b)).T,mode='reduced').Q
        output_basis=torch.linalg.qr(w,mode='reduced').Q
        core=dense(a@input_basis,b@input_basis,output_basis.T@w)
        core_energy=core.square().sum()
        energy_replay=float(abs(core_energy-energy)/energy.clamp_min(1e-30))
        x=torch.randn(8,1152,generator=torch.Generator().manual_seed(421))
        z=x@input_basis
        direct=((x@a.T)*(x@b.T))@w.T
        folded=torch.einsum('ni,oij,nj->no',z,core,z)@output_basis.T
        execution=float((direct-folded).norm()/direct.norm().clamp_min(1e-30))
        singular=torch.linalg.svdvals(core.reshape(count,-1))
        captures={str(k):float(singular[:k].square().sum()/core_energy) for k in (1,2,4,8)}
        row=dict(count=count,products=ids,group_energy=float(energy/total),individual_energy=float(individual/total),
            group_to_individual=float(energy/individual),old_group_energy=float(oldenergy/total),
            snapshot_cosine=cosine,deletion_error_increase=deletion,
            energy_replay=energy_replay,executor_replay=execution,output_captures=captures,
            core_shape=list(core.shape),input_basis_columns=input_basis.shape[1],output_basis_columns=output_basis.shape[1])
        worst_replay=max(worst_replay,energy_replay,execution);rows.append(row)
    last=rows[-1]
    result=dict(predictions=dict(pred_a_instrument=worst_replay<=1e-8,
        pred_b_cancellation=last['group_to_individual']<=.01,pred_c_stability=last['snapshot_cosine']>=.95,
        pred_d_joint_removal=last['deletion_error_increase']<=.005 and last['individual_energy']>=.5),
        rows=rows,current_source=audit['source'],previous_source=older,seconds=time.perf_counter()-start,
        scope='Post-result fixed product group inside an unconverged weight-only fit. '
              'Core is exact but not automatically a cheaper or identified semantic circuit.')
    with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
