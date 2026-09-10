"""CPU trained-weight pullback of the saved output partition, with no fit.

For one head h: O_P,h = (O q) q_h^T; O_R,h = O_h - O_P,h.
Both inherit the exact same native QK1/QK2 factors. Test whether the P scalar
value reader belongs to the R contribution's input row space, conditional on
this head's routing. This is not a whole-model semantic or causal certificate.
"""
import hashlib
import json
import time
from pathlib import Path
import torch
from correlative_route_read_write_v1 import fold


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()


def overlap(reader, value, remainder_writer, threshold):
    _,sv,vh=torch.linalg.svd(remainder_writer,full_matrices=False)
    keep=sv>threshold*sv[0]
    pulled=value.T@vh[keep].T
    basis=torch.linalg.qr(pulled,mode='reduced').Q
    residual=reader-basis@(basis.T@reader)
    return {'writer_rank':int(keep.sum()),'reader_relative_residual':float(residual.norm()/reader.norm().clamp_min(1e-30)),
            'reader_squared_overlap':float((basis.T@reader).square().sum()/reader.square().sum().clamp_min(1e-30)),
            'singular_values':sv.tolist()}


def main():
    start=time.perf_counter();torch.set_num_threads(2)
    poly=Path(__file__).resolve().parent;root=poly.parents[1]
    artifact_path=poly/'CORRELATIVE_REPLAYABLE_INTERFACE_V1_ARTIFACT.pt'
    parent=json.loads((poly/'CORRELATIVE_COMPLEMENT_SPLIT_V1_BINDING.json').read_text())
    checkpoint=Path(next(p for p in parent if p.endswith('pytorch_model.bin')))
    assert sha(checkpoint)==parent[str(checkpoint)]
    assert sha(artifact_path)==parent[str(artifact_path)]
    artifact=torch.load(artifact_path,map_location='cpu',weights_only=True)
    state=torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True)
    blocks=fold(artifact,state);reports=[];max_bridge=0.;max_fold=0.
    gen=torch.Generator().manual_seed(9111321)
    for layer,block in sorted(blocks.items()):
        pref=f'transformer.h.{layer}.attn.'
        output=state[pref+'c_proj.weight'].double()
        vl=state[pref+'c_v.weight'].double();v0=state['transformer.h.0.attn.c_v.weight'].double()
        lam=block['value_mix_lambda'];writer=block['writer']
        q=artifact['q'][(layer,'heads')].double().flatten()
        qnorm=float(q.norm());row={}
        for j,port in enumerate(block['ports']):
            h=port['head'];sl=slice(128*h,128*(h+1));qh=port['head_reader']
            oh=output[:,sl];op=writer[:,None]*qh[None,:];orr=oh-op
            value=torch.cat([(1-lam)*vl[sl],lam*v0[sl]],dim=1)
            reader=qh@value
            saved_reader=torch.cat([port['local_value_reader'],port['first_value_reader']])
            max_fold=max(max_fold,float((reader-saved_reader).abs().max()))
            report={'layer':layer,'head':h,'unit':port['unit'],'heads_in_block':len(block['ports']),
                    'block_q_norm':qnorm,'head_q_norm':float(qh.norm()),'value_mix_lambda':lam,
                    'value_input_dimension':2304,'value_rank':int(torch.linalg.matrix_rank(value,rtol=1e-6)),
                    'primary':overlap(reader,value,orr,1e-6)}
            report['rank_sensitivity']={str(t):overlap(reader,value,orr,t)['writer_rank'] for t in (1e-5,1e-7)}
            # Remove only stored-q normalization roundoff in a separate geometry control.
            ideal_op=(writer/qnorm)[:,None]*(qh/qnorm)[None,:]
            report['unit_normalized_q_control']=overlap(reader,value,oh-ideal_op,1e-6)
            inp=torch.randn(12,2304,generator=gen,dtype=torch.float64)
            route=torch.randn(12,1,generator=gen,dtype=torch.float64)
            native=(inp@value.T)@oh.T*route
            pp=(inp@reader)[:,None]*writer[None,:]*route
            rr=(inp@value.T)@orr.T*route
            error=float((native-pp-rr).norm()/native.norm().clamp_min(1e-30))
            max_bridge=max(max_bridge,error);report['synthetic_routed_split_relative_error']=error
            reports.append(report)
    assert len(reports)==26 and max_bridge<1e-10 and max_fold<1e-10
    full=[r for r in reports if r['primary']['writer_rank']==128]
    result={'schema':'correlative.ov_pullback.v1','reports':reports,
            'summary':{'heads':26,'blocks':14,'full_column_rank_remainder_writers':len(full),
                       'max_scalar_reader_residual_in_full_rank_remainders':max(r['primary']['reader_relative_residual'] for r in full),
                       'all_reader_squared_overlap_range':[min(r['primary']['reader_squared_overlap'] for r in reports),max(r['primary']['reader_squared_overlap'] for r in reports)],
                       'routed_split_max_relative_error':max_bridge,'saved_reader_max_absolute_error':max_fold},
            'routing_fact':'For this fixed output partition QK1, QK2, their four norm factors and native position transforms are identical within each head in both branches; only O changes.',
            'scope':'CPU trained-weight analysis. Value variables are actual normalized local and layer-0 attention inputs, treated as independent coordinates. Input row-space overlap is conditional per head, not equality of complete branch functions or proof of a semantic complement circuit. No native trajectories, fitting, causal selection or GPU use.',
            'checkpoint_sha256':parent[str(checkpoint)],'artifact_sha256':sha(artifact_path),
            'source_sha256':sha(__file__),'fold_source_sha256':sha(poly/'correlative_route_read_write_v1.py'),
            'wall_seconds':time.perf_counter()-start}
    out=poly/'CORRELATIVE_OV_PULLBACK_V1_RESULT.json'
    with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result['summary'],indent=2))
    print(json.dumps([{'unit':r['unit'],'heads_in_block':r['heads_in_block'],'rank':r['primary']['writer_rank'],
                      'overlap':r['primary']['reader_squared_overlap']} for r in reports],indent=2))


if __name__=='__main__':main()
