"""Distinguish fixed-feature response capacity from calibration-to-evaluation transfer.
Evaluation-fitted rows are explicit oracles, never exported or used as candidates.
"""
import hashlib,json,time
from pathlib import Path
from sparse_quartic_bank import features
import torch
P=Path(__file__).resolve().parent

from audit_fixed_cp_response_capacity import fit

def main(all_pool=False):
    torch.set_num_threads(2);torch.set_grad_enabled(False);start=time.monotonic()
    cache=torch.load(P/'NATIVE_QUARTIC_COVARIANCE_V1.pt',weights_only=True)['panels']
    extra=torch.load(P/'QUARTIC_ADDITIONAL_STATES_V1.pt',weights_only=True)
    labels=torch.load(P/'SENSITIVE_ROOT_CALIBRATION_V2.pt',weights_only=True)
    rowcache=P.parents[1]/'bilinear_quotient/.rowcache'
    for key,file,n in [('calibration','fineweb_n96_skip1200.pt',96),('evaluation','fineweb_n192_skip7000.pt',32)]:
        ids65=torch.load(rowcache/file,weights_only=True)[:n,:65]
        ids=ids65[:,:64]
        digest=lambda a:hashlib.sha256(a.contiguous().numpy().tobytes()).hexdigest()
        assert digest(ids)==labels['token_sha256'][key]
        if key=='calibration':
            assert digest(ids65[:32])==cache[0]['token_sha256']
            assert digest(ids65[32:96])==extra['token_sha256']
        else:assert digest(ids65)==cache[1]['token_sha256']
    xs=[torch.cat([cache[0]['rows'],extra['rows']]).double(),cache[1]['rows'].double()]
    scale=19054614563.464127
    targets=[a['target'][:,1].double()/scale for a in labels['panels']]
    weights=[a['weight'][:,1].double() for a in labels['panels']]
    matched=json.loads((P/'ROOT_MATCHED_READER_V1.json').read_text())['rows'][1]
    rec,don=torch.tensor(matched['pairs_flat']).T
    reference=torch.tensor([r['reference'] for r in matched['pair_rows']],dtype=torch.float64)/scale
    replay=float(((targets[1][don]-targets[1][rec])-reference).norm()/reference.norm());assert replay<1e-5
    rows=[]
    for seed in [1101,1102]:
        program=torch.load(P/f'SPARSE_SUPPORT_EXCHANGE_SEED{seed}_V1.pt',weights_only=True)
        if all_pool:
            original=torch.load(P/f'SPARSE_QUARTIC_BANK_SEED{seed}_V1.pt',weights_only=True)['pairs']
            candidates=torch.triu_indices(144,144,offset=1)
            occupied=set((original[0]*144+original[1]).tolist())
            remaining=candidates[:,torch.tensor([int(a)*144+int(b) not in occupied for a,b in candidates.T])]
            chosen=torch.randperm(remaining.shape[1],generator=torch.Generator().manual_seed(11700))[:256]
            program['pairs']=torch.cat([original,remaining[:,chosen]],1)
            assert program['pairs'].shape==(2,768)
        phis=[]
        for x in xs:
            phis.append(features(x,*[a.double() for a in program['factors']],program['pairs']))
        for arm,index,sensitive in [('calibration_uniform',0,False),('calibration_sensitive',0,True),('evaluation_sensitive_ORACLE',1,True),('evaluation_sensitive_pairs_ORACLE',1,True)]:
            w=weights[index] if sensitive else torch.ones_like(weights[index]);c,info=fit(phis[index],targets[index],w)
            if arm=='evaluation_sensitive_pairs_ORACLE':
                phi=phis[1];scales=phi.square().mean(0).sqrt();z=phi/scales
                w=weights[1]/weights[1].mean();G=z.T@(w[:,None]*z)
                D=z[don]-z[rec];cached_reference=targets[1][don]-targets[1][rec]
                transport=torch.linalg.solve(G,D.T)
                K=D@transport;K=(K+K.T)/2
                adjustment=torch.linalg.lstsq(K,cached_reference-D@(c*scales),driver='gelsd',rcond=1e-10)
                c=c+transport@adjustment.solution/scales
                pair_error=float((D@(c*scales)-cached_reference).norm()/cached_reference.norm())
                assert pair_error<1e-8,pair_error
                info['unconstrained_stationarity']=info.pop('stationarity')
                residual=G@(c*scales)-z.T@(w*targets[1])-D.T@adjustment.solution
                info.update(pair_constraint_rank=int(adjustment.rank),pair_constraint_error=pair_error,
                            constrained_stationarity=float(residual.norm()/(G.norm()*(c*scales).norm()).clamp_min(1e-30)))
                assert info['constrained_stationarity']<1e-10
            measurements=[]
            for phi,y,weight in zip(phis,targets,weights):
                pred=phi@c
                measurements.append(dict(value_error=float((pred-y).norm()/y.norm()),sensitive_error=float(((pred-y).square()@weight/(y.square()@weight)).sqrt())))
            pred=phis[1]@c
            rows.append(dict(seed=seed,arm=arm,solver=info,calibration=measurements[0],evaluation=measurements[1],same_token_response_error=float(((pred[don]-pred[rec])-reference).norm()/reference.norm())))
    result=dict(rows=rows,pair_replay=replay,seconds=time.monotonic()-start,scope='Root1 only, fixed shared quadratic directions and512learned rootpairs. SVD least squares rcond1e-12 with column scaling; no ridge. Calibration output labels are used in two arms: this is data-based diagnostic, not weights-only fitting. Evaluation-fitted oracle measures representational capacity on opened rows, NOT generalization or a deployable candidate. No coefficients exported. No full-model finite removal or OOD claim.')
    if all_pool:
        result['scope']=result['scope'].replace('and512learned rootpairs','andall768candidate rootpairs; a larger diagnostic dictionary, not the 512-product deployed budget')
    output='SHARED_POOL_RESPONSE_CAPACITY_V1.json' if all_pool else 'SHARED_RESPONSE_CAPACITY_V1.json'
    (P/output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--all-pool',action='store_true')
    main(all_pool=parser.parse_args().all_pool)
