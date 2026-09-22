"""Compress the frozen fitted CP parent, not refit directly to native weights."""
import json,time
from pathlib import Path
import torch
from cp_backward_pruning import prune,controls
from quartic_cp import cp_gram
from mixed_gaussian_cp import gram_dynamic
from noncentral_gaussian_cp import affine_moment
from gaussian_cp_derivative_gram import gram as derivative_gram
P=Path(__file__).resolve().parent
SCALE=19054614563.464127

def values(x,factors,c):
    phi=torch.ones(len(x),factors[0].shape[0],dtype=x.dtype)
    for f in factors:phi*=x@f.T
    return phi@c.T

def main():
    torch.set_num_threads(2);torch.set_grad_enabled(False);start=time.monotonic();checks=controls();saved=torch.load(P/'GAUSSIAN_CP_DATA_PROJECTIONS_V1.pt',weights_only=True);S=saved['projections']['covariance']['whitener'];mu=saved['mean'];x=torch.load(P/'NATIVE_QUARTIC_COVARIANCE_V1.pt',weights_only=True)['panels'][1]['rows'].double();label=torch.load(P/'SENSITIVE_ROOT_CALIBRATION_V2.pt',weights_only=True)['panels'][1];target=label['target'].double()/SCALE;weights=label['weight'].double();match=json.loads((P/'ROOT_MATCHED_READER_V1.json').read_text())['rows'][1];rec,don=torch.tensor(match['pairs_flat']).T;reference=torch.tensor([r['reference'] for r in match['pair_rows']],dtype=x.dtype)/SCALE
    rows=[];exports={}
    for seed in [1001,1002]:
        parent=torch.load(P/f'MIXED_CP_FEATURES_SEED{seed}_V1.pt',weights_only=True);f=[a.double() for a in parent['factors']];C=parent['coefficients'].double()/SCALE;G0=cp_gram(f,f);fw=[a@S for a in f];bias=[a@mu for a in f];G1=gram_dynamic(fw,bias,fw,bias);lam=parent['coefficient_weight'];meanphi=affine_moment(fw,bias);Gcenter=G1-meanphi[:,None]*meanphi[None,:]
        Gd=derivative_gram(fw,bias);energy_value=((C.T@C)*G1).sum();energy_derivative=((C.T@C)*Gd).sum()
        for metric,G in [('sobolev',.5*(G1/energy_value+Gd/energy_derivative)),('derivative',Gd/energy_derivative)]:
            snapshots,history=prune(G,C@G,[512,384,256,128]);exports[(seed,metric)]={}
            for count,snapshot in snapshots.items():
                ids=snapshot['indices'];c=snapshot['coefficients'];student=[a[ids] for a in f];pred=values(x,student,c);sens=((weights*(pred-target).square()).sum(0)/(weights*target.square()).sum(0)).sqrt();difference=-C.clone();difference[:,ids]+=c
                def error(g):return float((((difference.T@difference)*g).sum().clamp_min(0)/((C.T@C)*g).sum()).sqrt())
                row=dict(seed=seed,metric=metric,terms=count,products=3*count,stored_coefficients=4*count*1152+16*count+1152*16,parent_coefficient_error=error(G0),parent_gaussian_error=error(G1),native_text_error=float((pred-target).norm()/target.norm()),native_root1_sensitivity_error=float(sens[1]),native_root1_same_token_error=float(((pred[don,1]-pred[rec,1])-reference).norm()/reference.norm()),normal_residual=snapshot['normal_residual'],max_deletion_formula_error=max(r['formula_error'] for r in history))
                energy=(C@G1*C).sum(1);feature_error=((difference@G1*difference).sum(1).clamp_min(0)/energy).sqrt()
                row['parent_derivative_error']=error(Gd);row['parent_root1_derivative_error']=float(((difference[1]@Gd@difference[1]).clamp_min(0)/(C[1]@Gd@C[1])).sqrt())
                row['parent_centered_gaussian_error']=error(Gcenter);row['parent_gaussian_variance_fraction']=float(((C.T@C)*Gcenter).sum()/((C.T@C)*G1).sum());row['parent_root1_centered_gaussian_error']=float(((difference[1]@Gcenter@difference[1]).clamp_min(0)/(C[1]@Gcenter@C[1])).sqrt())
                row['parent_gaussian_output_energy_fraction']=(energy/energy.sum()).tolist();row['parent_gaussian_feature_errors']=feature_error.tolist()
                rows.append(row);exports[(seed,metric)][count]=dict(factors=[a.float() for a in student],coefficients=(c*SCALE).float(),writer=parent['writer'],indices=ids,seed=seed,degree=4);print(json.dumps(row),flush=True)
    primary=[r for r in rows if r['metric']=='sobolev' and r['terms']==256];baseline={(r['seed'],r['metric']):r for r in rows if r['terms']==512}
    predictions=dict(integrity=all(r['normal_residual']<1e-7 and r['max_deletion_formula_error']<1e-7 for r in rows),primary_parent_gaussian=all(r['parent_gaussian_error']<=.01 for r in primary),primary_native_retention=all(all(r[k]<=1.1*baseline[r['seed'],r['metric']][k] for k in ['native_text_error','native_root1_sensitivity_error','native_root1_same_token_error']) for r in primary),primary_component=all(r['native_root1_sensitivity_error']<=.1 and r['native_root1_same_token_error']<=.1 for r in primary))
    torch.save(exports,P/'SOBOLEV_CP_PRUNING_CANDIDATES_V1.pt');result=dict(predictions=predictions,rows=rows,controls=checks,seconds=time.monotonic()-start,scope='Backward greedy compression of frozen mixedCP512 fitted parents, not native target refitting. ExactshiftedGaussian value/derivative parentGrams; primaryequalnormalizedvalue+derivative, secondaryderivativeonly. Derivativesinwhitenedzcoordinates. Fixedbudgets andmetrics, allreported. Nativeopenedmetrics evaluationonly. PrimarySobolev256=768products; no finite-removal/OOD/adoption. Unit-feature-metric ridge1e-10. CP factors are retained, onlysupport/readouts change.')
    (P/'SOBOLEV_CP_PRUNING_V1.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
