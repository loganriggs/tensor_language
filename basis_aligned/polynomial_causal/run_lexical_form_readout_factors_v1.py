"""CPU-only scalar/complement/norm readout lattice on saved intervention states."""
import hashlib,itertools,json,time
from pathlib import Path
import torch
from paired_panel_bootstrap_v1 import PairedPanelBootstrap


def main():
    p=Path(__file__).resolve().parent;tic=time.perf_counter();torch.set_num_threads(2);torch.set_grad_enabled(False)
    binding=json.loads((p/'LEXICAL_FORM_READOUT_FACTORS_V1_BINDING.json').read_text())
    for path,sha in binding.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha
    saved=torch.load(p/'LEXICAL_FORM_INTERCHANGE_V1_STATES.pt',map_location='cpu',weights_only=True)
    e=torch.load(p/'GERUND_SHARED_READER_V1_COMPONENT.pt',map_location='cpu',weights_only=True)['e'].double()
    H=torch.tensor([[1,1,1,1],[-1,-1,1,1],[-1,1,-1,1],[1,-1,-1,1]],dtype=torch.float64)
    rho=lambda h:torch.sqrt(h.square().mean(-1)+torch.finfo(torch.float32).eps)
    lex=lambda z:torch.stack((z[:,2]-z[:,0],z[:,3]-z[:,1]),-1)
    center=lambda z:z-z.mean(-1,keepdim=True)
    reports={};checks={'unit_error':abs(float(e.norm())-1)};program={}
    for name in ['A1','A2']:
        U=torch.einsum('ba,nbd->nad',H,saved[name]['reader_components'])
        for after,before in [('FB','B'),('FX','X')]:
            label=name+'_'+after;bs=PairedPanelBootstrap(16,9111581+len(reports))
            h=[saved[name]['states'][k]['h'].double() for k in [before,after]];z=[saved[name]['states'][k]['selected_scores'].double() for k in [before,after]]
            s=[x@e for x in h];a=U@e;perp=[x-y[:,None]*e for x,y in zip(h,s)]
            c=[torch.einsum('nkd,nd->nk',U,x) for x in perp];r=[rho(x) for x in h]
            split=max(float((a*ss[:,None]+cc-torch.einsum('nkd,nd->nk',U,hh)).abs().max()) for ss,cc,hh in zip(s,c,h))
            read=lambda ss,cc,rr:30*torch.tanh((a*ss[:,None]+cc)/(30*rr[:,None]))
            pred={''.join(map(str,choice)):read(s[choice[0]],c[choice[1]],r[choice[2]]) for choice in itertools.product([0,1],repeat=3)}
            eta=perp[0].square().mean(-1);rp=torch.sqrt(eta+s[1].square()/1152+torch.finfo(torch.float32).eps)
            pred['physical']=read(s[1],c[0],rp)
            physical_state=h[0]+(s[1]-s[0])[:,None]*e
            norm_error=float((rp-rho(physical_state)).abs().max())
            endpoint=max(float((pred[k]-target).abs().max()) for k,target in [('000',z[0]),('111',z[1])])
            reference=lex(z[1])-lex(z[0]);ref_sq=reference.square().sum(-1);score_ref=center(z[1]-z[0]);arms={}
            for k,value in pred.items():
                error=(lex(value)-lex(z[0]))-reference;err_sq=error.square().sum(-1)
                arms[k]=dict(lexical_drift_error=float(error.norm()/reference.norm()),lexical_error_ci95=bs.relative_l2(err_sq.tolist(),ref_sq.tolist()),
                    error_squared_per_row=err_sq.tolist(),reference_squared_per_row=ref_sq.tolist(),selected_effect_error=float(center(value-z[1]).norm()/score_ref.norm()),
                    predicted_score_components=((value-z[0])@H.T/4).tolist())
            checks[label]=dict(numerator_split_max_abs=split,physical_norm_max_abs=norm_error,endpoint_logit_max_abs=endpoint,reference_norm=float(reference.norm()),finite=all(bool(torch.isfinite(x).all()) for x in pred.values()))
            reports[label]=dict(arms=arms,native_score_change_components=((z[1]-z[0])@H.T/4).tolist())
            program[label]=dict(a=a,c_base=c[0],s_base=s[0],perpendicular_mean_square=eta)
    valid=checks['unit_error']<=1e-10 and all(v['numerator_split_max_abs']<=1e-10 and v['physical_norm_max_abs']<=1e-10 and v['endpoint_logit_max_abs']<=1e-3 and v['reference_norm']>1e-4 and v['finite'] for k,v in checks.items() if isinstance(v,dict))
    held=lambda key:valid and all(v['arms'][key]['lexical_drift_error']<=.2 for v in reports.values())
    artifact=p/'LEXICAL_FORM_READOUT_FACTORS_V1_PROGRAM.pt';assert not artifact.exists();torch.save(program,artifact)
    out=dict(schema='lexical.form_readout_factors.v1',predictions={'pred_a_instrument':valid,'pred_b_physical_scalar':held('physical'),'pred_c_scalar_actual_norm':held('101'),'pred_d_complement_actual_norm':held('011')},reports=reports,checks=checks,
        source_sha256=hashlib.sha256((p/'LEXICAL_FORM_INTERCHANGE_V1_STATES.pt').read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),artifact_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
        wall_seconds=time.perf_counter()-tic,price=dict(native_forwards=0,gpu_accessed=False,checkpoint_loaded=False,stored_coefficients=sum(t.numel() for v in program.values() for t in v.values()),artifact_bytes=artifact.stat().st_size,native_weight_saving=0),
        scope='Conditional local readout of fixed e and native initial context; saved-state factorial evaluates explanations of form-induced lexical drift only. No native extraction, OOD or repair of prior selectivity/choice failures.')
    with (p/'LEXICAL_FORM_READOUT_FACTORS_V1_RESULT.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
    print(json.dumps(dict(predictions=out['predictions'],errors={n:{k:dict(lexical=v['lexical_drift_error'],selected=v['selected_effect_error']) for k,v in r['arms'].items() if k in ['physical','101','011','111']} for n,r in reports.items()},checks=checks,price=out['price'],wall_seconds=out['wall_seconds']),indent=2))


if __name__=='__main__':main()
