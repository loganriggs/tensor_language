"""Calibration-only descriptive predictors of frozen quartic residuals."""
import hashlib,json,time,math
import torch
from audit_conditional_residual_accounting import P,SCALE,load
from audit_local_quartic_followup import correction

def grouped(train,values,test,prior,strength=20):
    keys,inv=torch.unique(train,sorted=True,return_inverse=True)
    sums=values.new_zeros(len(keys),values.shape[1]);sums.index_add_(0,inv,values)
    count=torch.bincount(inv,minlength=len(keys)).to(values.dtype)[:,None]
    table=(sums+strength*prior)/(count+strength)
    idx=torch.searchsorted(keys,test);safe=idx.clamp_max(len(keys)-1)
    seen=(idx<len(keys))&(keys[safe]==test)
    out=prior.expand(len(test),-1).clone();out[seen]=table[safe[seen]]
    return out,seen

def predictors(token,position,value,newtoken,newposition):
    mean=value.mean(0,keepdim=True)
    pos,_=grouped(position,value,newposition,mean,0)
    trainpos,_=grouped(position,value,position,mean,0)
    tok,seen=grouped(token,value,newtoken,mean)
    add,_=grouped(token,value-trainpos,newtoken,torch.zeros_like(mean))
    return dict(constant=mean.expand(len(newtoken),-1),position=pos,token=tok,additive=pos+add),seen

def controls():
    t=torch.arange(8).repeat_interleave(4);p=torch.arange(4).repeat(8)
    v=torch.stack([t.double(),p.double()],1)
    pred,seen=predictors(t,p,v,torch.tensor([2,99]),torch.tensor([3,1]))
    assert seen.tolist()==[True,False]
    assert torch.equal(pred['token'][1],v.mean(0))
    assert pred['position'][0,1]==3 and pred['position'][1,1]==1
    exact,_=grouped(t,v,t,v.mean(0,keepdim=True),0)
    assert torch.equal(exact[:,0],t.double())
    assert torch.equal(pred['additive'][1],pred['position'][1])
    return dict(unseen_fallback=True,planted_token_signal=True,planted_position_signal=True)

def main():
    torch.set_num_threads(2);torch.set_grad_enabled(False);start=time.monotonic();checks=controls()
    root=P.parents[2];tokenpath=root/'basis_aligned/bilinear_quotient/.rowcache/fineweb_n96_skip1200.pt'
    tokens=torch.load(tokenpath,weights_only=True)[:96,:65]
    labels=torch.load(P/'SENSITIVE_ROOT_CALIBRATION_V2.pt',weights_only=True)
    cov=torch.load(P/'NATIVE_QUARTIC_COVARIANCE_V1.pt',weights_only=True)
    extra=torch.load(P/'QUARTIC_ADDITIONAL_STATES_V1.pt',weights_only=True)
    digest=lambda x:hashlib.sha256(x.contiguous().numpy().tobytes()).hexdigest()
    assert digest(tokens[:,:64])==labels['token_sha256']['calibration']
    assert digest(tokens[:32])==cov['panels'][0]['token_sha256']
    assert digest(tokens[32:])==extra['token_sha256']
    data=torch.load(P/'RESIDUAL_FRESH_STATES_V1.pt',weights_only=True)
    newtokens=torch.load(P/'RESIDUAL_FRESH_TOKENS_V1.pt',weights_only=True)
    assert digest(newtokens)==data['token_content_sha256']
    x=torch.cat([cov['panels'][0]['rows'],extra['rows']]).double();y=labels['panels'][0]['target'].double()/SCALE
    nx=data['rows'].double();ny=data['target'].double()/SCALE
    token=tokens[:,:64].flatten();ntoken=newtokens[:,:64].flatten();pos=torch.arange(64).repeat(96);npos=torch.arange(64).repeat(256)
    parent,ph=load('MIXED_CP_FEATURES_SEED1001_V1.pt')
    def base(xx):return torch.cat([torch.stack([b@a.T for a in parent['factors']]).prod(0)@parent['coefficients'].T/SCALE for b in xx.split(1024)])
    bx=base(x);bn=base(nx);scale=y[:,4:].square().mean(0).sqrt();rows=[]
    native=json.loads((P/'LOCAL_QUARTIC_RESIDUAL_NATIVE_V1.json').read_text())
    for seed in [25001,25002]:
        art,sha=load(f'LOCAL_QUARTIC_RESIDUAL_ADAM_SEED{seed}_V1.pt')
        receipt=next(r for r in native['rows'] if r['optimizer']=='adam' and r['seed']==seed)
        assert sha==receipt['sha256'] and art['parent_sha256']==ph
        residual=(y[:,4:]-bx[:,4:]-correction(art,x))/scale
        nr=(ny[:,4:]-bn[:,4:]-correction(art,nx))/scale
        preds,seen=predictors(token,pos,residual,ntoken,npos)
        risks,_=predictors(token,pos,residual.square().sum(1,keepdim=True),ntoken,npos)
        total=nr.square().sum();constant=(nr-preds['constant']).square().sum();result={}
        for name,pp in preds.items():
            e=(nr-pp).square();risk=risks[name][:,0]
            top=torch.argsort(risk,descending=True,stable=True)[:math.ceil(len(nr)*.1)]
            result[name]=dict(error_energy_over_original=float(e.sum()/total),error_energy_over_constant=float(e.sum()/constant),per_output_energy_over_original=(e.sum(0)/nr.square().sum(0)).tolist(),documents_improved=int((e.reshape(256,64,12).sum((1,2))<nr.square().reshape(256,64,12).sum((1,2))).sum()),top10_predicted_risk_error_share=None if name=='constant' else float(nr[top].square().sum()/total))
        rows.append(dict(seed=seed,sha256=sha,seen_token_fraction=float(seen.double().mean()),unseen_error_energy_share=float(nr[~seen].square().sum()/total),models=result))
        print(seed,result,flush=True)
    out=dict(checks=checks,rows=rows,predictions=dict(token=all(r['models']['token']['error_energy_over_constant']<=.8 for r in rows),position=all(r['models']['position']['error_energy_over_constant']<=.9 for r in rows),risk=all(r['models']['additive']['top10_predicted_risk_error_share']>=.3 for r in rows)),seconds=time.monotonic()-start,scope='Calibration-only residual and risk tables; opened evaluation; descriptive not causal; no candidate export or evaluation refit.')
    (P/'RESIDUAL_TOKEN_POSITION_V1.json').write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':main()
