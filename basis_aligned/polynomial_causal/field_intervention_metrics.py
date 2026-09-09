"""Shared field-intervention panels; separate mechanics from scientific gates."""
import torch


def panel(logits,arm,metadata,selected,target='answer'):
    selected=torch.as_tensor(selected,dtype=torch.bool)
    if not bool(selected.any()):raise ValueError('empty registered panel')
    answer=torch.tensor([m['answer'] for m in metadata]);desired=torch.tensor([m[target] for m in metadata])
    native=logits['native'][selected];actual=logits[arm][selected];answer=answer[selected];desired=desired[selected]
    p=native.softmax(-1);q=actual.softmax(-1);take=lambda a,index:a.gather(-1,index[:,None]).squeeze(-1)
    effect=actual-native;effect=effect-effect.mean(-1,keepdim=True)
    gold_change=float((take(q,answer)-take(p,answer)).mean())
    return {'n':len(native),'native_accuracy':float((native.argmax(-1)==answer).double().mean()),
            'target_accuracy':float((actual.argmax(-1)==desired).double().mean()),
            'target_probability_gain':float((take(q,desired)-take(p,desired)).mean()),
            'gold_probability_change':gold_change,'removal_gold_loss':-gold_change,
            'centered_effect_rms':float(effect.square().mean().sqrt())}


def correspondence(actual,expected):
    absolute=float((actual-expected).abs().max())
    relative=float((actual-expected).square().mean().sqrt())/max(float(expected.square().mean().sqrt()),1e-6)
    finite=bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
    return {'max_abs':absolute,'relative_rms':relative,'finite':finite,'passed':finite and absolute<=1e-9 and relative<=1e-10}


def controls():
    p=torch.tensor([[.8,.1,.1],[.1,.8,.1]],dtype=torch.float64)
    q=torch.tensor([[.1,.8,.1],[.1,.8,.1]],dtype=torch.float64)
    logits={'native':p.log(),'edit':q.log()};meta=[{'answer':0,'desired':1},{'answer':1,'desired':1}]
    target=panel(logits,'edit',meta,[True,False],'desired');control=panel(logits,'edit',meta,[False,True],'desired')
    checks={'signed_gain_and_damage':abs(target['target_probability_gain']-.7)<1e-12 and abs(target['removal_gold_loss']-.7)<1e-12,
            'separate_control_panel':control['gold_probability_change']==0 and control['target_accuracy']==1}
    shifted=panel({k:v+100 for k,v in logits.items()},'edit',meta,[True,False],'desired')
    checks['logit_offset_invariance']=abs(shifted['target_probability_gain']-target['target_probability_gain'])<1e-12
    try:panel(logits,'edit',meta,[False,False],'desired');checks['empty_panel_rejected']=False
    except ValueError:checks['empty_panel_rejected']=True
    checks['mechanical_only_gate']=correspondence(p,p)['passed'] and not correspondence(p,q)['passed']
    return {'passed':all(checks.values()),'checks':checks}
