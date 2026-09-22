"""CPU replay of hybrid exports, independent of the runner's dense readout assembly."""
import argparse,hashlib,json,time
import torch
from audit_conditional_residual_accounting import P,SCALE,load

def evaluate(artifact,x):
    # Evaluate each output's eight four-linear-form products directly.
    coeff=artifact['coefficients']/SCALE
    assert coeff.shape==(12,8) and artifact['output_start']==4
    terms=torch.ones(len(x),12,8,dtype=x.dtype)
    for factor in artifact['factors']:
        terms*=torch.einsum('nd,gkd->ngk',x,factor.reshape(12,8,x.shape[1]))
    return (terms*coeff).sum(-1)

def metrics(pred,y,pairs):
    r,d=pairs
    val=((pred-y).square().sum(0)/y.square().sum(0)).sqrt()
    change=pred[d]-pred[r];ref=y[d]-y[r]
    response=((change-ref).square().sum(0)/ref.square().sum(0)).sqrt()
    return dict(small_value_rms=float(val[4:].square().mean().sqrt()),small_response_rms=float(response[4:].square().mean().sqrt()),feature_value_errors=val.tolist(),feature_response_errors=response.tolist())

def check():
    torch.manual_seed(57001);x=torch.randn(11,7,dtype=torch.float64)
    a=dict(factors=[torch.randn(96,7,dtype=torch.float64) for _ in range(4)],coefficients=torch.randn(12,8,dtype=torch.float64),output_start=4)
    expected=torch.zeros(11,12,dtype=torch.float64)
    for g in range(12):
        for k in range(8):
            v=torch.ones(11,dtype=torch.float64)
            for f in a['factors']:v*=x@f[g*8+k]
            expected[:,g]+=v*a['coefficients'][g,k]/SCALE
    error=float((evaluate(a,x)-expected).norm()/expected.norm());assert error<1e-12
    return error

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true');args=parser.parse_args()
    torch.set_num_threads(2);torch.set_grad_enabled(False);start=time.monotonic();control=check()
    if args.check:print('Independent per-output export control PASS',control);return
    source=P/'HYBRID_LOCAL_QUARTIC_NATIVE_V1.json'
    if not source.exists():raise SystemExit('Native hybrid receipt pending; no audit result written.')
    receipt=json.loads(source.read_text());assert {r['seed'] for r in receipt['rows']}=={25001,25002} and len(receipt['rows'])==2
    data=torch.load(P/'RESIDUAL_FRESH_STATES_V1.pt',weights_only=True);x=data['rows'].double();y=data['target'].double()/SCALE
    pairs=torch.tensor(json.loads((P/'ALL_FEATURE_MATCHED_RESPONSES_V1.json').read_text())['pairs']).T
    parent,ph=load('MIXED_CP_FEATURES_SEED1001_V1.pt')
    base=torch.cat([torch.stack([xx@f.T for f in parent['factors']]).prod(0)@parent['coefficients'].T/SCALE for xx in x.split(1024)])
    rows=[]
    for record in receipt['rows']:
        seed=record['seed'];a,sha=load(f'HYBRID_LOCAL_QUARTIC_ADAM_SEED{seed}_V1.pt');assert sha==record['sha256'] and a['parent_sha256']==ph
        pred=base.clone();pred[:,4:]+=torch.cat([evaluate(a,xx) for xx in x.split(1024)])
        assert torch.equal(pred[:,:4],base[:,:4]);score=metrics(pred,y,pairs)
        replay=max(abs(v-record['fresh_metrics'][k]) for k,v in score.items() if isinstance(v,float))
        for k in ['feature_value_errors','feature_response_errors']:
            replay=max(replay,max(abs(a-b) for a,b in zip(score[k],record['fresh_metrics'][k])))
        assert replay<1e-6,replay
        rows.append(dict(seed=seed,sha256=sha,replay_error=replay,metrics=score,protected_outputs_equal=True))
    result=dict(rows=rows,synthetic_control_error=control,seconds=time.monotonic()-start,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),scope='CPUfloat64 replay of exported candidate values and matched differences on opened panel. Gaussian contractions not independently repeated. No refitting, OOD or causal claim.')
    (P/'HYBRID_EXPORT_AUDIT_V1.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
