"""Red-team native group agreement: context levels versus changes.
A group write replay<=1e-9. B centered8 pairedwrite relative<=.10 everyfamily.
C first-order native2token swap predictions relative<=.10 bothseeds/everyfamily.
No fit, selection, calibration or threshold repair.
"""
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from sparse_path_program_v1 import run
from sparse_path_stability_atlas_v1 import digest
from stable_path_native_cache_v1 import relative


@torch.no_grad()
def main():
    torch.set_num_threads(2);p=Path(__file__).parent;out=p/'PATH_GROUP_CHANGE_ACCOUNTING_V1.json';assert not out.exists()
    old=json.loads((p/'PATH_OUTPUT_GROUP_NATIVE_V1.json').read_text());assert old['pred_a']
    ap=p/'PATH_OUTPUT_SCHMIDT_V1.pt';assert digest(ap)==old['source_sha256']
    groups=[g for g in torch.load(ap,weights_only=True,map_location='cpu')['groups'] if g['metric']=='centered']
    binding=json.loads((p/'NATIVE_SUFFIX_UPSTREAM_V1_BINDING.json').read_text())['files'];assert all(digest(k)==v for k,v in binding.items())
    upfile=p/'NATIVE_SUFFIX_UPSTREAM_V1_PORTS.pt';assert digest(upfile)==json.loads((p/'NATIVE_SUFFIX_UPSTREAM_V1_RESULT.json').read_text())['cache_sha256']
    up=torch.load(upfile,weights_only=True,map_location='cpu')['ports']
    cache=torch.load(p/'NATIVE_RELATION_OUTPUT_FRESH_V1_ENDPOINTS.pt',weights_only=True,map_location='cpu')['ports']
    rows=json.loads((p/'NATIVE_RELATION_OUTPUT_FRESH_V1_ROWS.json').read_text())['rows']
    state=torch.load(next(k for k in binding if k.endswith('/pytorch_model.bin')),weights_only=True,mmap=True,map_location='cpu')
    u=state['lm_head.weight'].double();pre=cache['pre'];h=(pre+cache['native_output']).double()
    epsilon=torch.finfo(torch.float32).eps;denominator=pre.double().square().mean(-1)+epsilon
    reports=[];errors=[]
    for name,k in [('centered8',8),('full96',96),('stable18',18)]:
        if name=='stable18':
            stable=p/'STABLE_PATH_BANK_V1.pt';assert digest(stable)==json.loads((p/'STABLE_PATH_BANK_V1.json').read_text())['artifact_sha256']
            programs=torch.load(stable,weights_only=True,map_location='cpu')['programs']
            writes=[run(a,up['residual'].double(),up['head_values'].double(),denominator)['write'] for a in programs]
        else:
            writes=[]
            for g in groups:
                result=run(g['program'],up['residual'].double(),up['head_values'].double(),denominator)
                w=(result['amplitudes']@g['edge_to_group'][:,:k]@g['group_writer'][:,:k].T)/denominator[:,None]
                alt=dict(g['program']);alt['physical_writer']=g['group_writer'][:,:k]@g['edge_to_group'][:,:k].T
                errors.append(relative(w,run(alt,up['residual'].double(),up['head_values'].double(),denominator)['write']))
                writes.append(w)
        centered=[w-w.mean(0,keepdim=True) for w in writes];delta=[w[1::2]-w[::2] for w in writes]
        families=[]
        for family in sorted({r['family'] for r in rows}):
            ids=torch.tensor([i for i,r in enumerate(rows) if r['family']==family]);linear_errors=[];linear=[];exact=[]
            for change in delta:
                predictions=[];truth=[]
                for i in ids.tolist():
                    x=h[2*i];dx=change[i];readers=u[[rows[i]['donor_answer_id'],rows[i]['donor_foil_id']]]
                    rms=(x.square().mean()+epsilon).sqrt();normalized=x/rms
                    logits=readers@normalized;derivative=(readers@dx)/rms-logits*(x@dx)/(len(x)*rms.square())
                    gains=1-torch.tanh(logits/30).square();jvp=gains*derivative
                    new=x+dx;newlogit=readers@(new/(new.square().mean()+epsilon).sqrt())
                    change_logits=30*(torch.tanh(newlogit/30)-torch.tanh(logits/30))
                    predictions.append(jvp[0]-jvp[1]);truth.append(change_logits[0]-change_logits[1])
                a=torch.stack(predictions);b=torch.stack(truth);linear_errors.append(relative(a,b));linear.append(a);exact.append(b)
            families.append(dict(family=family,paired_write_relative_rms=relative(delta[0][ids],delta[1][ids]),
                paired_change_norm_over_endpoint_norm=[float(d[ids].norm()/w[(2*ids[:,None]+torch.tensor([0,1])).flatten()].norm()) for d,w in zip(delta,writes)],
                first_order_prediction_relative_rms=linear_errors,first_order_replica_relative_rms=relative(*linear),
                fp64_exact_swap_replica_relative_rms=relative(*exact)))
        reports.append(dict(name=name,level_relative_rms=relative(*writes),globally_centered_relative_rms=relative(*centered),
            mean_energy_fraction=[float(len(w)*w.mean(0).square().sum()/w.square().sum()) for w in writes],families=families))
    primary=reports[0]
    result=dict(pred_a=max(errors)<=1e-9,pred_b=all(f['paired_write_relative_rms']<=.10 for f in primary['families']),
        pred_c=all(max(f['first_order_prediction_relative_rms'])<=.10 for f in primary['families']),
        maximum_replay_error=max(errors),reports=reports,source_native_receipt_sha256=digest(p/'PATH_OUTPUT_GROUP_NATIVE_V1.json'),script_sha256=digest(__file__),
        scope='Diagnostic fp64 analytical native tail with native fp32 epsilon; original fp32 behavioral verdict retained. Centering and pair differences are diagnostics, not intervention changes.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));assert result['pred_a']


if __name__=='__main__':main()
