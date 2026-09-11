"""Capacity control for completed shared dictionaries, not new discovery.

A exact conditional/CP/executor/price<=1e-8; B both learned programs beat the
best native control by .01; C energy selection beats random by .01.
2645 native products use fewer floats and no sparse feature-graph indices.
"""
import hashlib,json,time
from pathlib import Path
import torch
from native_support_exchange_v1_audit import P,CK
from conditional_writer_spectral_v1 import solve
from structured_branch_amplitudes_v1 import inner


def main():
    torch.set_grad_enabled(False);torch.set_default_dtype(torch.float64);torch.set_num_threads(2)
    out=P/'MATCHED_PRICE_NATIVE_V1_AUDIT.json';assert not out.exists();start=time.perf_counter()
    parent=json.loads((P/'FULL_SUPPORT_EXCHANGE_V1_RESULT.json').read_text())
    learned=[r for r in parent['arms'] if r['mode']=='exchange'];assert len(learned)==2
    budget=min(r['price']['float_coefficients']-1152 for r in learned);count=budget//(3*1152);assert count==2645
    sd=torch.load(CK,weights_only=True,map_location='cpu',mmap=True)
    l,r,d=[sd[f'transformer.h.17.mlp.{k}.weight'].double() for k in ('Left','Right','Down')]
    metric=torch.load('/dev/shm/bilin18_native_product_energy_v1.pt',weights_only=True,map_location='cpu')['unembedding_gram'].double()
    wh=torch.linalg.cholesky(metric).T;w=wh@d;native=(l,r,w)
    total=json.loads((P/'NATIVE_READER_METRIC_V1_AUDIT.json').read_text())['native_total']
    energy=w.square().sum(0)*.5*(l.square().sum(1)*r.square().sum(1)+(l*r).sum(1).square())
    selections={'energy':energy.argsort(descending=True)[:count],
                'random':torch.randperm(4608,generator=torch.Generator().manual_seed(418))[:count]}
    rows=[];artifacts={}
    for label,ids in selections.items():
        a,b=l[ids],r[ids];old=d[:,ids];before=(a,b,wh@old)
        before_loss=float((inner(before,before)-2*inner(native,before)+total)/total)
        tic=time.perf_counter();writer,cross,gram,diagnostics=solve(d,l,r,a,b)
        weighted=wh@writer;proposal=(a,b,weighted)
        fitted=((weighted.T@weighted)*gram).sum();overlap=((w.T@weighted)*cross).sum()
        loss=float((fitted-2*overlap+total)/total)
        cp_loss=float((inner(proposal,proposal)-2*inner(native,proposal)+total)/total)
        # Replay grouped selected-native evaluation against materialized factors.
        x=torch.randn(8,1152,generator=torch.Generator().manual_seed(779))
        direct=((x@a.T)*(x@b.T))@writer.T
        original_features=((x@l.T)*(x@r.T))[:,ids]@writer.T
        execution=float((direct-original_features).norm()/direct.norm())
        row=dict(selection=label,products=count,retained_output_capture=1-before_loss,
            refitted_output_capture=1-loss,output_refit_gain=before_loss-loss,cp_replay=abs(cp_loss-loss),
            executor_replay=execution,solver=diagnostics,solve_and_score_seconds=time.perf_counter()-tic,
            matrix_float_coefficients=3*1152*count,bias_floats=1152,graph_indices=0)
        rows.append(row);artifacts[label]=dict(left=a,right=b,down=writer,bias=sd['transformer.h.17.mlp.Down_bias'].double(),native_indices=ids)
        print(json.dumps(row),flush=True)
    best=max(r['refitted_output_capture'] for r in rows)
    pred=dict(pred_a_instrument=all(max(r['cp_replay'],r['executor_replay'],r['solver']['normal_residual'])<=1e-8 and
        r['output_refit_gain']>=-1e-10 and r['matrix_float_coefficients']<=budget for r in rows),
        pred_b_learned_advantage=all(r['final_capture']>=best+.01 for r in learned),
        pred_c_selection_advantage=rows[0]['refitted_output_capture']>=rows[1]['refitted_output_capture']+.01)
    cache=Path('/dev/shm/bilin18_matched_price_native_v1.pt');assert not cache.exists();torch.save(artifacts,cache)
    result=dict(predictions=pred,rows=rows,learned_captures=[r['final_capture'] for r in learned],
        learned_minus_best_native=[r['final_capture']-best for r in learned],learned_matrix_float_budget=budget,
        learned_stored_graph_indices=1179648,cache=dict(path=str(cache),sha256=hashlib.sha256(cache.read_bytes()).hexdigest()),
        seconds=time.perf_counter()-start,body_forwards=0,
        scope='Conservative matched-float native selection control with exact outputs. No global support optimum, behavioral comparison or stable-circuit claim.')
    with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))


if __name__=='__main__':main()
