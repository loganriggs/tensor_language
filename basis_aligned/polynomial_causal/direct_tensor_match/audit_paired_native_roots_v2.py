import json,time
from pathlib import Path
import torch
from paired_root_compiler import compile_program,evaluate,cast,price
from shared_root_block_compiler import evaluate as original_evaluate
P=Path(__file__).resolve().parent

def main():
    torch.set_num_threads(2);torch.set_grad_enabled(False);start=time.monotonic();xs=[p['rows'].float() for p in torch.load(P/'NATIVE_QUARTIC_COVARIANCE_V1.pt',weights_only=True)['panels']];metric=torch.load(P/'MIDPOINT_CONTINUATION_NATIVE_OPERATOR_V1.pt',weights_only=True)['R_U'].double();records=[]
    for rank in [4,8,16,32]:
        source=torch.load(P/f'QUARTIC_RANK_BLOCK_{rank}_V1.pt',weights_only=True);source64={k:v.double() for k,v in source.items()};program,diagnostics=compile_program(source64,allow_rotations=True);archive=cast(program,torch.float32);errors=[];double_errors=[]
        for x in xs:
            reference=original_evaluate(source64,x.double())@metric.T;actual=evaluate(archive,x).double()@metric.T;high=evaluate(program,x.double())@metric.T;errors.append(float((actual-reference).norm()/reference.norm()));double_errors.append(float((high-reference).norm()/reference.norm()))
        cost=price(archive);path=P/f'PAIRED_ROOT_RANK_{rank}_V2.pt';torch.save(archive,path);row=dict(rank=rank,float32_replay=errors,float64_replay=double_errors,price=cost,pairs=diagnostics);records.append(row);print(json.dumps({k:v for k,v in row.items() if k!='pairs'}),flush=True)
    primary=next(r for r in records if r['rank']==16);pred=dict(replay=max(primary['float32_replay'])<1e-4,price=primary['price']['products']<=384 and primary['price']['stored_coefficients']<=330240)
    result=dict(records=records,predictions=pred,seconds=time.monotonic()-start,scope='Exact pair rewrite with fixed alternative output-pair bases, unchanged matrix replay tolerance; old naturalinputs and common unembeddingGram frame for replay. No freshnativebehavior, semantics oradoption. Singular/illconditioned pencils retain spectralfallback.')
    (P/'PAIRED_ROOT_NATIVE_V2.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(predictions=pred,seconds=result['seconds'])))
if __name__=='__main__':main()
