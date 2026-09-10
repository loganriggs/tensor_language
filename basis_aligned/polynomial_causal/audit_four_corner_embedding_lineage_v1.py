"""Native transport coefficients and a token-level disjoint-edit certificate.

Reads five small lambda tensors through mmap; no trained forward or GPU call.
The embedding identity is universal for per-token lookup maps, not a sample
activation-rank or final-output claim.
"""
import hashlib,json
from pathlib import Path
from types import SimpleNamespace
import torch
import four_corner_residual_lineage as L
P=Path(__file__).resolve().parent
CHECKPOINT=Path('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin')
SQUARES=P/'SEMANTIC_SQUARE_ROWS_V1_AUDIT.json'
OUT=P/'FOUR_CORNER_EMBEDDING_LINEAGE_V1_AUDIT.json'


def main():
    state=torch.load(CHECKPOINT,map_location='cpu',weights_only=True,mmap=True)
    lambdas=[state[f'transformer.h.{i}.lambdas'] for i in range(5)]
    proxy=SimpleNamespace(transformer=SimpleNamespace(h=[SimpleNamespace(lambdas=x) for x in lambdas]))
    coefficients=L.coefficients(proxy)
    data=json.loads(SQUARES.read_text());panels={};positions=0
    for panel,entry in data['panels'].items():
        checked=0
        for s in entry['squares']:
            t00=s['context0']['tokens0'];t01=s['context0']['tokens1'];t10=s['context1']['tokens0'];t11=s['context1']['tokens1']
            assert len(t00)==len(t01)==len(t10)==len(t11)
            for a,b,c,d in zip(t00,t01,t10,t11):
                assert sorted((a,d))==sorted((b,c));checked+=1
        positions+=checked;panels[panel]={'squares':entry['square_count'],'token_positions_certified':checked}
    result={'passed':True,'native_lambdas':[x.tolist() for x in lambdas],'raw_pre_mlp4_coefficients':coefficients,
            'panels':panels,'total_token_positions_certified':positions,'trained_forwards':0,'gpu_accessed':False,
            'square_manifest_sha256':hashlib.sha256(SQUARES.read_bytes()).hexdigest(),
            'lineage_source_sha256':hashlib.sha256(Path(L.__file__).read_bytes()).hexdigest(),
            'lambda_tensor_sha256':[hashlib.sha256(x.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest() for x in lambdas],
            'checkpoint':str(CHECKPOINT),
            'claim':'At every certified position the two diagonal token multisets match; any per-token lookup, including embedding RMS, has zero mixed difference in real arithmetic. Paired equal-edge subtraction is bitwise zero for identical native values.',
            'limits':'Transport coefficients are not importance. This does not cancel contextual attention/MLP writes or final-input normalization, and does not remove embedding dependence of other terms.'}
    with OUT.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))


if __name__=='__main__':main()
