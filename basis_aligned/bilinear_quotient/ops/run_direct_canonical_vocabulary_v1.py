#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_frame pred_b_common pred_c_extra
"""Map canonical output modes into vocabulary coordinates; weight-only audit.
pred_a_frame: output directions orthonormal and U=QR replay relative error<1e-5.
pred_b_common: each unit mode abs cosine with all-ones vocabulary vector<.2.
pred_c_extra: each mode energy on extra GPT2 vocabulary rows>=50257 is<.01.
Null: stable modes largely express generic shifts or extra-vocabulary artifacts.
No semantic labels, no causal claim, no fitting. Model's fixed vocabulary frame
is existing shared state, not a free new circuit parameter. Four directions.
"""
import os,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'
PLAN=dict(modes=4,valid_gpt2_tokens=50257,native_forwards=0)
def main():
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print(json.dumps(PLAN));return
 import torch,tiktoken
 from disk_guard import guard_torch_save
 torch.set_num_threads(4);torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False;out=P/'CANONICAL_VOCABULARY_V1.json';assert not out.exists();start=time.perf_counter();state=torch.load('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin',weights_only=True);U=state['lm_head.weight'].cuda().float();del state;Q,R=torch.linalg.qr(U);canonical=torch.load(P/'CANONICAL_ROOT_FEATURES_V1.pt',weights_only=True)['output_directions'].cuda().float();directions=Q@canonical;gram=directions.double().T@directions.double();orthogonality=float((gram-torch.eye(4,device='cuda')).norm());torch.manual_seed(2118);probe=torch.randn(1152,4,device='cuda');replay=float((U@probe-Q@(R@probe)).norm()/(U@probe).norm());enc=tiktoken.get_encoding('gpt2');rows=[]
 def label(i):return enc.decode([i]) if i<50257 else '<extra_vocab_'+str(i)+'>'
 for k in range(4):
  d=directions[:,k].double();norm=d.square().sum();positive=torch.topk(d,12).indices.tolist();negative=torch.topk(-d,12).indices.tolist();row=dict(mode=k,common_shift_cosine=float(d.sum()/(len(d)**.5*d.norm())),extra_vocabulary_energy_fraction=float(d[50257:].square().sum()/norm),top_positive=[dict(token_id=i,token=label(i),weight=float(d[i])) for i in positive],top_negative=[dict(token_id=i,token=label(i),weight=float(d[i])) for i in negative]);rows.append(row)
 pred=dict(pred_a_frame=orthogonality<1e-5 and replay<1e-5,pred_b_common=all(abs(r['common_shift_cosine'])<.2 for r in rows),pred_c_extra=all(r['extra_vocabulary_energy_fraction']<.01 for r in rows));guard_torch_save(dict(directions=directions.cpu(),scope='Interpretation only; fixed vocabulary frame applied to canonical modes.'),str(P/'CANONICAL_VOCABULARY_V1.pt'));out.write_text(json.dumps(dict(plan=PLAN,records=rows,predictions=pred,orthogonality_error=orthogonality,qr_replay_error=replay,seconds=time.perf_counter()-start,scope='Output effects of canonical combinations, no feature semantics inferred from token rankings. Signs arbitrary. Softcap remains explicit; common shift before softcap is not automatically probability-invariant.'),indent=2,ensure_ascii=False)+'\n');print(dict(predictions=pred,common_shift_cosines=[r['common_shift_cosine'] for r in rows],extra_energy=[r['extra_vocabulary_energy_fraction'] for r in rows]),flush=True)
if __name__=='__main__':main()
