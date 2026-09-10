"""CPU-only, deterministic document-separated Pile panel; no native model access."""
import hashlib,json
from pathlib import Path
import numpy as np
import torch
import tiktoken
from datasets import Dataset

P=Path(__file__).resolve().parent
SOURCE=Path('/workspace/.hf_home/datasets/NeelNanda___pile-10k/default/0.0.0/127bfedcd5047750df5ccf3a12979a47bfa0bafa/pile-10k-train.arrow')
def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    out=P/'MILLION_TOKEN_PANEL_V1_ROWS.pt';receipt=P/'MILLION_TOKEN_PANEL_V1_ROWS.json'
    assert not out.exists() and not receipt.exists()
    ds=Dataset.from_file(str(SOURCE));enc=tiktoken.get_encoding('gpt2');rng=np.random.default_rng(91162114)
    rows=[];docs=[];seen_doc=set();seen_prefix=set()
    for idx in rng.permutation(len(ds)):
        row=ds[int(idx)];text=row['text'];dh=hashlib.sha256(text.encode()).hexdigest()
        if dh in seen_doc:continue
        ids=enc.encode_ordinary(text)
        if len(ids)<513:continue
        prefix=np.asarray(ids[:513],dtype=np.int32);ph=hashlib.sha256(prefix.tobytes()).hexdigest()
        if ph in seen_prefix:continue
        rows.append(prefix);seen_doc.add(dh);seen_prefix.add(ph)
        docs.append(dict(source_row=int(idx),document_sha256=dh,prefix_sha256=ph,source_tokens=len(ids),subset=row['meta']['pile_set_name']))
        if len(rows)==2048:break
    assert len(rows)==2048
    positions=np.stack([np.sort(rng.choice(512,32,replace=False)) for _ in rows])
    spec=dict(rows=torch.from_numpy(np.stack(rows)).long(),positions=torch.from_numpy(positions).long(),train_rows=torch.arange(1600),validation_rows=torch.arange(1600,1824),test_rows=torch.arange(1824,2048))
    torch.save(spec,out)
    r=dict(schema='million.token.panel.rows.v1',source=str(SOURCE),source_sha256=digest(SOURCE),builder_sha256=digest(__file__),artifact_sha256=digest(out),seed=91162114,documents=docs,processed_input_tokens=2048*512,sampled_states=2048*32,split_documents=dict(train=1600,validation=224,test=224),scope='One prefix per distinct exact-text/hash document and distinct token prefix, length>=513. Long-document/prefix selection bias; near-duplicate documents not excluded. New corpus panel, not verified pretraining distribution or historically untouched corpus. No semantic labels. Test rows captured but not fitted or evaluated.')
    receipt.write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps({k:v for k,v in r.items() if k!='documents'}))

if __name__=='__main__':main()
