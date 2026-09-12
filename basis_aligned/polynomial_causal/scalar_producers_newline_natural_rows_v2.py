"""Ground-truth-token selection only; preserve authored panel for factorial audit."""
from pathlib import Path
import torch,json,hashlib,random
P=Path(__file__).resolve().parent
SOURCE=P.parent/'bilinear_quotient/.rowcache/fineweb_n192_skip11000.pt'
def main():
 expected='b1564bfd071418f401a816cb01e3d26b082a3e73ba858838f1c83c250db4d868';assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==expected
 data=torch.load(SOURCE,weights_only=True,map_location='cpu');assert data.shape==(192,513)
 eligible=[i for i in range(24,192) if bool((data[i,64:257]==198).any())];random.Random(9231726).shuffle(eligible);chosen=eligible[:32];assert len(chosen)==32
 authored=json.loads((P/'SCALAR_PRODUCERS_NEWLINE_V1_ROWS.json').read_text())['rows'];rows=[dict(r,pool='authored') for r in authored]
 for ordinal,i in enumerate(chosen):
  pos=64+int(torch.nonzero(data[i,64:257]==198)[0,0]);ids=data[i,:pos].tolist();assert data[i,pos]==198 and 64<=len(ids)<=256
  rows.append(dict(row_id=len(rows),pool='fineweb',family=ordinal//16,source_row=i,target_position=pos,ids=ids,newline_id=198,comma_id=11))
 out=P/'SCALAR_PRODUCERS_NEWLINE_NATURAL_V2_ROWS.json';assert not out.exists();out.write_text(json.dumps(dict(rows=rows,mean_token_rows=data[:24,:256].tolist(),source=str(SOURCE),source_sha=expected,seed=9231726,eligible_rows=len(eligible),scope='32ground-truth-nextNL FineWeb prefixes plus16unchanged authored controls. Disjoint cache row indices from24mean rows; document independence not established. No model outcomes used.'),indent=2)+'\n')
 print(dict(natural_rows=32,authored_rows=16,eligible=len(eligible),natural_min_tokens=min(len(r['ids']) for r in rows[16:]),natural_max_tokens=max(len(r['ids']) for r in rows[16:])))
if __name__=='__main__':main()
