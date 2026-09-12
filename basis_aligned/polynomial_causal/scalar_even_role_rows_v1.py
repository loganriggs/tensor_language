"""Independent writer/reader geography factorial, fixed without model scoring."""
from pathlib import Path
import tiktoken,json
from scalar_producers_context_transfer_rows_v1 import PAIRS
P=Path(__file__).resolve().parent

def main():
 enc=tiktoken.get_encoding('gpt2');rows=[]
 for order in range(2):
  for citypair,cities in enumerate((('Glasgow','Phoenix'),('Cambridge','Detroit'))):
   for concept,(uk,us,stem) in enumerate(PAIRS):
    for writer in range(2):
     for reader in range(2):
      clauses=[f'The writer lives in {cities[writer]}.',f'The reader lives in {cities[reader]}.']
      if order:clauses.reverse()
      text=' '.join(clauses)+f' Using the writer\'s local spelling, the message begins: "{stem}'
      ue,se=enc.encode(uk),enc.encode(us);assert len(ue)==len(se)==1
      rows.append(dict(row_id=len(rows),family=order,city_pair=citypair,concept=concept,writer=writer,reader=reader,writer_city=cities[writer],reader_city=cities[reader],text=text,ids=enc.encode(text),uk_id=ue[0],us_id=se[0],control_ids=[670,3946],writer_donor=len(rows)^2,reader_donor=len(rows)^1))
 for row in rows:
  for key in ('writer_donor','reader_donor'):
   donor=rows[row[key]];assert len(row['ids'])==len(donor['ids']);assert sum(a!=b for a,b in zip(row['ids'],donor['ids']))==1
   assert (row['reader']==donor['reader']) if key=='writer_donor' else (row['writer']==donor['writer'])
 prior=json.loads((P/'SCALAR_JOINT_KEY_CONFIRMATION_V1_ROWS.json').read_text())['regional'];overlap=sum(tuple(r['ids']) in {tuple(x['ids']) for x in prior} for r in rows)
 out=P/'SCALAR_EVEN_ROLE_V1_ROWS.json';assert not out.exists();out.write_text(json.dumps(dict(rows=rows,old_confirmation_exact_prompt_overlap=overlap,scope='96fixed writer/reader factorial prompts, two mentionorders; samecities/endpoints/instruction. Some earlier opposedcityprompts intentionally reused; independentportchanges new. No modelscorefilter, fit, or newcorpusOOD.'),indent=2)+'\n');print(dict(rows=len(rows),old_exact_prompt_overlap=overlap,min_tokens=min(len(r['ids']) for r in rows),max_tokens=max(len(r['ids']) for r in rows)))
if __name__=='__main__':main()
