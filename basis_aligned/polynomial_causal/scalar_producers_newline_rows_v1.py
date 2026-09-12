"""Fixed newline-service capability panel, independent of model outcomes."""
from pathlib import Path
import json,tiktoken
P=Path(__file__).resolve().parent
LISTS=[
'Shopping list:\n- bread\n- milk\n- eggs',
'Things to pack:\n1. passport\n2. tickets\n3. charger',
'Today\'s tasks:\n- read the notes\n- reply to email\n- wash the dishes',
'Inventory:\nItem 1: pens\nItem 2: paper\nItem 3: envelopes',
'Ingredients:\n2 cups flour\n1 cup milk\n2 eggs',
'Meeting agenda:\n1. Introductions\n2. Budget review\n3. Questions',
'Cities on the itinerary:\nParis\nBerlin\nRome',
'Weekly schedule:\nMonday: reading\nTuesday: writing\nWednesday: revision']
FORMS=[
'Contact details:\nName: Alice\nCity: Boston\nCountry: USA',
'Application form:\nFirst name: David\nLast name: Miller\nOccupation: Teacher',
'Book record:\nTitle: The River\nAuthor: Jane Smith\nYear: 2019',
'Order details:\nProduct: Notebook\nQuantity: 12\nStatus: Shipped',
'Event information:\nDate: Friday\nTime: 14:00\nLocation: Main Hall',
'Employee record:\nDepartment: Sales\nPosition: Manager\nOffice: London',
'Travel booking:\nDestination: Madrid\nDeparture: Monday\nPassengers: 2',
'Project summary:\nOwner: Engineering\nPriority: High\nStatus: Active']
def main():
 enc=tiktoken.get_encoding('gpt2');assert enc.encode('\n')==[198] and enc.encode(',')==[11]
 rows=[dict(row_id=8*family+i,family=family,text=text,ids=enc.encode(text),newline_id=198,comma_id=11) for family,texts in enumerate((LISTS,FORMS)) for i,text in enumerate(texts)]
 assert len(rows)==16 and len(set(r['text'] for r in rows))==16
 assert all('\n' in r['text'] and not r['text'].endswith('\n') for r in rows)
 out=P/'SCALAR_PRODUCERS_NEWLINE_V1_ROWS.json';assert not out.exists();out.write_text(json.dumps(dict(rows=rows,scope='Fixed8lists+8forms. Tokenization/structure checked; no native capability, positive control, or candidate effects evaluated.'),indent=2)+'\n');print(dict(rows=len(rows),min_tokens=min(len(r['ids']) for r in rows),max_tokens=max(len(r['ids']) for r in rows)))
if __name__=='__main__':main()
