import openreview, json

client = openreview.api.OpenReviewClient(baseurl='https://api2.openreview.net')

venue_id = 'ICML.cc/2026/Workshop/Mech_Interp'
notes = client.get_all_notes(content={'venueid': venue_id})

papers = [{
    'title': n.content.get('title', {}).get('value'),
    'authors': n.content.get('authors', {}).get('value'),
    'url': f'https://openreview.net/forum?id={n.forum}',
} for n in notes]

json.dump(papers, open('icml2026_mechinterp.json', 'w'), indent=2)
print(f'{len(papers)} papers')