"""Frozen four-domain validation rows for existing weight-only parent1 branches."""
import collections
import hashlib
import json
from pathlib import Path

import numpy as np
import tiktoken
import torch
from datasets import Dataset

DOMAINS = {'reference': ['Wikipedia (en)'], 'discussion': ['StackExchange'],
           'biomedical': ['PubMed Abstracts', 'PubMed Central'],
           'legal_patent': ['FreeLaw', 'USPTO Backgrounds']}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for data in iter(lambda: handle.read(1048576), b''):
            h.update(data)
    return h.hexdigest()


def main():
    torch.set_num_threads(2)
    p = Path(__file__).parent
    root = p.parents[1]
    stem = 'SHARED_NODE_PARENT1_CORPUS_SHIFT_V1'
    receipt_path = p/(stem+'_ROWS.json')
    assert not receipt_path.exists()
    old = json.loads((p/'MILLION_TOKEN_PANEL_V1_ROWS.json').read_text())
    source = Path(old['source'])
    assert digest(source)==old['source_sha256']
    dataset = Dataset.from_file(str(source))
    encoder = tiktoken.get_encoding('gpt2')
    forbidden_docs = {row['document_sha256'] for row in old['documents']}
    forbidden_prefixes = {row['prefix_sha256'] for row in old['documents']}
    forbidden_windows = set()
    for name in ('FINEWEB', 'SUPPRESSION'):
        panel = torch.load(p/f'SHARED_NODE_PARENT1_{name}_V1_ROWS.pt', weights_only=True, map_location='cpu')
        for row in panel['rows']:
            forbidden_windows.add(hashlib.sha256(row.numpy().tobytes()).hexdigest())
    families = panel['families']
    domain_of = {subset: domain for domain, subsets in DOMAINS.items() for subset in subsets}
    count = collections.Counter()
    excluded = collections.Counter()
    seen_docs, seen_prefixes, selected = set(), set(), []
    rng = np.random.default_rng(5401)
    visited = 0
    for index in rng.permutation(len(dataset)):
        visited += 1
        record = dataset[int(index)]
        subset = record['meta']['pile_set_name']
        domain = domain_of.get(subset)
        if domain is None or count[domain]>=32:
            continue
        text = record['text']
        document_hash = hashlib.sha256(text.encode()).hexdigest()
        if document_hash in forbidden_docs or document_hash in seen_docs:
            excluded['old_or_duplicate_document'] += 1
            continue
        ids = encoder.encode_ordinary(text)[:513]
        prefix_hash = hashlib.sha256(np.asarray(ids, dtype=np.int32).tobytes()).hexdigest()
        if prefix_hash in forbidden_prefixes or prefix_hash in seen_prefixes:
            excluded['old_or_duplicate_prefix'] += 1
            continue
        positions = [[i for i in range(128, len(ids)) if ids[i] in family] for family in families]
        controls = [i for i in range(128, len(ids)) if ids[i] not in families[0] and ids[i] not in families[1]]
        if not all(positions) or not controls:
            excluded['target_coverage'] += 1
            continue
        first = positions[0][0]
        targets = [first, min(positions[1], key=lambda i: (abs(i-first), i)),
                   min(controls, key=lambda i: (abs(i-first), i))]
        windows = [torch.tensor(ids[i-128:i+1], dtype=torch.long) for i in targets]
        hashes = [hashlib.sha256(row.numpy().tobytes()).hexdigest() for row in windows]
        if any(value in forbidden_windows for value in hashes):
            excluded['duplicate_endpoint'] += 1
            continue
        count[domain] += 1
        seen_docs.add(document_hash)
        seen_prefixes.add(prefix_hash)
        forbidden_windows.update(hashes)
        selected.append(dict(source_document_row=int(index), document_sha256=document_hash,
                             prefix_sha256=prefix_hash, source_length=len(ids), subset=subset,
                             domain=domain, ids=ids, positions=targets))
        if len(selected)==128:
            break
    summary = dict(domains=DOMAINS, selected_per_domain=dict(count), visited_source_rows=visited,
                   exclusions=dict(excluded), source=str(source), source_sha256=digest(source),
                   old_document_hashes_excluded=len(forbidden_docs), seed=5401,
                   coverage_held=all(count[domain]==32 for domain in DOMAINS),
                   scope='Distinct exact-text/hash documents outside the old2048document panel; fixed four-domain corpus shift from FineWeb. Not verified pretraining-disjoint, historically untouched, or near-duplicate-free.')
    if not summary['coverage_held']:
        receipt_path.write_text(json.dumps(summary, indent=2)+'\n')
        print(json.dumps(summary, indent=2))
        return
    selected = [selected[i] for i in rng.permutation(len(selected))]
    prefixes = torch.full((128, 513), 50256, dtype=torch.long)
    for i, item in enumerate(selected):
        prefixes[i, :item['source_length']] = torch.tensor(item['ids'])
    prefix_path = p/(stem+'_PREFIXES.pt')
    torch.save(prefixes, prefix_path)
    relative_source = str(prefix_path.relative_to(root))
    rows, metadata = [], []
    for family in range(3):
        for i, item in enumerate(selected):
            pos = item['positions'][family]
            rows.append(prefixes[i, pos-128:pos+1].clone())
            metadata.append(dict(document=item['source_document_row'], source=relative_source,
                                 source_row=i, family=family, target_position=pos,
                                 target_id=int(prefixes[i, pos]), source_length=item['source_length'],
                                 domain=item['domain'], subset=item['subset'],
                                 document_sha256=item['document_sha256'], prefix_sha256=item['prefix_sha256']))
    panel_path = p/(stem+'_ROWS.pt')
    torch.save(dict(rows=torch.stack(rows), metadata=metadata, families=families,
                    documents=[item['source_document_row'] for item in selected],
                    domain_labels=[item['domain'] for item in selected]), panel_path)
    summary.update(metadata=metadata, row_sources={relative_source:digest(prefix_path)},
                   artifact_sha256=digest(panel_path), builder_sha256=digest(__file__),
                   source_subset_counts=dict(collections.Counter(item['subset'] for item in selected)),
                   selected_length_range=[min(item['source_length'] for item in selected),max(item['source_length'] for item in selected)])
    receipt_path.write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps({key:value for key,value in summary.items() if key!='metadata'}, indent=2))


if __name__=='__main__':
    main()
