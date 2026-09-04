"""Tests for ops/repo_orphans.py.

The first sweep archived 67 files it should not have, because `corpus_files()` omitted
`polynomial_causal/*.py` -- so an artefact referenced only by a script in that directory looked
unreferenced. These tests pin the corpus's completeness, because a scan is only as honest as the
corpus it compares against and the failure mode is silent: you get a plausible list of "dead" files.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import repo_orphans as RO


def _dirs_represented():
    return {os.path.dirname(p) for p in RO.corpus_files()}


def test_corpus_includes_scripts_from_both_script_directories():
    """The regression: polynomial_causal's own .py files must be able to vouch for an artefact."""
    dirs = _dirs_represented()
    assert RO.BQ in dirs, 'bilinear_quotient root scripts missing from the corpus'
    assert RO.PC in dirs, 'polynomial_causal scripts missing from the corpus -- this caused 67 bad archives'
    assert os.path.join(RO.BQ, 'ops') in dirs, 'ops/ missing from the corpus'


def test_corpus_includes_python_from_polynomial_causal():
    py = [p for p in RO.corpus_files() if p.startswith(RO.PC) and p.endswith('.py')]
    assert len(py) > 100, f'expected polynomial_causal scripts in the corpus, found {len(py)}'


def test_corpus_includes_the_archive_so_archiving_cannot_cascade():
    arch = os.path.join(RO.BQ, 'archive')
    if not os.path.isdir(arch):
        return
    assert any(p.startswith(arch) for p in RO.corpus_files()), \
        'archive/ missing from the corpus: archiving would cascade until the root is empty'


def test_scan_partitions_orphans_into_the_two_tiers():
    cands, mentions, orphans, dead, ran_only = RO.scan()
    assert set(dead) | set(ran_only) == set(orphans)
    assert not (set(dead) & set(ran_only)), 'a file cannot be both never-run and run'
    assert set(orphans) <= set(cands)


def test_the_root_is_currently_a_fixed_point():
    """After a sweep the scan must converge: no orphans, or the last sweep was incomplete."""
    _, _, orphans, _, _ = RO.scan()
    assert not orphans, f'{len(orphans)} orphans at the root; run ops/archive_orphans.py'


def test_curated_documents_are_never_candidates():
    cands, _, _, _, _ = RO.scan()
    for keep in ('BILIN18_CONNECTION.md', 'queue.txt', 'BENCHMARK_BACKLOG.md'):
        assert keep not in cands, f'{keep} must never be archivable'
