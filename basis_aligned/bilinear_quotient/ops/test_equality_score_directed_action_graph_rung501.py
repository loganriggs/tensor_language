import copy
import sys
from pathlib import Path


OPS = Path(__file__).resolve().parent
if str(OPS) not in sys.path:
    sys.path.insert(0, str(OPS))

import equality_score_directed_action_graph_rung501 as rung


def _reader(cosine=.9, residual=.3):
    return {
        "cosine": cosine, "positive_fit_scale": 1.0, "scaled_residual": residual,
        "native_response_rms_over_native_write_rms": .1,
        "hybrid_response_rms_over_native_write_rms": .1, "tokens": 100,
    }


def _arm(score=True):
    copy_row = _reader(.9 if score else .2, .3 if score else .98)
    return {
        "equality_recovery": 1.0,
        "task_effect": {"cosine": .9, "positive_fit_scale": 1.0, "scaled_residual": .3},
        "off_target_signed_mean_hybrid_minus_native_nat": 0.0,
        "reader": {
            "copy_positive": copy_row,
            "noncopy_equality": _reader(.5, .85),
            "all_noncopy": _reader(.4, .9),
        },
    }


def _synthetic_partition():
    analysis = {"p": {}}
    for background in rung.BACKGROUNDS:
        analysis["p"][background] = {
            "score_donor": [copy.deepcopy(_arm(True)), copy.deepcopy(_arm(True))],
            "payload_donor": [copy.deepcopy(_arm(False)), copy.deepcopy(_arm(False))],
        }
    cross = {"p": [{cell: {"cosine": .9, "tokens": 100}
                    for cell in rung.CELLS} for _ in range(2)]}
    return analysis, cross


def test_exact_reader_report_and_price():
    stats = rung._empty_reader_stats()
    index = (0, 0, 0, 0, 0)
    stats["ref2"][index] = 9
    stats["hyb2"][index] = 36
    stats["cross"][index] = 18
    stats["write2"][index] = 100
    stats["tokens"][index] = 4
    report = rung._reader_report(stats, index)
    assert report["cosine"] == 1
    assert report["positive_fit_scale"] == .5
    assert report["scaled_residual"] == 0
    assert rung.FORWARDS_PER_BATCH == 65
    assert rung.DISCOVERY_FORWARDS == 8125


def test_partition_edge_requires_payload_rejection():
    analysis, background = _synthetic_partition()
    assert rung._partition_edge(analysis, background, "p", 0)["edge"]
    analysis["p"]["early_present"]["payload_donor"][0]["reader"]["copy_positive"] = (
        copy.deepcopy(
            analysis["p"]["early_present"]["score_donor"][0]["reader"]["copy_positive"]))
    assert not rung._partition_edge(analysis, background, "p", 0)["edge"]


def test_same_layer_candidates_are_bidirectional_but_others_are_causal_ordered():
    assert (2, 3) in rung.PAIRS and (3, 2) in rung.PAIRS
    assert (1, 0) not in rung.PAIRS and (2, 0) not in rung.PAIRS
    assert len(rung.PAIRS) == len(set(rung.PAIRS)) == 7
