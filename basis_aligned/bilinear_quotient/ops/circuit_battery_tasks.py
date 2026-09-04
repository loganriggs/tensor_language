"""Circuit battery task bank (CPU only, no model, no GPU).

ONE row generator for EVERY behaviour.  A behaviour is added by writing a single
`gen(rng, family, pools) -> dict` function plus its answer vocabulary; the four
counterfactual families, the three frozen splits, the construction checks, the
row hashing and the disjoint pools are produced mechanically and identically for
all tasks.  This exists so that a circuit costs a run, not a bespoke dataset.

Families (identical meaning for every task):
  A1  answer-changing: the donor perturbs the task's causal variable.
  A2  answer-changing: a SECOND, structurally different causal perturbation.
  P   answer-preserving: the donor perturbs a surface/non-causal slot only;
      base and donor have the SAME answer (active control).
  C   copy control: a prompt of the same surface form whose correct answer is a
      COPY of a visible token rather than the computed function (active control).

Splits FIT / SELECT / TEST use disjoint value pools (numeric starts partitioned
mod 3, word pools sliced) so held-out numbers are genuinely held out.
"""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from typing import Callable

import tiktoken

ENC = tiktoken.get_encoding("gpt2")
FAMILIES = ("A1", "A2", "P", "C")
SPLITS = ("FIT", "SELECT", "TEST")
SCHEMA = "circuit_battery_rows_v1"

WORDS = [
    "ember", "reef", "gulch", "marble", "cinder", "willow", "harbor", "quartz",
    "lantern", "bramble", "cobalt", "thistle", "meadow", "canyon", "ripple", "fennel",
    "amber", "juniper", "slate", "verdant", "pillar", "orchard", "tundra", "walnut",
    "beacon", "clover", "drift", "ledger", "mosaic", "nectar", "opal", "prairie",
    "quiver", "rustic", "saffron", "timber", "umber", "velvet", "whistle", "zenith",
]
WORDS += ["anchor", "basil", "copper", "granite", "hollow", "ivory", "kettle", "lemon",
          "maple", "olive", "pepper", "raven", "silver", "bronze", "moss", "nest", "oak",
          "pine", "sage", "thorn", "vine", "wheat", "daisy", "falcon", "jasmine", "kelp"]
# fillers may be multi-token (they are never answers); induction answers must be single tokens
SINGLE_WORDS = [w for w in WORDS if len(ENC.encode(" " + w)) == 1]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]
LETTERS = [chr(c) for c in range(ord("A"), ord("Z") + 1)]


def single(text: str) -> bool:
    return len(ENC.encode(text)) == 1


@dataclass
class Pools:
    """Split-disjoint value pools.  Numeric starts partition mod 3; words slice."""
    split: str

    @property
    def _k(self) -> int:
        return SPLITS.index(self.split)

    def nums(self, lo: int, hi: int) -> list[int]:
        return [n for n in range(lo, hi) if n % 3 == self._k]

    def words(self) -> list[str]:
        return WORDS[self._k::3]

    def single_words(self) -> list[str]:
        return SINGLE_WORDS[self._k::3]

    def days(self) -> list[str]:
        return DAYS

    def letters(self) -> list[str]:
        return LETTERS[self._k::3]


@dataclass
class Task:
    task_id: str
    description: str
    causal_variable: str
    answer_vocab: list[str]
    gen: Callable[[random.Random, str, Pools], dict | None]
    notes: str = ""
    # True for RETRIEVAL behaviours whose answer is a token already visible in the prompt
    # (induction): for those, "the answer is not copyable" is not an invariant of A1/A2.
    answer_visible_in_prompt: bool = False
    families: tuple[str, ...] = FAMILIES
    meta: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# generators.  Each returns {base_text, donor_text, base_answer, donor_answer,
# semantic_details}.  `family` is one of FAMILIES; returning None = skip draw.
# --------------------------------------------------------------------------- #

def _gen_numbered_list(rng, family, pools):
    ns = pools.nums(11, 90)
    n = rng.choice(ns)
    w = rng.sample(pools.words(), 4)
    base = f"{n}. {w[0]}\n{n + 1}. {w[1]}\n"
    if family == "A1":                       # shift the whole list (causal: last label)
        d = rng.choice([x for x in ns if x != n])
        return dict(base_text=base, donor_text=f"{d}. {w[0]}\n{d + 1}. {w[1]}\n",
                    base_answer=str(n + 2), donor_answer=str(d + 2),
                    semantic_details={"base_labels": [n, n + 1], "donor_labels": [d, d + 1]})
    if family == "A2":                       # third line: same start, longer list
        return dict(base_text=base, donor_text=f"{n}. {w[0]}\n{n + 1}. {w[1]}\n{n + 2}. {w[2]}\n",
                    base_answer=str(n + 2), donor_answer=str(n + 3),
                    semantic_details={"perturbation": "list_length"})
    if family == "P":                        # surface: swap the item words only
        return dict(base_text=base, donor_text=f"{n}. {w[2]}\n{n + 1}. {w[3]}\n",
                    base_answer=str(n + 2), donor_answer=str(n + 2),
                    semantic_details={"perturbation": "item_words"})
    d = rng.choice([x for x in ns if x != n])
    return dict(base_text=f"{n}. {w[0]}\n{n}. {w[1]}\n", donor_text=f"{d}. {w[0]}\n{d}. {w[1]}\n",
                base_answer=str(n), donor_answer=str(d),
                semantic_details={"control": "repeated_index_copy"})


def _gen_numeric_sequence(rng, family, pools):
    starts = pools.nums(3, 40)
    a = rng.choice(starts)
    step = rng.choice([2, 3])
    pre = rng.choice(pools.words()) + ": "   # shared surface prefix: widens the draw space
    _s = lambda s, k: " ".join(str(s + i * k) for i in range(4))
    seq = lambda s, k: pre + _s(s, k)
    if family == "A1":
        b = rng.choice([x for x in starts if x != a])
        return dict(base_text=seq(a, step), donor_text=seq(b, step),
                    base_answer=f" {a + 4 * step}", donor_answer=f" {b + 4 * step}",
                    semantic_details={"perturbation": "start", "step": step})
    if family == "A2":
        k2 = 5 if step != 5 else 2
        return dict(base_text=seq(a, step), donor_text=seq(a, k2),
                    base_answer=f" {a + 4 * step}", donor_answer=f" {a + 4 * k2}",
                    semantic_details={"perturbation": "step"})
    if family == "P":
        return dict(base_text="Sequence: " + _s(a, step), donor_text="Numbers: " + _s(a, step),
                    base_answer=f" {a + 4 * step}", donor_answer=f" {a + 4 * step}",
                    semantic_details={"perturbation": "prefix_word"})
    b = rng.choice([x for x in starts if x != a])
    return dict(base_text=seq(a, 0), donor_text=seq(b, 0),
                base_answer=f" {a}", donor_answer=f" {b}",
                semantic_details={"control": "constant_sequence_copy"})


def _gen_weekday(rng, family, pools):
    i = rng.randrange(0, 5)
    d = pools.days()
    pre = rng.choice(pools.words()) + ": "   # shared surface prefix: widens the draw space
    nxt = lambda j: f" {d[(j + 2) % 7]}"
    if family == "A1":
        j = rng.choice([x for x in range(0, 5) if x != i])
        return dict(base_text=f"{pre}{d[i]} {d[i + 1]}", donor_text=f"{pre}{d[j]} {d[j + 1]}",
                    base_answer=nxt(i), donor_answer=nxt(j),
                    semantic_details={"perturbation": "start_day"})
    if family == "A2":
        return dict(base_text=f"{pre}{d[i]} {d[i + 1]}",
                    donor_text=f"{pre}{d[i]} {d[i + 1]} {d[(i + 2) % 7]}",
                    base_answer=nxt(i), donor_answer=f" {d[(i + 3) % 7]}",
                    semantic_details={"perturbation": "sequence_length"})
    if family == "P":
        w = rng.sample(pools.words(), 2)
        return dict(base_text=f"{w[0]}: {d[i]} {d[i + 1]}", donor_text=f"{w[1]}: {d[i]} {d[i + 1]}",
                    base_answer=nxt(i), donor_answer=nxt(i),
                    semantic_details={"perturbation": "prefix_word"})
    j = rng.choice([x for x in range(0, 5) if x != i])
    return dict(base_text=f"{pre}{d[i]} {d[i]}", donor_text=f"{pre}{d[j]} {d[j]}",
                base_answer=f" {d[i]}", donor_answer=f" {d[j]}",
                semantic_details={"control": "repeated_day_copy"})


def _gen_induction(rng, family, pools):
    w = rng.sample(pools.single_words(), 6)
    ctx = lambda a, b, c, d, key: f" {a} {b} {c} {d} {key}"
    if family == "A1":                       # which token repeats -> which successor
        return dict(base_text=ctx(w[0], w[1], w[2], w[3], w[0]),
                    donor_text=ctx(w[0], w[1], w[2], w[3], w[2]),
                    base_answer=f" {w[1]}", donor_answer=f" {w[3]}",
                    semantic_details={"perturbation": "repeated_key"})
    if family == "A2":                       # same key, different successor payload
        return dict(base_text=ctx(w[0], w[1], w[2], w[3], w[0]),
                    donor_text=ctx(w[0], w[4], w[2], w[3], w[0]),
                    base_answer=f" {w[1]}", donor_answer=f" {w[4]}",
                    semantic_details={"perturbation": "payload"})
    if family == "P":                        # unrelated filler word changes
        return dict(base_text=ctx(w[0], w[1], w[2], w[3], w[0]),
                    donor_text=ctx(w[0], w[1], w[4], w[3], w[0]),
                    base_answer=f" {w[1]}", donor_answer=f" {w[1]}",
                    semantic_details={"perturbation": "filler"})
    return dict(base_text=f" {w[0]} {w[1]} {w[2]} {w[3]} {w[3]}",
                donor_text=f" {w[0]} {w[1]} {w[2]} {w[4]} {w[4]}",
                base_answer=f" {w[3]}", donor_answer=f" {w[4]}",
                semantic_details={"control": "immediate_repeat_copy"})


def _gen_addition(rng, family, pools):
    xs = pools.nums(11, 60)
    a = rng.choice(xs)
    b = rng.randrange(2, 8)
    if family == "A1":
        a2 = rng.choice([x for x in xs if x != a])
        return dict(base_text=f"{a} + {b} =", donor_text=f"{a2} + {b} =",
                    base_answer=f" {a + b}", donor_answer=f" {a2 + b}",
                    semantic_details={"perturbation": "first_operand"})
    if family == "A2":
        b2 = rng.choice([x for x in range(2, 8) if x != b])
        return dict(base_text=f"{a} + {b} =", donor_text=f"{a} + {b2} =",
                    base_answer=f" {a + b}", donor_answer=f" {a + b2}",
                    semantic_details={"perturbation": "second_operand"})
    if family == "P":
        return dict(base_text=f"{a} + {b} =", donor_text=f"{b} + {a} =",
                    base_answer=f" {a + b}", donor_answer=f" {a + b}",
                    semantic_details={"perturbation": "operand_order"})
    a2 = rng.choice([x for x in xs if x != a])
    return dict(base_text=f"{a} + 0 =", donor_text=f"{a2} + 0 =",
                base_answer=f" {a}", donor_answer=f" {a2}",
                semantic_details={"control": "add_zero_copy"})


def _gen_letter_list(rng, family, pools):
    ls = pools.letters()
    ls = [c for c in ls if ord(c) < ord("W")]
    c = rng.choice(ls)
    w = rng.sample(pools.words(), 4)
    nxt = lambda ch, k=2: chr(ord(ch) + k)
    base = f"{c}. {w[0]}\n{nxt(c, 1)}. {w[1]}\n"
    if family == "A1":
        c2 = rng.choice([x for x in ls if x != c])
        return dict(base_text=base, donor_text=f"{c2}. {w[0]}\n{nxt(c2, 1)}. {w[1]}\n",
                    base_answer=nxt(c), donor_answer=nxt(c2),
                    semantic_details={"perturbation": "start_letter"})
    if family == "A2":
        return dict(base_text=base,
                    donor_text=f"{c}. {w[0]}\n{nxt(c, 1)}. {w[1]}\n{nxt(c, 2)}. {w[2]}\n",
                    base_answer=nxt(c), donor_answer=nxt(c, 3),
                    semantic_details={"perturbation": "list_length"})
    if family == "P":
        return dict(base_text=base, donor_text=f"{c}. {w[2]}\n{nxt(c, 1)}. {w[3]}\n",
                    base_answer=nxt(c), donor_answer=nxt(c),
                    semantic_details={"perturbation": "item_words"})
    c2 = rng.choice([x for x in ls if x != c])
    return dict(base_text=f"{c}. {w[0]}\n{c}. {w[1]}\n", donor_text=f"{c2}. {w[0]}\n{c2}. {w[1]}\n",
                base_answer=c, donor_answer=c2,
                semantic_details={"control": "repeated_letter_copy"})


def _gen_countdown(rng, family, pools):
    starts = pools.nums(12, 80)
    a = rng.choice(starts)
    pre = rng.choice(pools.words()) + ": "   # shared surface prefix: widens the draw space
    _s = lambda s: " ".join(str(s - i) for i in range(4))
    seq = lambda s: pre + _s(s)
    if family == "A1":
        b = rng.choice([x for x in starts if x != a])
        return dict(base_text=seq(a), donor_text=seq(b),
                    base_answer=f" {a - 4}", donor_answer=f" {b - 4}",
                    semantic_details={"perturbation": "start"})
    if family == "A2":                       # direction flip: ascending from the same value
        return dict(base_text=seq(a), donor_text=pre + " ".join(str(a + i) for i in range(4)),
                    base_answer=f" {a - 4}", donor_answer=f" {a + 4}",
                    semantic_details={"perturbation": "direction"})
    if family == "P":
        w = rng.sample(pools.words(), 2)
        return dict(base_text=f"{w[0]}: " + _s(a), donor_text=f"{w[1]}: " + _s(a),
                    base_answer=f" {a - 4}", donor_answer=f" {a - 4}",
                    semantic_details={"perturbation": "prefix_word"})
    b = rng.choice([x for x in starts if x != a])
    return dict(base_text=pre + " ".join([str(a)] * 4), donor_text=pre + " ".join([str(b)] * 4),
                base_answer=f" {a}", donor_answer=f" {b}",
                semantic_details={"control": "constant_sequence_copy"})


def _gen_month(rng, family, pools):
    i = rng.randrange(0, 9)
    m = MONTHS
    pre = rng.choice(pools.words()) + ": "   # shared surface prefix: widens the draw space
    if family == "A1":
        j = rng.choice([x for x in range(0, 9) if x != i])
        return dict(base_text=f"{pre}{m[i]} {m[i + 1]}", donor_text=f"{pre}{m[j]} {m[j + 1]}",
                    base_answer=f" {m[i + 2]}", donor_answer=f" {m[j + 2]}",
                    semantic_details={"perturbation": "start_month"})
    if family == "A2":
        return dict(base_text=f"{pre}{m[i]} {m[i + 1]}",
                    donor_text=f"{pre}{m[i]} {m[i + 1]} {m[i + 2]}",
                    base_answer=f" {m[i + 2]}", donor_answer=f" {m[i + 3]}",
                    semantic_details={"perturbation": "sequence_length"})
    if family == "P":
        w = rng.sample(pools.words(), 2)
        return dict(base_text=f"{w[0]}: {m[i]} {m[i + 1]}", donor_text=f"{w[1]}: {m[i]} {m[i + 1]}",
                    base_answer=f" {m[i + 2]}", donor_answer=f" {m[i + 2]}",
                    semantic_details={"perturbation": "prefix_word"})
    j = rng.choice([x for x in range(0, 9) if x != i])
    return dict(base_text=f"{pre}{m[i]} {m[i]}", donor_text=f"{pre}{m[j]} {m[j]}",
                base_answer=f" {m[i]}", donor_answer=f" {m[j]}",
                semantic_details={"control": "repeated_month_copy"})




# --- bank v2 additions: behaviours selected by a measured native-capability scan --------- #

ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "IX", "X", "XI", "XII"]
COUNT_WORDS = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
               "eleven", "twelve"]
KEYS = ["Chapter", "Section", "Item", "Part", "Step", "page", "Page", "Figure", "Table"]
LOWER = [chr(c) for c in range(ord("a"), ord("t") + 1)]


def _gen_paren_list(rng, family, pools):
    """Same task as the numbered list, different surface separator: 'N) word'."""
    ns = pools.nums(11, 90)
    n = rng.choice(ns)
    w = rng.sample(pools.words(), 4)
    base = f"{n}) {w[0]}\n{n + 1}) {w[1]}\n"
    if family == "A1":
        d = rng.choice([x for x in ns if x != n])
        return dict(base_text=base, donor_text=f"{d}) {w[0]}\n{d + 1}) {w[1]}\n",
                    base_answer=str(n + 2), donor_answer=str(d + 2),
                    semantic_details={"perturbation": "list_start"})
    if family == "A2":
        return dict(base_text=base, donor_text=f"{n}) {w[0]}\n{n + 1}) {w[1]}\n{n + 2}) {w[2]}\n",
                    base_answer=str(n + 2), donor_answer=str(n + 3),
                    semantic_details={"perturbation": "list_length"})
    if family == "P":
        return dict(base_text=base, donor_text=f"{n}) {w[2]}\n{n + 1}) {w[3]}\n",
                    base_answer=str(n + 2), donor_answer=str(n + 2),
                    semantic_details={"perturbation": "item_words"})
    d = rng.choice([x for x in ns if x != n])
    return dict(base_text=f"{n}) {w[0]}\n{n}) {w[1]}\n", donor_text=f"{d}) {w[0]}\n{d}) {w[1]}\n",
                base_answer=str(n), donor_answer=str(d),
                semantic_details={"control": "repeated_index_copy"})


def _gen_keyed_line(rng, family, pools):
    """'Chapter 4 / Chapter 5 / Chapter' -> ' 6'.  The key word is a pure surface slot."""
    ns = pools.nums(2, 60)
    n = rng.choice(ns)
    k, k2 = rng.sample(KEYS, 2)
    line = lambda key, a: f"{key} {a}\n{key} {a + 1}\n{key}"
    if family == "A1":
        d = rng.choice([x for x in ns if x != n])
        return dict(base_text=line(k, n), donor_text=line(k, d),
                    base_answer=f" {n + 2}", donor_answer=f" {d + 2}",
                    semantic_details={"perturbation": "counter_start", "key": k})
    if family == "A2":
        return dict(base_text=line(k, n), donor_text=f"{k} {n}\n{k} {n + 1}\n{k} {n + 2}\n{k}",
                    base_answer=f" {n + 2}", donor_answer=f" {n + 3}",
                    semantic_details={"perturbation": "run_length"})
    if family == "P":
        return dict(base_text=line(k, n), donor_text=line(k2, n),
                    base_answer=f" {n + 2}", donor_answer=f" {n + 2}",
                    semantic_details={"perturbation": "key_word"})
    d = rng.choice([x for x in ns if x != n])
    return dict(base_text=f"{k} {n}\n{k} {n}\n{k}", donor_text=f"{k} {d}\n{k} {d}\n{k}",
                base_answer=f" {n}", donor_answer=f" {d}",
                semantic_details={"control": "repeated_counter_copy"})


def _gen_roman_list(rng, family, pools):
    idx = list(range(len(ROMAN) - 3))[pools._k::3]
    i = rng.choice(idx)
    w = rng.sample(pools.words(), 4)
    lst = lambda j, ws: f"{ROMAN[j]}. {ws[0]}\n{ROMAN[j + 1]}. {ws[1]}\n"
    if family == "A1":
        j = rng.choice([x for x in idx if x != i])
        return dict(base_text=lst(i, w), donor_text=lst(j, w),
                    base_answer=ROMAN[i + 2], donor_answer=ROMAN[j + 2],
                    semantic_details={"perturbation": "list_start"})
    if family == "A2":
        return dict(base_text=lst(i, w),
                    donor_text=lst(i, w) + f"{ROMAN[i + 2]}. {w[2]}\n",
                    base_answer=ROMAN[i + 2], donor_answer=ROMAN[i + 3],
                    semantic_details={"perturbation": "list_length"})
    if family == "P":
        return dict(base_text=lst(i, w), donor_text=lst(i, w[2:]),
                    base_answer=ROMAN[i + 2], donor_answer=ROMAN[i + 2],
                    semantic_details={"perturbation": "item_words"})
    j = rng.choice([x for x in idx if x != i])
    return dict(base_text=f"{ROMAN[i]}. {w[0]}\n{ROMAN[i]}. {w[1]}\n",
                donor_text=f"{ROMAN[j]}. {w[0]}\n{ROMAN[j]}. {w[1]}\n",
                base_answer=ROMAN[i], donor_answer=ROMAN[j],
                semantic_details={"control": "repeated_numeral_copy"})


def _gen_numeric_run(rng, family, pools):
    """Bare ascending run with step != 1; the model's behaviour is LAST + 1 (measured), so the
    causal variable is the last number, not the step -- the sharp contrast with a step-continuer."""
    starts = pools.nums(3, 60)
    a = rng.choice(starts)
    k = rng.choice([2, 3])
    run = lambda s, step: " ".join(str(s + i * step) for i in range(4))
    last = lambda s, step: s + 3 * step
    if family == "A1":
        b = rng.choice([x for x in starts if x != a])
        return dict(base_text=run(a, k), donor_text=run(b, k),
                    base_answer=f" {last(a, k) + 1}", donor_answer=f" {last(b, k) + 1}",
                    semantic_details={"perturbation": "run_start", "step": k})
    if family == "A2":                       # change ONLY the last term: isolates the causal slot
        d = rng.choice([x for x in range(a + 3 * k - 6, a + 3 * k + 7) if x != last(a, k) and x > 0])
        base = run(a, k)
        return dict(base_text=base, donor_text=" ".join(base.split()[:-1] + [str(d)]),
                    base_answer=f" {last(a, k) + 1}", donor_answer=f" {d + 1}",
                    semantic_details={"perturbation": "final_term_only"})
    if family == "P":
        w = rng.sample(pools.words(), 2)
        return dict(base_text=f"{w[0]}: " + run(a, k), donor_text=f"{w[1]}: " + run(a, k),
                    base_answer=f" {last(a, k) + 1}", donor_answer=f" {last(a, k) + 1}",
                    semantic_details={"perturbation": "prefix_word"})
    b = rng.choice([x for x in starts if x != a])
    return dict(base_text=" ".join([str(a)] * 4), donor_text=" ".join([str(b)] * 4),
                base_answer=f" {a}", donor_answer=f" {b}",
                semantic_details={"control": "constant_run_copy"})


def _gen_counting_words(rng, family, pools):
    idx = list(range(len(COUNT_WORDS) - 4))[pools._k::3]
    i = rng.choice(idx)
    c = COUNT_WORDS
    pre = rng.choice(pools.words()) + ": "   # shared surface prefix: widens the draw space
    run = lambda j: pre + f"{c[j]} {c[j + 1]} {c[j + 2]}"
    if family == "A1":
        j = rng.choice([x for x in idx if x != i])
        return dict(base_text=run(i), donor_text=run(j),
                    base_answer=f" {c[i + 3]}", donor_answer=f" {c[j + 3]}",
                    semantic_details={"perturbation": "run_start"})
    if family == "A2":
        return dict(base_text=run(i), donor_text=run(i) + f" {c[i + 3]}",
                    base_answer=f" {c[i + 3]}", donor_answer=f" {c[i + 4]}",
                    semantic_details={"perturbation": "run_length"})
    if family == "P":
        w = rng.sample(pools.words(), 2)
        body = run(i)[len(pre):]
        return dict(base_text=f"{w[0]}: " + body, donor_text=f"{w[1]}: " + body,
                    base_answer=f" {c[i + 3]}", donor_answer=f" {c[i + 3]}",
                    semantic_details={"perturbation": "prefix_word"})
    j = rng.choice([x for x in idx if x != i])
    return dict(base_text=pre + f"{c[i]} {c[i]} {c[i]}", donor_text=pre + f"{c[j]} {c[j]} {c[j]}",
                base_answer=f" {c[i]}", donor_answer=f" {c[j]}",
                semantic_details={"control": "repeated_word_copy"})


def _gen_alphabet_run(rng, family, pools):
    idx = list(range(len(LOWER) - 5))[pools._k::3]
    i = rng.choice(idx)
    L = LOWER
    pre = rng.choice(pools.words()) + ": "   # shared surface prefix: widens the draw space
    run = lambda j: pre + f"{L[j]} {L[j + 1]} {L[j + 2]} {L[j + 3]}"
    if family == "A1":
        j = rng.choice([x for x in idx if x != i])
        return dict(base_text=run(i), donor_text=run(j),
                    base_answer=f" {L[i + 4]}", donor_answer=f" {L[j + 4]}",
                    semantic_details={"perturbation": "run_start"})
    if family == "A2":
        return dict(base_text=run(i), donor_text=run(i) + f" {L[i + 4]}",
                    base_answer=f" {L[i + 4]}", donor_answer=f" {L[i + 5]}",
                    semantic_details={"perturbation": "run_length"})
    if family == "P":
        w = rng.sample(pools.words(), 2)
        body = run(i)[len(pre):]
        return dict(base_text=f"{w[0]}: " + body, donor_text=f"{w[1]}: " + body,
                    base_answer=f" {L[i + 4]}", donor_answer=f" {L[i + 4]}",
                    semantic_details={"perturbation": "prefix_word"})
    j = rng.choice([x for x in idx if x != i])
    return dict(base_text=pre + f"{L[i]} {L[i]} {L[i]} {L[i]}",
                donor_text=pre + f"{L[j]} {L[j]} {L[j]} {L[j]}",
                base_answer=f" {L[i]}", donor_answer=f" {L[j]}",
                semantic_details={"control": "repeated_letter_copy"})


def _gen_bracket(rng, family, pools):
    """'(a [b' -> ']': close the INNERMOST still-open bracket.  No copy control (the answer is
    never a visible token), so this task declares families without C."""
    pairs = [("(", ")"), ("[", "]"), ("{", "}")]
    w = rng.sample(pools.words(), 4)
    (o1, _c1), (o2, c2) = rng.sample(pairs, 2)
    base = f"{o1}{w[0]} {o2}{w[1]}"
    if family == "A1":                       # swap which bracket is innermost
        return dict(base_text=base, donor_text=f"{o2}{w[0]} {o1}{w[1]}",
                    base_answer=c2, donor_answer=_c1,
                    semantic_details={"perturbation": "innermost_bracket_type"})
    if family == "A2":                       # a third, different innermost level
        (o3, c3) = [p for p in pairs if p[0] not in (o1, o2)][0]
        return dict(base_text=base, donor_text=f"{base} {o3}{w[2]}",
                    base_answer=c2, donor_answer=c3,
                    semantic_details={"perturbation": "extra_level"})
    return dict(base_text=base, donor_text=f"{o1}{w[2]} {o2}{w[3]}",
                base_answer=c2, donor_answer=c2,
                semantic_details={"perturbation": "content_words"})


def _gen_verbatim_repeat(rng, family, pools):
    """' quartz quartz quartz' -> ' quartz': a pure copy behaviour, the C-control of the bank
    promoted to a circuit in its own right (families without C: it IS the copy task)."""
    w = rng.sample(pools.single_words(), 4)
    if family == "A1":
        return dict(base_text=f" {w[0]} {w[0]} {w[0]}", donor_text=f" {w[1]} {w[1]} {w[1]}",
                    base_answer=f" {w[0]}", donor_answer=f" {w[1]}",
                    semantic_details={"perturbation": "repeated_token"})
    if family == "A2":                       # the repeat starts later: only the last token repeats
        return dict(base_text=f" {w[0]} {w[0]} {w[0]}", donor_text=f" {w[2]} {w[3]} {w[3]}",
                    base_answer=f" {w[0]}", donor_answer=f" {w[3]}",
                    semantic_details={"perturbation": "repeat_onset"})
    return dict(base_text=f" {w[0]} {w[0]} {w[0]}", donor_text=f" {w[2]} {w[0]} {w[0]}",
                base_answer=f" {w[0]}", donor_answer=f" {w[0]}",
                semantic_details={"perturbation": "leading_filler"})


NUMBERS = [str(i) for i in range(0, 100)] + [f" {i}" for i in range(0, 100)]

TASKS: dict[str, Task] = {
    "numbered_list.index_successor": Task(
        "numbered_list.index_successor",
        "next line label of a numbered list = last visible label + 1",
        "last visible list label", NUMBERS, _gen_numbered_list,
        notes="R567/R576/§2808 lineage; C is R567's repeated-index control."),
    "numeric_sequence.continuation": Task(
        "numeric_sequence.continuation",
        "continue an arithmetic progression of four terms",
        "(last term, step)", NUMBERS, _gen_numeric_sequence),
    "numeric_sequence.countdown": Task(
        "numeric_sequence.countdown",
        "continue a descending run of four terms",
        "(last term, sign of step)", NUMBERS, _gen_countdown),
    "arithmetic.small_addition": Task(
        "arithmetic.small_addition",
        "two-operand small addition, answer after '='",
        "(operand values)", NUMBERS, _gen_addition),
    "weekday.successor": Task(
        "weekday.successor", "next weekday after a run of weekdays",
        "last named weekday", [f" {d}" for d in DAYS], _gen_weekday),
    "month.successor": Task(
        "month.successor", "next month after a run of months",
        "last named month", [f" {m}" for m in MONTHS], _gen_month),
    "letter_list.index_successor": Task(
        "letter_list.index_successor", "next line label of an A./B. lettered list",
        "last visible list letter", LETTERS + [f" {c}" for c in LETTERS], _gen_letter_list),
    "paren_list.index_successor": Task(
        "paren_list.index_successor", "next line label of an 'N) item' list",
        "last visible list label", NUMBERS, _gen_paren_list,
        notes="surface variant of numbered_list: tests re-use of the same writer/readers"),
    "keyed_line.counter_successor": Task(
        "keyed_line.counter_successor", "next counter after 'Chapter 4 / Chapter 5 / Chapter'",
        "last counter value", NUMBERS, _gen_keyed_line,
        notes="P perturbs the key word, so P is a strong surface control"),
    "roman_list.index_successor": Task(
        "roman_list.index_successor", "next line label of a roman-numeral list",
        "last visible roman numeral", ROMAN + [f" {r}" for r in ROMAN], _gen_roman_list),
    "numeric_run.last_plus_one": Task(
        "numeric_run.last_plus_one", "continuation of a bare ascending run: LAST number + 1",
        "last number in the run", NUMBERS, _gen_numeric_run,
        notes="the measured behaviour: with step 2 or 3 the model answers last+1, NOT last+step"),
    "counting_words.successor": Task(
        "counting_words.successor", "next number word after 'one two three'",
        "last number word", [f" {w}" for w in COUNT_WORDS], _gen_counting_words),
    "alphabet_run.successor": Task(
        "alphabet_run.successor", "next letter after 'a b c d'",
        "last letter in the run", [f" {c}" for c in LOWER], _gen_alphabet_run),
    "bracket.close_innermost": Task(
        "bracket.close_innermost", "closing bracket matching the innermost open bracket",
        "type of the innermost open bracket", [")", "]", "}"], _gen_bracket,
        notes="no copy control: the answer is never a visible token",
        families=("A1", "A2", "P")),
    "verbatim_repeat.copy": Task(
        "verbatim_repeat.copy", "continue a verbatim repetition of one token",
        "identity of the repeated token", [f" {w}" for w in SINGLE_WORDS], _gen_verbatim_repeat,
        notes="pure copy behaviour; no separate copy control (it is one)",
        answer_visible_in_prompt=True, families=("A1", "A2", "P")),
    "induction.copy_successor": Task(
        "induction.copy_successor",
        "token following the earlier occurrence of the repeated final token",
        "identity of the repeated key token", [f" {w}" for w in SINGLE_WORDS], _gen_induction,
        notes="retrieval, not computation: the answer is a visible token by construction",
        answer_visible_in_prompt=True),
}


def _row(task: Task, family: str, split: str, d: dict) -> dict | None:
    bt, dt = d["base_text"], d["donor_text"]
    ba, da = d["base_answer"], d["donor_answer"]
    checks = {
        "single_token_answers": single(ba) and single(da),
        "base_roundtrip": ENC.decode(ENC.encode(bt)) == bt,
        "donor_roundtrip": ENC.decode(ENC.encode(dt)) == dt,
        "distinct_prompts": bt != dt,
        "family_answer_invariant": (ba != da) if family in ("A1", "A2", "C") else (ba == da),
        "answers_in_vocab": ba in task.answer_vocab and da in task.answer_vocab,
    }
    if not all(checks.values()):
        return None
    bi, di = ENC.encode(bt), ENC.encode(dt)
    row = {
        "task_id": task.task_id, "family_id": f"{task.task_id}/{family}", "family": family,
        "split": split, "role": "interchange", "answer_changes": ba != da,
        "base_text": bt, "base_ids": bi, "base_answer": ba, "base_answer_id": ENC.encode(ba)[0],
        "donor_text": dt, "donor_ids": di, "donor_answer": da, "donor_answer_id": ENC.encode(da)[0],
        "semantic_details": d.get("semantic_details", {}),
        "construction_checks": checks,
    }
    row["row_id"] = hashlib.sha256(
        json.dumps({k: row[k] for k in ("task_id", "family", "split", "base_text", "donor_text",
                                        "base_answer", "donor_answer")},
                   sort_keys=True).encode()).hexdigest()
    return row


def build_rows(task_id: str, per_cell: int = 24, seed: int = 2808) -> list[dict]:
    """Deterministic rows for one task: FAMILIES x SPLITS x per_cell, deduped."""
    task = TASKS[task_id]
    out: list[dict] = []
    for si, split in enumerate(SPLITS):
        pools = Pools(split)
        for fi, family in enumerate(task.families):
            rng = random.Random(hash((seed, task_id, split, family)) & 0xFFFFFFFF)
            seen, made, tries = set(), 0, 0
            while made < per_cell and tries < per_cell * 60:
                tries += 1
                d = task.gen(rng, family, pools)
                if d is None:
                    continue
                row = _row(task, family, split, d)
                if row is None or row["row_id"] in seen:
                    continue
                seen.add(row["row_id"])
                out.append(row)
                made += 1
            if made < per_cell:
                raise RuntimeError(f"{task_id}/{family}/{split}: only {made}/{per_cell} valid rows")
    return out


def candidate_strings(task_id: str) -> list[str]:
    return list(TASKS[task_id].answer_vocab)


def bank_digest() -> dict:
    """Hash of the whole bank so a battery receipt pins the dataset it used."""
    src = open(__file__, "rb").read()
    return {"schema": SCHEMA, "tasks": sorted(TASKS),
            "source_sha256": hashlib.sha256(src).hexdigest()}


if __name__ == "__main__":
    for tid in TASKS:
        rows = build_rows(tid, per_cell=8)
        print(f"{tid:38s} {len(rows):4d} rows  e.g. {rows[0]['base_text']!r} -> "
              f"{rows[0]['base_answer']!r} | donor {rows[0]['donor_text']!r} -> {rows[0]['donor_answer']!r}")
