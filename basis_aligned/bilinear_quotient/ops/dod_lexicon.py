#!/usr/bin/env python3
# BQGATE: LIBRARY -- registry of every lexicon used by the DoD batteries and the corpus lists; freshness check.
"""`used_words()` returns every word that appears in the corpus's shared lists (objects, both reporter members,
adjectives, places, tasks) or in any panel this lane has used (aspectual v1 lexicon and templates, lexicon 3/4,
temporal/narrative/number/modal panels). `fresh(candidates, n)` returns the first n single-token candidates not in
that set, or raises listing the shortfall. Written after review 5: four freshness mistakes came from ad-hoc checks."""
from __future__ import annotations
import aspectual_dod_lib as L
import circuit_fast_screen_candidates as lex
try:
    import circuit_fast_screen_canonical_control_v2 as canon
except Exception:  # pragma: no cover
    canon = None

PANELS = {
    "aspectual_v1_agents": L.AGENTS, "aspectual_v1_periods": L.PERIODS,
    "lexicon3_agents": ("nun", "cook", "waiter", "broker", "tutor", "pastor", "referee", "witness", "tenant", "landlord", "dealer", "printer", "author", "hunter", "knight", "wizard"),
    "lexicon3_periods": ("recess", "vacation", "retreat", "rally", "parade", "banquet", "ceremony", "exam", "lecture", "seminar", "workshop", "interview", "race", "battle", "war", "epidemic"),
    "lexicon4_agents": ("vendor", "maid", "peasant", "pirate", "robber", "widow", "orphan", "prophet", "emperor", "princess", "count", "general", "colonel", "sergeant", "deputy", "thief"),
    "lexicon4_places": ("canyon", "harbor", "island", "forest", "garden", "tower", "bridge", "cabin", "river", "ocean", "road", "valley", "field", "stable", "chapel", "tavern"),
    "temporal_places": ("lake", "hill", "gate", "barn", "mill", "dock", "cliff", "beach", "pond", "fence", "well", "shed", "ridge", "cave", "creek", "marsh"),
    "narrative_focus_v42": ("ancient", "broken", "hidden", "rusty", "sacred", "shallow", "steep", "wooden", "crooked", "frozen", "hollow", "humble", "lonely", "modest", "noble", "painted"),
    "narrative_v43_subjects": ("clown", "spy", "shepherd", "brewer", "fisherman"),
    "narrative_v43_places": ("palace", "castle", "temple", "prison", "hospital", "library", "museum", "station", "factory", "school", "church", "cottage", "quarry", "pier", "fountain", "tunnel"),
    "narrative_v43_focus": ("rugged", "sandy", "secret", "silent", "sturdy", "winding", "yellow", "golden", "tidy", "vast", "sunny", "grand", "ruined", "empty", "royal", "modern"),
    "pronoun_gender_v71_nouns": ("hero", "heroine", "god", "goddess", "grandson", "granddaughter", "boyfriend", "girlfriend", "spokesman", "spokeswoman", "dad", "mom", "bull", "cow", "groom", "bride", "male", "female", "guy", "gal"),
    "pronoun_gender_v71_objects": ("compass", "ladder", "hammer", "mirror", "candle", "saddle", "helmet", "anchor", "bucket", "shovel"),
    "pronoun_number_v76_agents": ("trader", "critic", "senator", "diver", "wrestler", "cyclist", "physicist", "biologist", "economist", "philosopher", "programmer", "developer", "consultant", "investor", "marine", "commander"),
    "pronoun_number_v76_objects": ("coin", "rope", "torch", "bell", "jar", "drum", "crown", "sword", "shield", "purse", "wallet", "ticket", "spoon", "knife", "bowl", "pillow"),
    "selection_v85_agents": ("major", "recruit", "veteran", "cleric", "elder", "apostle", "comedian", "musician", "patient", "customer", "client", "tourist", "passenger", "commuter", "pedestrian", "citizen"),
    "selection_v85_objects": ("blanket", "wagon", "barrel", "cradle", "scroll", "brush", "needle", "lamp", "box", "bag", "cart", "flag", "map", "key", "tent", "boat"),
    "number_v62_places": ("canal", "lodge", "inn", "mine", "farm", "arena", "bakery", "brewery", "cellar", "clinic", "depot", "gallery", "hangar", "plaza", "reef", "shrine"),
}


def used_words(exclude: tuple[str, ...] = ()) -> set[str]:
    """`exclude` names PANELS entries to leave out: a runner checking its OWN registered panel passes its panel names,
    so registering a panel after the fact does not break that runner's replay."""
    words = set(lex._OBJECTS) | {p[0] for p in lex._REPORTERS} | {p[1] for p in lex._REPORTERS} | set(lex._ADJECTIVES)
    for attr in ("_PLACES", "_TASKS", "_SUBJECTS", "_ALTERNATES"):
        words |= set(getattr(canon, attr, ()) or ())
    words |= L._PRIOR_AGENTS | L._PRIOR_PERIODS
    for k, v in PANELS.items():
        if k not in exclude:
            words |= set(v)
    return words


def fresh(candidates, n: int, *, plural: bool = False, exclude: tuple[str, ...] = ()) -> tuple[str, ...]:
    """`exclude`: the caller's own PANELS entries (registered after its run), so its replay still picks the same words."""
    used = used_words(exclude)
    out = []
    for w in candidates:
        if w in used or w in out:
            continue
        if len(L.ENCODING.encode(" " + w)) != 1:
            continue
        if plural and len(L.ENCODING.encode(" " + w + "s")) != 1:
            continue
        out.append(w)
        if len(out) == n:
            return tuple(out)
    raise L.RowError(f"only {len(out)} fresh single-token candidates of {n} requested: {out}")
