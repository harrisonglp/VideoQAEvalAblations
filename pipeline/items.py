"""Build the evidence-bank items: DATA/items/evidence_bank_v1.jsonl.

    python -m pipeline.items

Each question becomes one item: {set, id, film, question, reference, aliases, answer_type, [tolerance | tolerance_s],
evidence_type, evidence, intended_sufficient, temporal_type}. `evidence` holds the cited intervals in seconds;
`intended_sufficient` lists the input sets that each should answer the question, derived from its evidence type, and
`temporal_type` how its evidence is spread over the film (see evals/evidence_bank_v1/FORMAT.md).
"""
from .common import DATA, EVALS, read_jsonl, write_jsonl

SET = "evidence_bank_v1"
# evidence sources: name in evidence_type -> (`source` in evidence intervals, input that carries it)
SOURCES = {"dialogue": ("said", "D"), "visual": ("seen", "V"), "text": ("read", "R"), "audio": ("heard", "A")}
SCENE_GAP_S, MOMENT_S = 60, 10


def parse_evidence_type(t: str) -> tuple[str, list[str]]:
    """'visual' -> ('single', ['visual']); 'dialogue or text' -> ('or', [...]); 'dialogue and visual' -> ('and', [...]);
    'prior' / 'unanswerable' -> (t, []). Raises ValueError for anything outside the vocabulary in FORMAT.md."""
    if t in ("prior", "unanswerable"):
        return t, []
    for kind in ("or", "and"):
        parts = t.split(f" {kind} ")
        if len(parts) == 2:
            if any(p not in SOURCES for p in parts) or parts[0] == parts[1]:
                raise ValueError(f"unknown evidence type {t!r}")
            if list(SOURCES).index(parts[0]) > list(SOURCES).index(parts[1]):
                raise ValueError(f"{t!r}: write sources in the order {', '.join(SOURCES)}")
            return kind, parts
    if t in SOURCES:
        return "single", [t]
    raise ValueError(f"unknown evidence type {t!r}")


def sufficient(t: str) -> list[list[str]]:
    """Input sets that should each suffice (T title, D transcript, V frames, R on-screen text, A sounds)."""
    kind, parts = parse_evidence_type(t)
    if kind == "prior":
        return [["T"]]
    if kind == "unanswerable":
        return []
    inputs = [SOURCES[p][1] for p in parts]
    return [inputs] if kind == "and" else [[x] for x in inputs]


def _sec(t) -> float:
    if isinstance(t, (int, float)):
        return float(t)
    v = 0.0
    for p in str(t).split(":"):
        v = v * 60 + float(p)
    return v


def temporal_type(ev: list, all_: bool = False) -> str:
    """Intervals at most SCENE_GAP_S apart belong to one scene. One scene spanning <= MOMENT_S is a moment, otherwise a
    scene; several scenes are movie-any (any one suffices) or multi-scene (all needed); no interval is movie-whole."""
    if not ev:
        return "movie-whole"
    spans = sorted((e["t_start_s"], e["t_end_s"]) for e in ev)
    scenes = [list(spans[0])]
    for a, z in spans[1:]:
        if a - scenes[-1][1] <= SCENE_GAP_S:
            scenes[-1][1] = max(scenes[-1][1], z)
        else:
            scenes.append([a, z])
    if len(scenes) > 1:
        return "multi-scene" if all_ else "movie-any"
    return "moment" if scenes[0][1] - scenes[0][0] <= MOMENT_S else "scene"


def intervals(q: dict) -> list:
    return [{"t_start_s": _sec(e["start"]), "t_end_s": _sec(e["end"]), "source": e["source"]} for e in q.get("evidence", [])]


def open_items(questions: list) -> list:
    out = []
    for q in questions:
        ev = intervals(q)
        it = {"set": SET, "id": q["id"], "film": q["film"], "question": q["question"], "reference": q["answer"],
              "aliases": q.get("aliases", []), "answer_type": q.get("answer_type", "text")}
        it.update({k: q[k] for k in ("tolerance", "tolerance_s") if k in q})
        it.update(evidence_type=q["evidence_type"], evidence=ev, intended_sufficient=sufficient(q["evidence_type"]),
                  temporal_type=temporal_type(ev, q.get("all", False)))
        out.append(it)
    return out


def main():
    items = open_items(read_jsonl(EVALS / SET / "questions.jsonl"))
    write_jsonl(DATA / "items" / f"{SET}.jsonl", items)
    print(f"{SET}: {len(items)} items from {len({it['film'] for it in items})} films")


if __name__ == "__main__":
    main()
