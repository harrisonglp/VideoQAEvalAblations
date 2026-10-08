"""How much of each question's evidence the extracted inputs carry: results/evidence_bank_v1/fidelity.json and .csv.

    python -m pipeline.fidelity [--films mf2_17,...]

Questions are written from the film (pictures, dialogue, written text, sounds); the tracks are machine-made. For every
cited evidence interval:
    said   the transcript (speech-only) has a line within ± PAD_S whose text states the answer (questions of type
           `dialogue`) or any line at all (other types)
    seen   at least one of the 256 sampled frames falls inside the interval (± 1 s): the reader can see the moment
    read   the on-screen text track has an entry overlapping the interval (± PAD_S) that states the answer
    heard  the sound track has an entry overlapping the interval (± PAD_S); for questions of type `audio`, one that names
           the sound
A question's evidence is reachable when, for each source its type requires, at least one cited interval is captured
("X or Y": either source; "X and Y": both). Questions that cite no evidence are left out.
"""
import argparse
import csv
from collections import defaultdict

from .common import DATA, RESULTS, read_json, read_jsonl, write_json
from .conditions import answer_variants, clean_line, mentions
from .items import SET, SOURCES, parse_evidence_type

PAD_S = 5


def captured(src: str, a: float, z: float, film: dict, variants: list, must_state: bool) -> bool:
    if src == "said":
        hits = [c for l in film["lines"] if a - PAD_S <= l["t"] <= z + PAD_S and (c := clean_line(l["text"]))]
        return any(mentions(variants, t) for t in hits) if must_state else bool(hits)
    if src == "seen":
        return any(a - 1 <= f["t"] <= z + 1 for f in film["frames"])
    if src == "read":
        return any(s["start"] <= z + PAD_S and s["end"] >= a - PAD_S and mentions(variants, s["text"]) for s in film.get("screens", []))
    if src == "heard":
        hits = [s["sound"] for s in film.get("sounds", []) if s["start"] <= z + PAD_S and s["end"] >= a - PAD_S]
        return any(mentions(variants, t) for t in hits) if must_state else bool(hits)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--films", default="", help="comma-separated video ids; default: every film")
    a = ap.parse_args()
    name = SET
    items = [it for it in read_jsonl(DATA / "items" / f"{name}.jsonl") if it["evidence"]]
    if a.films:
        items = [it for it in items if it["film"] in set(a.films.split(","))]
    films = {}
    for vid in {it["film"] for it in items}:
        f = read_json(DATA / "films" / vid / "film.json")
        if (p := DATA / "ocr" / f"{vid}.json").exists():
            f["screens"] = read_json(p)["screens"]
        if (p := DATA / "sounds" / f"{vid}.json").exists():
            f["sounds"] = read_json(p)["events"]
        films[vid] = f
    per_source, per_type, rows = defaultdict(lambda: [0, 0]), defaultdict(lambda: [0, 0]), []
    for it in items:
        kind, parts = parse_evidence_type(it["evidence_type"])
        variants = answer_variants(it["reference"], it.get("aliases", []), it.get("answer_type", "text"))
        film, got = films[it["film"]], defaultdict(bool)
        for e in it["evidence"]:
            must = (e["source"] == "said" and kind == "single") or (e["source"] == "heard" and kind == "single")
            c = captured(e["source"], e["t_start_s"], e["t_end_s"], film, variants, must)
            per_source[e["source"]][0] += c
            per_source[e["source"]][1] += 1
            got[e["source"]] |= c
        need = [SOURCES[p][0] for p in parts]
        ok = any(got[s] for s in need) if kind == "or" else all(got[s] for s in need)
        per_type[it["evidence_type"]][0] += ok
        per_type[it["evidence_type"]][1] += 1
        rows.append([it["id"], it["film"], it["evidence_type"], int(ok), *(int(got[s]) for s in ("said", "seen", "read", "heard"))])
    rate = lambda a, n: {"captured": a, "n": n, "rate": round(a / n, 3) if n else None}
    out = {"set": name, "items": len(items), "reachable": rate(sum(r[3] for r in rows), len(rows)),
           "by_source": {k: rate(*v) for k, v in sorted(per_source.items())},
           "by_type": {k: rate(*v) for k, v in sorted(per_type.items())}}
    (RESULTS / name).mkdir(parents=True, exist_ok=True)
    write_json(RESULTS / name / "fidelity.json", out)
    with open(RESULTS / name / "fidelity.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "film", "evidence_type", "reachable", "said", "seen", "read", "heard"])
        w.writerows(rows)
    print(out)


if __name__ == "__main__":
    main()
