"""Check evidence-bank questions against FORMAT.md and against the evidence each input actually contains.

    python -m pipeline.validate evals/evidence_bank_v1/questions.jsonl [--films mf2_17,...] [--films-json FILE]
        [--data DIR] [--report report.json] [--format-only]

Errors (the question cannot be used as written):
    missing fields, malformed or duplicate id, unknown film, evidence type outside the vocabulary, intervals that are
    malformed, reversed or outside the film, cited frames that are not among the film's frames or lie outside their
    interval (± 1.5 s), sources in `evidence` that differ from the sources named by `evidence_type`, no evidence on a
    question whose type needs it, evidence on a `prior` or `unanswerable` question, an `unanswerable` answer other
    than "not shown in the film", `all` with fewer than two intervals, a non-numeric `number` answer.
Warnings (to be checked by hand):
    unknown   a field outside KNOWN
    leak      the answer (or an alias) occurs in an input that should not carry it: in the transcript or the on-screen
              text track when the question's type excludes that source, within NEAR_S of the cited evidence (or
              anywhere, for questions without evidence); elsewhere in the film it is reported as a weak leak
    support   a `read` interval whose on-screen text (± PAD_S) does not state the answer; a `said` interval whose
              transcript lines (± PAD_S) do not state it verbatim (may be a paraphrase; not checked for "X and Y"
              types); a `heard` interval with no sound tag (± PAD_S)
    citation  the answer also occurs in a cited input (transcript or on-screen text track) at times the question does
              not cite (± PAD_S); every occurrence should be cited
Inputs are read from DIR/materials/<video_id>/{transcript,screens,sounds,frames}.tsv (pipeline.materials) and
DIR/films/<video_id>/film.json (pipeline.films); DIR defaults to DATA (<repo>/data or $PAPER1_DATA).
"""
import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

from .common import DATA, EVALS, fmt_time, read_json, read_jsonl, write_json
from .conditions import _norm, answer_variants
from .items import SOURCES, _sec, intervals, parse_evidence_type, temporal_type

REQUIRED = ("id", "film", "question", "answer", "evidence_type", "evidence")
KNOWN = set(REQUIRED) | {"aliases", "answer_type", "tolerance", "tolerance_s", "all", "checks"}
UNANSWERABLE = "not shown in the film"
PAD_S = {"said": 5, "read": 3, "heard": 4}  # s around a cited interval within which its source is searched
NEAR_S = 120  # a leak within this distance of the cited evidence is likely to be about the same moment


def _rows(path):
    if not path.exists():
        return None
    with open(path, newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


class Film:
    def __init__(self, root, vid):
        m = root / "materials" / vid
        self.duration = read_json(root / "films" / vid / "film.json")["duration_s"]
        self.lines = [(float(r["t_s"]), _norm(r["text"])) for r in _rows(m / "transcript.tsv")]
        self.frames = [float(r["t_s"]) for r in _rows(m / "frames.tsv")]
        s = _rows(m / "screens.tsv")
        self.screens = None if s is None else [(_sec(r["start"]), _sec(r["end"]), _norm(r["text"])) for r in s]
        h = _rows(m / "sounds.tsv")
        self.sounds = None if h is None else [(_sec(r["t0"]), _sec(r["t1"]), r["labels"]) for r in h]


def _variants(q) -> list[str]:
    return answer_variants(q["answer"], q.get("aliases", []), q.get("answer_type", "text"))


def _hit(variants, text) -> str | None:
    for v in variants:
        if re.search(rf"(?<!\w){re.escape(v)}(?!\w)", text):
            return v
    return None


def check(q: dict, films: dict, data_root, seen_ids: set, format_only: bool = False) -> tuple[list, list]:
    err, warn = [], []
    for k in REQUIRED:
        if k not in q:
            err.append(f"missing field {k!r}")
    if err:
        return err, warn
    for k in sorted(set(q) - KNOWN):
        warn.append(f"unknown field {k!r}")
    meta = films.get(q["film"])
    if meta is None:
        return [f"film {q['film']!r} is not in films.json"], warn
    if not re.fullmatch(rf"{meta['code']}_\d{{3}}", q["id"]):
        err.append(f"id must look like {meta['code']}_001")
    if q["id"] in seen_ids:
        err.append("duplicate id")
    seen_ids.add(q["id"])
    try:
        kind, parts = parse_evidence_type(q["evidence_type"])
    except ValueError as e:
        return err + [str(e)], warn
    film = Film(data_root, q["film"]) if not format_only else None
    duration = film.duration if film else films[q["film"]].get("duration_s", 10 ** 6)
    ev = q["evidence"]
    if kind in ("prior", "unanswerable"):
        if ev:
            err.append(f"{kind} items cite no evidence")
        if kind == "unanswerable" and q["answer"] != UNANSWERABLE:
            err.append(f"unanswerable items answer {UNANSWERABLE!r}")
    else:
        if not ev:
            err.append("no evidence intervals")
        cited = {e.get("source") for e in ev}
        want = {SOURCES[p][0] for p in parts}
        if cited != want:
            err.append(f"evidence sources {sorted(cited)} do not match evidence_type (needs {sorted(want)})")
    for i, e in enumerate(ev):
        try:
            a, z = _sec(e["start"]), _sec(e["end"])
        except (KeyError, ValueError):
            err.append(f"evidence[{i}]: start/end must be mm:ss or h:mm:ss")
            continue
        if not (0 <= a <= z <= duration + 1):
            err.append(f"evidence[{i}]: {e['start']}-{e['end']} outside 0-{duration:.0f} s or reversed")
        for k in e.get("frames", []) if film else []:
            if not (isinstance(k, int) and 0 <= k < len(film.frames)):
                err.append(f"evidence[{i}]: frame {k!r} is not one of the {len(film.frames)} frames")
            elif not (a - 1.5 <= film.frames[k] <= z + 1.5):
                err.append(f"evidence[{i}]: frame {k} is at {film.frames[k]:.0f} s, outside {e['start']}-{e['end']}")
    if q.get("all") and len(ev) < 2:
        err.append("'all' needs at least two intervals")
    if q.get("answer_type") == "number":
        try:
            float(q["answer"])
        except ValueError:
            err.append("number answers are written as digits")
    if kind == "unanswerable" or format_only:
        return err, warn

    variants, named = _variants(q), set(parts)
    spans = [(_sec(e["start"]), _sec(e["end"])) for e in ev if "start" in e and "end" in e]
    near = lambda t0, t1: not spans or any(t0 <= z + NEAR_S and t1 >= a - NEAR_S for a, z in spans)
    channels = [("transcript", "dialogue", [(t, t, x) for t, x in film.lines]), ("on-screen text track", "text", film.screens)]
    for name, source, timed in channels:
        if source in named or timed is None:
            continue
        hits = [(t0, t1, v) for t0, t1, x in timed if (v := _hit(variants, x))]
        close = [h for h in hits if near(h[0], h[1])]
        if close:
            warn.append(f"leak: {close[0][2]!r} is in the {name} near the evidence ({fmt_time(close[0][0])})")
        elif hits:
            warn.append(f"leak (weak): {hits[0][2]!r} is in the {name} elsewhere in the film ({fmt_time(hits[0][0])})")
    for e in ev:
        a, z, src = _sec(e["start"]), _sec(e["end"]), e.get("source")
        pad = PAD_S.get(src, 0)
        if src == "read":
            if film.screens is None:
                warn.append("support: no on-screen text track built yet; read evidence unchecked")
            elif not _hit(variants, " ".join(t for s0, s1, t in film.screens if s0 <= z + pad and s1 >= a - pad)):
                warn.append(f"support: answer not in the on-screen text track at {e['start']}-{e['end']}")
        elif src == "said" and kind != "and":
            if not _hit(variants, " ".join(t for t0, t in film.lines if a - pad <= t0 <= z + pad)):
                warn.append(f"support: answer not verbatim in the transcript at {e['start']}-{e['end']} (paraphrase?)")
        elif src == "heard":
            if film.sounds is None:
                warn.append("support: no sound tags built yet; heard evidence unchecked")
            elif not any(s0 <= z + pad and s1 >= a - pad for s0, s1, _ in film.sounds):
                warn.append(f"support: no sound tag at {e['start']}-{e['end']}")
    for src, timed in (("said", [(t, t, x) for t, x in film.lines]), ("read", film.screens)):
        cited = [(_sec(e["start"]) - PAD_S[src], _sec(e["end"]) + PAD_S[src]) for e in ev if e.get("source") == src]
        if not cited or timed is None:
            continue
        uncited = [t0 for t0, t1, x in timed if _hit(variants, x) and not any(t0 <= z and t1 >= a for a, z in cited)]
        if uncited:
            warn.append(f"citation: the answer is also {'said' if src == 'said' else 'on screen'} at "
                        f"{', '.join(fmt_time(t) for t in uncited[:4])}{' …' if len(uncited) > 4 else ''} "
                        f"({len(uncited)} uncited); cite every occurrence")
    return err, warn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("questions")
    ap.add_argument("--films", default="", help="comma-separated video ids; default: every film")
    ap.add_argument("--films-json", default=str(EVALS / "evidence_bank_v1" / "films.json"))
    ap.add_argument("--data", default=str(DATA))
    ap.add_argument("--report")
    ap.add_argument("--format-only", action="store_true",
                    help="only the entry format (for questions written from the film, before the tracks exist)")
    a = ap.parse_args()
    films = {f["video_id"]: f for f in read_json(a.films_json)}
    qs, seen, report = read_jsonl(a.questions), set(), []
    if a.films:
        qs = [q for q in qs if q.get("film") in set(a.films.split(","))]
    for q in qs:
        err, warn = check(q, films, Path(a.data), seen, a.format_only)
        report.append({"id": q.get("id"), "errors": err, "warnings": warn})
        for m in err:
            print(f"ERROR {q.get('id')}: {m}")
        for m in warn:
            print(f"warn  {q.get('id')}: {m}")
    ok = [q for q, r in zip(qs, report) if not r["errors"]]
    by = lambda f: dict(Counter(map(f, ok)).most_common())
    summary = {"items": len(qs), "valid": len(ok), "with_warnings": sum(bool(r["warnings"]) for r in report),
               "films": by(lambda q: q["film"]), "evidence_type": by(lambda q: q["evidence_type"]),
               "temporal_type": by(lambda q: temporal_type(intervals(q), q.get("all", False))),
               "answer_type": by(lambda q: q.get("answer_type", "text"))}
    print(json.dumps(summary, indent=1))
    if a.report:
        write_json(a.report, {"summary": summary, "items": report})
    raise SystemExit(1 if len(ok) < len(qs) else 0)


if __name__ == "__main__":
    main()
