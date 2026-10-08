"""Compare a fresh extraction root with a reference root (e.g. <repo>/data), per film and lane.

    python scripts/extract_check.py <reference DATA> <fresh DATA> [<video_id> ...] [--films-json FILE] [--report FILE]

Run with the ocr_env interpreter ($SHARED_ROOT/envs/ocr_env/bin/python): the frames comparison needs numpy and
Pillow, everything else is standard library. Films: the ids given, else those of --films-json, else every film with an
output in either root. Model outputs can differ slightly between runs on a GPU, so each lane reports agreement levels
as well as identity:
    speech             lines (ref/new); VAD segments with the same (start, end); of those, the share with the same
                       text; word-level similarity of the whole transcript (difflib ratio)
    screen_text        screens and intervals (`items`) identical; else screens matched exactly (Jaccard over (start,
                       end, text)), matched with the same text and start within 1 s, and word-level similarity of all
                       screen text
    frames             sampled frames byte-identical; masked boxes identical (text.json); masked frames byte-identical,
                       mean absolute pixel difference (0-255) over all frames and its maximum over frames, and the share
                       of pixels differing by more than 32 (a box moved or appeared)
    sounds             events identical; else exact-match Jaccard over (start, end, sound) and labelled-time agreement
                       (per label, intersection over union of the time it covers, summed over labels)
    validation_sounds  windows identical; else Jaccard over (t0, label)
    film               identical apart from `provenance`
    materials          files identical
For each JSON output, `provenance` is "same", "none" (in neither root), {"only in": root}, or the module, hash, model
and settings that differ. A lane missing from either root is reported as such. Exit status 0; read the report.
"""
import argparse
import difflib
import json
import sys
from pathlib import Path


def load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def ratio(a: list, b: list) -> float:
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio() if a or b else 1.0


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a | b else 1.0


def speech(ref: Path, new: Path, vid: str) -> dict:
    a, b = load(ref / "transcripts" / f"{vid}.json"), load(new / "transcripts" / f"{vid}.json")
    if a is None or b is None:
        return {"missing": [n for n, x in (("ref", a), ("new", b)) if x is None]}
    la, lb = a["lines"], b["lines"]
    if la == lb:
        return {"identical": True, "lines": len(la)}
    by_span = {(l["start"], l["end"]): l["text"] for l in lb}
    same_span = [l for l in la if (l["start"], l["end"]) in by_span]
    return {"identical": False, "lines": [len(la), len(lb)],
            "same_segments": round(len(same_span) / max(1, len(la)), 4),
            "same_text_of_same_segments": round(sum(by_span[(l["start"], l["end"])] == l["text"] for l in same_span)
                                                / max(1, len(same_span)), 4),
            "word_similarity": round(ratio(" ".join(l["text"] for l in la).split(),
                                           " ".join(l["text"] for l in lb).split()), 4)}


def screen_text(ref: Path, new: Path, vid: str) -> dict:
    a, b = load(ref / "ocr" / f"{vid}.json"), load(new / "ocr" / f"{vid}.json")
    if a is None or b is None:
        return {"missing": [n for n, x in (("ref", a), ("new", b)) if x is None]}
    sa, sb = a["screens"], b["screens"]
    if sa == sb and a["items"] == b["items"]:
        return {"identical": True, "screens": len(sa), "items": len(a["items"])}
    key = lambda s: (s["start"], s["end"], s["text"])
    near = sum(any(s["text"] == o["text"] and abs(s["start"] - o["start"]) <= 1.0 for o in sb) for s in sa)
    return {"identical": False, "screens": [len(sa), len(sb)], "items": [len(a["items"]), len(b["items"])],
            "screens_exact_jaccard": round(jaccard({key(s) for s in sa}, {key(s) for s in sb}), 4),
            "screens_same_text_within_1s": round(near / max(1, len(sa)), 4),
            "word_similarity": round(ratio(" ".join(s["text"] for s in sa).split(),
                                           " ".join(s["text"] for s in sb).split()), 4)}


def frames(ref: Path, new: Path, vid: str) -> dict:
    ra, rb = ref / "films" / vid, new / "films" / vid
    ta, tb = load(ra / "text.json"), load(rb / "text.json")
    if ta is None or tb is None:
        return {"missing": [n for n, x in (("ref", ta), ("new", tb)) if x is None]}
    try:
        import numpy as np
        from PIL import Image
    except ImportError:
        return {"skipped": "needs numpy and Pillow (run with envs/ocr_env)"}
    files = sorted(ta["frames"])
    same_bytes = lambda d, f: (ra / d / f).exists() and (rb / d / f).exists() \
        and (ra / d / f).read_bytes() == (rb / d / f).read_bytes()
    diffs, big, pixels = [], 0, 0
    for f in files:
        if not (rb / "frames_notext" / f).exists():
            diffs.append(255.0)
            continue
        x = np.asarray(Image.open(ra / "frames_notext" / f).convert("RGB"), np.int16)
        y = np.asarray(Image.open(rb / "frames_notext" / f).convert("RGB"), np.int16)
        if x.shape != y.shape:
            diffs.append(255.0)
            continue
        d = np.abs(x - y)
        diffs.append(float(d.mean()))
        big, pixels = big + int((d.max(axis=2) > 32).sum()), pixels + d.shape[0] * d.shape[1]
    counts = {"frames_identical": sum(same_bytes("frames", f) for f in files),
              "masked_boxes_identical": sum(ta["frames"][f]["masked"] == tb["frames"].get(f, {}).get("masked")
                                            for f in files),
              "masked_frames_identical": sum(same_bytes("frames_notext", f) for f in files)}
    return {"identical": len(tb["frames"]) == len(files) and all(n == len(files) for n in counts.values()),
            "frames": [len(files), len(tb["frames"])], **counts,
            "mean_abs_pixel_diff": round(sum(diffs) / max(1, len(diffs)), 4),
            "max_frame_mean_abs_diff": round(max(diffs, default=0.0), 4),
            "pixels_diff_gt_32": round(big / max(1, pixels), 6),
            "summary": [ta["summary"], tb["summary"]] if ta["summary"] != tb["summary"] else ta["summary"]}


def _covered(events: list, label: str) -> list:
    spans = sorted((e["start"], e["end"]) for e in events if e["sound"] == label)
    out = []
    for s, e in spans:
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def _length(spans: list) -> float:
    return sum(e - s for s, e in spans)


def _intersection(a: list, b: list) -> float:
    return sum(max(0.0, min(e1, e2) - max(s1, s2)) for s1, e1 in a for s2, e2 in b)


def sounds(ref: Path, new: Path, vid: str) -> dict:
    a, b = load(ref / "sounds" / f"{vid}.json"), load(new / "sounds" / f"{vid}.json")
    if a is None or b is None:
        return {"missing": [n for n, x in (("ref", a), ("new", b)) if x is None]}
    ea, eb = a["events"], b["events"]
    if ea == eb:
        return {"identical": True, "events": len(ea)}
    inter = union = 0.0
    for label in {e["sound"] for e in ea + eb}:
        ca, cb = _covered(ea, label), _covered(eb, label)
        i = _intersection(ca, cb)
        inter, union = inter + i, union + _length(ca) + _length(cb) - i
    key = lambda e: (e["start"], e["end"], e["sound"])
    return {"identical": False, "events": [len(ea), len(eb)],
            "events_exact_jaccard": round(jaccard({key(e) for e in ea}, {key(e) for e in eb}), 4),
            "labelled_time_iou": round(inter / union, 4) if union else 1.0}


def validation_sounds(ref: Path, new: Path, vid: str) -> dict:
    a, b = load(ref / "sound_events" / f"{vid}.json"), load(new / "sound_events" / f"{vid}.json")
    if a is None or b is None:
        return {"missing": [n for n, x in (("ref", a), ("new", b)) if x is None]}
    if a["events"] == b["events"]:
        return {"identical": True, "windows": len(a["events"])}
    tags = lambda d: {(e["t0"], l) for e in d["events"] for l, _ in e["labels"]}
    return {"identical": False, "windows": [len(a["events"]), len(b["events"])],
            "tags_jaccard": round(jaccard(tags(a), tags(b)), 4)}


def files_identical(ref: Path, new: Path, rel: str) -> dict:
    pa, pb = ref / rel, new / rel
    names = sorted({p.name for d in (pa, pb) if d.is_dir() for p in d.iterdir()}) if pa.is_dir() or pb.is_dir() else [""]
    missing = [n for n in names if not ((pa / n).exists() and (pb / n).exists())]
    if missing:
        return {"missing": missing}
    differ = [n for n in names if (pa / n).read_bytes() != (pb / n).read_bytes()]
    return {"identical": not differ, **({"differ": differ} if differ else {})}


def film(ref: Path, new: Path, vid: str) -> dict:
    a, b = load(ref / "films" / vid / "film.json"), load(new / "films" / vid / "film.json")
    if a is None or b is None:
        return {"missing": [n for n, x in (("ref", a), ("new", b)) if x is None]}
    for d in (a, b):
        d.pop("provenance", None)
    if a == b:
        return {"identical": True}
    return {"identical": False, **{k: [len(a[k]), len(b[k])] for k in ("lines", "frames")}, "asr": [a["asr"], b["asr"]],
            "duration_s": [a["duration_s"], b["duration_s"]]}


def materials(ref: Path, new: Path, vid: str) -> dict:
    return files_identical(ref, new, f"materials/{vid}")


CHECKS = {"speech": speech, "screen_text": screen_text, "frames": frames, "sounds": sounds,
          "validation_sounds": validation_sounds, "film": film, "materials": materials}
OUTPUTS = {"speech": "transcripts/{}.json", "screen_text": "ocr/{}.json", "frames": "films/{}/text.json",
           "sounds": "sounds/{}.json", "validation_sounds": "sound_events/{}.json", "film": "films/{}/film.json"}


def provenance(ref: Path, new: Path, lane: str, vid: str):
    a, b = ((load(root / OUTPUTS[lane].format(vid)) or {}).get("provenance") for root in (ref, new))
    if a == b:
        return "same" if a else "none"
    if not (a and b):
        return {"only in": "new" if b else "ref"}
    out = {k: [a[k], b[k]] for k in ("module", "sha256", "model") if a[k] != b[k]}
    sa, sb = a["settings"], b["settings"]
    keys = sorted(k for k in sa.keys() | sb.keys() if sa.get(k) != sb.get(k))
    return {**out, **({"settings": keys} if keys else {})}


def film_ids(*roots: Path) -> list[str]:
    ids = set()
    for root in roots:
        for sub in ("transcripts", "ocr", "sounds"):
            ids |= {p.stem for p in (root / sub).glob("*.json")}
        ids |= {p.name for p in (root / "films").glob("*") if p.is_dir()}
    return sorted(ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("reference")
    ap.add_argument("fresh")
    ap.add_argument("video_ids", nargs="*")
    ap.add_argument("--films-json")
    ap.add_argument("--report", help="write the full comparison as JSON")
    a = ap.parse_args()
    ref, new = Path(a.reference), Path(a.fresh)
    for root in (ref, new):
        if not root.is_dir():
            sys.exit(f"not a directory: {root}")
    ids = a.video_ids or ([f["video_id"] for f in json.loads(Path(a.films_json).read_text())] if a.films_json
                          else film_ids(ref, new))
    report = {}
    for vid in ids:
        report[vid] = {}
        for lane, check in CHECKS.items():
            r = report[vid][lane] = check(ref, new, vid)
            if lane == "validation_sounds" and r.get("missing") == ["ref", "new"]:
                continue
            if lane in OUTPUTS and "missing" not in r:
                r["provenance"] = provenance(ref, new, lane, vid)
            print(f"{vid}  {lane:<17} {json.dumps(r, ensure_ascii=False)}", flush=True)
    same = {lane: sum(bool(report[v][lane].get("identical")) for v in ids) for lane in CHECKS}
    print(f"identical, of {len(ids)} films: " + ", ".join(f"{k} {n}" for k, n in same.items()), flush=True)
    if a.report:
        Path(a.report).write_text(json.dumps(report, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
