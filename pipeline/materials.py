"""Per-film validation tables, the input of pipeline.validate: DATA/materials/<video_id>/.

    python -m pipeline.materials evals/evidence_bank_v1/films.json [...] [--only <video_id> ...]

One tab-separated file per source, so that a question can be checked against exactly what each source contains
(t_s in seconds to 1 decimal; every other time as mm:ss or h:mm:ss):
    transcript.tsv  t_s, mmss, text     the transcript lines of film.json as the readers receive them
                                        (conditions.clean_line applied, lines it empties dropped)
    screens.tsv     start, end, text    the screen-text lane (DATA/ocr/<video_id>.json), if built
    sounds.tsv      t0, t1, labels      the validation sound tags with their probabilities
                                        (DATA/sound_events/<video_id>.json, never shown to the readers), if built
    frames.tsv      k, file, t_s, mmss  the sampled frames of film.json (images in DATA/films/<video_id>/frames/);
                                        written last, so it marks a complete export
"""
import argparse
import csv
import json

from .common import DATA, fmt_time, read_json
from .conditions import clean_line


def _tsv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)


def export(vid: str) -> dict:
    film = read_json(DATA / "films" / vid / "film.json")
    out = DATA / "materials" / vid
    out.mkdir(parents=True, exist_ok=True)
    lines = [(round(l["t"], 1), fmt_time(l["t"]), c) for l in film["lines"] if (c := clean_line(l["text"]))]
    _tsv(out / "transcript.tsv", ["t_s", "mmss", "text"], lines)
    counts = {"lines": len(lines), "frames": len(film["frames"])}
    if (ocr := DATA / "ocr" / f"{vid}.json").exists():
        screens = read_json(ocr)["screens"]
        _tsv(out / "screens.tsv", ["start", "end", "text"], [(fmt_time(s["start"]), fmt_time(s["end"]), s["text"]) for s in screens])
        counts["screens"] = len(screens)
    if (snd := DATA / "sound_events" / f"{vid}.json").exists():
        events = read_json(snd)["events"]
        _tsv(out / "sounds.tsv", ["t0", "t1", "labels"],
             [(fmt_time(e["t0"]), fmt_time(e["t1"]), "; ".join(f"{l} {p:.2f}" for l, p in e["labels"])) for e in events])
        counts["sound_windows"] = len(events)
    _tsv(out / "frames.tsv", ["k", "file", "t_s", "mmss"],
         [(k, fr["file"], round(fr["t"], 1), fmt_time(fr["t"])) for k, fr in enumerate(film["frames"])])
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("films_json", nargs="+")
    ap.add_argument("--only", nargs="+", help="video ids to export; default: every film listed")
    a = ap.parse_args()
    for path in a.films_json:
        for f in read_json(path):
            if a.only and f["video_id"] not in a.only:
                continue
            print(f["video_id"], json.dumps(export(f["video_id"])), flush=True)


if __name__ == "__main__":
    main()
