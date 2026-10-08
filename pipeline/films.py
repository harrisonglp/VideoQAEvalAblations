"""The film file every condition reads: DATA/films/<video_id>/film.json, beside the frames of pipeline.frames.

    python -m pipeline.films evals/evidence_bank_v1/films.json [...] [--only <video_id> ...]

film.json = {id, title, year, duration_s, frames: [{t, file}], lines: [{t, text}], asr, provenance}
    title, year     from the films list
    duration_s      of the source video (ffprobe), rounded to 1 ms
    frames          the frames lane's frames in time order (DATA/films/<video_id>/text.json, pipeline.frames)
    lines, asr      the speech lane's lines and model (DATA/transcripts/<video_id>.json, pipeline.speech),
                    t = segment start
Transcript text is stored as transcribed; the speech-only filter (conditions.clean_line) is applied at prompt time.
At run time pipeline.run adds the screen-text lane (DATA/ocr/) and the sounds lane (DATA/sounds/).
"""
import argparse

from .common import DATA, duration, provenance, read_json, source_video, write_json


def build(video_id: str, title: str, year: int) -> dict:
    root, transcript = DATA / "films" / video_id, DATA / "transcripts" / f"{video_id}.json"
    for need in (transcript, root / "text.json"):
        if not need.exists():
            raise SystemExit(f"{video_id}: {need} is missing (run pipeline.extract)")
    t = read_json(transcript)
    film = {"id": video_id, "title": title, "year": year, "duration_s": round(duration(source_video(video_id)), 3),
            "frames": [{"t": f["t"], "file": name} for name, f in read_json(root / "text.json")["frames"].items()],
            "lines": [{"t": float(l["start"]), "text": l["text"].strip()} for l in t["lines"] if l["text"].strip()],
            "asr": t["model"], "provenance": provenance(__file__, None)}
    write_json(root / "film.json", film)
    return film


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("films_json", nargs="+")
    ap.add_argument("--only", nargs="+", help="video ids to build; default: every film listed")
    a = ap.parse_args()
    for path in a.films_json:
        for f in read_json(path):
            if a.only and f["video_id"] not in a.only:
                continue
            film = build(f["video_id"], f["title"], f["year"])
            print(f"{film['id']}: {len(film['frames'])} frames, {len(film['lines'])} lines, {film['duration_s'] / 60:.1f} min")


if __name__ == "__main__":
    main()
