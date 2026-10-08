"""Screen-text lane ("read", input R): what is written on screen, as timestamped entries, one per screen of text.

The film is sampled at 1 frame/s, independently of the 256 frames, so that text shown for only a few seconds (inserts,
signs, letters, intertitles) is not lost between frames. Each sample is read by PaddleOCR; repeated reads of the same
string are merged into intervals, unreadable and fragmentary intervals are dropped, and intervals that appear together
are joined in reading order into one entry per screen. Readers see each entry as `[start–end] On screen: text`.
pipeline.frames uses the same reader, the readability rule (`readable`) and the intervals to mask text in the frames.

Model
    PaddleOCR 3.7.0 (PaddleX 3.7.2), PP-OCRv6_medium_det + PP-OCRv6_medium_rec, engine "transformers" (PyTorch);
    document orientation, unwarping and text-line orientation disabled (film frames are upright and flat)
Settings (constants below)
    FPS 1.0 (--fps)      samples per second of film
    WIDTH 960            samples scaled to 960 px wide, square pixels, height rounded to even
    THUMB, DIFF          a sample is read only when its 32x24 grey thumbnail differs from the last read sample by a
                         mean absolute difference >= 2.0 (0-255); otherwise the previous detections are carried over
    KEEP_SCORE 0.5       recognised strings below this score are discarded at detection
    GAP_S 2.0            an interval closes when its string has not been seen for more than 2 s
    SIMILAR 0.8          a string extends the most similar open interval not yet extended in the same sample when the
                         difflib ratio of their normalised texts (lower case, every run of characters other than a-z
                         and 0-9 -> one space) is >= 0.8; otherwise it opens a new interval
    MIN_SCORE 0.8        an interval is kept when its mean score is >= 0.8, its text is readable (>= 2 ASCII letters
    SOLO_SCORE 0.95      or digits), and it was seen in >= 2 samples or once with score >= 0.95
    FRAGMENT_COVER 0.5   an interval inside another with more samples, whose box covers >= 50% of its box, is dropped
    JOIN_S 1.0           intervals starting within 1 s of a screen's first interval join that screen
An interval's text is its spelling with the highest summed score, its score the mean, its box the per-coordinate
median, its end the last sighting + 1/FPS. Within a screen, rows are formed by vertical centre (an interval joins a row
when its centre is within half its height of the row's centre) and read left to right.
Input
    the source video (common.source_video), decoded by ffmpeg
Output
    DATA/ocr/<video_id>.json = {video_id, engine, fps, width, samples, samples_read,
                                items: [{start, end, text, score, n, box}], screens: [{start, end, text}], provenance}
    `box` is [x0, y0, x1, y1] as fractions of the frame, so it applies to frames of any size.
    DATA/ocr/<video_id>.samples.jsonl: per-sample detections {t, texts, regions}, for audit and --remerge.
Run
    $SHARED_ROOT/envs/ocr_env/bin/python -m pipeline.screen_text <video_id> [...] [--fps 1] [--device gpu:0] [--remerge]
    from the repo root, on a GPU; or through pipeline.extract. --remerge rebuilds `items` and `screens` from the saved
    samples without running OCR.
"""
import argparse
import difflib
import json
import re
import subprocess
from collections import Counter

import cv2
import numpy as np

from .common import DATA, provenance, source_video, write_json

DET, REC = "PP-OCRv6_medium_det", "PP-OCRv6_medium_rec"
MODEL = f"PaddleOCR {DET} + {REC}"
READER = {"text_detection_model_name": DET, "text_recognition_model_name": REC, "use_doc_orientation_classify": False,
          "use_doc_unwarping": False, "use_textline_orientation": False, "engine": "transformers"}
FPS = 1.0              # text shown for 1 s or longer is sampled at least once
WIDTH = 960            # OCR input width, about twice the 448 px frames, so small text keeps enough pixels
THUMB = (32, 24)       # change-gate thumbnail (w, h), grey
DIFF = 2.0             # mean absolute thumbnail difference (0-255) below which a sample is not re-read
KEEP_SCORE = 0.5       # recognition score below which a read is discarded
GAP_S = 2.0            # unseen for longer than this, an interval closes
SIMILAR = 0.8          # character similarity at which a read extends an open interval (absorbs OCR misreadings)
MIN_SCORE = 0.8        # mean score an interval needs to be kept
SOLO_SCORE = 0.95      # score at which an interval seen once is kept
FRAGMENT_COVER = 0.5   # box coverage by a longer-lived interval at which an interval is dropped as its fragment
JOIN_S = 1.0           # start difference within which intervals belong to one screen


def readable(text: str, score: float, min_score: float) -> bool:
    """A confident read with at least two ASCII letters or digits. The recogniser is multilingual and turns texture
    into single CJK characters; the films are in English."""
    return score >= min_score and sum(c.isascii() and c.isalnum() for c in text) >= 2


def reader(device: str = "gpu:0"):
    from paddleocr import PaddleOCR
    return PaddleOCR(**READER, device=device)


def detect(ocr, image: np.ndarray) -> tuple[list, list]:
    """(strings read with score >= KEEP_SCORE [{text, score, box}], all detected text regions [box]); boxes in
    fractions of the frame."""
    h, w = image.shape[:2]
    r = ocr.predict(image)[0]
    frac = lambda b: [round(float(b[0]) / w, 4), round(float(b[1]) / h, 4), round(float(b[2]) / w, 4), round(float(b[3]) / h, 4)]
    poly_box = lambda p: frac([min(x for x, _ in p), min(y for _, y in p), max(x for x, _ in p), max(y for _, y in p)])
    texts = [{"text": t.strip(), "score": round(float(s), 3), "box": frac(b)}
             for t, s, b in zip(r["rec_texts"], r["rec_scores"], r["rec_boxes"]) if t.strip() and s >= KEEP_SCORE]
    return texts, [poly_box(p) for p in r["dt_polys"]]


def frames(path, fps: float):
    """Yield (t, BGR frame) at `fps`, scaled to WIDTH with square pixels."""
    probe = json.loads(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                       "stream=width,height,sample_aspect_ratio", "-of", "json", str(path)],
                                      capture_output=True, check=True, text=True).stdout)["streams"][0]
    sar = probe.get("sample_aspect_ratio", "1:1")
    num, den = (int(x) for x in sar.split(":")) if sar not in ("0:1", "N/A") else (1, 1)
    height = 2 * round(WIDTH * probe["height"] / (probe["width"] * num / den) / 2)
    cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(path), "-vf", f"fps={fps},scale={WIDTH}:{height}",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    size = WIDTH * height * 3
    with subprocess.Popen(cmd, stdout=subprocess.PIPE) as p:
        k = 0
        while len(buf := p.stdout.read(size)) == size:
            yield k / fps, np.frombuffer(buf, np.uint8).reshape(height, WIDTH, 3)
            k += 1


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def overlap(a: list, b: list) -> float:
    """Fraction of box b covered by box a."""
    w, h = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    area = (b[2] - b[0]) * (b[3] - b[1])
    return w * h / area if w > 0 and h > 0 and area > 0 else 0.0


def merge(samples: list[dict], fps: float) -> list[dict]:
    """Per-sample detections [{t, texts}] in time order -> kept intervals [{start, end, text, score, n, box}]."""
    open_, done = [], []
    for s in samples:
        t = s["t"]
        done += [tr for tr in open_ if tr["last"] < t - GAP_S]
        open_ = [tr for tr in open_ if tr["last"] >= t - GAP_S]
        for d in s["texts"]:
            key = _norm(d["text"])
            if not key:
                continue
            free = [tr for tr in open_ if tr["last"] < t]
            best = max(free, key=lambda tr: difflib.SequenceMatcher(None, key, tr["key"]).ratio(), default=None)
            if best and difflib.SequenceMatcher(None, key, best["key"]).ratio() >= SIMILAR:
                tr = best
            else:
                tr = {"start": t, "key": key, "variants": Counter(), "scores": [], "boxes": []}
                open_.append(tr)
            tr["last"] = t
            tr["variants"][d["text"]] += d["score"]
            tr["scores"].append(d["score"])
            tr["boxes"].append(d["box"])
    items = []
    for tr in sorted(done + open_, key=lambda tr: (tr["start"], np.median([b[1] for b in tr["boxes"]]))):
        text = tr["variants"].most_common(1)[0][0]
        score, n = float(np.mean(tr["scores"])), len(tr["scores"])
        if readable(text, score, MIN_SCORE) and (n >= 2 or max(tr["scores"]) >= SOLO_SCORE):
            items.append({"start": tr["start"], "end": tr["last"] + 1 / fps, "text": text, "score": round(score, 3), "n": n,
                          "box": [round(float(v), 4) for v in np.median(tr["boxes"], axis=0)]})
    return [it for it in items if not any(o is not it and o["n"] > it["n"] and o["start"] <= it["start"] and it["end"] <= o["end"]
                                          and overlap(o["box"], it["box"]) >= FRAGMENT_COVER for o in items)]


def screens(items: list[dict]) -> list[dict]:
    """Join intervals that appear together into one entry, in reading order (rows top to bottom, left to right)."""
    groups = []
    for it in sorted(items, key=lambda i: i["start"]):
        if groups and it["start"] - groups[-1][0]["start"] <= JOIN_S:
            groups[-1].append(it)
        else:
            groups.append([it])
    out = []
    for g in groups:
        rows = []
        for it in sorted(g, key=lambda i: (i["box"][1] + i["box"][3]) / 2):
            mid, height = (it["box"][1] + it["box"][3]) / 2, it["box"][3] - it["box"][1]
            if rows and abs(mid - rows[-1][0]) < height / 2:
                rows[-1][1].append(it)
            else:
                rows.append((mid, [it]))
        text = " ".join(" ".join(i["text"] for i in sorted(r, key=lambda i: i["box"][0])) for _, r in rows)
        out.append({"start": min(i["start"] for i in g), "end": max(i["end"] for i in g), "text": text})
    return out


def settings(fps: float, device) -> dict:
    """The settings recorded in the output's provenance."""
    return {"reader": READER, "device": device, "fps": fps, "width": WIDTH, "thumb": THUMB, "diff": DIFF,
            "keep_score": KEEP_SCORE, "gap_s": GAP_S, "similar": SIMILAR, "min_score": MIN_SCORE,
            "solo_score": SOLO_SCORE, "fragment_cover": FRAGMENT_COVER, "join_s": JOIN_S}


def track(ocr, video_id: str, fps: float, device: str) -> dict:
    out = DATA / "ocr"
    out.mkdir(parents=True, exist_ok=True)
    samples, last, read = [], None, 0
    with open(out / f"{video_id}.samples.jsonl", "w") as log:
        for t, img in frames(source_video(video_id), fps):
            thumb = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), THUMB, interpolation=cv2.INTER_AREA).astype(np.float32)
            if last is None or np.abs(thumb - last[0]).mean() >= DIFF:
                texts, regions = detect(ocr, img)
                last, read = (thumb, texts, regions), read + 1
            s = {"t": t, "texts": last[1], "regions": last[2]}
            samples.append(s)
            log.write(json.dumps(s, ensure_ascii=False) + "\n")
            if len(samples) % 600 == 0:
                print(f"{video_id}: {len(samples)} s decoded, {read} read", flush=True)
    items = merge(samples, fps)
    result = {"video_id": video_id, "engine": MODEL, "fps": fps, "width": WIDTH, "samples": len(samples),
              "samples_read": read, "items": items, "screens": screens(items),
              "provenance": provenance(__file__, MODEL, **settings(fps, device))}
    write_json(out / f"{video_id}.json", result)
    return result


def remerge(video_id: str) -> dict:
    """Rebuild `items` and `screens` from the saved per-sample detections."""
    path = DATA / "ocr" / f"{video_id}.json"
    result = json.loads(path.read_text())
    samples = [json.loads(l) for l in open(DATA / "ocr" / f"{video_id}.samples.jsonl")]
    result["items"] = merge(samples, result["fps"])
    result["screens"] = screens(result["items"])
    device = result.get("provenance", {}).get("settings", {}).get("device")  # of the run that read the samples
    result["provenance"] = provenance(__file__, MODEL, **settings(result["fps"], device))
    write_json(path, result)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video_ids", nargs="+")
    ap.add_argument("--fps", type=float, default=FPS)
    ap.add_argument("--device", default="gpu:0")
    ap.add_argument("--remerge", action="store_true", help="rebuild from saved samples, without running OCR")
    a = ap.parse_args()
    ocr = None if a.remerge else reader(a.device)
    for vid in a.video_ids:
        r = remerge(vid) if a.remerge else track(ocr, vid, a.fps, a.device)
        print(f"{vid}: {len(r['items'])} text intervals from {r['samples']} s ({r['samples_read']} read)", flush=True)


if __name__ == "__main__":
    main()
