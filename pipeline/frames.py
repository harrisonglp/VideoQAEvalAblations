"""Frames lane ("seen", input V): what is seen, as 256 frames sampled uniformly through the film, with the text on
screen blacked out.

Sampling
    Frame k = 0 … 255 is taken at (k + 0.5) · duration / 256 (duration from ffprobe, times rounded to 1 ms), so frames
    are evenly spaced and none falls on the first or last instant; one ffmpeg seek per frame (-threads 1 -ss <t>
    -i <video> -frames:v 1), SEEKS in parallel, a seek that yields no frame skipped with a warning; pixels made square
    (the sample aspect ratio applied, so the frame has the film's display aspect ratio), then long edge 448 px and short
    edge in proportion and even (448 × 336 for 4:3 films); JPEG at ffmpeg -q:v 3; named <kkkk>_<t_ms>.jpg.
    With the same ffmpeg build the JPEGs are byte-identical from run to run.
Masking
    V must not carry written text, which the screen-text lane supplies separately (R). For each frame:
    1. it is upscaled (bicubic) to screen_text.WIDTH (960 px) and read with the screen-text reader (PP-OCRv6 medium
       detector and recogniser, see pipeline.screen_text);
    2. masked regions are the boxes of the strings read in this frame that are readable at MASK_SCORE, plus the boxes
       of the screen-text intervals (before joining into screens) that cover the frame's time (start <= t < end) and
       overlap a detector region of this frame (more than TRACK_OVERLAP of either box covered by the other). Detector
       regions without a readable string are not masked: on film frames they are mostly texture (foliage, patterned
       cloth, hands), and masking them would remove visual evidence;
    3. each region is padded by PAD · frame width on every side, clipped to the frame and filled black;
    4. the masked frame is read again with the same reader and rule; strings still readable are recorded as
       `residual`, not masked further.
Settings (constants below)
    N_FRAMES 256, EDGE 448, JPEG_Q 3, SEEKS 16; MASK_SCORE 0.6, TRACK_OVERLAP 0.3, PAD 0.006, JPEG_QUALITY 95,
    FILL black
Input
    the source video (common.source_video); DATA/ocr/<video_id>.json (`items`) from pipeline.screen_text
Output (DATA/films/<video_id>/)
    frames/           the sampled frames (V of the natural-frames condition tdv_fullframes)
    frames_notext/    regions black: V of every other condition; JPEG quality 95
    frames_textonly/  everything except the regions black: diagnostic of what a reader could read itself
    text.json         {video_id, pad_frac, summary: {frames, frames_masked, mean_area_masked_frames,
                       frames_with_residual_text}, frames: {file: {t, text, masked, area, residual}}, provenance};
                      `frames` lists the sampled frames in time order (pipeline.films lists them in film.json),
                      `masked` holds the unpadded boxes as fractions of the frame, `area` the masked fraction
Run
    $SHARED_ROOT/envs/ocr_env/bin/python -m pipeline.frames <video_id> [...] [--device gpu:0]
    from the repo root, on a GPU, after pipeline.screen_text; or through pipeline.extract
"""
import argparse
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image, ImageDraw

from .common import DATA, duration, provenance, read_json, source_video, write_json
from .screen_text import KEEP_SCORE, MODEL, READER, WIDTH, detect, overlap, readable, reader

N_FRAMES = 256       # frames per film
EDGE = 448           # long edge of a frame in px
# non-square pixels are resampled to square first (a no-op for square pixels); then the long edge to EDGE, the short
# edge in proportion and even
SCALE = f"scale=iw*sar:ih,setsar=1,scale='if(gt(iw,ih),{EDGE},-2)':'if(gt(iw,ih),-2,{EDGE})'"
JPEG_Q = 3           # ffmpeg -q:v of the sampled frames
SEEKS = 16           # ffmpeg seeks in parallel, each single-threaded
MASK_SCORE = 0.6     # a read in a single frame counts as text from this score (no repeats to confirm it)
TRACK_OVERLAP = 0.3  # a screen-text interval is masked only where the detector also finds text in this frame
PAD = 0.006          # fraction of frame width added on every side (about 2.7 px at 448 px), to cover glyph edges
JPEG_QUALITY = 95    # Pillow JPEG quality of the masked and text-only frames
FILL = (0, 0, 0)     # mask colour, black


def times(duration_s: float) -> list[float]:
    """Midpoints of N_FRAMES equal slices of the film, rounded to 1 ms."""
    return [round((k + 0.5) * duration_s / N_FRAMES, 3) for k in range(N_FRAMES)] if duration_s > 0 else []


def grab(video, t: float, path) -> bool:
    """The frame at t s, scaled, to `path`; False if the seek yields no frame."""
    tmp = path.with_suffix(".tmp.jpg")
    subprocess.run(["nice", "-n", "10", "ffmpeg", "-nostdin", "-v", "error", "-y", "-threads", "1", "-ss", f"{t:.3f}",
                    "-i", str(video), "-frames:v", "1", "-vf", SCALE, "-q:v", str(JPEG_Q), str(tmp)],
                   check=False, stdin=subprocess.DEVNULL)
    if not tmp.exists():
        return False
    os.replace(tmp, path)
    return True


def sample(video_id: str) -> list[dict]:
    """DATA/films/<video_id>/frames/ <- the frames of the source video; [{t, file}] in time order."""
    video, out = source_video(video_id), DATA / "films" / video_id / "frames"
    ts = times(duration(video))
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.jpg"):
        old.unlink()

    def one(kt):
        k, t = kt
        name = f"{k:04d}_{int(round(t * 1000))}.jpg"
        return {"t": t, "file": name} if grab(video, t, out / name) else None

    with ThreadPoolExecutor(SEEKS) as pool:
        frames = [f for f in pool.map(one, enumerate(ts)) if f]
    if len(frames) < len(ts):
        print(f"{video_id}: WARNING: {len(ts) - len(frames)} of {len(ts)} seeks produced no frame", flush=True)
    if not frames:
        raise SystemExit(f"{video_id}: no frames from {video}")
    return frames


def regions_at(t: float, track: list[dict], found: list[list[float]]) -> list[list[float]]:
    """Boxes of screen-text intervals covering t that overlap a detector region of this frame."""
    return [it["box"] for it in track if it["start"] <= t < it["end"]
            and any(overlap(it["box"], r) > TRACK_OVERLAP or overlap(r, it["box"]) > TRACK_OVERLAP for r in found)]


def as_array(img: Image.Image) -> np.ndarray:
    """The frame as the reader's input: upscaled to WIDTH, BGR."""
    big = img.resize((WIDTH, round(img.height * WIDTH / img.width)), Image.BICUBIC)
    return np.asarray(big)[:, :, ::-1].copy()


def pixel_boxes(boxes: list[list[float]], w: int, h: int) -> list[tuple]:
    """Fractional boxes -> pixel rectangles (x0, y0, x1, y1), padded by PAD · w, clipped to the frame, rounded."""
    pad = PAD * w
    return [tuple(round(v) for v in (max(0, b[0] * w - pad), max(0, b[1] * h - pad), min(w, b[2] * w + pad),
                                     min(h, b[3] * h + pad))) for b in boxes]


def apply_mask(img: Image.Image, rects: list[tuple]) -> tuple[Image.Image, Image.Image, float]:
    """(frame with the rectangles filled, frame with everything else filled, fraction of the frame masked)."""
    notext, textonly = img.copy(), Image.new("RGB", img.size)
    draw, covered = ImageDraw.Draw(notext), np.zeros((img.height, img.width), bool)
    for box in rects:
        draw.rectangle(box, fill=FILL)
        textonly.paste(img.crop(box), box[:2])
        covered[box[1]:box[3], box[0]:box[2]] = True
    return notext, textonly, round(float(covered.mean()), 4)


def mask_frame(ocr, img: Image.Image, t: float, track: list[dict]) -> tuple[Image.Image, Image.Image, dict]:
    """(masked frame, text-only frame, its text.json record)."""
    texts, found = detect(ocr, as_array(img))
    texts = [d for d in texts if readable(d["text"], d["score"], MASK_SCORE)]
    boxes = [d["box"] for d in texts] + regions_at(t, track, found)
    notext, textonly, area = apply_mask(img, pixel_boxes(boxes, img.width, img.height))
    residual = [d["text"] for d in detect(ocr, as_array(notext))[0] if readable(d["text"], d["score"], MASK_SCORE)]
    return notext, textonly, {"t": t, "text": [d["text"] for d in texts], "masked": boxes, "area": area,
                              "residual": residual}


def mask(ocr, video_id: str, device: str) -> dict:
    root = DATA / "films" / video_id
    track = read_json(DATA / "ocr" / f"{video_id}.json")["items"]
    frames = sample(video_id)
    (root / "frames_notext").mkdir(exist_ok=True)
    (root / "frames_textonly").mkdir(exist_ok=True)
    per_frame = {}
    for f in frames:
        img = Image.open(root / "frames" / f["file"]).convert("RGB")
        notext, textonly, per_frame[f["file"]] = mask_frame(ocr, img, f["t"], track)
        notext.save(root / "frames_notext" / f["file"], quality=JPEG_QUALITY)
        textonly.save(root / "frames_textonly" / f["file"], quality=JPEG_QUALITY)
    masked = [v for v in per_frame.values() if v["masked"]]
    summary = {"frames": len(per_frame), "frames_masked": len(masked),
               "mean_area_masked_frames": round(float(np.mean([v["area"] for v in masked])), 4) if masked else 0.0,
               "frames_with_residual_text": sum(bool(v["residual"]) for v in per_frame.values())}
    write_json(root / "text.json", {"video_id": video_id, "pad_frac": PAD, "summary": summary, "frames": per_frame,
                                    "provenance": provenance(__file__, MODEL, frames=N_FRAMES, edge=EDGE, jpeg_q=JPEG_Q,
                                                             reader=READER, device=device, width=WIDTH,
                                                             keep_score=KEEP_SCORE, mask_score=MASK_SCORE,
                                                             track_overlap=TRACK_OVERLAP, pad=PAD,
                                                             jpeg_quality=JPEG_QUALITY, fill=FILL)})
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video_ids", nargs="+")
    ap.add_argument("--device", default="gpu:0")
    a = ap.parse_args()
    ocr = reader(a.device)
    for vid in a.video_ids:
        print(f"{vid}: {mask(ocr, vid, a.device)}", flush=True)


if __name__ == "__main__":
    main()
