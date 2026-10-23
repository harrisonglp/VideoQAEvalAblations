"""Paths and small I/O helpers shared by the pipeline modules."""
import hashlib
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SHARED = Path(os.environ.get("SHARED_ROOT", "/data/cinesage"))  # environments, model caches, source videos
RAW = SHARED / "raw"  # source videos
CONFIGS, EVALS, DATA, RESULTS = ROOT / "configs", ROOT / "evals", ROOT / "data", ROOT / "results"
if os.environ.get("PAPER1_DATA"):  # build generated inputs into another root, leaving <repo>/data untouched
    DATA = Path(os.environ["PAPER1_DATA"]).absolute()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, ensure_ascii=False))


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_profile(name: str) -> dict:
    """The KEY=VALUE settings of configs/models/<name>.env (a value may be single-quoted). A profile's file name is
    its served model name."""
    out = {}
    path = CONFIGS / "models" / f"{name}.env"
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            out[k] = v[1:-1] if len(v) >= 2 and v[0] == v[-1] == "'" else v
    return out


# the reply the prompt asks for ({"id": ..., "answer": ...}); with ANSWER_SCHEMA=1 the server constrains decoding to it
ANSWER_SCHEMA = {"type": "json_schema", "json_schema": {"name": "answer", "strict": True, "schema": {
    "type": "object", "properties": {"id": {"type": "string"}, "answer": {"type": "string"}},
    "required": ["id", "answer"], "additionalProperties": False}}}


def request_options(model: str) -> tuple[dict, int | None]:
    """(fields added to every request of this model, its answer token limit or None for the default), from its
    profile's REQUEST_EXTRA (JSON), ANSWER_SCHEMA (1: replies constrained to the answer JSON, as response_format) and
    MAX_TOKENS. Without REQUEST_EXTRA, thinking is turned off through the chat template, as for the first reader."""
    p = read_profile(model)
    extra = json.loads(p["REQUEST_EXTRA"]) if p.get("REQUEST_EXTRA") else {"chat_template_kwargs": {"enable_thinking": False}}
    if p.get("ANSWER_SCHEMA") == "1":
        extra = {**extra, "response_format": ANSWER_SCHEMA}
    return extra, int(p["MAX_TOKENS"]) if p.get("MAX_TOKENS") else None


def source_video(video_id: str) -> Path:
    """Path of the source video, looked up in every evals/*/films.json."""
    for listing in sorted(EVALS.glob("*/films.json")):
        for f in read_json(listing):
            if f["video_id"] == video_id:
                return RAW / f["video"]
    raise SystemExit(f"{video_id}: not listed in any evals/*/films.json")


def duration(path) -> float:
    """Duration of a video file in seconds: the container duration reported by ffprobe."""
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)],
                         capture_output=True, text=True, check=True).stdout
    return float(json.loads(out)["format"]["duration"])


def provenance(module_file: str, model, **settings) -> dict:
    """The `provenance` object of an output: the module that wrote it (path in the repo and the first 16 hex
    characters of its file's sha256), the model, and every setting that affects the output."""
    path = Path(module_file).resolve()
    return {"module": path.relative_to(ROOT).as_posix(), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()[:16],
            "model": model, "settings": settings}


def audio(path, sr: int):
    """The soundtrack of a video file as mono float32 samples in [-1, 1) at `sr` Hz (ffmpeg, 16-bit PCM)."""
    import numpy as np  # here, not at the top: the client-environment modules need only the standard library
    raw = subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(path), "-vn", "-ac", "1",
                          "-ar", str(sr), "-f", "s16le", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0


def fmt_time(t: float) -> str:
    """Seconds -> "mm:ss" or "h:mm:ss", rounded to the nearest second (the timestamp format used in every prompt)."""
    t = int(round(t))
    h, m, s = t // 3600, t % 3600 // 60, t % 60
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"
