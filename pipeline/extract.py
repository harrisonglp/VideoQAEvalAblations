"""Build the inputs of one or more films from their source videos, step by step in dependency order.

    python -m pipeline.extract [<video_id> ...] [--films-json evals/evidence_bank_v1/films.json] [--validation]
        [--force]

Without video ids, every film of --films-json. Per film, in this order:
    speech             DATA/transcripts/<id>.json           pipeline.speech             asr_parakeet_env
    screen_text        DATA/ocr/<id>.json                   pipeline.screen_text        ocr_nemotron_env
    frames             DATA/films/<id>/text.json, frames*/  pipeline.frames             ocr_nemotron_env
    sounds             DATA/sounds/<id>.json                pipeline.sounds             panns_env
    validation_sounds  DATA/sound_events/<id>.json          pipeline.validation_sounds  florence_env  (--validation)
    film               DATA/films/<id>/film.json            pipeline.films              client
    materials          DATA/materials/<id>/                 pipeline.materials          client        (--validation)
frames reads the screen_text output, film the speech and frames outputs, materials film.json and, when built, the
screen_text and validation_sounds outputs. validation_sounds and materials serve bank validation only (no reader
receives them). A step is skipped when its output exists, unless --force; frames, film and materials are also rebuilt
when a step they read ran, and a step whose input is missing is blocked. Each step runs as a subprocess with its
environment's interpreter (PYTHON; default $SHARED_ROOT/envs/<env>/bin/python), from the repo root, so the model
libraries of different lanes never share a process. One line per step and film: id, step, status, wall time (model
loading included). The exit status is 1 if any step failed or was blocked.

DATA is <repo>/data, or $PAPER1_DATA (common.DATA), and is passed on to every step. Lanes read model files from the
caches under the shared root (LANE_ENV).
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from .common import DATA, ROOT, SHARED, read_json


ENVS = SHARED / "envs"
# environment: interpreter, overridden by the variable named
PYTHON = {
    "asr_parakeet_env": os.environ.get("PAPER1_PY_SPEECH", str(ENVS / "asr_parakeet_env/bin/python")),
    "ocr_nemotron_env": os.environ.get("PAPER1_PY_OCR", str(ENVS / "ocr_nemotron_env/bin/python")),
    "panns_env": os.environ.get("PAPER1_PY_SOUNDS", str(ENVS / "panns_env/bin/python")),
    "florence_env": os.environ.get("PAPER1_PY_VALIDATION", str(ENVS / "florence_env/bin/python")),
    "client": os.environ.get("PAPER1_PY_CLIENT", str(ENVS / "client/bin/python")),
}

# step: (module, environment, arguments, output marking it done, outputs it needs, steps whose outputs it reads)
STEPS = {
    "speech": ("pipeline.speech", "asr_parakeet_env", ["{id}"], "transcripts/{id}.json", [], []),
    "screen_text": ("pipeline.screen_text", "ocr_nemotron_env", ["{id}", "--device", "cuda:0"], "ocr/{id}.json",
                    [], []),
    "frames": ("pipeline.frames", "ocr_nemotron_env", ["{id}", "--device", "cuda:0"], "films/{id}/text.json",
               ["ocr/{id}.json"], ["screen_text"]),
    "sounds": ("pipeline.sounds", "panns_env", ["{id}", "--device", "cuda"], "sounds/{id}.json", [], []),
    "validation_sounds": ("pipeline.validation_sounds", "florence_env", ["{id}", "--device", "cuda", "--hop", "3"],
                          "sound_events/{id}.json", [], []),
    "film": ("pipeline.films", "client", ["{films}", "--only", "{id}"], "films/{id}/film.json",
             ["transcripts/{id}.json", "films/{id}/text.json"], ["speech", "frames"]),
    "materials": ("pipeline.materials", "client", ["{films}", "--only", "{id}"], "materials/{id}/frames.tsv",
                  ["films/{id}/film.json"], ["film", "screen_text", "validation_sounds"]),
}
VALIDATION = ["validation_sounds", "materials"]  # run only with --validation

# Every step gets DATA as PAPER1_DATA and the repo on PYTHONPATH. Model caches stay under the shared root: PANNs reads
# $HOME/panns_data/, htdemucs $XDG_CACHE_HOME/torch/hub/checkpoints/, Hugging Face models $HF_HOME (HF_HOME and
# HF_HUB_OFFLINE are taken from the caller when set).
HOME = SHARED / "cache" / "serving-home"
LANE_ENV = {"HOME": str(HOME), "XDG_CACHE_HOME": str(HOME / ".cache"),
            "HF_HOME": os.environ.get("HF_HOME", str(SHARED / "hf-cache")),
            "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE", "1"),
            "PAPER1_DATA": str(DATA),
            "PYTHONPATH": os.pathsep.join(p for p in (str(ROOT), os.environ.get("PYTHONPATH")) if p)}


def run_step(step: str, vid: str, films_json: Path) -> tuple[str, float]:
    module, env, args, _, _, _ = STEPS[step]
    cmd = [PYTHON[env], "-m", module, *(a.format(id=vid, films=films_json) for a in args)]
    t0 = time.time()
    r = subprocess.run(cmd, cwd=ROOT, env={**os.environ, **LANE_ENV})
    return ("ran" if r.returncode == 0 else f"failed (exit {r.returncode})"), time.time() - t0


def extract(vid: str, steps: list[str], films_json: Path, force: bool) -> bool:
    ran, ok = set(), True
    for step in steps:
        _, _, _, output, needs, reads = STEPS[step]
        missing = [n.format(id=vid) for n in needs if not (DATA / n.format(id=vid)).exists()]
        dt = 0.0
        if (DATA / output.format(id=vid)).exists() and not force and not ran & set(reads):
            status = "exists"
        elif missing:
            status, ok = f"blocked (missing {', '.join(missing)})", False
        else:
            status, dt = run_step(step, vid, films_json)
            if status == "ran":
                ran.add(step)
            else:
                ok = False
        print(f"{vid}  {step:<17} {status:<8} {dt:7.1f} s", flush=True)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video_ids", nargs="*", help="default: every film of --films-json")
    ap.add_argument("--films-json", default="evals/evidence_bank_v1/films.json")
    ap.add_argument("--validation", action="store_true", help="also build the bank-validation inputs: "
                                                               + ", ".join(VALIDATION))
    ap.add_argument("--force", action="store_true", help="rebuild outputs that exist")
    a = ap.parse_args()
    steps = [s for s in STEPS if a.validation or s not in VALIDATION]
    films_json = Path(a.films_json)
    films_json = (films_json if films_json.exists() else ROOT / films_json).absolute()
    listed = [f["video_id"] for f in read_json(films_json)]
    if unknown := [v for v in a.video_ids if v not in listed]:
        sys.exit(f"not in {films_json}: {', '.join(unknown)}")
    print(f"DATA = {DATA}; steps {','.join(steps)}", flush=True)
    failed = [vid for vid in a.video_ids or listed if not extract(vid, steps, films_json, a.force)]
    if failed:
        sys.exit(f"failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
