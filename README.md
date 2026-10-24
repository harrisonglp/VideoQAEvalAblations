# Which evidence does a question about a film depend on?

When a model answers a question about a full-length film, which evidence does the answer depend on: what is said,
seen, read on screen or heard? Each film is turned into four machine-made inputs ("lanes"). A reader model answers the
questions of an evidence bank under controlled conditions: each lane given or withheld, the whole film or only
excerpts around the cited evidence, or the film with that evidence cut out. The conditions under which a question is
still answered show which evidence it needs. The question set (1,000 questions about 50 public-domain films) is in
[`evals/evidence_bank_v1/`](evals/evidence_bank_v1/README.md).

## Quick start

```bash
scripts/run_pipeline.sh --films mf2_17,mf2_9 --workers 2   # a few films, end to end
scripts/run_pipeline.sh                                    # every film of the bank
```

The script extracts the lanes, builds the items and checks the questions, runs the reader under every condition of the
main and the evidence-necessity experiment and writes `results/evidence_bank_v1/qwen36-27b/summary.json` and
`units.csv`; every step resumes where it stopped and logs to `logs/`. To see exactly what the reader receives for one
question under one condition:

```bash
python3 scripts/show_prompts.py w_tdv hh53_007 [--answer "a green vest"]
```

## Lanes

| Lane (input) | Module | Models | Output (under `data/`) |
|---|---|---|---|
| speech (D, "said") | `pipeline/speech.py` | htdemucs vocals stem → Silero VAD; Qwen3-ASR-1.7B on the original mix | `transcripts/<id>.json` |
| screen text (R, "read") | `pipeline/screen_text.py` | PP-OCRv6 medium (PaddleOCR, PyTorch engine), 1 frame/s | `ocr/<id>.json` |
| frames (V, "seen") | `pipeline/frames.py` | 256 frames sampled uniformly (ffmpeg); text masked with the screen-text reader | `films/<id>/frames_notext/` (+ `frames/`, `frames_textonly/`, `text.json`) |
| sounds (A, "heard") | `pipeline/sounds.py` | PANNs CNN14 (AudioSet), music 0.25, other labels 0.4 | `sounds/<id>.json` |

`pipeline/films.py` then writes `films/<id>/film.json` (title, year, duration, frame list, transcript lines), the
file every condition reads; the screen-text and sound lanes are added at run time. Each module's docstring states its
models and versions, every setting with its value and reason, inputs, outputs and command.

```bash
python -m pipeline.extract [<id> ...] [--validation]       # default: every film of evals/evidence_bank_v1/films.json
```

`extract` runs each lane in its own environment (table below), in dependency order. Finished steps are skipped
(`--force` rebuilds them), a step whose input is missing is reported as blocked, and one line per step and film gives
its status and wall time. `--validation` adds two outputs that no reader receives: `validation_sounds`
(`sound_events/<id>.json`, a second AudioSet tagger used to check `heard` evidence, deliberately different from the
sounds lane) and `materials` (`materials/<id>/*.tsv`, read by `pipeline.validate`).

## Reader, conditions and scoring

- **Reader:** Qwen3.6-27B (FP8) served by vLLM 0.28 (`configs/models/qwen36-27b.env`: 262,144-token context, fixed
  32 GiB KV cache, prefix caching, at most 4 sequences). Thinking is disabled, replies are limited to 128 tokens, and
  temperature, seed, top-p (0.95), top-k (20) and min-p (0) are sent with each request (`pipeline/run.py`).
- **Conditions** (`configs/conditions.json`, implemented in `pipeline/conditions.py`): question only; title only; the
  title with every combination of lanes, over the whole film and restricted to excerpts (cited evidence ± 60 s); all
  lanes with the cited evidence and every mention of the answer (numbers up to 12 excepted) cut out, and a placebo cut
  of the same length elsewhere; and the transcript with unmasked frames, over the whole film and in excerpts. What each one is, how they're named and how
  they're used to sort questions: [`configs/README.md`](configs/README.md).
- **Scoring:** rules first (`conditions.score_open`: normalised match against the answer and its aliases, numbers in
  digits or words within a tolerance, times, abstention for unanswerable questions); the answers the rules cannot
  decide are judged by the reader itself, without the film (`conditions.judge_prompt`, greedy).

| Experiment (`configs/experiments/`) | Questions | Conditions | Runs |
|---|---|---|---|
| `evidence_bank_v1.json` | all 1,000 | the 34 whole-film, excerpt and control conditions | 1, greedy |
| `evidence_bank_v1_necessity.json` | 300 (15 films, stratified) | evidence cut and placebo | 1, greedy |
| `evidence_bank_v1_reliability.json` | 200 (10 films, drawn at random) | 23 of the main conditions (not the whole-film lane pairs and triples) | 4, temperature 0.7 |
| `evidence_bank_v1_screen.json` | all 1,000 | question only, title only | 4, temperature 0.7 |
| `evidence_bank_v1_panel.json` | all 1,000, per evaluation reader | question only, title only, the 15 excerpt lane sets, the whole film with one lane or all lanes | 1, greedy, fixed judge |
| `evidence_bank_v1_panel_wholefilm.json` | all 1,000, per evaluation reader | the whole-film lane pairs and triples | 1, greedy, fixed judge |
| `evidence_bank_v1_smoke.json` | 40 (2 films) | as the panel experiment, to test a reader before the panel run | 1, greedy, fixed judge |

```bash
pipeline/serve.sh start configs/models/qwen36-27b.env
python -m pipeline.run --experiment configs/experiments/evidence_bank_v1.json --model qwen36-27b \
    --url http://127.0.0.1:8010/v1/chat/completions [--films ...] [--conditions ...]      # resumable
python -m pipeline.analyze configs/experiments/evidence_bank_v1.json qwen36-27b         # summary.json, units.csv
pipeline/serve.sh stop configs/models/qwen36-27b.env
```

The necessity experiment writes to the same results folder and is analysed with the main experiment. Before a run:
`python -m pipeline.items` (items), `python -m pipeline.validate evals/evidence_bank_v1/questions.jsonl` (questions
against the extracted inputs) and `python -m pipeline.fidelity` (how much of each question's evidence the inputs carry).

## Panel: several readers, labels verified or reassigned

The evaluation readers (`configs/models/`: `gemma4-31b`, `nemotron35-super-vl`, `command-a-plus`, `muse-spark-open`)
each run the panel experiment. Each profile pins the model's revision and sets its request options: thinking off, or
the lowest reasoning effort for `muse-spark-open`, whose reasoning cannot be turned off, and replies constrained to the
answer's JSON schema. Answers the rules cannot score are left for one fixed judge, so grading is the same for every
reader:

```bash
python -m pipeline.run --experiment configs/experiments/evidence_bank_v1_panel.json --model gemma4-31b \
    --url http://127.0.0.1:8011/v1/chat/completions --judge none                     # per reader, resumable
python -m pipeline.judge --experiment configs/experiments/evidence_bank_v1_panel.json --model gemma4-31b \
    --judge-model gemma4-31b --url http://127.0.0.1:8011/v1/chat/completions         # the judge, for every reader
python -m pipeline.analyze configs/experiments/evidence_bank_v1_panel.json gemma4-31b  # per reader
python -m pipeline.panel configs/experiments/panel_v1.json                             # results/evidence_bank_v1/panel/
```

`pipeline.panel` (`configs/experiments/panel_v1.json`) types every question from the readers' excerpt ablations,
verifies or reassigns its synthetic label to the evidence source the readers do best with (rule `best`), and
measures leave-one-model-out agreement: labels reassigned from one, two or three readers against the measured type
of a reader that took no part, before and after, averaged over the folds, with bootstrap intervals over films. Folds
with two training readers are also reported per three-reader panel. It also writes each reader's accuracy profile,
the single-lane matrix, the agreement between readers (the ceiling) and a routing check. Outputs: `summary.json`,
`labels.csv`, `folds.csv`, `profiles.csv`. Two variants write next to it: `panel_v1_threshold.json` (rule
`threshold`, into `panel_threshold/`) and `panel_v1_without_muse.json` (the three other readers, into
`panel_without_muse/`). The whole-film pairs and triples (`evidence_bank_v1_panel_wholefilm.json`) were not run for
the panel.

Results (`results/evidence_bank_v1/panel/summary.json`): with two training readers, the synthetic labels agree with
the held-out reader's measured type on 65.5% of the questions it types (95% CI 62.9-67.9), and the reassigned labels
on 72.8% (70.7-74.9), averaged over the 12 folds (four three-reader panels); 70.3% with one training reader and 76.1%
with three.

## Models

| Model | Version | File | sha256 |
|---|---|---|---|
| htdemucs (Demucs v4) | demucs 4.0.1, weights 955717e8 | `955717e8-8726e21a.th` | `8726e21a993978c7ba086d3872e7608d7d5bfca646ca4aca459ffda844faa8b4` |
| Silero VAD | silero-vad 6.2.3 | `silero_vad.jit` | `e1122837f4154c511485fe0b9c64455f7b929c96fbb8d79fbdb336383ebd3720` |
| Qwen/Qwen3-ASR-1.7B | snapshot 7278e1e70fe206f11671096ffdd38061171dd6e5 | `model-00001-of-00002.safetensors` | `a4cd1f1a04d90b757dc7f7dd26254e69a013b19e80efe590a83c6a3bde8608d6` |
| | | `model-00002-of-00002.safetensors` | `6e0b9d9e09e2e0238e7ef3cc8a484ab387e91b90f1900bedf88bc92d7929ccfc` |
| PP-OCRv6_medium_det | PaddleOCR 3.7.0 official model, safetensors | `model.safetensors` | `bd393266c02e1a680b1b34c301d5d0d81e6290440b7f8ab0f5d5032276b17eb1` |
| PP-OCRv6_medium_rec | PaddleOCR 3.7.0 official model, safetensors | `model.safetensors` | `5f43c16f2a684b1d2284662178bdb604febd3d6bfdb5ca73828d08d0f7c0c3e9` |
| PANNs CNN14 (AudioSet) | panns_inference 0.1.1 | `Cnn14_mAP=0.431.pth` | `0dc499e40e9761ef5ea061ffc77697697f277f6a960894903df3ada000e34b31` |
| MIT/ast-finetuned-audioset-10-10-0.4593 (validation only) | snapshot f826b80d28226b62986cc218e5cec390b1096902 | `model.safetensors` | `ae0c1e2ad4e1381d851fa9bf298ba13ebc9c5a914cdee2dbe427a6583869924d` |
| Qwen/Qwen3.6-27B-FP8 (reader) | snapshot e89b16ebf1988b3d6befa7de50abc2d76f26eb09 | 80 files | `envs/models.sha256` |
| google/gemma-4-31B-it (panel reader, judge) | snapshot 842da3794eaa0b77d5f08bae87a17459d91ff475 | all files | `envs/panel_models/gemma4-31b.sha256` |
| nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4 (panel reader) | snapshot 59b8721646c38111a242da18218859fb2b454322 | all files | `envs/panel_models/nemotron35-super-vl.sha256` |
| CohereLabs/command-a-plus-05-2026-w4a4 (panel reader) | snapshot ebd2d72c0f9a84389c6b057136d38c435aea2700 | all files | `envs/panel_models/command-a-plus.sha256` |
| meta-models/Muse-Spark-Open-816B-A42B-NVFP4 (panel reader) | snapshot 4239e4db19376904f2e82c167e78f70abb5455fe | all files | `envs/panel_models/muse-spark-open.sha256` |

`envs/models.sha256` lists every model file (with configs, tokenizers and preprocessors) by its path under the shared
root: `cd /data/cinesage && sha256sum -c <repo>/envs/models.sha256`. The lanes read the models offline from the caches
that `pipeline.extract` sets (set the same variables to run a lane module directly): `HF_HOME=/data/cinesage/hf-cache`
and `HF_HUB_OFFLINE=1` (Qwen3-ASR, AST); `HOME=/data/cinesage/cache/serving-home` (PANNs in `panns_data/`) with
`XDG_CACHE_HOME=$HOME/.cache` (htdemucs in `torch/hub/checkpoints/`); `PADDLE_PDX_CACHE_HOME=/data/cinesage/cache/paddlex`
(PaddleOCR, with the other variables of `extract.ENV_EXTRA`). The panel readers' lists in `envs/panel_models/` are
relative to each model's snapshot folder: `cd <snapshot> && sha256sum -c <repo>/envs/panel_models/<profile>.sha256`.

## Environments and hardware

| Environment | Python | Used by | Packages |
|---|---|---|---|
| `client` | 3.10.20 | `extract`, `films`, `materials`, `items`, `validate`, `fidelity`, `run`, `analyze`, `page_cache` (standard library only) | `envs/client.txt` |
| `asr_qwen_env` | 3.11.15 | `speech` (demucs, silero-vad, qwen-asr, PyTorch) | `envs/asr_qwen_env.txt` |
| `ocr_env` | 3.12.3 | `screen_text`, `frames` (PaddleOCR, OpenCV, Pillow); `scripts/extract_check.py` | `envs/ocr_env.txt` |
| `panns_env` | 3.12.3 | `sounds` (panns_inference) | `envs/panns_env.txt` |
| `florence_env` | 3.10.20 | `validation_sounds` (Transformers) | `envs/florence_env.txt` |
| `vllm` | 3.11.15 | `serve.sh` (vLLM 0.28.0) | `envs/vllm.txt` |
| `vllm32` | 3.11.17 | `serve.sh` for the panel readers (vLLM 0.32.0, CUDA 12.9 build) | `envs/vllm32.txt` |
| `muse` | container | `serve.sh` for `muse-spark-open` (Meta's vLLM container, CUDA 13.0) | `envs/muse.txt` |

The environments live in `$SHARED_ROOT/envs/` (default `/data/cinesage/envs/`); `pipeline.extract` finds them there
(override one with `PAPER1_PY_CLIENT`, `PAPER1_PY_SPEECH`, `PAPER1_PY_OCR`, `PAPER1_PY_SOUNDS` or
`PAPER1_PY_VALIDATION`). Each `envs/<environment>.txt` is its `pip freeze`, headed by its Python version; rebuild one
with `python<version> -m venv <env> && <env>/bin/pip install -r envs/<env>.txt --extra-index-url
https://download.pytorch.org/whl/cu130`. Every lane decodes the film with the system's ffmpeg and ffprobe
(6.1.1-3ubuntu5+esm8); another version can change the frame JPEGs and the decoded audio. Hardware: NVIDIA DGX Spark
(GB10, aarch64, CUDA 13, driver 580.82.09), Ubuntu 24.04.3. On it, with three films extracted at a time, a film of 75
to 90 minutes takes (medians) 14 min for speech, 10 min for screen text, 1 min for frames, under 1 min for sounds and
2 min for the validation tags. The reader answers the main experiment at about 23 questions a minute; an evidence-cut
prompt (about 62,000 tokens, little of it shared with other questions) takes minutes.

## Provenance and re-runs

Every lane output and `film.json` ends with a `provenance` object: the module that wrote it (`module`, and `sha256`,
the first 16 hex characters of the module file's sha256), its `model`, and the `settings` that affect the output,
command-line values included. To verify a re-run, extract into a fresh root and compare it with the reference:

```bash
PAPER1_DATA=/tmp/rerun python -m pipeline.extract <id> ...
/data/cinesage/envs/ocr_env/bin/python scripts/extract_check.py data /tmp/rerun <id> ... --report /tmp/rerun/check.json
```

For each film and lane it reports whether the outputs are identical, how closely they agree if not, and whether their
provenance matches. Frame sampling, `film.json` and `materials` are deterministic for the same inputs and ffmpeg, as
are htdemucs' random shifts of each separation piece (seeded per film, `speech.SHIFT_SEED`). The models run on the GPU
(Qwen3-ASR in bfloat16), whose kernels are not bitwise reproducible, so texts and scores can differ slightly between
runs and a segment, read or label near its threshold can flip.

## Layout

```
pipeline/   speech, screen_text, frames, sounds (lanes); validation_sounds; films, materials; extract (entry point);
            items, conditions, run, judge, analyze, panel, validate, fidelity; common; page_cache; serve.sh
configs/    conditions.json, experiments/, models/ (vLLM profile)
evals/      evidence_bank_v1/ (questions, films, README, FORMAT)
envs/       package lists of the environments; models.sha256
scripts/    run_pipeline.sh, show_prompts.py, extract_check.py
tests/      unit tests (python3 -m unittest discover tests)
data/       generated inputs (rebuildable; PAPER1_DATA=<dir> builds into another root)
results/    run records and analyses
```

Shared resources are read from `/data/cinesage` (override with `SHARED_ROOT`): `raw/` (source videos), `envs/`,
`hf-cache/`, `cache/serving-home/`, `cache/paddlex/`. The reader server accepts local images only from under this
root, so the frames in `data/` (or `PAPER1_DATA`) must resolve to paths inside it.
