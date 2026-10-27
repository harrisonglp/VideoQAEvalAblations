# Which evidence does a question about a film depend on?

When a model answers a question about a full-length film, which evidence does the answer depend on: what is said,
seen, read on screen or heard? Each film is turned into four machine-made inputs ("lanes"). A reader model answers the
questions of an evidence bank under controlled conditions: each lane given or withheld, the whole film or only
excerpts around the cited evidence, or the film with that evidence cut out. The conditions under which a question is
still answered show which evidence it needs. The question set (1,000 questions about 50 public-domain films) is in
[`evals/evidence_bank_v1/`](evals/evidence_bank_v1/README.md).

## How the experiment works

The experiment separates three things that are easy to confuse:

1. **The synthetic label** is the evidence type assigned when the question was written, such as `dialogue`,
   `visual`, `dialogue or text` or `dialogue and visual`. It lives in
   [`evals/evidence_bank_v1/questions.jsonl`](evals/evidence_bank_v1/questions.jsonl).
2. **A reader's measured type** comes from its answers under the 16 excerpt cells: title only plus every non-empty
   combination of speech (D), frames (V), screen text (R) and sounds (A). For example, a question is measured as
   `visual` if frames alone are the best sufficient source.
3. **The panel label** pools the ablations of several readers. It verifies the synthetic label when they agree,
   reassigns it when the pooled measured type differs, and keeps it unchanged when the models cannot determine a
   type.

The full path through the repository is:

1. The bank defines 1,000 questions, their reference answers, synthetic labels and cited evidence intervals.
2. `pipeline.extract` turns each of the 50 films into four independent evidence lanes under `data/`.
3. `pipeline.items` turns the bank into the normalized items consumed by the runner.
4. `pipeline.run` sends every item to one reader under every condition and writes one JSONL file per condition.
5. Rule-based scoring handles exact answers, aliases, numbers, times and abstentions. `pipeline.judge` sends only
   undecided short answers to the fixed judge.
6. `pipeline.analyze` summarizes one reader. `pipeline.panel` pools readers, reassigns labels and performs the
   leave-one-model-out test used in the paper.

In leave-one-model-out evaluation, labels are reassigned from one, two or three readers and compared with the
measured type of a reader that was not used during reassignment. “Before” is agreement between the synthetic label
and the held-out reader; “after” is agreement after reassignment. The paper's main result trains on two readers and
tests on the third, averaged over 12 folds (four three-reader panels).

## Where everything is

| Path | What it contains |
|---|---|
| [`evals/evidence_bank_v1/`](evals/evidence_bank_v1/README.md) | The released benchmark: 1,000 questions, 50-film manifest, writing/review protocol and examples |
| [`evals/evidence_bank_v1/FORMAT.md`](evals/evidence_bank_v1/FORMAT.md) | Field-by-field question schema, evidence intervals, evidence-type vocabulary and scoring rules |
| [`configs/README.md`](configs/README.md) | How condition names work (`td`, `w_tdvra`, and so on), how measured types are assigned and which conditions each experiment runs |
| `configs/conditions.json` | Machine-readable definitions of the inputs and transforms in every condition |
| `configs/experiments/` | Lists of questions, films, conditions, repeats and panel readers for each experiment |
| `configs/models/` | Pinned reader checkpoints, ports, context limits, KV-cache sizes and server/request options |
| `pipeline/` | Extraction, prompt construction, running, judging and analysis code; the main entry points are listed below |
| `results/evidence_bank_v1/` | Final reader answers, per-reader analyses, panel labels, folds, profiles and sensitivity analyses |
| `envs/` | Reproducible package lists and SHA-256 manifests for lane and reader models |
| `scripts/` | End-to-end runner, prompt viewer and extraction comparison tool |
| `tests/` | Unit tests for conditions, scoring and panel assignment (`python3 -m unittest discover tests`) |
| `data/` | Generated evidence lanes. It is intentionally not tracked; rebuild it from the source films or point `PAPER1_DATA` at another copy |

The most useful pipeline modules are:

| Module | Responsibility |
|---|---|
| `pipeline/extract.py` | Runs all lane extractors in dependency order |
| `pipeline/speech.py`, `screen_text.py`, `frames.py`, `sounds.py` | Build the four evidence lanes |
| `pipeline/items.py` | Normalizes the question bank into runnable items |
| `pipeline/conditions.py` | Builds prompts, applies ablations, parses answers and implements rule scoring |
| `pipeline/run.py` | Runs one reader and writes resumable condition JSONL records |
| `pipeline/judge.py` | Applies the same fixed judge to undecided answers from every reader |
| `pipeline/analyze.py` | Produces one reader's accuracy, measured types, coverage and related statistics |
| `pipeline/panel.py` | Pools readers, verifies/reassigns labels, bootstraps films, computes leave-one-model-out agreement and routing |
| `pipeline/serve.sh` | Starts and stops a reader's vLLM server from its model profile |

## Quick start

This command exercises extraction and one reader end to end. It expects the source videos and model caches under
`$SHARED_ROOT` (default `/data/cinesage`); the public-domain videos are not stored in git. The final four-reader panel
uses the separate commands in “Panel: several readers, labels verified or reassigned” below.

```bash
scripts/run_pipeline.sh --films mf2_17,mf2_9 --workers 2   # a few films, end to end
scripts/run_pipeline.sh                                    # every film of the bank
```

The script extracts the lanes, builds the items and checks the questions, runs the reader under every condition of the
main and the evidence-necessity experiment and writes `results/evidence_bank_v1/gemma4-31b/summary.json` and
`units.csv`; every step resumes where it stopped and logs to `logs/`. To see exactly what the reader receives for one
question under one condition:

```bash
python3 scripts/show_prompts.py w_tdv hh53_007 [--answer "a green vest"]
```

## Lanes

| Lane (input) | Module | Models | Output (under `data/`) |
|---|---|---|---|
| speech (D, "said") | `pipeline/speech.py` | htdemucs vocals stem → Silero VAD; Parakeet TDT 0.6B v3 on the original mix | `transcripts/<id>.json` |
| screen text (R, "read") | `pipeline/screen_text.py` | Nemotron OCR v2 (English, line level), 1 frame/s | `ocr/<id>.json` |
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

- **Readers:** the evaluation readers of `configs/models/` (Gemma 4 31B, Nemotron 3.5 Super-VL, Command A+, Muse
  Spark Open), each served by vLLM from a profile that pins its revision (131,072-token context, fixed KV cache,
  prefix caching). Thinking is disabled (Muse Spark Open runs at its lowest reasoning effort), replies are constrained
  to the answer's JSON schema and limited to 128 tokens (1,024 for Muse Spark Open), and temperature, seed, top-p
  (0.95), top-k (20) and min-p (0) are sent with each request (`pipeline/run.py`).
- **Conditions** (`configs/conditions.json`, implemented in `pipeline/conditions.py`): question only; title only; the
  title with every combination of lanes, over the whole film and restricted to excerpts (cited evidence ± 60 s); all
  lanes with the cited evidence and every mention of the answer (numbers up to 12 excepted) cut out, and a placebo cut
  of the same length elsewhere; and the transcript with unmasked frames, over the whole film and in excerpts. What each one is, how they're named and how
  they're used to sort questions: [`configs/README.md`](configs/README.md).
- **Scoring:** rules first (`conditions.score_open`: normalised match against the answer and its aliases, numbers in
  digits or words within a tolerance, times, abstention for unanswerable questions); the answers the rules cannot
  decide are judged without the film (`conditions.judge_prompt`, greedy), by the reader itself (`--judge self`) or,
  for the panel, by one fixed judge (`pipeline.judge`).

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
pipeline/serve.sh start configs/models/gemma4-31b.env
python -m pipeline.run --experiment configs/experiments/evidence_bank_v1.json --model gemma4-31b \
    --url http://127.0.0.1:8011/v1/chat/completions [--films ...] [--conditions ...]      # resumable
python -m pipeline.analyze configs/experiments/evidence_bank_v1.json gemma4-31b         # summary.json, units.csv
pipeline/serve.sh stop configs/models/gemma4-31b.env
```

The necessity experiment writes to the same results folder and is analysed with the main experiment. Before a run:
`python -m pipeline.items` (items), `python -m pipeline.validate evals/evidence_bank_v1/questions.jsonl` (questions
against the extracted inputs) and `python -m pipeline.fidelity` (how much of each question's evidence the inputs carry).

## How results are organized

The committed final results have this structure:

```text
results/evidence_bank_v1/
├── validate.json                  question and evidence validation report
├── fidelity.json / fidelity.csv   whether cited evidence reached the generated lanes
├── gemma4-31b/                    one folder per evaluation reader
│   ├── q.jsonl                    one answer record per question for this condition
│   ├── t.jsonl
│   ├── ...                        the other 20 panel conditions
│   ├── summary.json               reader-level accuracies, measured types and coverage
│   └── units.csv                  per-question scores and measured types
├── nemotron35-super-vl/
├── command-a-plus/
├── muse-spark-open/
├── panel/                         main “best” reassignment rule
│   ├── summary.json               headline results, CIs, ceilings, transitions and routing
│   ├── labels.csv                 synthetic, per-reader and final panel type per question
│   ├── folds.csv                  every leave-one-model-out fold
│   └── profiles.csv               reader × condition accuracy
├── panel_threshold/               sensitivity analysis with the threshold rule
└── panel_without_muse/            three-reader sensitivity analysis
```

Each `<reader>/<condition>.jsonl` line is one `(condition, question, repeat)` call. Important fields are:

| Field | Meaning |
|---|---|
| `condition`, `model`, `film`, `id`, `rep` | Identity of the call |
| `gold` | Reference answer |
| `text` | Raw constrained model reply |
| `pred` | Parsed short answer used for scoring |
| `correct` | Final score, `1` or `0` |
| `judged`, `judge_model` | Present when rule scoring deferred to the fixed judge |
| `n_frames`, `frames` | Number and kind of frames in the request |
| `latency_s`, `prompt_tokens`, `cached_tokens` | Runtime and token accounting |
| `request_sha256` | Fingerprint of the exact request; `scripts/show_prompts.py` rebuilds and prints it |
| `error` | Present only if the request failed |

The question-only, title-only and five whole-film conditions contain 1,000 records each. The 15 excerpt conditions
contain 798 because prior and unanswerable questions do not cite evidence intervals. A complete panel reader therefore
has 7 × 1,000 + 15 × 798 = 18,970 records across 22 files.

For the paper's headline numbers, read `panel/summary.json` → `leave_one_model_out.train_2`. For question-level
label changes, use `panel/labels.csv`. For each held-out/train combination, use `panel/folds.csv`. For the evidence
profile plotted in the paper, use `panel/profiles.csv` and `panel/summary.json` → `single_lane_matrix`.

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

Final results are in `results/evidence_bank_v1/`. With two training readers, the synthetic labels agree with the
held-out reader's measured type on 66.4% of the questions it types (95% CI 63.8--68.7), and the reassigned labels on
72.5% (70.6--74.4), averaged over 12 folds (four three-reader panels). Reassigning from one reader reaches 70.1%;
from three, 74.3%. The threshold-rule sensitivity gives 71.9% with two readers, and the three-reader panel without
Muse gives 75.0% (from 67.5% before reassignment).


## Models

| Model | Version | File | sha256 |
|---|---|---|---|
| htdemucs (Demucs v4) | demucs 4.0.1, weights 955717e8 | `955717e8-8726e21a.th` | `8726e21a993978c7ba086d3872e7608d7d5bfca646ca4aca459ffda844faa8b4` |
| Silero VAD | silero-vad 6.2.3 | `silero_vad.jit` | `e1122837f4154c511485fe0b9c64455f7b929c96fbb8d79fbdb336383ebd3720` |
| nvidia/parakeet-tdt-0.6b-v3 | snapshot 541d1f99c6b0c3cd0b11a95167540bb8edefd82b | `model.safetensors` | `3a2026366188c8c68598edbbff92f8d11590a08e0ae2e6775544e7b07d6a5e11` |
| nvidia/nemotron-ocr-v2 (`v2_english`) | snapshot 0e83e83f17943524b90afa6c0fd82ac2bc1a40ca | `detector.pth`, `recognizer.pth`, `relational.pth` | `envs/models.sha256` |
| PANNs CNN14 (AudioSet) | panns_inference 0.1.1 | `Cnn14_mAP=0.431.pth` | `0dc499e40e9761ef5ea061ffc77697697f277f6a960894903df3ada000e34b31` |
| MIT/ast-finetuned-audioset-10-10-0.4593 (validation only) | snapshot f826b80d28226b62986cc218e5cec390b1096902 | `model.safetensors` | `ae0c1e2ad4e1381d851fa9bf298ba13ebc9c5a914cdee2dbe427a6583869924d` |
| google/gemma-4-31B-it (panel reader, judge) | snapshot 842da3794eaa0b77d5f08bae87a17459d91ff475 | all files | `envs/panel_models/gemma4-31b.sha256` |
| nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4 (panel reader) | snapshot 59b8721646c38111a242da18218859fb2b454322 | all files | `envs/panel_models/nemotron35-super-vl.sha256` |
| CohereLabs/command-a-plus-05-2026-w4a4 (panel reader) | snapshot ebd2d72c0f9a84389c6b057136d38c435aea2700 | all files | `envs/panel_models/command-a-plus.sha256` |
| meta-models/Muse-Spark-Open-816B-A42B-NVFP4 (panel reader) | snapshot 4239e4db19376904f2e82c167e78f70abb5455fe | all files | `envs/panel_models/muse-spark-open.sha256` |

`envs/models.sha256` lists every model file (with configs, tokenizers and preprocessors) by its path under the shared
root: `cd /data/cinesage && sha256sum -c <repo>/envs/models.sha256`. The lanes read the models offline from the caches
that `pipeline.extract` sets (set the same variables to run a lane module directly): `HF_HOME=/data/cinesage/hf-cache`
and `HF_HUB_OFFLINE=1` (Parakeet, Nemotron OCR, AST); `HOME=/data/cinesage/cache/serving-home` (PANNs in
`panns_data/`) with `XDG_CACHE_HOME=$HOME/.cache` (htdemucs in `torch/hub/checkpoints/`). The panel readers' lists in `envs/panel_models/` are
relative to each model's snapshot folder: `cd <snapshot> && sha256sum -c <repo>/envs/panel_models/<profile>.sha256`.

## Environments and hardware

| Environment | Python | Used by | Packages |
|---|---|---|---|
| `client` | 3.10.20 | `extract`, `films`, `materials`, `items`, `validate`, `fidelity`, `run`, `analyze`, `page_cache` (standard library only) | `envs/client.txt` |
| `asr_parakeet_env` | 3.12.3 | `speech` (demucs, silero-vad, Transformers, PyTorch) | `envs/asr_parakeet_env.txt` |
| `ocr_nemotron_env` | 3.12.3 | `screen_text`, `frames` (nemotron-ocr, OpenCV, Pillow); `scripts/extract_check.py` | `envs/ocr_nemotron_env.txt` |
| `panns_env` | 3.12.3 | `sounds` (panns_inference) | `envs/panns_env.txt` |
| `florence_env` | 3.10.20 | `validation_sounds` (Transformers) | `envs/florence_env.txt` |
| `vllm32` | 3.11.17 | `serve.sh` for the panel readers (vLLM 0.32.0, CUDA 12.9 build) | `envs/vllm32.txt` |
| `muse` | container | `serve.sh` for `muse-spark-open` (Meta's vLLM container, CUDA 13.0) | `envs/muse.txt` |

The environments live in `$SHARED_ROOT/envs/` (default `/data/cinesage/envs/`); `pipeline.extract` finds them there
(override one with `PAPER1_PY_CLIENT`, `PAPER1_PY_SPEECH`, `PAPER1_PY_OCR`, `PAPER1_PY_SOUNDS` or
`PAPER1_PY_VALIDATION`). Each `envs/<environment>.txt` is its `pip freeze`, headed by its Python version; rebuild one
with `python<version> -m venv <env> && <env>/bin/pip install -r envs/<env>.txt --extra-index-url
https://download.pytorch.org/whl/cu130`. `nemotron-ocr` is not on PyPI: install it from the `nemotron-ocr/` folder of
the Nemotron OCR v2 snapshot with `pip install --no-build-isolation .` after the rest, with `nvcc` (CUDA 13) on `PATH`,
`hatchling` installed and the Python headers on `CPATH`; its build script asks for C++17, and torch 2.14's headers need
`-std=c++20` there. Every lane decodes the film with the system's ffmpeg and ffprobe
(6.1.1-3ubuntu5+esm8); another version can change the frame JPEGs and the decoded audio. Hardware: the lanes on an
NVIDIA DGX Spark (GB10, aarch64, CUDA 13, driver 580.82.09), Ubuntu 24.04.3; the panel readers on NVIDIA B200 GPUs (one
per reader, four for Muse Spark Open).

## Provenance and re-runs

Every lane output and `film.json` ends with a `provenance` object: the module that wrote it (`module`, and `sha256`,
the first 16 hex characters of the module file's sha256), its `model`, and the `settings` that affect the output,
command-line values included. To verify a re-run, extract into a fresh root and compare it with the reference:

```bash
PAPER1_DATA=/tmp/rerun python -m pipeline.extract <id> ...
/data/cinesage/envs/ocr_nemotron_env/bin/python scripts/extract_check.py data /tmp/rerun <id> ... --report /tmp/rerun/check.json
```

For each film and lane it reports whether the outputs are identical, how closely they agree if not, and whether their
provenance matches. Frame sampling, `film.json` and `materials` are deterministic for the same inputs and ffmpeg, as
are htdemucs' random shifts of each separation piece (seeded per film, `speech.SHIFT_SEED`). The models run on the GPU
(Nemotron OCR in mixed precision), whose kernels are not bitwise reproducible, so texts and scores can differ slightly
between runs and a segment, read or label near its threshold can flip.

## Shared resources

Shared resources are read from `/data/cinesage` (override with `SHARED_ROOT`): `raw/` (source videos), `envs/`,
`hf-cache/`, `cache/serving-home/`. The reader server accepts local images only from under this
root, so the frames in `data/` (or `PAPER1_DATA`) must resolve to paths inside it.
