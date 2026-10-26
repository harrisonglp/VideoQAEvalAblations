#!/bin/bash
# End to end on the evidence bank: extract the four inputs of each film, build the items and check the questions, run
# the reader under every condition of the main and the evidence-necessity experiment, and analyse.
#
#   scripts/run_pipeline.sh [--films all|id,id,...] [--workers N]
#
# Steps (progress in logs/pipeline.log; the output of steps 1-3 and 5 in logs/<step>.log, of step 4 in logs/run.log
# and logs/serve.log):
#   1. extract   speech, on-screen text, frames and sounds of each film, and the validation materials
#                (pipeline.extract --validation), N films at a time (default 3)
#   2. items     data/items/evidence_bank_v1.jsonl (pipeline.items)
#   3. checks    the questions against the extracted inputs (pipeline.validate) and how much of each question's
#                evidence the inputs carry (pipeline.fidelity)
#   4. reader    the main experiment and the evidence-necessity experiment (pipeline.run), three conditions at a time,
#                with the reader server started for this step and stopped when the script exits (pipeline/serve.sh)
#   5. analysis  results/evidence_bank_v1/<model>/summary.json and units.csv (pipeline.analyze)
# Every step resumes where it stopped: finished extraction steps and scored answers are skipped.
set -euo pipefail
cd "$(dirname "$0")/.."
SHARED=${SHARED_ROOT:-/data/cinesage}
PY=$SHARED/envs/client/bin/python
MODEL=gemma4-31b
PROFILE=configs/models/$MODEL.env
URL=http://127.0.0.1:$(sed -n 's/^PORT=//p' "$PROFILE")/v1/chat/completions

FILMS=all
WORKERS=3
while [ $# -gt 0 ]; do
  case $1 in
    --films) FILMS=$2; shift 2 ;;
    --workers) WORKERS=$2; shift 2 ;;
    *) echo "usage: $0 [--films all|id,id,...] [--workers N]" >&2; exit 2 ;;
  esac
done
if [ "$FILMS" = all ]; then
  IDS=$("$PY" -c "import json; print(' '.join(f['video_id'] for f in json.load(open('evals/evidence_bank_v1/films.json'))))")
  SUBSET=()
else
  IDS=${FILMS//,/ }
  SUBSET=(--films "$FILMS")
fi

mkdir -p logs results/evidence_bank_v1
log() { echo "$(date '+%F %T') $*" | tee -a logs/pipeline.log; }
conditions() { "$PY" -c "import json, sys; print('\n'.join(json.load(open(sys.argv[1]))['conditions']))" "$1"; }

log "1/5 extract: $(wc -w <<< "$IDS") films, $WORKERS at a time"
tr ' ' '\n' <<< "$IDS" | xargs -P "$WORKERS" -I{} "$PY" -m pipeline.extract {} --validation >> logs/extract.log 2>&1

log "2/5 items"
"$PY" -m pipeline.items >> logs/items.log 2>&1

log "3/5 checks"
"$PY" -m pipeline.validate evals/evidence_bank_v1/questions.jsonl "${SUBSET[@]}" \
  --report results/evidence_bank_v1/validate.json >> logs/checks.log 2>&1
"$PY" -m pipeline.fidelity "${SUBSET[@]}" >> logs/checks.log 2>&1

log "4/5 reader"
trap 'bash pipeline/serve.sh stop "$PROFILE" >> logs/serve.log 2>&1' EXIT
bash pipeline/serve.sh start "$PROFILE" >> logs/serve.log 2>&1
for exp in evidence_bank_v1 evidence_bank_v1_necessity; do
  cfg=configs/experiments/$exp.json
  conditions "$cfg" | xargs -P 3 -I{} "$PY" -m pipeline.run --experiment "$cfg" --model "$MODEL" --url "$URL" \
    --conditions {} "${SUBSET[@]}" >> logs/run.log 2>&1
  log "  $exp done"
done

log "5/5 analysis"
"$PY" -m pipeline.analyze configs/experiments/evidence_bank_v1.json "$MODEL" >> logs/analysis.log 2>&1
log "done: results/evidence_bank_v1/$MODEL/summary.json"
