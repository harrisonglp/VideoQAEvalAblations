"""Judge the answers that the scoring rules could not decide, with one fixed judge model for every reader.

    python -m pipeline.judge --experiment configs/experiments/evidence_bank_v1_panel.json --model gemma4-31b \
        --judge-model gemma4-31b --url http://127.0.0.1:8011/v1/chat/completions [--concurrency 8]

Readers run with `pipeline.run --judge none` leave such answers with `pending_judge`. This pass sends each of them to
the judge model (served at --url), without the film, with the same prompt and decoding as the inline judge
(conditions.judge_prompt, run.judge), and rewrites the reader's result files in place (atomically): `correct`, the
verdict in `judged`, and `judge_model`. A call that fails five times stays pending for the next pass. Using one judge
for every reader keeps grading the same across readers, so their scores differ only by what they answered.
"""
import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor

from .common import DATA, RESULTS, read_json, read_jsonl
from .run import judge


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--model", required=True, help="the reader whose answers are judged")
    ap.add_argument("--judge-model", required=True, help="served name of the judge (configs/models/<name>.env)")
    ap.add_argument("--url", required=True, help="the judge's chat-completions URL")
    ap.add_argument("--concurrency", type=int, default=8)
    a = ap.parse_args()
    exp = read_json(a.experiment)
    items = {it["id"]: it for it in read_jsonl(DATA / "items" / f"{exp['items']}.jsonl")}
    folder = RESULTS / exp.get("name", exp["items"]) / a.model
    total = left = 0
    for path in sorted(folder.glob("*.jsonl")):
        recs = read_jsonl(path)
        todo = [r for r in recs if r.get("pending_judge") and r.get("correct") is None]
        if not todo:
            continue

        def one(r):
            score, verdict = judge(a.url, a.judge_model, items[r["id"]], r["pred"])
            if score is not None:
                r.update(correct=score, judged=verdict, judge_model=a.judge_model)
                r.pop("pending_judge")
            return score is not None

        with ThreadPoolExecutor(a.concurrency) as pool:
            ok = sum(pool.map(one, todo))
        tmp = path.with_suffix(".jsonl.tmp")
        with open(tmp, "w") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in recs)
        os.replace(tmp, path)
        total += ok
        left += len(todo) - ok
        print(f"{path.stem}: judged {ok} of {len(todo)}", flush=True)
    print(f"{a.model}: {total} judged by {a.judge_model}, {left} still pending")


if __name__ == "__main__":
    main()
