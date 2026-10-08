"""Show exactly what the reader receives for one question under one condition: the request parameters (one line per
repeat of the experiment, with the request's fingerprint as pipeline.run stores it in `request_sha256`), the system
message and the user message, with each image part shown as <image: films/<id>/<frame dir>/<file>>. With --answer,
also the judge prompt for that answer and its rule score (score_open; "judge" when the rules leave it to the judge).

    python3 scripts/show_prompts.py <condition> <question id> [--experiment configs/experiments/evidence_bank_v1.json]
        [--model qwen36-27b] [--answer "..."]

Films are read from DATA (data/, or $PAPER1_DATA) as pipeline.run loads them; the fingerprint matches a stored record's
only when the data root and the model name are the run's.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline import conditions as C, run
from pipeline.common import CONFIGS, DATA, EVALS, read_json, read_jsonl
from pipeline.items import SET, open_items


def load_film(fid, cond):
    path = DATA / "films" / fid / "film.json"
    if path.exists():
        film = read_json(path)
    elif not any(x in cond["inputs"] for x in "DVRA"):
        f = next(f for f in read_json(EVALS / SET / "films.json") if f["video_id"] == fid)
        film = {"id": fid, "title": f["title"], "year": f["year"]}
    else:
        raise SystemExit(f"{fid}: no {path} (set PAPER1_DATA to the data root)")
    if (ocr := DATA / "ocr" / f"{fid}.json").exists():
        film["screens"] = read_json(ocr)["screens"]
    if (snd := DATA / "sounds" / f"{fid}.json").exists():
        film["sounds"] = read_json(snd)["events"]
    return film


def shown(part):
    if part["type"] == "text":
        return part["text"]
    path = Path(part["image_url"]["url"].removeprefix("file://"))
    return f"<image: {path.relative_to(DATA).as_posix()}>"


def main():
    ap = argparse.ArgumentParser(description="Show the reader's prompt for one question under one condition.")
    ap.add_argument("condition")
    ap.add_argument("id", help="question id, e.g. ns37_002")
    ap.add_argument("--experiment", default=str(CONFIGS / "experiments" / "evidence_bank_v1.json"))
    ap.add_argument("--model", default="qwen36-27b", help="served model name, as given to pipeline.run")
    ap.add_argument("--answer", help="an answer to score and to show the judge prompt for")
    a = ap.parse_args()
    exp = read_json(a.experiment)
    conds = read_json(CONFIGS / "conditions.json")["conditions"]
    if a.condition not in conds:
        raise SystemExit(f"{a.condition}: no such condition in configs/conditions.json")
    items = {it["id"]: it for it in open_items(read_jsonl(EVALS / SET / "questions.jsonl"))}
    if a.id not in items:
        raise SystemExit(f"{a.id}: no such question in evals/{SET}/questions.jsonl")
    item, name = items[a.id], Path(a.experiment).name
    if a.condition not in exp["conditions"]:
        print(f"note: {name} does not run {a.condition}", file=sys.stderr)
    if exp.get("films") and item["film"] not in exp["films"]:
        print(f"note: {name} does not include film {item['film']}", file=sys.stderr)
    cond = {**conds[a.condition], "frames": conds[a.condition].get("frames", exp["frames"])}
    if not run.applies(cond, item):
        print(f"{a.condition} does not apply to {a.id}: {cond['transform']} needs cited evidence, and this "
              f"{item['evidence_type']} question cites none.")
        return
    film = load_film(item["film"], cond)
    ev = C.events(film, cond, item)
    bodies = [run.request(item, cond, film, ev, run.FRAME_DIRS[cond["frames"]], rep, a.model) for rep in exp["repeats"]]
    system, user = bodies[0]["messages"]
    print("== request parameters")
    for body in bodies:
        print(json.dumps({k: v for k, v in body.items() if k != "messages"}), f"request_sha256={run.fingerprint(body)}")
    print(f"\n== system message\n{system['content']}")
    print("\n== user message\n" + "".join(map(shown, user["content"])))
    if a.answer is not None:
        print("\n== judge prompt")
        for m in C.judge_prompt(item, a.answer):
            print(f"-- {m['role']}\n{m['content']}")
        score = C.score_open(item, a.answer)
        print(f"\n== rule score\n{'judge' if score is None else score}")


if __name__ == "__main__":
    main()
