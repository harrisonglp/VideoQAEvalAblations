"""Run the reader on the evidence bank under each condition of an experiment, against an OpenAI-compatible server
(vLLM).

    python -m pipeline.run --experiment configs/experiments/evidence_bank_v1.json --model gemma4-31b \
        --url http://127.0.0.1:8011/v1/chat/completions [--conditions q,t] [--films mf2_17,...] [--ids hh53_007,...] \
        [--concurrency 4]

Writes one record per (condition, question, repeat) to results/<experiment name>/<model>/<condition>.jsonl: the reply,
the parsed answer, its score, the judge's verdict when the scoring rules cannot decide, latency, token counts and the
request's fingerprint (`request_sha256`, see `fingerprint`; scripts/show_prompts.py prints the same value for the
request it rebuilds, so a stored answer can be traced to the exact prompt).
The experiment names its item set, the frame set shown as V (`frames`), the decoding settings (`repeats`), its
conditions and optionally a subset of films (`films`); the results folder is its `name` (default: the item set).
Questions are processed film by film, so that the questions of a film reuse the cached film prefix (the first
WARMUP_CALLS of a film, from the model's profile, default 1, are sent one at a time to fill the cache); a re-run skips
the calls that already have a scored record. Answers that the rules cannot score are judged by the same model,
without the film (conditions.judge_prompt), or, with `--judge none`, stored with `pending_judge` for a later pass with
one fixed judge model (pipeline.judge); a call that fails five times is left without a score and retried on the next
run. Each model's reasoning, sampling and answer-limit settings come from its profile, configs/models/<model>.env.
"""
import argparse
import hashlib
import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

from . import conditions as C
from .common import CONFIGS, DATA, EVALS, RESULTS, read_json, read_jsonl, read_profile, request_options

FRAME_DIRS = {"full": "frames", "notext": "frames_notext"}
# sent with every request, with the temperature and seed of the repeat: vLLM fills unset sampling parameters from the
# model's own generation config
SAMPLING = {"top_p": 0.95, "top_k": 20, "min_p": 0.0, "repetition_penalty": 1.0, "presence_penalty": 0.0,
            "frequency_penalty": 0.0}
PER_ITEM = ("cut_evidence", "cut_placebo", "window_only")  # transforms that depend on the question's evidence
MAX_TOKENS, JUDGE_MAX_TOKENS, ATTEMPTS = 128, 32, 5
# what reasoning models put before the answer when the server does not separate it: the answer follows the last marker
REASONING_ENDS = ("</think>", "<channel|>", "<|END_THINKING|>")


def post(url, body, timeout=7200):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}: {e.read()[:300]!r}"}
    except (urllib.error.URLError, OSError, TimeoutError) as e:
        return {"error": f"connection: {e}"}


def complete(url, body):
    """(reply text, usage with the finish reason and seed), or (None, the last error) after ATTEMPTS failed calls. A
    sampled reply that runs out of context or has no answer is asked again with the next seed."""
    for attempt in range(ATTEMPTS):
        j = post(url, body)
        if "choices" in j:
            choice = j["choices"][0]
            text = choice["message"].get("content") or ""
            for end in REASONING_ENDS:
                text = text.split(end)[-1]
            text = text.strip()
            if body.get("temperature") and (choice.get("finish_reason") == "length" or not text) and attempt + 1 < ATTEMPTS:
                body = {**body, "seed": body["seed"] + 1}
                continue
            return text, {**(j.get("usage") or {}), "finish_reason": choice.get("finish_reason"), "seed": body.get("seed")}
        time.sleep(10 * (attempt + 1))
    return None, j.get("error")


def applies(cond, item):
    """Transforms that depend on the cited evidence apply only to questions that cite evidence."""
    return cond.get("transform") not in PER_ITEM or bool(item["evidence"])


def request(item, cond, film, ev, frame_dir, rep, model):
    """The chat-completion request for one question under one condition."""
    image = lambda x: {"type": "image_url", "image_url": {"url": f"file://{DATA / 'films' / x[0] / frame_dir / x[1]}"}}
    extra, max_tokens = request_options(model)
    limit = {} if max_tokens == 0 else {"max_tokens": max_tokens or MAX_TOKENS}
    return {"model": model, **SAMPLING, **rep, **limit, **extra,
            "messages": [{"role": "system", "content": C.SYSTEM},
                         {"role": "user", "content": C.film_parts(cond, film, ev, image) + [{"type": "text", "text": C.question(item)}]}]}


def fingerprint(body) -> str:
    """First 16 hex digits of the sha256 of the request as canonical JSON (keys sorted). The image parts are absolute
    file paths, so the same request rebuilt from another data root has another fingerprint."""
    return hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]


def judge(url, model, item, answer):
    """(score, verdict) from the model as judge; (None, None) when the call fails. The verdict is free text, so a
    profile's answer schema is not applied."""
    extra, _ = request_options(model)
    extra.pop("response_format", None)
    text, _ = complete(url, {"model": model, "temperature": 0.0, "max_tokens": JUDGE_MAX_TOKENS, **extra,
                             "messages": C.judge_prompt(item, answer)})
    if text is None:
        return None, None
    v = C.parse_verdict(text)
    return (0, "unclear") if v is None else (v, text[:20])


def done_calls(path):
    """Calls with a scored record, or with an answer waiting for the fixed judge."""
    if not path.exists():
        return set()
    return {(r["film"], r["id"], r["rep"]) for r in map(json.loads, open(path))
            if r.get("correct") is not None or r.get("pending_judge")}


def run_condition(name, cond, items, films, exp, model, url, concurrency, judge_mode="self"):
    out = RESULTS / exp.get("name", exp["items"]) / model / f"{name}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done, lock = done_calls(out), Lock()
    reps = exp["repeats"]
    cond = {**cond, "frames": cond.get("frames", exp["frames"])}
    frame_dir = FRAME_DIRS[cond["frames"]]
    per_item = cond.get("transform") in PER_ITEM
    # Only whole-film conditions give every question of a film the same prefix; those run film by film, and models with
    # sliding-window layers reuse the prefix only from the third request on, so they warm up with two. The others (no
    # film, or excerpts per question) put every film's questions into one batch.
    shared = not per_item and any(x in cond["inputs"] for x in "DVRA")
    warmup = int(read_profile(model).get("WARMUP_CALLS", 1)) if shared else 0
    with open(out, "a") as fh:
        for group in ([[f] for f in sorted(films)] if shared else [sorted(films)]):
            calls = [(fid, it, k) for fid in group for it in items if it["film"] == fid and applies(cond, it)
                     for k in range(len(reps)) if (fid, it["id"], k) not in done]
            if not calls:
                continue
            evs = {fid: None if per_item else C.events(films[fid], cond) for fid in group}

            def call(c):
                fid, it, k = c
                e = C.events(films[fid], cond, it) if per_item else evs[fid]
                rec = {"condition": name, "model": model, "film": fid, "id": it["id"], "rep": k, "gold": it["reference"],
                       "n_frames": sum(x[0] == "frame" for x in e), "frames": cond["frames"]}
                body = request(it, cond, films[fid], e, frame_dir, reps[k], model)
                rec["request_sha256"] = fingerprint(body)
                t0 = time.time()
                text, usage = complete(url, body)
                if text is None:
                    rec.update(correct=None, error=usage)
                else:
                    pred = C.parse_answer(text)
                    correct = C.score_open(it, pred)
                    if correct is None and judge_mode == "none":
                        rec["pending_judge"] = True
                    elif correct is None:
                        correct, verdict = judge(url, model, it, pred)
                        rec["judged"] = verdict
                        if correct is None:
                            rec["error"] = "judge call failed"
                    rec.update(text=text[:500], pred=pred, correct=correct, latency_s=round(time.time() - t0, 2),
                               prompt_tokens=usage.get("prompt_tokens"),
                               cached_tokens=(usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
                               completion_tokens=usage.get("completion_tokens"), finish_reason=usage.get("finish_reason"),
                               seed=usage.get("seed"))
                with lock:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fh.flush()
                return rec

            t0 = time.time()
            recs = [call(c) for c in calls[:warmup]]  # the first calls fill the prefix cache that the others reuse
            with ThreadPoolExecutor(concurrency) as pool:
                recs += list(pool.map(call, calls[warmup:]))
            ok = [r for r in recs if r["correct"] is not None]
            pending = sum(bool(r.get("pending_judge")) for r in recs)
            print(f"{time.strftime('%T')} {model} {name} {group[0] if shared else 'all films'}: "
                  f"{sum(r['correct'] for r in ok)}/{len(ok)} correct, "
                  f"{sum('judged' in r for r in recs)} judged, {pending} for the judge, "
                  f"{len(recs) - len(ok) - pending} failed, {time.time() - t0:.0f} s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--model", required=True, help="served model name (configs/models/<model>.env)")
    ap.add_argument("--url", required=True)
    ap.add_argument("--conditions", default="", help="comma-separated subset; default: all in the experiment")
    ap.add_argument("--films", default="", help="comma-separated video ids; default: the experiment's films, or all")
    ap.add_argument("--ids", default="", help="comma-separated question ids; default: all")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--judge", choices=("self", "none"), default="self",
                    help="self: the reader judges the answers the rules cannot score; none: leave them for pipeline.judge")
    a = ap.parse_args()
    exp = read_json(a.experiment)
    conds = read_json(CONFIGS / "conditions.json")["conditions"]
    items = read_jsonl(DATA / "items" / f"{exp['items']}.jsonl")
    for key, wanted in (("film", exp.get("films")), ("film", a.films.split(",") if a.films else None),
                        ("id", a.ids.split(",") if a.ids else None)):
        if wanted:
            items = [it for it in items if it[key] in set(wanted)]
    listed = {f["video_id"]: f for f in read_json(EVALS / exp["items"] / "films.json")}
    films = {}
    for f in sorted({it["film"] for it in items}):
        path = DATA / "films" / f / "film.json"
        # the question-only and title-only conditions need no extracted input: title and year come from the film list
        films[f] = read_json(path) if path.exists() else {"id": f, "title": listed[f]["title"], "year": listed[f]["year"]}
    for f, film in films.items():
        if (ocr := DATA / "ocr" / f"{f}.json").exists():
            film["screens"] = read_json(ocr)["screens"]
        if (snd := DATA / "sounds" / f"{f}.json").exists():
            film["sounds"] = read_json(snd)["events"]
    for name in (a.conditions.split(",") if a.conditions else exp["conditions"]):
        run_condition(name, conds[name], items, films, exp, a.model, a.url, a.concurrency, a.judge)


if __name__ == "__main__":
    main()
