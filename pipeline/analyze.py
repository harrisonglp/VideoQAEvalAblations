"""Statistics of one experiment for one model: results/<name>/<model>/summary.json and units.csv.

    python -m pipeline.analyze configs/experiments/evidence_bank_v1.json gemma4-31b

Every condition of configs/conditions.json that has a results file in the experiment's folder is analysed, so the
evidence cut and its placebo, run as their own experiment under the same name, are included.
Unit of analysis
    A question; its score in a condition is the fraction of its runs answered correctly. Each condition is scored on
    the questions it applies to (excerpt and cut conditions only on questions that cite evidence); `coverage` says how
    many. Records of questions that are not in the item set are skipped and counted.
Uncertainty
    Nonparametric bootstrap over films (10,000 resamples, seed 12345): all questions of a drawn film enter together.
    Every condition is scored on the same resamples, so differences from the reference condition are paired (on the
    questions both cover). Two-sided bootstrap p-values, Holm-adjusted over all conditions.
Decomposition
    With the title in every cell, the exact Shapley contribution of each input (transcript D, frames V, on-screen text
    R, sounds A) relative to the title alone, and the pairwise interactions, over the largest complete factorial (DVRA,
    DVR or DV) of the whole-film conditions (t, td, ..., tdvra) and of the excerpt conditions (t, w_td, ..., w_tdvra),
    where the conditions form one (in the main experiment only the excerpt conditions do). Estimates are on the
    answerable questions that every cell covers, 95% intervals from the first 2,000 bootstrap resamples.
Evidence type
    An input set suffices for a question when its score is >= `sufficient`. The measured type follows from the smallest
    sufficient sets: guessable (the question alone, q, suffices), prior (the title alone, t), one source, "X or Y" (each
    alone suffices), "X and Y" (only together); "any source" when three or more inputs each suffice, "several pairs" or
    "three or more sources" when no single input does and several pairs or none suffice; "undetermined" when all
    inputs together do not suffice. Patterns in which an input set suffices but a superset does not are flagged as
    unstable. Agreement with the intended type, Cohen's kappa and the confusion matrix are computed on the determined
    questions, by type and by family, separately for each factorial. Unanswerable questions are scored as the
    abstention rate per condition.
Evidence necessity
    For the evidence cut (transform cut_evidence), the share of the questions solved with all inputs (the reference
    condition) that fail once the cited evidence is removed, by intended type, and the same share under the placebo cut
    (transform cut_placebo) on the same questions.
"""
import collections
import csv
import json
import math
import random
import sys
from itertools import combinations

from .common import CONFIGS, DATA, RESULTS, read_json, read_jsonl, write_json
from .items import SOURCES, parse_evidence_type

B, SEED = 10_000, 12345
LETTER_NAME = {v[1]: k for k, v in SOURCES.items()}  # D -> dialogue, V -> visual, R -> text, A -> audio


def unit_scores(path, known: set) -> tuple[dict, int]:
    """({question id: score}, records skipped because their question is not in the item set) for one condition."""
    recs, skipped = {}, 0
    for r in map(json.loads, open(path)):
        if r.get("correct") is None:
            continue
        if r["id"] not in known:
            skipped += 1
            continue
        recs.setdefault((r["id"], r["rep"]), r["correct"])
    runs = collections.defaultdict(list)
    for (iid, _), c in sorted(recs.items(), key=lambda kv: kv[0][1]):
        runs[iid].append(c)
    return {iid: sum(v) / len(v) for iid, v in runs.items()}, skipped


def cell(sub, prefix=""):
    """Condition name of an input subset with the title in every cell: () -> t, ('D', 'V') -> tdv; excerpt cells get
    `prefix` unless they hold no film input."""
    name = "t" + "".join(x.lower() for x in "DVRA" if x in sub)
    return prefix + name if prefix and sub else name


def factorial(S, prefix=""):
    """The largest of DVRA, DVR and DV whose 2^n cells are all present, or None."""
    for F in ("DVRA", "DVR", "DV"):
        if all(cell(sub, prefix) in S for k in range(len(F) + 1) for sub in combinations(F, k)):
            return F
    return None


def shapley(v: dict, F: str) -> dict:
    n, res = len(F), {}
    for i in F:
        rest = [p for p in F if p != i]
        res[i] = sum(math.factorial(k) * math.factorial(n - k - 1) / math.factorial(n) *
                     (v[frozenset(sub) | {i}] - v[frozenset(sub)]) for k in range(n) for sub in combinations(rest, k))
    for a, b in combinations(F, 2):
        res[f"{a}x{b}"] = v[frozenset((a, b))] - v[frozenset((a,))] - v[frozenset((b,))] + v[frozenset()]
    return res


def measured_type(ok: dict, F: str, question_only: bool | None) -> tuple[str, bool]:
    """ok: {frozenset(inputs): sufficient?} over the factorial F with the title in every cell, so ok[frozenset()] is
    the title alone; `question_only` says whether the question alone suffices. Returns (type, unstable)."""
    unstable = any(ok[a] and not ok[b] for a in ok for b in ok if a < b)
    if not ok[frozenset(F)]:
        return "undetermined", unstable
    if question_only:
        return "guessable", unstable
    if ok[frozenset()]:
        return "prior", unstable
    single = [x for x in F if ok[frozenset(x)]]
    if len(single) == 1:
        return LETTER_NAME[single[0]], unstable
    if len(single) == 2:
        return f"{LETTER_NAME[single[0]]} or {LETTER_NAME[single[1]]}", unstable
    if single:
        return "any source", unstable
    pairs = [(a, b) for a, b in combinations(F, 2) if ok[frozenset((a, b))]]
    if len(pairs) == 1:
        return f"{LETTER_NAME[pairs[0][0]]} and {LETTER_NAME[pairs[0][1]]}", unstable
    return ("several pairs" if pairs else "three or more sources"), unstable


def intended_type(evidence_type: str, F: str) -> str:
    """The intended type restricted to the inputs of the factorial F; 'untestable' when it needs an input outside F."""
    kind, parts = parse_evidence_type(evidence_type)
    if kind in ("prior", "unanswerable"):
        return kind
    letters = [SOURCES[p][1] for p in parts]
    have = [x for x in letters if x in F]
    if kind == "and":
        return evidence_type if len(have) == 2 else "untestable"
    if not have:
        return "untestable"
    return " or ".join(LETTER_NAME[x] for x in have)


def family(t: str) -> str:
    if t in ("prior", "guessable", "undetermined", "unanswerable"):
        return t
    if t in SOURCES:
        return "single"
    if t in ("any source", "several pairs", "three or more sources"):
        return "other"
    return "or" if " or " in t else "and"


def agreement(pairs: list) -> dict:
    if not pairs:
        return {}
    conf = collections.Counter(pairs)
    n = len(pairs)
    po = sum(v for (a, b), v in conf.items() if a == b) / n
    pa, pb = collections.Counter(a for a, _ in pairs), collections.Counter(b for _, b in pairs)
    pe = sum(pa[k] * pb[k] for k in set(pa) | set(pb)) / n ** 2
    return {"n": n, "agreement": round(po, 3), "kappa": round((po - pe) / (1 - pe), 3) if pe < 1 else None,
            "confusion": {f"{a} -> {b}": v for (a, b), v in sorted(conf.items())}}


def main():
    exp, model = read_json(sys.argv[1]), sys.argv[2]
    conds = read_json(CONFIGS / "conditions.json")["conditions"]
    items = read_jsonl(DATA / "items" / f"{exp['items']}.jsonl")
    by_id = {it["id"]: it for it in items}
    out_dir = RESULTS / exp.get("name", exp["items"]) / model
    pending = {c: n for c in conds if (out_dir / f"{c}.jsonl").exists()
               and (n := sum(1 for r in read_jsonl(out_dir / f"{c}.jsonl") if r.get("pending_judge")))}
    if pending:
        sys.exit(f"answers still waiting for the judge (run pipeline.judge first): {pending}")
    S, skipped = {}, {}
    for c in conds:
        if (out_dir / f"{c}.jsonl").exists():
            S[c], n = unit_scores(out_dir / f"{c}.jsonl", set(by_id))
            if n:
                skipped[c] = n
    S = {c: s for c, s in S.items() if s}
    ref_name, thr = exp["reference"], exp["sufficient"]
    units = sorted(S[ref_name])
    rng = random.Random(SEED)
    films = collections.defaultdict(list)
    for i, u in enumerate(units):
        films[by_id[u]["film"]].append(i)
    order = sorted(films)
    boot = [[i for f in [rng.choice(order) for _ in films] for i in films[f]] for _ in range(B)]
    full = list(range(len(units)))

    def score(s, sample, within=None):  # accuracy (%) on the sampled questions that s covers
        vals = [s[units[i]] for i in sample if units[i] in s and (within is None or units[i] in within)]
        return 100 * (sum(vals) / len(vals)) if vals else float("nan")

    def summarize(f):
        bs = sorted(x for x in (f(s) for s in boot) if x == x)
        return f(full), [round(bs[int(0.025 * len(bs))], 1), round(bs[int(0.975 * len(bs)) - 1], 1)], bs

    ref = S[ref_name]
    rows = {}
    for c, s in S.items():
        acc, ci, _ = summarize(lambda x, s=s: score(s, x))
        d, dci, dbs = summarize(lambda x, s=s: score(s, x) - score(ref, x, within=s))
        crossing = sum(1 for v in dbs if (v >= 0 if d < 0 else v <= 0))
        common = [u for u in units if u in s]
        rows[c] = {"accuracy": round(acc, 1), "ci": ci, "coverage": len(common), "delta": round(d, 1), "delta_ci": dci,
                   "p": None if c == ref_name else min(1.0, 2 * (crossing + 1) / (len(dbs) + 1)),
                   "lost": sum(ref[u] >= thr > s[u] for u in common), "gained": sum(s[u] >= thr > ref[u] for u in common)}
    tests = sorted((v["p"], c) for c, v in rows.items() if v["p"] is not None)
    adj = 0.0
    for i, (p, c) in enumerate(tests):  # Holm step-down
        adj = max(adj, min(1.0, (len(tests) - i) * p))
        rows[c]["p_holm"] = round(adj, 4)
    summary = {"set": exp["items"], "model": model, "reference": ref_name, "questions": len(units),
               "bootstrap": {"resamples": B, "seed": SEED, "over": "films"}, "conditions": rows}
    if skipped:
        summary["skipped_records"] = skipped

    answerable = [u for u in units if by_id[u]["evidence_type"] != "unanswerable"]
    types = {}
    for prefix, label in (("", "whole_film"), ("w_", "window")):
        F = factorial(S, prefix)
        if not F:
            continue
        cells = {frozenset(sub): cell(sub, prefix) for k in range(len(F) + 1) for sub in combinations(F, k)}
        on_set = {u for u in answerable if all(u in S[c] for c in cells.values())}
        on_idx = [i for i in full if units[i] in on_set]
        if not on_set:
            continue
        val = lambda x: {k: score(S[c], x, within=on_set) for k, c in cells.items()}
        pt = shapley(val(on_idx), F)
        bs = [shapley(val(x), F) for x in boot[:2000]]
        summary.setdefault("shapley", {})[label] = {
            "inputs": F, "n": len(on_set), **{k: {"value": round(pt[k], 1), "ci": [round(sorted(b[k] for b in bs)[50], 1),
                                                                                  round(sorted(b[k] for b in bs)[1949], 1)]} for k in pt}}
        pairs, fam_pairs, unstable, undetermined, untestable = [], [], 0, 0, 0
        for u in answerable:
            q_ok = S["q"][u] >= thr if "q" in S and u in S["q"] else None
            if u in on_set:
                m, flag = measured_type({k: S[c][u] >= thr for k, c in cells.items()}, F, q_ok)
            else:  # not in every cell (excerpts: questions that cite no interval): only the film-free cells apply
                t_ok = "t" in S and u in S["t"] and S["t"][u] >= thr
                m, flag = ("guessable" if q_ok else "prior" if t_ok else "undetermined"), False
            want = intended_type(by_id[u]["evidence_type"], F)
            types.setdefault(u, {})[label] = m
            unstable += flag
            if want == "untestable":
                untestable += 1
            elif m == "undetermined":
                undetermined += 1
            else:
                pairs.append((want, m))
                fam_pairs.append((family(want), family(m)))
        summary.setdefault("evidence_type", {})[label] = {
            "inputs": F, "threshold": thr, "units": len(answerable), "untestable": untestable, "undetermined": undetermined,
            "unstable": unstable, "coverage": round(len(pairs) / max(1, len(answerable) - untestable), 3),
            "by_type": agreement(pairs), "by_family": agreement(fam_pairs)}

    unans = [u for u in units if by_id[u]["evidence_type"] == "unanswerable"]
    if unans:
        summary["unanswerable_abstention"] = {c: {"rate": round(sum(s[u] for u in unans if u in s) / n, 3), "n": n}
                                              for c, s in S.items() if (n := sum(u in s for u in unans))}

    placebo = next((c for c in S if conds[c].get("transform") == "cut_placebo"), None)
    for c in (c for c in S if conds[c].get("transform") == "cut_evidence"):
        nec = collections.defaultdict(lambda: [0, 0, 0, 0])  # broken, solved, placebo broken, placebo solved
        for u in answerable:
            if u in S[c] and ref.get(u, 0) >= thr:
                k = by_id[u]["evidence_type"]
                nec[k][0] += S[c][u] < thr
                nec[k][1] += 1
                if placebo and u in S[placebo]:
                    nec[k][2] += S[placebo][u] < thr
                    nec[k][3] += 1
        summary.setdefault("evidence_necessity", {})[c] = {
            k: {"broken": a, "solved": n, "rate": round(a / n, 3),
                **({"placebo_rate": round(pa / pn, 3), "net": round(a / n - pa / pn, 3)} if pn else {})}
            for k, (a, n, pa, pn) in sorted(nec.items())}

    recs = [r for c in S for r in read_jsonl(out_dir / f"{c}.jsonl") if "finish_reason" in r]
    if recs:  # reply lengths (reasoning included), replies asked again with a later seed, and replies still without an answer
        tokens = sorted(r["completion_tokens"] or 0 for r in recs)
        summary["generation"] = {
            "records": len(recs), "completion_tokens_median": tokens[len(tokens) // 2],
            "completion_tokens_p99": tokens[int(0.99 * (len(tokens) - 1))],
            "retried": sum(r.get("seed") not in (None, exp["repeats"][r["rep"]].get("seed")) for r in recs),
            "unresolved": sum(r["finish_reason"] == "length" or not r.get("pred") for r in recs)}

    write_json(out_dir / "summary.json", summary)
    with open(out_dir / "units.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["question", "film", "evidence_type", "measured_whole_film", "measured_window", *S])
        for u in units:
            it = by_id[u]
            w.writerow([u, it["film"], it["evidence_type"], types.get(u, {}).get("whole_film", ""),
                        types.get(u, {}).get("window", ""), *(round(S[c][u], 3) if u in S[c] else "" for c in S)])
    for c, v in rows.items():
        print(f"{c:16} {v['accuracy']:5.1f} {v['ci']}  n {v['coverage']:4d}  Δ {v['delta']:+5.1f} {v['delta_ci']}  p_holm {v.get('p_holm', '—')}")
    for label, v in summary.get("evidence_type", {}).items():
        print(label, json.dumps({k: v[k] for k in ("units", "untestable", "undetermined", "unstable", "coverage")}),
              json.dumps({k: v["by_family"].get(k) for k in ("n", "agreement", "kappa")}))


if __name__ == "__main__":
    main()
