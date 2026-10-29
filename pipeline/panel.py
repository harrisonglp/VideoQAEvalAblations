"""Panel analysis: evidence labels verified or reassigned from several readers' ablations, and how well the labels
agree with a reader that took no part in the reassignment.

    python -m pipeline.panel configs/experiments/panel_v1.json

The panel configuration names the experiment the readers ran (its results folder holds one folder per reader), the
readers, the sufficiency threshold `tau`, the training-set sizes of the folds and the bootstrap size.

Measured type
    From the 16 excerpt cells (the title alone and the 15 lane sets). A cell's score is the mean over the readers
    pooled (for one reader: 0 or 1, one recorded run), and it suffices when the score is >= tau. Rule "best" (default):
    undetermined if all lanes do not suffice, guessable if the question alone does, prior if the title does; else,
    among the single lanes that suffice, those with the highest score (one: that lane; two: "X or Y"; more: any
    source); else, among the lane pairs that suffice, those with the highest score (one: "X and Y"; more: several
    pairs); else three or more sources. A lane all pooled readers answer with thus beats one only some answer with. For
    one reader it equals pipeline.analyze's measured type. Rule "threshold": analyze's measured type on the pooled
    sufficiency (every lane that suffices counts, so pooling broadens types; kept as a sensitivity analysis).
    Questions without cited intervals (prior questions) are typed from the question-only and title-only cells.
Reassignment
    A question's synthetic label (its intended evidence type) is verified when the readers' pooled measured type
    equals it, reassigned to the pooled type when that is determined and different, and kept when the pooled type is
    undetermined. Unanswerable questions are not typed.
Leave-one-model-out
    For every held-out reader h and every training set T of the given sizes drawn from the other readers: on the
    answerable questions that h types (h's measured type is determined) and whose label is testable, the agreement of
    the synthetic label with h's type (before) and of the label reassigned from T (after). Agreement is exact type and,
    separately, label family; Cohen's kappa as in pipeline.analyze. Fold results are averaged per training-set size.
    95% intervals: bootstrap over films (all questions of a drawn film together), the same resamples for before and
    after, so their difference is paired. With more than three readers, the folds of training size 2 are also
    reported per three-reader panel (each panel: three folds, every reader held out once).
Comparisons
    Other labels scored on the same folds: "fixed rule" ("X or Y" -> X, "X and Y" -> Y, no readers), "two-source only"
    (the reassigned label for "or" and "and" labels, the synthetic label otherwise) and "type map" (each synthetic
    label replaced by the type T most often reassigns it to on the other films). Agreement before and after is also
    given per label family, pooled over the folds.
Also
    The ceiling: agreement between two readers' measured types. Inconsistent cells: per reader, the questions where a
    lane set suffices but a larger one does not (single runs make some of these). Profiles: every reader's accuracy per condition on
    the answerable questions, and the single-lane matrix (labelled single-source type x lane given alone, excerpts).
    Routing: each question goes to the reader that is most accurate with all lanes over the whole film on questions of
    the same evidence source (chosen on the other films), by synthetic and by reassigned label (pooled from all
    readers, so the reassigned labels have seen every reader).

Outputs in results/<experiment name>/panel/ (or the configuration's `out` folder): summary.json, labels.csv (per
question: label, each reader's measured type, pooled type, status), folds.csv (one row per fold), profiles.csv (reader x
condition accuracy).
"""
import collections
import csv
import random
import sys
from itertools import combinations

from .analyze import LETTER_NAME, agreement, cell, family, intended_type, measured_type, unit_scores
from .common import CONFIGS, DATA, RESULTS, read_json, read_jsonl, write_json

F = "DVRA"
SINGLE = ("dialogue", "visual", "text", "audio")
LANE_OF = {"dialogue": "w_td", "visual": "w_tv", "text": "w_tr", "audio": "w_ta"}
CELLS = {frozenset(sub): cell(sub, "w_") for k in range(len(F) + 1) for sub in combinations(F, k)}


def best_type(p: dict, q_ok, tau: float) -> str:
    """Rule "best" on pooled cell scores p {frozenset(lanes): score} (the empty set is the title alone)."""
    if p[frozenset(F)] < tau:
        return "undetermined"
    if q_ok:
        return "guessable"
    if p[frozenset()] >= tau:
        return "prior"
    for size in (1, 2):
        ok = {s: p[s] for s in p if len(s) == size and p[s] >= tau}
        if ok:
            top = max(ok.values())
            best = sorted([s for s, v in ok.items() if v == top], key=lambda s: [F.index(x) for x in s])
            names = [" and ".join(LETTER_NAME[x] for x in sorted(s, key=F.index)) for s in best]
            if size == 1:
                return names[0] if len(names) == 1 else f"{names[0]} or {names[1]}" if len(names) == 2 else "any source"
            return names[0] if len(names) == 1 else "several pairs"
    return "three or more sources"


def type_of(scores: dict, readers, u, tau: float, on_set: set, rule: str = "best") -> str:
    """Measured type of question u from the mean score of `readers` (scores[reader][condition][question])."""
    def pooled(c):
        vals = [scores[m][c][u] for m in readers if u in scores[m].get(c, {})]
        return sum(vals) / len(vals) if vals else None
    q = pooled("q")
    q_ok = None if q is None else q >= tau
    if u in on_set:
        if rule == "best":
            return best_type({k: pooled(c) for k, c in CELLS.items()}, q_ok, tau)
        return measured_type({k: pooled(c) >= tau for k, c in CELLS.items()}, F, q_ok)[0]
    t = pooled("t")
    return "guessable" if q_ok else "prior" if t is not None and t >= tau else "undetermined"


def boot_mean(per_fold, films_order, B, seed):
    """Point estimate and 95% interval of the fold-averaged agreement (before, after, after - before). per_fold: list
    of {film: [n, matches before, matches after]}."""
    def stat(draw):
        b = a = 0.0
        for fold in per_fold:
            n = sb = sa = 0
            for f in draw:
                if f in fold:
                    n += fold[f][0]; sb += fold[f][1]; sa += fold[f][2]
            b += sb / n if n else 0
            a += sa / n if n else 0
        return b / len(per_fold), a / len(per_fold)
    rng = random.Random(seed)
    pt = stat(films_order)
    bs = [stat([rng.choice(films_order) for _ in films_order]) for _ in range(B)]
    ci = lambda xs: [round(100 * sorted(xs)[int(0.025 * B)], 1), round(100 * sorted(xs)[int(0.975 * B) - 1], 1)]
    return {"before": round(100 * pt[0], 1), "before_ci": ci([x[0] for x in bs]),
            "after": round(100 * pt[1], 1), "after_ci": ci([x[1] for x in bs]),
            "gain": round(100 * (pt[1] - pt[0]), 1), "gain_ci": ci([x[1] - x[0] for x in bs])}


def fixed_rule(label: str) -> str:
    """"X or Y" -> X, "X and Y" -> Y, any other label unchanged."""
    if " or " in label:
        return label.split(" or ")[0]
    return label.split(" and ")[1] if " and " in label else label


def by_panel(rows, readers):
    """Folds of training size 2 grouped by the three readers they involve (the training pair and the held-out
    reader), so each group is a three-reader leave-one-model-out with three folds. rows: (fold row, per film)."""
    groups = collections.defaultdict(list)
    for row, per_film in rows:
        names = set(row["train"].split("+")) | {row["held_out"]}
        groups["+".join(m for m in readers if m in names)].append((row, per_film))
    return dict(groups)


def main():
    cfg = read_json(sys.argv[1])
    exp = read_json(CONFIGS.parent / cfg["experiment"])
    tau, B, seed, rule = cfg.get("tau", 0.5), cfg.get("bootstrap", 2000), cfg.get("seed", 12345), cfg.get("rule", "best")
    items = {it["id"]: it for it in read_jsonl(DATA / "items" / f"{exp['items']}.jsonl")}
    if exp.get("films"):
        items = {u: it for u, it in items.items() if it["film"] in set(exp["films"])}
    base = RESULTS / exp.get("name", exp["items"])
    conds = read_json(CONFIGS / "conditions.json")["conditions"]
    scores = {}
    for m in cfg["readers"]:
        folder = base / m
        if not folder.exists():
            print(f"no results for {m}, left out")
            continue
        pending = [p.stem for p in folder.glob("*.jsonl") if any(r.get("pending_judge") for r in read_jsonl(p))]
        if pending:
            sys.exit(f"{m}: answers still waiting for the judge in {pending} (run pipeline.judge first)")
        scores[m] = {c: unit_scores(folder / f"{c}.jsonl", set(items))[0] for c in conds if (folder / f"{c}.jsonl").exists()}
    readers = [m for m in cfg["readers"] if m in scores and all(c in scores[m] for c in CELLS.values()) and "q" in scores[m]]
    if len(readers) < 2:
        sys.exit(f"need at least two readers with every excerpt cell; have {readers}")
    answerable = sorted(u for u, it in items.items() if it["evidence_type"] != "unanswerable")
    on_set = {u for u in answerable if all(u in scores[m][c] for m in readers for c in CELLS.values())}
    want = {u: intended_type(items[u]["evidence_type"], F) for u in answerable}
    own = {m: {u: type_of(scores, [m], u, tau, on_set, rule) for u in answerable} for m in readers}
    film_of = {u: items[u]["film"] for u in answerable}
    films_order = sorted(set(film_of.values()))

    def reassign(T):
        out = {}
        for u in answerable:
            p = type_of(scores, T, u, tau, on_set, rule)
            out[u] = (want[u], "kept") if p == "undetermined" else (p, "verified" if p == want[u] else "reassigned")
        return out

    testable = [u for u in answerable if want[u] != "untestable"]
    folds, by_size = [], collections.defaultdict(list)
    compare, by_fam = collections.defaultdict(lambda: collections.defaultdict(list)), collections.defaultdict(dict)
    for h in readers:
        others = [m for m in readers if m != h]
        for k in cfg.get("train_sizes", [2]):
            for T in combinations(others, k):
                new = reassign(T)
                base_u = [u for u in testable if own[h][u] != "undetermined"]
                moved = collections.defaultdict(collections.Counter)
                for u in testable:
                    if new[u][1] != "kept":
                        moved[(want[u], film_of[u])][new[u][0]] += 1

                def type_map(u):
                    c = sum((n for (w, f), n in moved.items() if w == want[u] and f != film_of[u]), collections.Counter())
                    return c.most_common(1)[0][0] if c else want[u]
                labels = {"fixed rule": lambda u: fixed_rule(want[u]),
                          "two-source only": lambda u: new[u][0] if family(want[u]) in ("or", "and") else want[u],
                          "type map": type_map}
                per_film = collections.defaultdict(lambda: [0, 0, 0])
                alt = {name: collections.defaultdict(lambda: [0, 0, 0]) for name in labels}
                fam = by_fam[k]
                for u in base_u:
                    r = per_film[film_of[u]]
                    r[0] += 1; r[1] += want[u] == own[h][u]; r[2] += new[u][0] == own[h][u]
                    for name, lab in labels.items():
                        a = alt[name][film_of[u]]
                        a[0] += 1; a[1] += want[u] == own[h][u]
                        a[2] += lab(u) == own[h][u]
                    f = fam.setdefault(family(want[u]), [0, 0, 0])
                    f[0] += 1; f[1] += want[u] == own[h][u]; f[2] += new[u][0] == own[h][u]
                for name in labels:
                    compare[k][name].append(dict(alt[name]))
                status = collections.Counter(new[u][1] for u in testable)
                before = agreement([(want[u], own[h][u]) for u in base_u])
                after = agreement([(new[u][0], own[h][u]) for u in base_u])
                fam_b = agreement([(family(want[u]), family(own[h][u])) for u in base_u])
                fam_a = agreement([(family(new[u][0]), family(own[h][u])) for u in base_u])
                row = {"held_out": h, "train": "+".join(T), "train_size": k, "n": len(base_u),
                       "coverage": round(len(base_u) / len(testable), 3),
                       "agree_before": before.get("agreement"), "agree_after": after.get("agreement"),
                       "kappa_before": before.get("kappa"), "kappa_after": after.get("kappa"),
                       "family_before": fam_b.get("agreement"), "family_after": fam_a.get("agreement"),
                       "verified": status["verified"], "reassigned": status["reassigned"], "kept": status["kept"]}
                folds.append(row)
                by_size[k].append((row, dict(per_film)))

    summary = {"experiment": exp.get("name", exp["items"]), "readers": readers, "tau": tau, "rule": rule,
               "bootstrap": {"resamples": B, "seed": seed, "over": "films"}, "questions": len(answerable),
               "typed_by_excerpts": len(on_set), "leave_one_model_out": {}}
    for k, rows in sorted(by_size.items()):
        res = boot_mean([pf for _, pf in rows], films_order, B, seed)
        mean = lambda key: round(sum(r[key] for r, _ in rows if r[key] is not None) / len(rows), 3)
        summary["leave_one_model_out"][f"train_{k}"] = {
            "folds": len(rows), **res, "kappa_before": mean("kappa_before"), "kappa_after": mean("kappa_after"),
            "family_before": round(100 * mean("family_before"), 1), "family_after": round(100 * mean("family_after"), 1),
            "coverage": mean("coverage"),
            "comparisons": {name: {key: v for key, v in boot_mean(pfs, films_order, B, seed).items()
                                   if not key.startswith("before")} for name, pfs in compare[k].items()},
            "by_family": {f: {"n": n, "before": round(100 * b / n, 1), "after": round(100 * a / n, 1)}
                          for f, (n, b, a) in sorted(by_fam[k].items())}}
    if len(readers) > 3 and 2 in by_size:
        summary["leave_one_model_out"]["train_2"]["by_panel"] = {
            name: {"folds": len(rows), **boot_mean([pf for _, pf in rows], films_order, B, seed)}
            for name, rows in by_panel(by_size[2], readers).items()}

    final = reassign(readers)
    summary["status_all_readers"] = dict(collections.Counter(s for u, (_, s) in final.items() if want[u] != "untestable"))
    summary["transitions_all_readers"] = dict(collections.Counter(
        f"{want[u]} -> {lab}" for u, (lab, s) in final.items() if s == "reassigned").most_common(25))
    summary["status_by_family"] = {fam: dict(collections.Counter(final[u][1] for u in answerable if family(want[u]) == fam))
                                   for fam in sorted({family(w) for w in want.values()})}
    summary["ceiling"] = {f"{a}~{b}": agreement([(own[a][u], own[b][u]) for u in answerable
                                                if own[a][u] != "undetermined" and own[b][u] != "undetermined"])
                          for a, b in combinations(readers, 2)}
    for v in summary["ceiling"].values():
        v.pop("confusion", None)
    summary["inconsistent_cells"] = {
        m: sum(any(scores[m][CELLS[a]][u] >= tau > scores[m][CELLS[b]][u] for a in CELLS for b in CELLS if a < b)
               for u in on_set) for m in readers}

    profiles, matrix = [], {}
    for m in readers:
        for c, s in scores[m].items():
            vals = [s[u] for u in answerable if u in s]
            if vals:
                profiles.append({"reader": m, "condition": c, "accuracy": round(100 * sum(vals) / len(vals), 1), "n": len(vals)})
        matrix[m] = {t: {lane: round(100 * sum(scores[m][lane][u] for u in us) / len(us), 1)
                         for lane in [*LANE_OF.values(), "w_tdvra"]}
                     for t in SINGLE if (us := [u for u in on_set if items[u]["evidence_type"] == t])}
    summary["single_lane_matrix"] = matrix

    if all("tdvra" in scores[m] for m in readers):
        src = lambda lab: lab if lab in SINGLE else "other"
        routing = {}
        pool = [u for u in answerable if all(u in scores[m]["tdvra"] for m in readers)]
        for name, lab in (("synthetic", {u: want[u] for u in pool}), ("reassigned", {u: final[u][0] for u in pool})):
            hits = []
            for u in pool:
                best = max(readers, key=lambda m: (sum(scores[m]["tdvra"][v] for v in pool if film_of[v] != film_of[u]
                                                       and src(lab[v]) == src(lab[u])), -readers.index(m)))
                hits.append(scores[best]["tdvra"][u])
            routing[f"routed_by_{name}_label"] = round(100 * sum(hits) / len(hits), 1)
        for m in readers:
            routing[m] = round(100 * sum(scores[m]["tdvra"][u] for u in pool) / len(pool), 1)
        routing["oracle_any_reader"] = round(100 * sum(max(scores[m]["tdvra"][u] for m in readers) for u in pool) / len(pool), 1)
        routing["n"] = len(pool)
        summary["routing_whole_film_all_lanes"] = routing

    out = base / cfg.get("out", "panel")
    write_json(out / "summary.json", summary)
    with open(out / "labels.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["question", "film", "evidence_type", *[f"measured_{m}" for m in readers], "panel_type", "status"])
        for u in answerable:
            w.writerow([u, film_of[u], items[u]["evidence_type"], *[own[m][u] for m in readers], final[u][0],
                        "untestable" if want[u] == "untestable" else final[u][1]])
    for name, rows in (("folds.csv", folds), ("profiles.csv", profiles)):
        if rows:
            with open(out / name, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
    for k, v in summary["leave_one_model_out"].items():
        print(f"{k}: before {v['before']} {v['before_ci']}  after {v['after']} {v['after_ci']}  gain {v['gain']} {v['gain_ci']}  "
              f"({v['folds']} folds, coverage {v['coverage']})")
        for name, c in v["comparisons"].items():
            print(f"    {name}: {c['after']} {c['after_ci']}  gain {c['gain']} {c['gain_ci']}")
    print("status:", summary["status_all_readers"])


if __name__ == "__main__":
    main()
