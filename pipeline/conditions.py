"""Evidence conditions: which inputs the reader receives about a film, how the prompt is rendered from them, and how
answers are scored.

A film has a speech transcript (timestamped lines), 256 frames sampled uniformly through it, an on-screen text track
(screens of text with their time spans) and a sound track (sound labels with their time spans). A condition gives any
subset of the title (T), transcript (D), frames (V), on-screen text (R) and sounds (A), either for the whole film or
transformed:

    window_only   only what lies within pad_s of the question's cited evidence intervals; a marker stands for each
                  stretch left out between two excerpts
    cut_evidence  the whole film with the cited evidence intervals (± pad_s) removed and, with answer_mentions, every
                  transcript line (± MENTION_PAD_S), screen of text and sound label that states the answer, wherever it
                  is in the film; number answers up to SMALL_NUMBER keep their mentions, because their words occur
                  throughout a film in other senses. Each removed stretch is replaced by a marker
    cut_placebo   the same stretches moved to random places away from the evidence and from every mention of the
                  answer (seeded per question): the control for cut_evidence

The prompt interleaves the inputs in time order, each entry preceded by its timestamp; at equal times a frame comes
first. Answers are scored by rule where possible (score_open); otherwise the reader judges its own answer against the
reference, without the film (judge_prompt).
"""
import hashlib
import json
import random
import re

from .common import fmt_time

SYSTEM = "You are a helpful assistant answering questions about a movie. Output valid JSON."
ABSTAIN_ANSWER = "not shown in the film"
OPEN_FORMAT = ("\n\nAnswer with a short answer only: a word, a short phrase, a number, a name or a time (mm:ss). If the "
               f"answer is not said, shown, written or heard in the movie, answer \"{ABSTAIN_ANSWER}\". If uncertain, give "
               "your best guess."
               "\n\nOutput ONLY valid JSON in this exact format:\n  {{\"id\": {id}, \"answer\": \"...\"}}")
MENTION_PAD_S = 1.0  # a transcript line is a point in time: a mention of the answer is deleted with this margin each side
SMALL_NUMBER = 12

# ---------------------------------------------------------------------------------------------- transcript policy
# The transcript input carries speech only. Bracketed annotations and music symbols ("[applause]", "♪ ... ♪") would let
# a text-only reader answer sound questions, so they are removed. Lines with no Latin letter are dropped too: the
# recogniser writes some interjections and grunts in Chinese characters ("嗯。", "哦。").
_TAG_LINE = re.compile(r"^\s*(?:[\[\(][^\]\)]*[\]\)]|[♪♫].*|[.\-…\s]+)\s*$")
_INLINE_TAG = re.compile(r"\s*(?:[\[\(][^\]\)]{1,40}[\]\)]|[♪♫])\s*")
_LATIN = re.compile(r"[A-Za-z\u00C0-\u024F]")


def clean_line(text: str):
    """The transcript line as the reader sees it, or None to drop it."""
    if _TAG_LINE.match(text) or not _LATIN.search(text):
        return None
    return _INLINE_TAG.sub(" ", text).strip() or None


# ---------------------------------------------------------------------------------------------- evidence
_ORDER = {"frame": 0, "text": 1, "sound": 2, "line": 3, "gap": 4}


def _span(e) -> tuple[float, float]:
    """Time span of an event: frames and lines are points; screens of text and sound labels last from start to end."""
    return (e[1], e[2][0]) if e[0] in ("text", "sound") else (e[1], e[1])


def _merge(spans: list) -> list:
    out = []
    for a, z in sorted(tuple(s) for s in spans):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], z)
        else:
            out.append([a, z])
    return out


def _cut(ev: list, spans: list) -> list:
    """Delete every event that overlaps a span, leaving one marker per span in time order."""
    for a, z in reversed(_merge(spans)):
        kept = [e for e in ev if e[0] == "gap" or not (_span(e)[0] <= z and _span(e)[1] >= a)]
        pos = next((i for i, e in enumerate(kept) if e[1] > z), len(kept))
        ev = kept[:pos] + [("gap", a, f"{fmt_time(a)}–{fmt_time(z)}")] + kept[pos:]
    return ev


def _mention_span(e) -> tuple[float, float]:
    a, z = _span(e)
    return (a - MENTION_PAD_S, z + MENTION_PAD_S) if e[0] == "line" else (a, z)


def _small_number(item: dict) -> bool:
    try:
        return item.get("answer_type") == "number" and abs(float(item["reference"])) <= SMALL_NUMBER
    except ValueError:
        return False


def _placebo_spans(item: dict, spans: list, duration: float, seed: int, avoid=()) -> list:
    """Spans of the same lengths as `spans`, placed uniformly at random where they overlap no cited evidence (± 60 s),
    none of the `avoid` spans (± 10 s) and no span placed before; a span not placed in 1,000 draws is dropped."""
    keep_out = _merge([(e["t_start_s"] - 60, e["t_end_s"] + 60) for e in item.get("evidence", [])]
                      + [(a - 10, z + 10) for a, z in avoid])
    rng = random.Random(int(hashlib.sha256(f"{seed}:{item['id']}".encode()).hexdigest()[:8], 16))
    out = []
    for a, z in spans:
        length = z - a
        for _ in range(1000):
            s = rng.uniform(0, max(0.0, duration - length))
            if not any(s <= q and s + length >= p for p, q in keep_out + out):
                out.append([s, s + length])
                break
    return out


def answer_variants(answer: str, aliases=(), answer_type: str = "text") -> list[str]:
    """Normalised strings that state the answer: the answer and aliases, and numbers as digits and number words.
    Strings shorter than 3 characters (other than digits) match too widely to be informative and are left out."""
    out = [n for a in [answer, *aliases] if (n := _norm(a)) and n != _norm(ABSTAIN_ANSWER)]
    if answer_type == "number":
        try:
            v = float(answer)
            out.append(str(int(v)) if v.is_integer() else str(v))
            out += [w for w, x in _NUMBER_WORDS.items() if x == v and w != "none"]
        except ValueError:
            pass
    return sorted({v for v in out if len(v) >= 3 or v.isdigit()})


def mentions(variants: list[str], text: str) -> bool:
    t = _norm(text)
    return any(re.search(rf"(?<!\w){re.escape(v)}(?!\w)", t) for v in variants)


def events(film: dict, cond: dict, item: dict | None = None) -> list:
    """Ordered evidence events for one film under one condition:
    ("frame", t, (video_id, file)) | ("line", t, text) | ("text", start, (end, text)) | ("sound", start, (end, label))
    | ("gap", t, label). `item` is needed for the transforms that depend on the question's cited evidence."""
    D, V, R, A = (x in cond["inputs"] for x in "DVRA")
    frames = film["frames"] if V else []
    lines = [{"t": l["t"], "text": c} for l in film["lines"] if (c := clean_line(l["text"]))] if D else []
    if R and "screens" not in film:
        raise ValueError(f"{film['id']}: no on-screen text track (run pipeline.screen_text)")
    screens = film.get("screens", []) if R else []
    if A and "sounds" not in film:
        raise ValueError(f"{film['id']}: no sound track (run pipeline.sounds)")
    sounds = film.get("sounds", []) if A else []
    ev = sorted([("frame", f["t"], (film["id"], f["file"])) for f in frames] + [("line", l["t"], l["text"]) for l in lines]
                + [("text", s["start"], (s["end"], s["text"])) for s in screens]
                + [("sound", s["start"], (s["end"], s["sound"])) for s in sounds], key=lambda e: (e[1], _ORDER[e[0]]))
    tf = cond.get("transform")
    if tf == "window_only":
        pad = cond.get("pad_s", 60)
        keep = _merge([(max(0.0, e["t_start_s"] - pad), min(film["duration_s"], e["t_end_s"] + pad))
                       for e in item.get("evidence", [])])
        ev = [e for e in ev if any(_span(e)[0] <= z and _span(e)[1] >= a for a, z in keep)]
        for (_, z0), (a1, _) in zip(keep, keep[1:]):  # the marker precedes everything that starts after an excerpt
            pos = next((i for i, e in enumerate(ev) if e[1] > z0), len(ev))
            ev.insert(pos, ("gap", z0, f"{fmt_time(z0)}–{fmt_time(a1)}"))
    elif tf in ("cut_evidence", "cut_placebo"):
        pad = cond.get("pad_s", 10)
        spans = [(max(0.0, e["t_start_s"] - pad), e["t_end_s"] + pad) for e in item.get("evidence", [])]
        hits = []
        if cond.get("answer_mentions") and not _small_number(item):
            v = answer_variants(item["reference"], item.get("aliases", []), item.get("answer_type", "text"))
            hits = [_mention_span(e) for e in ev
                    if e[0] in ("line", "text", "sound") and mentions(v, e[2] if e[0] == "line" else e[2][1])]
        spans = _merge(spans + hits)
        if tf == "cut_placebo":
            spans = _placebo_spans(item, spans, film["duration_s"], cond.get("seed", 12345), avoid=hits)
        ev = _cut(ev, spans)
    return ev


# ---------------------------------------------------------------------------------------------- prompt
def _join(parts):
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]


def header(cond: dict, title: str, n_frames: int) -> str:
    T, D, V, R, A = (x in cond["inputs"] for x in "TDVRA")
    if not (D or V or R or A):
        return (f"Based on your knowledge of the movie {title}, answer the question below.\n" if T
                else "Answer the question below about a movie.\n")
    window = cond.get("transform") == "window_only"
    if window:
        subject = f"excerpts of the movie {title}" if T else "excerpts of a movie"
    else:
        subject = f"the full-length movie {title}" if T else "a full-length movie"
    masked = ", with the text detected on screen blacked out" if cond.get("frames") == "notext" else ""
    s = "s" if n_frames != 1 else ""
    where = "from these excerpts" if window else "sampled uniformly from it"
    vis = f"{n_frames} frame{s} {where}{masked}, each preceded by its [mm:ss] timestamp"
    speech = "its speech, transcribed automatically (one timestamped line per speech segment; no speaker names)"
    written = "the text that appears on screen, read automatically (one entry per screen of text, with the time span it is visible)"
    heard = ("the sounds other than speech, as labels from an automatic sound tagger (one entry per label, with the time "
             "span it is detected)")
    parts = [p for p, on in ((vis, V), (speech, D), (written, R), (heard, A)) if on]
    body = parts[0] if len(parts) == 1 else f"{parts[0]}, interleaved in time order with {_join(parts[1:])}"
    obj = "them" if window else "it"
    verb = f"Watch and read {obj} carefully" if V and (D or R or A) else f"Watch {obj} carefully" if V else f"Read {obj} carefully"
    return f"Below {'are' if window else 'is'} {subject}: {body}. {verb} and answer the question below.\n"


def film_parts(cond: dict, film: dict, ev: list, image) -> list:
    """Chat content parts for the film part of the prompt; image((video_id, file)) returns an image part."""
    n = sum(e[0] == "frame" for e in ev)
    parts, buf = [{"type": "text", "text": header(cond, f"{film['title']} ({film['year']})", n)}], []
    for kind, t, x in ev:
        if kind == "frame":
            buf.append(f"<frame [{fmt_time(t)}]>\n")
            parts.append({"type": "text", "text": "".join(buf)})
            parts.append(image(x))
            buf = []
        elif kind == "line":
            buf.append(f"[{fmt_time(t)}] {x}\n")
        elif kind == "text":
            buf.append(f"[{fmt_time(t)}–{fmt_time(x[0])}] On screen: {x[1]}\n")
        elif kind == "sound":
            buf.append(f"[{fmt_time(t)}–{fmt_time(x[0])}] Sound: {x[1]}\n")
        else:
            buf.append(f"[{x}] (this part of the movie has been removed)\n")
    if any(x in cond["inputs"] for x in "DVRA"):
        end = "END OF EXCERPTS" if cond.get("transform") == "window_only" else "END OF MOVIE"
        # lines after the last frame go between the footer's leading newline and the rest of it
        parts.append({"type": "text", "text": ("\n" + "".join(buf) + f"{end}\n\n") if buf and n else "".join(buf) + f"\n{end}\n\n"})
    out = []
    for p in parts:  # merge adjacent text parts
        if p["type"] == "text" and out and out[-1]["type"] == "text":
            out[-1] = {"type": "text", "text": out[-1]["text"] + p["text"]}
        elif p["type"] != "text" or p["text"]:
            out.append(p)
    return out


def question(item: dict) -> str:
    return f"Question: {item['question']}" + OPEN_FORMAT.format(id=json.dumps(item["id"]))


# ---------------------------------------------------------------------------------------------- answers
def parse_answer(text: str):
    """The `answer` of the last JSON object in the reply, or else its first line."""
    text = text or ""
    for obj in reversed(re.findall(r"\{[^{}]*\}", text)):
        try:
            d = json.loads(obj)
        except ValueError:
            continue
        if isinstance(d, dict) and "answer" in d:
            return str(d["answer"]).strip()
    return text.strip().splitlines()[0][:200] if text.strip() else None


_NUMBER_WORDS = {"none": 0, **{w: i for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve "
                                     "thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty".split())}}
_ABSTAINED = re.compile(r"not (?:shown|stated|said|mentioned|specified|given|visible|revealed|in the (?:film|movie))|"
                        r"(?:does not|doesn't|did not|didn't|never) (?:say|show|mention|reveal|state|specify|tell)|"
                        r"cannot be determined|can't be determined|no information|unknown|not known|unclear", re.I)
_HEDGED = re.compile(r"\b(?:not|no|never|neither|nor)\b|\bor\b|/", re.I)


def abstained(answer: str) -> bool:
    return bool(_ABSTAINED.search(answer or ""))


def _norm(s: str) -> str:
    s = re.sub(r"[^\w\s]", " ", str(s).lower())
    return " ".join(w for w in s.split() if w not in ("a", "an", "the"))


_UNITS = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve thirteen "
                                     "fourteen fifteen sixteen seventeen eighteen nineteen".split())}
_TENS = {w: 10 * i for i, w in enumerate("twenty thirty forty fifty sixty seventy eighty ninety".split(), 2)}
_SCALES = {"hundred": 100, "thousand": 1000, "million": 10 ** 6, "billion": 10 ** 9}
_TIMES = {"once": 1, "twice": 2, "thrice": 3}


def _tokens(s: str) -> list:
    s = re.sub(r"(?<=\d)[, ](?=\d{3}\b)", "", str(s).lower())  # 30,000 and 30 000 -> 30000
    s = re.sub(r"\b(\d{1,2}):00\b", r"\1", s)  # an hour written as a clock time: 11:00 -> 11
    return re.findall(r"\d+(?:\.\d+)?|[a-z]+|[,;/]", s.replace("-", " "))


def _words_value(toks: list):
    """(value, tokens used) of a run of English number words: ["thirty", "six", "and", "a", "half"] -> (36.5, 5).
    A unit word ends the run after another unit ("three four" is two numbers), as does a tens word after a unit or
    tens word."""
    total = cur = 0.0
    seen, i, last = False, 0, None
    while i < len(toks):
        t, nxt = toks[i], toks[i + 1:i + 3]
        if t in _UNITS and last == "unit" or t in _TENS and last in ("unit", "ten") or t in _UNITS and last == "ten" and _UNITS[t] >= 10:
            break
        if t in _UNITS or t in _TENS:
            cur += _UNITS.get(t, 0) + _TENS.get(t, 0)
            last = "unit" if t in _UNITS else "ten"
        elif t == "a" and nxt[:1] and nxt[0] in (*_SCALES, "dozen") and not seen:
            cur = 1
        elif t == "hundred" and (seen or cur):
            cur, last = (cur or 1) * 100, "scale"
        elif t in _SCALES and (seen or cur):
            total, cur, last = total + (cur or 1) * _SCALES[t], 0.0, "scale"
        elif t == "dozen" and (seen or cur):
            cur, last = (cur or 1) * 12, "scale"
        elif t == "and" and nxt == ["a", "half"] and (seen or cur):
            return total + cur + 0.5, i + 3
        elif t == "and" and seen and nxt[:1] and (nxt[0] in _UNITS or nxt[0] in _TENS):
            pass  # "two hundred and fifty"
        else:
            break
        seen = seen or t in _UNITS or t in _TENS or t in _SCALES or t == "dozen"
        i += 1
    return (total + cur, i) if seen else (None, 0)


def _numbers(s: str) -> list:
    """Every number in the text, in digits ("30,000", "2.5", "30 thousand") or words ("thirty-six and a half", "twice")."""
    toks, out, i = _tokens(s), [], 0
    while i < len(toks):
        t = toks[i]
        if t[0].isdigit():
            v, i = float(t), i + 1
            if i < len(toks) and toks[i] in _SCALES:
                v, i = v * _SCALES[toks[i]], i + 1
            if toks[i:i + 3] == ["and", "a", "half"]:
                v, i = v + 0.5, i + 3
            out.append(v)
        elif t in _TIMES:
            out.append(float(_TIMES[t]))
            i += 1
        elif t == "none":
            out.append(0.0)
            i += 1
        else:
            v, used = _words_value(toks[i:])
            if v is None:
                i += 1
            else:
                out.append(v)
                i += used
    return out


def _first_time(s: str):
    m = re.search(r"\b(\d{1,2}):(\d{2})(?::(\d{2}))?\b", str(s))
    if not m:
        return None
    a, b, c = (int(x) if x else None for x in m.groups())
    return a * 3600 + b * 60 + c if c is not None else a * 60 + b


def score_open(item: dict, answer: str | None):
    """1 or 0 when the answer can be scored by rule, None when it needs the judge.
    An answer equal to the reference or an accepted variant (after normalisation) is correct. Otherwise an abstention
    ("not shown in the film" and paraphrases) is correct exactly for unanswerable questions. A text answer that contains
    the reference but also a negation or an alternative ("not a church, a barn"; "black or white") goes to the judge
    instead of being accepted by containment."""
    if not answer:
        return 0
    a = _norm(answer)
    refs = [item["reference"], *item.get("aliases", [])]
    if any(a == _norm(r) for r in refs if _norm(r)):
        return 1
    if item.get("evidence_type") == "unanswerable" or item["reference"] == ABSTAIN_ANSWER:
        return int(abstained(answer))
    if abstained(answer):
        return 0
    kind = item.get("answer_type", "text")
    hedged = _HEDGED.search(answer) and not any(_HEDGED.search(r) for r in refs)
    found = _numbers(answer) if kind == "number" else []
    if kind == "number" and (len(found) > 1 or hedged):
        return None  # several numbers or a negation ("2 or 3", "3-4", "not three, four", "not 3"): the judge decides
    if not hedged and any(f" {r} " in f" {a} " for r in map(_norm, refs) if r):
        return 1
    if kind == "number":
        return None if not found else int(abs(found[0] - float(item["reference"])) <= item.get("tolerance", 0))
    if kind == "time":
        t, ref = _first_time(answer), _first_time(item["reference"])
        return None if t is None else int(abs(t - ref) <= item.get("tolerance_s", 120))
    return None


def judge_prompt(item: dict, answer: str) -> list:
    refs = "; ".join([item["reference"], *item.get("aliases", [])])
    number = ""
    if item.get("answer_type") == "number":
        tol = item.get("tolerance", 0)
        value = f"within {tol:g} of the reference" if tol else "equal to the reference"
        number = f" A single value {value} is correct; an answer that hedges between several values is not."
    return [{"role": "system", "content": "You grade answers to questions about a film against a reference answer. Reply with one word: correct or incorrect."},
            {"role": "user", "content": f"Question: {item['question']}\nReference answer (accepted variants): {refs}\n"
                                        f"Answer to grade: {answer}\n\nIs the answer correct, i.e. does it mean the same as the reference?{number} "
                                        f"Reply with exactly one word: correct or incorrect."}]


def parse_verdict(text: str):
    """1 for a "correct" verdict, 0 for "incorrect" (or "not correct"), None when the reply is neither."""
    m = re.search(r"\b(incorrect|not correct|correct)\b", (text or "").lower())
    return None if not m else int(m.group(1) == "correct")
