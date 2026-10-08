# Evidence bank v1: entry format

`questions.jsonl` holds the 1,000 questions (50 films, 20 per film), one JSON object per line. How the questions were
written and reviewed is described in `README.md`.

## Fields

| Field | Required | Meaning |
|---|---|---|
| `id` | yes | `<film code>_<nnn>`, unique (e.g. `hh53_007`) |
| `film` | yes | video id, as in `films.json` |
| `question` | yes | the question as the reader sees it; identifies the scene without giving the answer away |
| `answer` | yes | the reference answer, a string; `number` answers in digits; `"not shown in the film"` for `unanswerable` questions |
| `evidence_type` | yes | which evidence answers it (vocabulary below) |
| `evidence` | yes | where that evidence is in the film: a list of intervals (below); empty for `prior` and `unanswerable` questions only |
| `aliases` | no | other phrasings that count as correct |
| `answer_type` | no | `text` (default), `number` or `time` |
| `tolerance` | no | for `number` answers, the largest accepted difference from `answer` (default 0) |
| `tolerance_s` | no | for `time` answers, the largest accepted difference in seconds (default 120) |
| `all` | no | `true` if the answer needs all the listed intervals together (a count across scenes, a change between scenes); default: any one interval suffices |

Each **evidence interval** has `start` and `end`, times in the film (`mm:ss` or `h:mm:ss`; equal for a single frame or
sound), and `source`: `said`, `seen`, `read` or `heard`. It may add `frames`, the indices (0–255) of the sampled frames
inside the interval that show the evidence, and a `note` describing the evidence. Notes may give the times of a line
in the transcripts and subtitles, the labels and probabilities of the independent sound tagger (AST, see `README.md`)
and sampled frames as `k<index>`; the pipeline does not read them.

## Evidence type vocabulary

There are four evidence sources, each with its own input to ablate:

| Source | Name in `evidence_type` | `source` in evidence | Input |
|---|---|---|---|
| what is said | `dialogue` | `said` | the transcript |
| what is seen | `visual` | `seen` | the 256 frames, with on-screen text masked |
| what is written on screen (signs, titles, letters, intertitles, credits) | `text` | `read` | the on-screen text track (OCR at 1 frame/s) |
| non-speech sound | `audio` | `heard` | the sound labels |

Intervals are times in the film, found by watching it; they do not depend on any of the pipeline's tracks.

`evidence_type` is one source (`dialogue`, `visual`, `text`, `audio`), two sources joined by `or` (each suffices,
e.g. `dialogue or text`) or `and` (both needed together, e.g. `dialogue and visual`), or one of:
- `prior`: a documented fact about the film that appears in no input (e.g. an award, a remake, the source novel); the
  title suffices;
- `unanswerable`: the answer is never said, shown, written or heard in the film; `answer` is `"not shown in the film"`.

Sources are written in the order dialogue, visual, text, audio: `visual or audio`, not `audio or visual`.

## Derived by the pipeline

- **Sufficient input sets**, from `evidence_type`. These define the predicted outcome of each input condition.
- **Temporal type**, from `evidence`. Intervals at most 60 s apart belong to one scene:

| Evidence | Temporal type |
|---|---|
| none | `movie-whole` |
| one scene spanning ≤ 10 s | `moment` |
| one longer scene | `scene` |
| several scenes | `movie-any`, or `multi-scene` when `all` is true |

- **Excerpts and evidence cut**, from `evidence`. The excerpt conditions (`w_t…`) keep only the cited evidence ± 60 s.
  The evidence cut (`tdvra_cut`) deletes the cited evidence ± 10 s and every mention of the answer anywhere in the film
  (a transcript line, screen of text or sound label containing `answer` or an alias; mentions of number answers up to
  12 are kept). Its placebo (`tdvra_placebo`) deletes spans of the same lengths at random places away from the evidence
  and the mentions.

## Scoring

An answer is scored by the first rule that applies:

1. An empty answer is incorrect.
2. An exact match with `answer` or an alias, after normalisation (lower case; punctuation and the articles "a", "an"
   and "the" removed), is correct.
3. `unanswerable` questions: an abstention ("not shown in the film" or a paraphrase) is correct; any other answer is
   incorrect.
4. Other questions: an abstention is incorrect.
5. `number` questions: an answer with several numbers or a hedge (a negation or an alternative: "not", "no", "never",
   "neither", "nor", "or", "/") goes to the judge. Hedge words count only when neither `answer` nor any alias
   contains one.
6. An answer containing `answer` or an alias (normalised, as whole words) and no hedge is correct.
7. `number` questions: if the answer contains a number (digits or words), it is correct when the first number is
   within `tolerance` of `answer` and incorrect otherwise.
8. `time` questions: if the answer contains a time (`mm:ss` or `h:mm:ss`), it is correct when the first time is within
   `tolerance_s` of `answer` and incorrect otherwise.
9. Anything else goes to the judge.

**The judge** is the reader model at temperature 0. It sees the question, `answer` with its `aliases` and the answer to
grade, but not the film, and replies `correct` or `incorrect`; any other reply counts as incorrect. For `number`
questions it is told the tolerance.

## Examples

```json
{"id": "ns37_005", "film": "mf2_17",
 "question": "What colour are the telephones in the young woman's New York hotel suite?",
 "answer": "white", "aliases": ["ivory", "cream", "off-white"], "answer_type": "text", "evidence_type": "visual",
 "evidence": [{"start": "45:38", "end": "46:14", "source": "seen", "frames": [159], "note": "the man in a brown fedora who finds her note phones from the suite: ivory-white handset and white cord (seen 45:40-45:41, 45:50, 46:10)"},
              {"start": "1:06:30", "end": "1:06:36", "source": "seen", "frames": [231], "note": "ivory-white cradle telephone on the nightstand beside her bed (again at 1:06:58, top right)"}]}
```
→ `visual`, `movie-any`: two scenes, either suffices.

```json
{"id": "dc61_001", "film": "mf2_34",
 "question": "What is the very first sound heard in the film, over the studio logos and before the title music starts?",
 "answer": "a gong", "aliases": ["gong", "a gong being struck", "a struck gong"], "evidence_type": "audio",
 "evidence": [{"start": "00:13", "end": "00:19", "source": "heard", "note": "AST: Gong 0.78 (00:12-00:16), 0.92 (00:15-00:19); the track is digitally silent 00:00-00:12 (the Lakeshore logo, 00:03-00:11, is silent); the gong strikes at 00:13 as the Pathe-America globe logo fades in (00:14-00:18) and decays into the title music (Music 0.72 from 00:18)"}]}
```
→ `audio`, `moment`.

```json
{"id": "mb47_006", "film": "mf2_14",
 "question": "How many times in the film does the photographer say he 'ain't exactly a cream puff'?",
 "answer": "2", "aliases": ["two", "twice"], "answer_type": "number", "tolerance": 0, "evidence_type": "dialogue", "all": true,
 "evidence": [{"start": "03:02", "end": "03:05", "source": "said", "note": "to the reporters on death row: 'I ain't exactly a cream puff. Cigarette?'"},
              {"start": "1:00:10", "end": "1:00:13", "source": "said", "note": "to the brunette in his studio: 'Well, I ain't exactly a cream puff sister.' (transcript 1:00:09)"}]}
```
→ `dialogue`, `multi-scene`: a count that needs both scenes (`all`); a `number` answer with `tolerance` 0.

```json
{"id": "dt45_024", "film": "mf2_49",
 "question": "What slogan is written above the place Vera picks when she says 'Let's try this place in the middle of the block'?",
 "answer": "Cash for Your Car", "aliases": ["cash for your car"], "evidence_type": "dialogue and text",
 "evidence": [{"start": "52:45", "end": "52:47", "source": "said", "note": "identifies the place: Vera, 'Let's try this place in the middle of the block.'"},
              {"start": "52:48", "end": "52:54", "source": "read", "note": "big roof sign 'CASH FOR YOUR CAR' over the used-car lot they pull into"}],
 "all": true}
```
→ `dialogue and text`, `moment`: the line identifies the place and the sign gives the answer; both are needed.

```json
{"id": "hh53_019", "film": "mf2_43",
 "question": "What is the name of Roy Collins's wife?",
 "answer": "not shown in the film", "aliases": ["not stated", "the film does not say", "not shown", "not mentioned"],
 "evidence_type": "unanswerable", "evidence": []}
```
→ `unanswerable` control, `movie-whole`.
