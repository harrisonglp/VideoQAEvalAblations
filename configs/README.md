# Conditions

Every question gets asked many times. Each time, the reader gets a different slice of the film: maybe just the
transcript, maybe the frames and the sounds, maybe only the couple of minutes around the moment that answers it. Each
of those slices is a *condition*. Look at which conditions a question is answered under and you can tell what evidence
it actually needs.

The conditions are defined in [`conditions.json`](conditions.json) and built in
[`pipeline/conditions.py`](../pipeline/conditions.py). Results are stored one file per condition, so
`results/evidence_bank_v1/gemma4-31b/w_tdv.jsonl` holds every answer given under `w_tdv`.

## Reading the names

A name is made of letters, one per input, always in the order t, d, v, r, a:

| Letter | Input | What the reader gets |
|---|---|---|
| `t` | title | the film's title and year |
| `d` | dialogue | the speech transcript, one timestamped line per segment, no speaker names |
| `v` | visual | 256 frames sampled evenly through the film, with the text detected on screen blacked out |
| `r` | read | the on-screen text found by OCR, with when each screen of text is visible |
| `a` | audio | sound labels from an audio tagger (music, gunshot, applause...), with when each is heard |

So `td` is the title and transcript, `tvra` is everything except the transcript, and `tdvra` is everything. Inputs
are interleaved in time order in the prompt, each entry with its timestamp.

The frames have the text masked so that "what's on screen" and "what's written on screen" stay separate inputs. If
the frames kept their text, a frames-only condition could answer questions about signs and letters, and you couldn't
tell the two apart.

## The groups

**Baselines**

- `q`: the question alone. No title, no film. Anything answered here is a guess (or the question gives it away).
- `t`: the title and year only. Anything answered here the model already knew about the film.

**Whole film** (15 conditions): the title plus every combination of the four lanes over the entire film:
`td`, `tv`, `tr`, `ta`, `tdv`, `tdr`, `tda`, `tvr`, `tva`, `tra`, `tdvr`, `tdva`, `tdra`, `tvra`, `tdvra`.

**Excerpts** (15 conditions, prefix `w_`): the same 15 combinations, but only what falls within 60 seconds of the
evidence the question cites. When a question cites several moments, the excerpts are joined with a marker for each
stretch that was skipped. These are the same frames, lines and labels the whole film has, just fewer of them: a
typical excerpt has about 8 of the 256 frames. Only questions that cite evidence get excerpt conditions (798 of the
1,000).

The excerpts are where the sorting really happens. With the right minutes in front of it the reader isn't hunting
through a 90-minute film, so whether a lane is enough on its own comes down to whether that lane carries the answer.

**Necessity** (whole film, all inputs)

- `tdvra_cut`: everything, minus the evidence. The cited moments (plus 10 s either side) are cut out, and so is every
  transcript line, screen of text and sound label anywhere in the film that states the answer. Small numbers (up to
  12) are left alone because words like "two" or "three" show up everywhere. Each cut is replaced by a marker so the
  reader knows something is missing.
- `tdvra_placebo`: the same amount of film cut, but from random places away from the evidence and from any mention of
  the answer.

If a question is answered with the whole film, fails after the cut and survives the placebo, the cited evidence really
is what the answer depends on. It's not that the film just got shorter.

**Masking check**

- `tdv_fullframes`: title, transcript and frames *with* their on-screen text visible, and no OCR or sound lanes. Next to
  `tdv` it shows what the masking takes away; next to `tdvra` it shows how much the OCR lane adds over reading text
  off the frames yourself.
- `w_tdv_fullframes`: the same, restricted to the excerpts, so it can be compared with `w_tdv`.

## How a question gets sorted

A question counts as answered under a condition when its score there is at least 0.5 (with one run per reader that
just means correct). The measured evidence type is the smallest set of lanes that answers it:

- right under `q`: *guessable*
- right under `t`: *prior* (known without watching)
- right with one lane only, say `w_ta` but not `w_td`, `w_tv` or `w_tr`: a *sound* question
- right with either of two lanes alone: "X or Y"
- right only when two lanes are together: "X and Y"
- wrong even with `w_tdvra`: undetermined (the reader can't answer it at all)

Compare that with the type the question was written with and you get a check on the labels. `pipeline/analyze.py`
does all this, plus Shapley values for how much each lane adds, and writes `summary.json` and `units.csv` (one row
per question, one column per condition).

## Which experiment runs which

| Experiment | Conditions |
|---|---|
| `evidence_bank_v1` | `q`, `t`, all 15 whole-film, all 15 excerpt, `tdv_fullframes`, `w_tdv_fullframes` (34) |
| `evidence_bank_v1_necessity` | `tdvra_cut`, `tdvra_placebo` (15 films) |
| `evidence_bank_v1_reliability` | the main ones minus the whole-film pairs and triples (23), 4 sampled runs |
| `evidence_bank_v1_screen` | `q`, `t`, 4 sampled runs |
| `evidence_bank_v1_panel` | `q`, `t`, the 5 whole-film single-lane and all-lane sets, all 15 excerpt sets, for every evaluation reader |
| `evidence_bank_v1_panel_wholefilm` | the 10 whole-film pairs and triples, for every evaluation reader |
| `evidence_bank_v1_smoke` | as the panel experiment, on 2 films |

To see exactly what the reader gets for a question under a condition:

```bash
python3 scripts/show_prompts.py w_tdv hh53_007
```
