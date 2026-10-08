# Evidence bank v1: writing and review protocol

Open short-answer questions about fifty public-domain films, 20 per film (1,000 in total), each labelled with the
evidence that answers it and where that evidence is in the film. The entry format is defined in `FORMAT.md`; this file
describes how the questions are written, checked and reviewed. The bank validates the ablation method: each question's
evidence type predicts which input conditions can answer it.

## Films

`films.json` lists the films (video id, code, title, year, source collection, video file): 28 from MF2 and 22 from
CHNB, released 1929–1963, across comedies, westerns, musicals, thrillers, war films, dramas and an animated feature.
Only the films are used; the collections' own questions are not.

## Materials

Questions are written from the film itself, not from the pipeline's tracks, so that whether each track captures the
evidence can be measured (rule 3). Writers and reviewers work from the video (480p), contact sheets and full frames,
ASR transcripts (Whisper), the human subtitles realigned to the audio where they match the cut, and a sound-event
tagger independent of the one that produces the readers' sound labels (AST: AudioSet labels in 4 s windows, used to
locate sounds). The times of the 256 frames the readers receive are known (frame k at (k + 0.5) × duration / 256), so
visual evidence can be placed where sampling reaches it.

| Source | What the reader gets |
|---|---|
| said | the transcript, speech only |
| seen | the 256 frames, with on-screen text blacked out |
| read | the on-screen text track (OCR at 1 frame/s, one entry per screen of text) |
| heard | the sound labels: non-speech sound events from an automatic tagger |

## Writing rules

1. **One short, checkable answer.** A word, name, number or short phrase. `aliases` cover every reasonable phrasing,
   including number words and spelling variants. Avoid questions whose correct answers could be phrased in many
   incomparable ways.
2. **The question locates its moment through the intended source and never gives the answer away.**
   - `visual` and `audio` questions describe people by appearance or role, not by names that are only spoken or
     written; using such a name turns the question into `dialogue and visual` or `text and visual`.
   - `dialogue` questions may locate the moment by what is said.
   - No times in the question.
3. **The evidence is defined by the film, not by the pipeline.** Questions are written by watching the film (its
   pictures, its dialogue, its written text and its sounds). Whether each track of the pipeline captures the evidence is
   measured afterwards, not assumed.
   - `seen` evidence is clearly visible for a sustained time (a shot of several seconds, or several shots), so that
     uniform frame sampling has a fair chance to include it.
   - `read` evidence is written on screen legibly (signs, inserts, letters, intertitles, credits).
   - `said` evidence is spoken (timed against the transcripts and subtitles, and the picture where they disagree).
   - `heard` evidence is a non-speech sound located with the independent sound tagger (confident, p ≥ 0.5, or repeated
     tags that fit the scene) or captioned in brackets in the human subtitles; counts come from separated runs of
     tagged windows, stated in the interval `note`.
4. **The evidence is nowhere else.** Each question is checked against every input its type excludes, across the whole
   film, for the answer and its synonyms: the transcript, the on-screen text track (credits, newspapers, signs,
   intertitles), the frames near the evidence and wherever the fact could recur, and the sound tags.
   - Silent-era intertitles are written text (`read`), even when they carry dialogue.
   - Something visible on screen that makes a sound (a gun fired in frame) is `visual or audio`, not `audio`.
5. **Not guessable.** Before writing the answer down, try to answer the question without the film. If the answer is the
   obvious default (a period telephone is black; a western has horses), reject it. `prior` questions are exempt.
6. **Combined types are tested, not assumed.**
   - `X or Y`: confirm that each source alone answers it.
   - `X and Y`: confirm that neither source alone answers it; typically one source identifies the moment or person
     and the other supplies the attribute.
7. **Controls.**
   - `prior`: a well-documented fact about the film (source novel, studio, award, a lead actor's role) that appears in
     **no** input, including the credits in the on-screen text track.
   - `unanswerable`: a natural question about the story whose answer is never said, shown, written or heard; the
     answer is `"not shown in the film"`.
8. **Natural, varied phrasing.** Questions read like a viewer's questions. Do not reuse a stock template per evidence
   type ("What sound other than speech…", "What colour is…"); the wording must not reveal the evidence type more than
   the content requires.
9. **Variety.** Within a film, spread questions over the whole running time, over characters and settings, over
   answer kinds (object, colour, number, name, place, action, reason, sound) and over temporal types (single moments,
   scenes, facts that recur, counts and changes across scenes).

## Composition (20 questions per film, 1,000 in total)

| evidence_type | target per film | questions |
|---|---|---|
| dialogue | 3 | 196 |
| visual | 3 | 162 |
| text | 2 | 107 |
| audio | 2 | 48 |
| two sources joined by `or` | 3 | 134 |
| two sources joined by `and` | 3 | 151 |
| prior | 1 | 50 |
| unanswerable | 3 | 152 |

Two-source questions may pair any two sources; the target was at least one `or` question per film involving text or
audio, and `and` questions involving text or audio across the film set. Text or audio is involved in 53 `or` questions
(39 films) and 36 `and` questions (28 films). Films that lack a source (little dialogue in Gulliver's Travels, few
distinct sound effects or little legible in-story text in many early films) shift questions to the nearest available
type. Audio falls short most: 48 questions are audio alone and 75 involve audio in any form.

Across the bank: 256 `moment`, 324 `scene`, 143 `movie-any`, 75 `multi-scene` and 202 `movie-whole` questions (the
`prior` and `unanswerable` questions), 151 `number` answers and one `time` answer. Every film has at least 4 `moment`
and 5 `scene` questions. Writing aimed at 3 multi-interval questions (`movie-any` or `multi-scene`) and 3 `number`
answers per film; after the closed-book revisions (review step 5), which replaced mostly guessable small counts,
Cyrano de Bergerac, Little Lord Fauntleroy and The Hitch-Hiker have 2 multi-interval questions, and 11 films have 2
`number` answers. Per film, at most one question's text evidence is only in the credits, at most one asks for a cast
member, and at most one is a count answered 2 or 3.

## Review

1. **Writing:** one writer per film, from the film, following the rules above; each question was checked against every
   input its type excludes.
2. **Independent review:** a second reviewer, who did not write the film's questions, re-derives every answer from the
   film, attempts each question from each single source to confirm the intended type, and checks wording, aliases,
   leakage and guessability. Verdicts: keep, fix (with the change) or drop (with the reason and a replacement
   question). Of the 1,000 questions, 385 were kept, 376 fixed and 239 dropped and replaced.
3. **Bank-level checks and final pass:** quoted lines are timed against the transcripts and subtitles, visual evidence
   against the 256 frame times and text evidence against the credits; the per-film caps, arithmetic answers (rewritten
   to ask for the stated fact) and answer leaks are checked across the bank. Every flagged question is re-checked
   against the film and fixed, replaced or confirmed with a reason: 119 flagged, 27 fixed, 17 replaced, 75 confirmed.
4. **Automatic checks:** a validator enforces `FORMAT.md` and reports possible leaks (the answer in the transcript or
   the on-screen text track when the type excludes that input) and missing support (a `read` answer not in the text
   track at the cited times). All 1,000 questions are valid; 322 carry warnings, mostly support that cannot be
   verified automatically (a paraphrased quote, an answer not verbatim in a track) and uncited occurrences of the answer
   in a cited input, plus weak leaks (a common answer word elsewhere in a track). The 15 questions with a near leak
   (within 2 min of the evidence) were each checked against the film and kept: the matched word does not answer them.
5. **Closed-book screen:** the study's reader attempts every question without the film: from the question alone and
   from the title and year (4 sampled runs each). Questions it answers correctly in at least 2 of the 4 runs of either
   condition are revised against the film, except `prior` and `unanswerable` questions: 66 of the 798
   evidence-bearing questions (36 films), of which 40 were reworded and 26 replaced. The revised questions are screened
   again, and the bank is then frozen (hashed) before the main runs.
