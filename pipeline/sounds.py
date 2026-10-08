"""Sounds lane ("heard", input A): what is heard other than speech, as timestamped AudioSet labels.

The soundtrack is tagged with an AudioSet classifier in short overlapping windows; a label is kept in a window when it
is not a speech, scene or generic parent label and its probability reaches the threshold for its kind (music or
other). Consecutive windows keeping the same label are merged into one event. Readers see each event as
`[start–end] Sound: label`. One sound can produce several events with related labels (e.g. bell and church bell).
The tagger must differ from the one in pipeline.validation_sounds (AST), which is used to check heard evidence.

Model
    PANNs CNN14 AudioSet tagger (panns_inference 0.1.1, checkpoint Cnn14_mAP=0.431.pth, sha256 0dc499e4…, read from
    $HOME/panns_data/), clip-wise sigmoid outputs over the 527 AudioSet labels
Settings (constants below)
    SR 32000               mono, the rate CNN14 was trained at
    WINDOW_S 4, HOP_S 3    window starts 0, 3, 6, … while start < length − 4 s (the final partial window is not
                           tagged); the same windows as the validation tagger
    MUSIC_THRESHOLD 0.25   for the 157 music labels: AudioSet "Singing" … "Humming" and "Music" … "Scary music";
                           keeps instrument and singing detail
    NONMUSIC_THRESHOLD 0.4 for every other label (--nonmusic-threshold). On 148 non-speech sounds captioned by hand in
                           four 30-minute stretches, 0.4 instead of 0.25 for every label cut wrong or unsupported label
                           runs from 155 to 63 and the captioned sounds named from 66 to 58
    SPEECH_AND_SCENE       21 labels never kept: speech reaches readers through the transcript; room, scene and noise
                           labels name no sound event
    GENERIC_PARENTS        18 ontology parents never kept: they only repeat a more specific child label
    BATCH 64               windows per inference call
Merging: a run of a label starts at the start of the first window keeping it and ends at the end (start + 4 s) of the
last; a window without the label closes the run only when its start is more than HOP_S after the run's end, so up to
two consecutive windows without the label are bridged. Event times are rounded to 0.1 s, labels lower-cased, events
sorted by start, then label.
Input
    the source video (common.source_video), decoded by ffmpeg
Output
    DATA/sounds/<video_id>.json (or --out-dir) = {video_id, model, window_s, hop_s, music_threshold,
                                                  nonmusic_threshold, events: [{start, end, sound}], provenance}
Run
    $SHARED_ROOT/envs/panns_env/bin/python -m pipeline.sounds <video_id> [...] --device cuda [--nonmusic-threshold 0.4]
        [--out-dir DIR]
    from the repo root, with HOME holding panns_data/ (pipeline.extract sets HOME to the shared model cache); or
    through pipeline.extract. --device defaults to cpu.
"""
import argparse
from pathlib import Path

import numpy as np

from .common import DATA, audio, provenance, source_video, write_json

MODEL = "PANNs CNN14 (AudioSet, Kong et al. 2020)"
SR = 32000                # the rate CNN14 was trained at
WINDOW_S, HOP_S = 4, 3    # the windows of the validation tagger, so the two taggers' runs line up
MUSIC_THRESHOLD = 0.25    # keeps instrument and singing detail; the false labels at 0.25 are mostly non-music
NONMUSIC_THRESHOLD = 0.4  # more than halves the false labels of 0.25 (animals, birds) at a small loss of recall
BATCH = 64                # windows per inference call
MUSIC_RANGES = (("Singing", "Humming"), ("Music", "Scary music"))  # AudioSet label order: the singing and music branches
SPEECH_AND_SCENE = {  # speech reaches readers through the transcript; scene and noise labels name no sound event
    "Speech", "Male speech, man speaking", "Female speech, woman speaking", "Child speech, kid speaking",
    "Conversation", "Narration, monologue", "Babbling", "Speech synthesizer", "Whispering", "Chatter",
    "Inside, small room", "Inside, large room or hall", "Inside, public space", "Outside, urban or manmade",
    "Outside, rural or natural", "Silence", "Sound effect", "Noise", "Environmental noise", "Static", "Hum"}
GENERIC_PARENTS = {  # ontology parents that only repeat a more specific child label
    "Animal", "Domestic animals, pets", "Livestock, farm animals, working animals", "Wild animals", "Vehicle",
    "Motor vehicle (road)", "Mechanisms", "Tools", "Liquid", "Wood", "Glass", "Domestic sounds, home sounds",
    "Musical instrument", "Plucked string instrument", "Bowed string instrument", "Brass instrument",
    "Percussion", "Keyboard (musical)"}


def music_ids(labels: list[str]) -> set[int]:
    return set().union(*(range(labels.index(a), labels.index(b) + 1) for a, b in MUSIC_RANGES))


def kept(p: np.ndarray, labels: list[str], music: set[int], nonmusic_threshold: float) -> set[str]:
    """Labels kept in one window with probabilities p."""
    return {labels[i] for i in np.flatnonzero(p >= min(MUSIC_THRESHOLD, nonmusic_threshold))
            if labels[i] not in SPEECH_AND_SCENE | GENERIC_PARENTS
            and p[i] >= (MUSIC_THRESHOLD if i in music else nonmusic_threshold)}


def merge(windows: list[tuple[float, set[str]]]) -> list[dict]:
    """(window start in s, labels kept) in time order -> events [{start, end, sound}] sorted by start, then label."""
    runs, events = {}, []
    for t0, labs in windows:
        for lab in list(runs):
            if lab not in labs and t0 - runs[lab][1] > HOP_S:
                events.append({"start": round(runs[lab][0], 1), "end": round(runs[lab][1], 1), "sound": lab.lower()})
                del runs[lab]
        for lab in labs:
            runs[lab] = [runs[lab][0] if lab in runs else t0, t0 + WINDOW_S]
    events += [{"start": round(r[0], 1), "end": round(r[1], 1), "sound": lab.lower()} for lab, r in runs.items()]
    events.sort(key=lambda e: (e["start"], e["sound"]))
    return events


def tag(tagger, labels: list[str], x: np.ndarray, nonmusic_threshold: float) -> tuple[list, list]:
    """(window starts in samples, events) for mono audio x at SR."""
    music = music_ids(labels)
    win, hop = WINDOW_S * SR, HOP_S * SR
    starts = list(range(0, max(1, len(x) - win), hop))
    windows = []
    for b in range(0, len(starts), BATCH):
        probs, _ = tagger.inference(np.stack([x[s:s + win] for s in starts[b:b + BATCH]]))
        windows += [(s / SR, kept(p, labels, music, nonmusic_threshold)) for s, p in zip(starts[b:b + BATCH], probs)]
    return starts, merge(windows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video_ids", nargs="+")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--nonmusic-threshold", type=float, default=NONMUSIC_THRESHOLD)
    ap.add_argument("--out-dir", default=str(DATA / "sounds"))
    a = ap.parse_args()
    from panns_inference import AudioTagging
    from panns_inference.config import labels
    tagger = AudioTagging(checkpoint_path=None, device=a.device)
    prov = provenance(__file__, MODEL, sr=SR, window_s=WINDOW_S, hop_s=HOP_S, music_threshold=MUSIC_THRESHOLD,
                      nonmusic_threshold=a.nonmusic_threshold, music_ranges=MUSIC_RANGES,
                      speech_and_scene=sorted(SPEECH_AND_SCENE), generic_parents=sorted(GENERIC_PARENTS), batch=BATCH,
                      device=a.device)
    for vid in a.video_ids:
        starts, events = tag(tagger, labels, audio(source_video(vid), SR), a.nonmusic_threshold)
        write_json(Path(a.out_dir) / f"{vid}.json",
                   {"video_id": vid, "model": MODEL, "window_s": WINDOW_S, "hop_s": HOP_S,
                    "music_threshold": MUSIC_THRESHOLD, "nonmusic_threshold": a.nonmusic_threshold, "events": events,
                    "provenance": prov})
        print(f"{vid}: {len(events)} sound events from {len(starts)} windows", flush=True)


if __name__ == "__main__":
    main()
