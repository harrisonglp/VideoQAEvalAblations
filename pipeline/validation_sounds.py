"""Validation sound tags (never shown to a reader): AudioSet tags from a second tagger, used to locate and check
`heard` evidence when questions are written and validated (pipeline.materials -> sounds.tsv -> pipeline.validate).

Readers receive only the sounds lane (pipeline.sounds, PANNs); this tagger must stay different from it, so that
heard evidence is not checked against the track under test.

Model
    Audio Spectrogram Transformer MIT/ast-finetuned-audioset-10-10-0.4593 (snapshot f826b80d, transformers 4.46.3)
Settings (constants below)
    SR 16000                mono 16-bit PCM, the AST feature extractor's rate
    WINDOW_S 4, --hop 3     4 s windows every 3 s; starts while start < length − 4 s
    TOP_K 8                 per window, those of the 8 highest-scoring labels that reach THRESHOLD and are not in
    THRESHOLD 0.25          SPEECH (11 speech, room and silence labels)
    SPARSE_THRESHOLD 0.08   any of the 16 SPARSE labels (gunfire, explosions, whistles, bells, sirens, thunder, drums,
                            ...) from p >= 0.08, so that isolated events can be located and counted
    BATCH 16                windows per forward pass
Input
    the source video, decoded once by ffmpeg to DATA/audio/<video_id>.wav (kept as a cache)
Output
    DATA/sound_events/<video_id>.json = {video_id, model, window_s, hop_s, threshold, sparse_threshold,
                                         events: [{t0, t1, labels: [[label, p]]}], provenance}, one event per window
                                         with a label
Run
    $SHARED_ROOT/envs/florence_env/bin/python -m pipeline.validation_sounds <video_id> [...] --device cuda --hop 3
    from the repo root, with the model in the Hugging Face cache (HF_HOME); or pipeline.extract --validation
"""
import argparse
import subprocess
import wave

import numpy as np
import torch
from transformers import ASTFeatureExtractor, ASTForAudioClassification

from .common import DATA, provenance, source_video, write_json

MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"
SR = 16000                # the AST feature extractor's rate
WINDOW_S = 4              # the sounds lane uses the same windows, so the two taggers' runs line up
HOP_S = 3.0               # default of --hop
THRESHOLD = 0.25          # minimum p among the TOP_K labels of a window
TOP_K = 8
SPARSE_THRESHOLD = 0.08   # a short event in a 4 s window scores low; kept so that it can be found and counted
BATCH = 16                # windows per forward pass
SPEECH = {"Speech", "Male speech, man speaking", "Female speech, woman speaking", "Child speech, kid speaking",
          "Conversation", "Narration, monologue", "Babbling", "Speech synthesizer", "Inside, small room",
          "Inside, large room or hall", "Silence"}
SPARSE = {"Gunshot, gunfire", "Machine gun", "Explosion", "Artillery fire", "Cap gun", "Fusillade", "Whistle", "Bell",
          "Church bell", "Siren", "Boom", "Burst, pop", "Thunder", "Bugle", "Drum", "Marching"}


def load_wav(video_id: str) -> tuple[int, np.ndarray]:
    wav = DATA / "audio" / f"{video_id}.wav"
    if not wav.exists():
        wav.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(source_video(video_id)), "-vn",
                        "-ac", "1", "-ar", str(SR), "-c:a", "pcm_s16le", str(wav)], check=True)
    with wave.open(str(wav)) as w:
        return w.getframerate(), np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0


def tag(fe, model, device: str, sr: int, x: np.ndarray, hop_s: float) -> list[dict]:
    labels = model.config.id2label
    sparse = [i for i, l in labels.items() if l in SPARSE]
    win, hop = int(WINDOW_S * sr), int(hop_s * sr)
    starts = list(range(0, max(1, len(x) - win), hop))
    events = []
    for b in range(0, len(starts), BATCH):
        batch = starts[b:b + BATCH]
        feats = fe([x[s:s + win] for s in batch], sampling_rate=sr, return_tensors="pt")
        with torch.no_grad():
            probs = torch.sigmoid(model(**{k: v.to(device) for k, v in feats.items()}).logits).float().cpu().numpy()
        for s, p in zip(batch, probs):
            top = [(labels[i], float(p[i])) for i in np.argsort(-p)[:TOP_K] if p[i] >= THRESHOLD and labels[i] not in SPEECH]
            top += [(labels[i], float(p[i])) for i in sparse if SPARSE_THRESHOLD <= p[i] and labels[i] not in dict(top)]
            if top:
                events.append({"t0": round(s / sr, 1), "t1": round((s + win) / sr, 1), "labels": [[l, round(v, 3)] for l, v in top]})
    return events


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video_ids", nargs="+")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--hop", type=float, default=HOP_S)
    a = ap.parse_args()
    fe = ASTFeatureExtractor.from_pretrained(MODEL)
    model = ASTForAudioClassification.from_pretrained(MODEL).eval().to(a.device)
    prov = provenance(__file__, MODEL, sr=SR, window_s=WINDOW_S, hop_s=a.hop, top_k=TOP_K, threshold=THRESHOLD,
                      sparse_threshold=SPARSE_THRESHOLD, speech=sorted(SPEECH), sparse=sorted(SPARSE), batch=BATCH,
                      device=a.device)
    for vid in a.video_ids:
        sr, x = load_wav(vid)
        events = tag(fe, model, a.device, sr, x, a.hop)
        write_json(DATA / "sound_events" / f"{vid}.json",
                   {"video_id": vid, "model": MODEL, "window_s": WINDOW_S, "hop_s": a.hop, "threshold": THRESHOLD,
                    "sparse_threshold": SPARSE_THRESHOLD, "events": events, "provenance": prov})
        print(f"{vid}: {len(events)} windows with non-speech sound")


if __name__ == "__main__":
    main()
