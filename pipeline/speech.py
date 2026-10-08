"""Speech lane ("said", input D): what is said in the film, as timestamped transcript lines.

Speech is located with Silero VAD on a vocals stem separated by htdemucs, so that music and effects neither hide speech
from the VAD nor pass for it, and only the speech segments are transcribed, from the original mix. Each segment
becomes one line; its language is identified per segment, so non-English speech is transcribed in its own language.
Readers see each line at its segment start, rounded to the second, with no speaker label, after the speech-only filter
(conditions.clean_line) applied at prompt time.

Models
    htdemucs (Demucs v4, package demucs 4.0.1, weights 955717e8) on cuda:0: the vocals stem
    Silero VAD (package silero-vad 6.2.3), get_speech_timestamps: speech segments, on the stem
    Qwen/Qwen3-ASR-1.7B (snapshot 7278e1e7, package qwen-asr 0.0.6), bfloat16 on cuda:0: the text, from the mix
Vocals stem
    The soundtrack is decoded as stereo float32 at the separator's 44.1 kHz and normalised with the mean and standard
    deviation of the whole track's mono mix. It is separated in pieces of PIECE_S, each with CONTEXT_S of the
    neighbouring audio on both sides, by apply_model(split=True, overlap=OVERLAP), whose default shifts=1 moves each
    piece by a random offset of up to 0.5 s; the offsets come from Python's random module, seeded with SHIFT_SEED at
    the start of each film, so the stem is reproducible. The vocals source is de-normalised, averaged to mono, cut to
    its piece and resampled to 16 kHz (torchaudio.functional.resample); the pieces are concatenated.
Settings (constants below)
    SEPARATOR      htdemucs
    PIECE_S        600 s per separation piece
    CONTEXT_S      10 s of context on each side of a piece
    OVERLAP        0.25 between apply_model's segments within a piece
    SHIFT_SEED     0, the seed of the random shift offsets
    SR             16000 Hz mono, the input rate of Silero VAD and Qwen3-ASR; the mix as 16-bit PCM scaled to [-1, 1)
    VAD            min_silence_duration_ms 400, speech_pad_ms 200, max_speech_duration_s 20; other parameters at the
                   package defaults (threshold 0.5, min_speech_duration_ms 250, 512-sample window)
    LANGUAGE       None: language identified per segment
    MAX_BATCH      16 segments per inference batch
    MAX_NEW_TOKENS 256 per segment
    --batch        64 segments per transcribe() call (CALL_SEGMENTS)
Input
    the source video (common.source_video), decoded by ffmpeg: stereo at 44.1 kHz for the stem, mono at 16 kHz for
    the recogniser
Output
    DATA/transcripts/<video_id>.json = {video_id, model, vad: {VAD..., input: "htdemucs vocals stem"},
                                        lines: [{start, end, text}], provenance}
    one line per VAD segment, start and end in seconds (2 decimals); segments transcribed as empty are dropped
Run
    $SHARED_ROOT/envs/asr_qwen_env/bin/python -m pipeline.speech <video_id> [...] [--batch 64]
    from the repo root, on a GPU, with Qwen3-ASR in the Hugging Face cache (HF_HOME) and the htdemucs weights in the
    torch hub cache ($XDG_CACHE_HOME/torch/hub/checkpoints); or through pipeline.extract
"""
import argparse
import random
import subprocess

import numpy as np
import torch
import torchaudio

from .common import DATA, audio, provenance, source_video, write_json

MODEL = "Qwen/Qwen3-ASR-1.7B"
SEPARATOR = "htdemucs"
DEVICE = "cuda:0"
PIECE_S = 600         # separation piece, so memory does not grow with film length
CONTEXT_S = 10        # separated with the piece and cut afterwards, so the pieces join without edge effects
OVERLAP = 0.25        # between apply_model's segments within a piece
SHIFT_SEED = 0
SR = 16000            # the rate both Silero VAD and Qwen3-ASR expect
VAD = {"min_silence_duration_ms": 400,  # a pause of 0.4 s ends a segment, so lines follow phrase breaks
       "speech_pad_ms": 200,            # 0.2 s added at both ends so word onsets and endings are not cut
       "max_speech_duration_s": 20}     # longer speech is split, so each line stays close to its time in the film
LANGUAGE = None       # identify the language per segment; do not force English
DTYPE = torch.bfloat16
MAX_BATCH = 16        # segments per GPU inference batch
MAX_NEW_TOKENS = 256  # per segment; 20 s of speech needs far fewer
CALL_SEGMENTS = 64    # segments per transcribe() call (--batch); the model splits them into batches of MAX_BATCH


def load_audio(video_id: str) -> np.ndarray:
    return audio(source_video(video_id), SR)


def stereo(path, sr: int) -> torch.Tensor:
    """The soundtrack of a video file as a (2, n) float32 tensor at `sr` Hz (ffmpeg)."""
    raw = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-vn", "-ac", "2", "-ar", str(sr),
                          "-f", "f32le", "-"], capture_output=True, check=True).stdout
    return torch.from_numpy(np.frombuffer(raw, np.float32).reshape(-1, 2).T.copy())


def vocals(sep, wav: torch.Tensor) -> np.ndarray:
    """The vocals stem of a (2, n) soundtrack at sep.samplerate, as mono float32 at SR."""
    from demucs.apply import apply_model
    sr, k = sep.samplerate, sep.sources.index("vocals")
    ref = wav.mean(0)
    mu, sd = ref.mean(), ref.std()
    n, step, ctx = wav.shape[1], PIECE_S * sr, CONTEXT_S * sr
    pieces = []
    random.seed(SHIFT_SEED)
    for a in range(0, n, step):
        lo, hi = max(0, a - ctx), min(n, a + step + ctx)
        with torch.no_grad():
            v = apply_model(sep, ((wav[:, lo:hi] - mu) / sd).to(DEVICE)[None], split=True, overlap=OVERLAP,
                            progress=False)[0, k]
        v = (v * sd + mu).mean(0)[a - lo:a - lo + min(step, n - a)]
        pieces.append(torchaudio.functional.resample(v, sr, SR).cpu())
    return torch.cat(pieces).numpy().astype(np.float32)


def transcribe(model, vad, x: np.ndarray, stem: np.ndarray, call_segments: int = CALL_SEGMENTS) -> tuple[list, list]:
    """(VAD segments in samples, lines [{start, end, text}]): segments found on the vocals stem, transcribed from the
    mix x; both mono at SR."""
    from silero_vad import get_speech_timestamps
    segs = get_speech_timestamps(torch.from_numpy(stem), vad, sampling_rate=SR, **VAD)
    lines = []
    for i in range(0, len(segs), call_segments):
        chunk = segs[i:i + call_segments]
        results = model.transcribe(audio=[(x[g["start"]:g["end"]], SR) for g in chunk], language=LANGUAGE)
        lines += [{"start": round(g["start"] / SR, 2), "end": round(g["end"] / SR, 2), "text": r.text.strip()}
                  for g, r in zip(chunk, results) if r.text.strip()]
    return segs, lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video_ids", nargs="+")
    ap.add_argument("--batch", type=int, default=CALL_SEGMENTS)
    a = ap.parse_args()
    from demucs.pretrained import get_model
    from qwen_asr import Qwen3ASRModel
    from silero_vad import load_silero_vad
    sep = get_model(SEPARATOR).to(DEVICE).eval()
    vad = load_silero_vad()
    model = Qwen3ASRModel.from_pretrained(MODEL, dtype=DTYPE, device_map=DEVICE, max_inference_batch_size=MAX_BATCH,
                                          max_new_tokens=MAX_NEW_TOKENS)
    prov = provenance(__file__, {"separator": SEPARATOR, "vad": "Silero VAD", "asr": MODEL},
                      separator_sr=sep.samplerate, piece_s=PIECE_S, context_s=CONTEXT_S, overlap=OVERLAP,
                      shift_seed=SHIFT_SEED, sr=SR, vad=VAD, language=LANGUAGE, dtype=str(DTYPE), device=DEVICE,
                      max_batch=MAX_BATCH, max_new_tokens=MAX_NEW_TOKENS, call_segments=a.batch)
    for vid in a.video_ids:
        stem = vocals(sep, stereo(source_video(vid), sep.samplerate))
        segs, lines = transcribe(model, vad, load_audio(vid), stem, a.batch)
        write_json(DATA / "transcripts" / f"{vid}.json",
                   {"video_id": vid, "model": MODEL, "vad": {**VAD, "input": f"{SEPARATOR} vocals stem"},
                    "lines": lines, "provenance": prov})
        print(f"{vid}: {len(segs)} speech segments, {len(lines)} lines", flush=True)


if __name__ == "__main__":
    main()
