#!/usr/bin/env python3
"""
Transcribe + diarize audio with WhisperX, writing speaker-labeled text.

Drop audio files into /data/input; transcripts appear in /data/output as:
  <name>.txt   speaker-grouped, human-readable (rename SPEAKER_00 -> real names)
  <name>.srt   subtitle form, one cue per segment, speaker-prefixed

Behaviour is env-driven (see .env.example). Runs as a watcher by default
(stays up, model stays loaded in VRAM/RAM -> fast subsequent files), or
processes the files passed as CLI args once and exits.
"""
import os
import sys
import time
import json
import traceback
from pathlib import Path

import whisperx

# ---- config from environment -------------------------------------------------
def _int_or_none(v):
    v = (v or "").strip()
    return int(v) if v else None

MODEL        = os.getenv("MODEL", "large-v3")
LANGUAGE     = os.getenv("LANGUAGE", "pl").strip() or None      # None = autodetect
DEVICE       = os.getenv("DEVICE", "cuda")
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "float16")             # cpu: use int8
BATCH_SIZE   = int(os.getenv("BATCH_SIZE", "16"))
DIARIZE      = os.getenv("DIARIZE", "1") not in ("0", "false", "False", "")
HF_TOKEN     = os.getenv("HF_TOKEN", "").strip()
MIN_SPEAKERS = _int_or_none(os.getenv("MIN_SPEAKERS"))
MAX_SPEAKERS = _int_or_none(os.getenv("MAX_SPEAKERS"))
WATCH        = os.getenv("WATCH", "1") not in ("0", "false", "False", "")
POLL_SECONDS = int(os.getenv("POLL_SECONDS", "5"))

IN_DIR  = Path(os.getenv("INPUT_DIR", "/data/input"))
OUT_DIR = Path(os.getenv("OUTPUT_DIR", "/data/output"))

AUDIO_EXT = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus", ".aac",
             ".mp4", ".mkv", ".mov", ".webm", ".wma"}


def log(msg):
    print(f"[stt] {msg}", flush=True)


# ---- output formatting -------------------------------------------------------
def fmt_ts(seconds, srt=False):
    seconds = float(seconds or 0)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if srt:
        ms = int((seconds - int(seconds)) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
    return f"{h:02d}:{m:02d}:{s:02d}"


def write_txt(segments, path):
    """Speaker-grouped, readable transcript. Consecutive same-speaker lines merge."""
    blocks = []
    cur_spk = object()  # sentinel
    buf = []
    start = None
    for seg in segments:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        spk = seg.get("speaker", "SPEAKER_??")
        if spk != cur_spk:
            if buf:
                blocks.append((cur_spk, start, " ".join(buf)))
            cur_spk, start, buf = spk, seg.get("start"), [text]
        else:
            buf.append(text)
    if buf:
        blocks.append((cur_spk, start, " ".join(buf)))

    with path.open("w", encoding="utf-8") as f:
        for spk, start, text in blocks:
            f.write(f"[{spk}]  {fmt_ts(start)}\n{text}\n\n")


def write_srt(segments, path):
    with path.open("w", encoding="utf-8") as f:
        n = 0
        for seg in segments:
            text = (seg.get("text") or "").strip()
            if not text:
                continue
            n += 1
            spk = seg.get("speaker")
            prefix = f"[{spk}] " if spk else ""
            f.write(f"{n}\n{fmt_ts(seg.get('start'), srt=True)} --> "
                    f"{fmt_ts(seg.get('end'), srt=True)}\n{prefix}{text}\n\n")


# ---- model loading (once) ----------------------------------------------------
class Pipeline:
    def __init__(self):
        log(f"device={DEVICE} compute_type={COMPUTE_TYPE} model={MODEL} "
            f"language={LANGUAGE or 'auto'} diarize={DIARIZE}")
        self.asr = whisperx.load_model(
            MODEL, DEVICE, compute_type=COMPUTE_TYPE, language=LANGUAGE)

        # Alignment model is language-specific; load lazily per detected language.
        self._align_cache = {}

        self.diarizer = None
        if DIARIZE:
            if not HF_TOKEN:
                log("DIARIZE=1 but HF_TOKEN is empty -> diarization DISABLED. "
                    "Set HF_TOKEN and accept the pyannote model licenses to enable.")
            else:
                self.diarizer = _load_diarizer(HF_TOKEN, DEVICE)
                log("diarization pipeline loaded")

    def _align(self, segments, lang, audio):
        try:
            if lang not in self._align_cache:
                self._align_cache[lang] = whisperx.load_align_model(
                    language_code=lang, device=DEVICE)
            amodel, meta = self._align_cache[lang]
            return whisperx.align(segments, amodel, meta, audio, DEVICE,
                                  return_char_alignments=False)["segments"]
        except Exception as e:
            log(f"alignment skipped for language '{lang}': {e}")
            return segments

    def run(self, audio_path):
        audio = whisperx.load_audio(str(audio_path))
        result = self.asr.transcribe(audio, batch_size=BATCH_SIZE)
        lang = result.get("language", LANGUAGE) or "en"

        segments = self._align(result["segments"], lang, audio)

        if self.diarizer is not None:
            try:
                diar = self.diarizer(
                    audio, min_speakers=MIN_SPEAKERS, max_speakers=MAX_SPEAKERS)
                merged = _assign_speakers(diar, {"segments": segments})
                segments = merged["segments"]
            except Exception as e:
                log(f"diarization failed (writing transcript without speakers): {e}")
        return segments


def _load_diarizer(token, device):
    # Class location moved across whisperx versions; try the known spots.
    try:
        from whisperx.diarize import DiarizationPipeline
    except Exception:
        from whisperx import DiarizationPipeline  # older layout
    return DiarizationPipeline(use_auth_token=token, device=device)


def _assign_speakers(diar_segments, result):
    try:
        from whisperx.diarize import assign_word_speakers
    except Exception:
        from whisperx import assign_word_speakers
    return assign_word_speakers(diar_segments, result)


# ---- file handling -----------------------------------------------------------
def out_paths(audio_path):
    stem = audio_path.stem
    return OUT_DIR / f"{stem}.txt", OUT_DIR / f"{stem}.srt"


def already_done(audio_path):
    txt, _ = out_paths(audio_path)
    return txt.exists()


def stable(path, prev_sizes):
    """True once a file's size has stopped changing (avoids half-copied files)."""
    try:
        size = path.stat().st_size
    except OSError:
        return False
    ok = prev_sizes.get(path) == size and size > 0
    prev_sizes[path] = size
    return ok


def process(pipe, audio_path):
    txt, srt = out_paths(audio_path)
    log(f"transcribing: {audio_path.name}")
    t0 = time.monotonic()
    segments = pipe.run(audio_path)
    write_txt(segments, txt)
    write_srt(segments, srt)
    log(f"done: {txt.name}  ({time.monotonic() - t0:.0f}s)")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pipe = Pipeline()

    args = [Path(a) for a in sys.argv[1:]]
    if args:  # one-shot mode
        for p in args:
            try:
                process(pipe, p)
            except Exception:
                log(f"ERROR on {p}:\n{traceback.format_exc()}")
        return

    IN_DIR.mkdir(parents=True, exist_ok=True)
    log(f"watching {IN_DIR} (drop audio files here)")
    prev_sizes = {}
    while True:
        for p in sorted(IN_DIR.iterdir()):
            if (p.is_file() and p.suffix.lower() in AUDIO_EXT
                    and not already_done(p) and stable(p, prev_sizes)):
                try:
                    process(pipe, p)
                except Exception:
                    log(f"ERROR on {p.name}:\n{traceback.format_exc()}")
        if not WATCH:
            break
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
