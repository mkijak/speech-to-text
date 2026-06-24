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
import re
import sys
import gc
import time
import json
import inspect
import traceback
from pathlib import Path

import whisperx

try:
    import torch
except Exception:  # pragma: no cover - torch is always present with whisperx
    torch = None

# ---- config from environment -------------------------------------------------
def _int_or_none(v):
    v = (v or "").strip()
    return int(v) if v else None

MODEL        = os.getenv("MODEL", "large-v3")
LANGUAGE     = os.getenv("LANGUAGE", "pl").strip() or None      # None = autodetect
DEVICE       = os.getenv("DEVICE", "cuda")
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "float16")             # cpu: use int8
BATCH_SIZE   = int(os.getenv("BATCH_SIZE", "16"))
# Quality knobs (you have GPU headroom). Higher beam = more accurate, a bit slower.
BEAM_SIZE    = int(os.getenv("BEAM_SIZE", "5"))
# Prime the model with domain vocabulary / names so it spells jargon correctly.
INITIAL_PROMPT = os.getenv("INITIAL_PROMPT", "").strip() or None
DIARIZE      = os.getenv("DIARIZE", "1") not in ("0", "false", "False", "")
HF_TOKEN     = os.getenv("HF_TOKEN", "").strip()
# Pin the diarization model. Empty = whisperx default (currently
# pyannote/speaker-diarization-community-1). Set e.g. pyannote/speaker-diarization-3.1
# to force a model you've already been granted access to.
DIARIZE_MODEL = os.getenv("DIARIZE_MODEL", "").strip() or None
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


def _free():
    """Release cached VRAM so the next stage/file gets a clean, unfragmented heap.
    Without this the caching allocator hoards memory across files -> 'worked once,
    then OOM' on the second recording."""
    gc.collect()
    if torch is not None and DEVICE == "cuda":
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


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
        asr_options = {"beam_size": BEAM_SIZE}
        if INITIAL_PROMPT:
            asr_options["initial_prompt"] = INITIAL_PROMPT
        self.asr = whisperx.load_model(
            MODEL, DEVICE, compute_type=COMPUTE_TYPE, language=LANGUAGE,
            asr_options=asr_options)

        # Alignment model is language-specific; load lazily per detected language.
        self._align_cache = {}

        self.diarizer = None
        if DIARIZE:
            if not HF_TOKEN:
                log("DIARIZE=1 but HF_TOKEN is empty -> diarization DISABLED. "
                    "Set HF_TOKEN and accept the pyannote model licenses to enable.")
            else:
                self.diarizer = _load_diarizer(HF_TOKEN, DEVICE, DIARIZE_MODEL)
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

    def _apply_options(self, prompt, beam):
        # Per-file overrides of the ASR options that were baked in at load time.
        # Option container differs across versions (NamedTuple vs dataclass), so
        # set each key the most compatible way and ignore unknown ones.
        wanted = {}
        if prompt is not None:
            wanted["initial_prompt"] = prompt
        if beam is not None:
            wanted["beam_size"] = beam
        for k, v in wanted.items():
            try:
                self.asr.options = self.asr.options._replace(**{k: v})
            except Exception:
                try:
                    setattr(self.asr.options, k, v)
                except Exception:
                    pass

    def run(self, audio_path, min_speakers=None, max_speakers=None,
            prompt=None, language=None, beam=None):
        self._apply_options(prompt, beam)
        audio = whisperx.load_audio(str(audio_path))
        kwargs = {"batch_size": BATCH_SIZE}
        if language:
            kwargs["language"] = language
        try:
            result = self.asr.transcribe(audio, **kwargs)
        except TypeError:
            result = self.asr.transcribe(audio, batch_size=BATCH_SIZE)
        lang = result.get("language", language or LANGUAGE) or "en"
        _free()  # release the decode batch before loading the alignment model

        segments = self._align(result["segments"], lang, audio)
        _free()

        if self.diarizer is not None:
            try:
                diar = self.diarizer(
                    audio, min_speakers=min_speakers, max_speakers=max_speakers)
                merged = _assign_speakers(diar, {"segments": segments})
                segments = merged["segments"]
            except Exception as e:
                log(f"diarization failed (writing transcript without speakers): {e}")
        del audio
        _free()  # clean heap before the next file's transcribe batch
        return segments


def _load_diarizer(token, device, model_name=None):
    # Class location moved across whisperx versions; try the known spots.
    try:
        from whisperx.diarize import DiarizationPipeline
    except Exception:
        from whisperx import DiarizationPipeline  # older layout
    # Signatures drift across versions: the auth kwarg was renamed
    # use_auth_token -> token, and we only pass model_name if accepted.
    params = inspect.signature(DiarizationPipeline.__init__).parameters
    kwargs = {"device": device}
    if model_name and "model_name" in params:
        kwargs["model_name"] = model_name
    if "use_auth_token" in params:
        kwargs["use_auth_token"] = token
    elif "token" in params:
        kwargs["token"] = token
    return DiarizationPipeline(**kwargs)


def _assign_speakers(diar_segments, result):
    try:
        from whisperx.diarize import assign_word_speakers
    except Exception:
        from whisperx import assign_word_speakers
    return assign_word_speakers(diar_segments, result)


# ---- file handling -----------------------------------------------------------
# Speaker hint in the filename: "3spk" = exactly 3, "2-4spk" = a range.
SPK_RE = re.compile(r"(\d+)\s*(?:-\s*(\d+))?\s*spk", re.I)


def parse_speakers(stem):
    """Return (min_speakers, max_speakers, clean_stem) from a filename stem.

    Reads a "<n>spk" / "<a>-<b>spk" marker anywhere in the name and strips it
    from the output name. Falls back to the env defaults when no marker present.
    """
    m = SPK_RE.search(stem)
    if not m:
        return MIN_SPEAKERS, MAX_SPEAKERS, stem
    lo = int(m.group(1))
    hi = int(m.group(2)) if m.group(2) else lo
    clean = (stem[:m.start()] + stem[m.end():]).strip(" ._-") or stem
    return lo, hi, clean


def load_instructions(audio_path):
    """Optional per-file instruction sidecar: a '<name>.txt' next to the audio
    with simple key=value lines (# comments allowed), e.g.:

        speakers = 4            # exact count, or a range like 2-5
        prompt   = ...          # vocabulary / context for THIS recording
        language = pl           # optional, overrides default
        beam     = 10           # optional, overrides default

    Returns a dict of the keys present (empty if no sidecar)."""
    _, _, clean = parse_speakers(audio_path.stem)
    for cand in (
        audio_path.with_name(audio_path.stem + ".txt"),
        audio_path.with_name(clean + ".txt"),
    ):
        try:
            if not cand.is_file():
                continue
            data = {}
            for line in cand.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                data[k.strip().lower()] = v.strip()
            return data
        except OSError:
            pass
    return {}


def _speakers_from_str(s):
    m = re.match(r"\s*(\d+)\s*(?:-\s*(\d+))?\s*$", s or "")
    if not m:
        return None, None
    lo = int(m.group(1))
    hi = int(m.group(2)) if m.group(2) else lo
    return lo, hi


def out_paths(stem):
    return OUT_DIR / f"{stem}.txt", OUT_DIR / f"{stem}.srt"


def already_done(audio_path):
    _, _, stem = parse_speakers(audio_path.stem)
    txt, _ = out_paths(stem)
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
    # Filename speaker marker is the fallback; the instruction file overrides it.
    lo, hi, stem = parse_speakers(audio_path.stem)
    instr = load_instructions(audio_path)
    if "speakers" in instr:
        s_lo, s_hi = _speakers_from_str(instr["speakers"])
        if s_lo is not None:
            lo, hi = s_lo, s_hi
    prompt = instr.get("prompt") or INITIAL_PROMPT
    language = instr.get("language") or LANGUAGE
    beam = int(instr["beam"]) if instr.get("beam", "").isdigit() else BEAM_SIZE

    txt, srt = out_paths(stem)
    if lo is None and hi is None:
        spk = "auto"
    elif lo == hi:
        spk = str(lo)
    else:
        spk = f"{lo}-{hi}"
    extras = []
    if instr:
        extras.append("instructions")
    if prompt:
        extras.append("prompt")
    tag = f"  [{', '.join(extras)}]" if extras else ""
    log(f"transcribing: {audio_path.name}  (speakers: {spk}, beam: {beam}){tag}")
    t0 = time.monotonic()
    segments = pipe.run(audio_path, min_speakers=lo, max_speakers=hi,
                        prompt=prompt, language=language, beam=beam)
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
