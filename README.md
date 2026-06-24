# speech-to-text

Standalone, containerized **WhisperX** pipeline: audio in → speaker-labeled
transcript out. Transcription (`large-v3`) **and** diarization (who-said-what)
in one step. Runs on GPU or CPU.

```
speech-to-text/
  input/   ← drop audio here (mp3, wav, m4a, flac, ogg, mp4, mkv, ...)
  output/  ← <name>.txt (speaker-grouped) and <name>.srt appear here
```

Speakers come out as `SPEAKER_00`, `SPEAKER_01`, … — find-and-replace them with
real names afterward.

---

## 1. One-time setup

**Diarization token (for speaker labels).** Free, ~2 min:
1. Create a token: <https://huggingface.co/settings/tokens>
2. Click **Agree and access** on both gated models:
   - <https://huggingface.co/pyannote/segmentation-3.0>
   - <https://huggingface.co/pyannote/speaker-diarization-3.1>
3. `cp env.example .env` and put the token in `HF_TOKEN=`.

Without a token it still transcribes — it just won't split speakers.

---

## 2A. Run on RTX 5xxx — GPU

For Blackwell script uses CUDA 12.8 + cu128 PyTorch.

1. Install **Docker Desktop** and the current **NVIDIA Windows driver**. In Docker
   Desktop: Settings → General → *Use the WSL 2 based engine* (default). GPU passthrough
   to WSL2/containers works automatically with a recent driver.
2. Verify the GPU is visible to Docker (PowerShell):
   ```powershell
   docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi
   ```
   You should see the gpu. If not, update the driver / restart Docker Desktop.
3. From this folder:
   ```powershell
   docker compose --profile gpu up -d --build
   ```
   First build downloads a few GB and compiles nothing scary; first transcription
   also downloads `large-v3` (~3 GB) into a cached volume (only once).
4. Drop an audio file into `speech-to-text\input\`. Watch progress:
   ```powershell
   docker compose --profile gpu logs -f
   ```
   The transcript appears in `output\`.

## 2B. Run anywhere — CPU (smaller model)

No GPU required. Uses `small` + int8 by default so it's actually usable.

```bash
docker compose --profile cpu up -d --build
```

Bump quality at the cost of speed by setting `CPU_MODEL=medium` (or `large-v3` if
you're patient) in `.env`. On CPU, `large-v3` on a 40-min file can take ~30+ min.

---

## Usage notes

- **It's a watch folder.** The container stays running with the model loaded
- **Re-running:** a file is skipped if its `.txt` already exists in `output/`.
  Delete the `.txt` to redo it.
- **Stop / start:** `docker compose --profile gpu down` / `... up -d`.
- **One-shot instead of watching:** set `WATCH=0` in `.env` — it processes
  everything in `input/` once and exits.
- **Known speaker count** improves diarization: set `MIN_SPEAKERS`/`MAX_SPEAKERS`
  in `.env` (e.g. both `2` for a 1-on-1 call).

## Tuning

| Setting | What it does |
|---|---|
| `MODEL` / `CPU_MODEL` | `base` < `small` < `medium` < `large-v3` (quality vs speed) |
| `LANGUAGE` | `pl`, `en`, … or empty to autodetect |
| `COMPUTE_TYPE` | GPU: `float16`; CPU: `int8` (set automatically per profile) |
| `BATCH_SIZE` | Higher = faster on GPU, more VRAM |
| `MIN_SPEAKERS` / `MAX_SPEAKERS` | Constrain the number of speakers |
