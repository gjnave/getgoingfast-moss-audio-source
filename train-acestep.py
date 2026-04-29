from __future__ import annotations

import csv
import json
import os
import re
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable

import gradio as gr

from src.hf_inference import MossAudioHFInference, read_env_model_id, resolve_device


AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a", ".aac"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}
MEDIA_EXTENSIONS = AUDIO_EXTENSIONS | VIDEO_EXTENSIONS

DEFAULT_DATASET_DIR = Path(__file__).resolve().parent / "ace_step_dataset"
DEFAULT_PROMPT = (
    "Create a concise ACE-Step LoRA training caption for this audio. "
    "Describe the genre, instruments, mood, tempo feel, vocals, production style, "
    "and notable sound events. Do not mention that you are an AI."
)
WHISPER_MODELS = {}


def clean_caption(text: str) -> str:
    return " ".join((text or "").replace("\r", " ").replace("\n", " ").split())


def clean_dataset_name(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", (text or "").strip()).strip("-")
    return cleaned or "moss_ace_dataset"


def format_lrc_time(seconds: float) -> str:
    minutes = int(seconds // 60)
    remaining = seconds - (minutes * 60)
    return f"{minutes:02d}:{remaining:05.2f}"


def format_range_time(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    remaining = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{remaining:06.3f}"


def media_files_in_folder(folder: Path, recursive: bool) -> list[Path]:
    pattern = "**/*" if recursive else "*"
    files = [
        item
        for item in folder.glob(pattern)
        if item.is_file() and item.suffix.lower() in MEDIA_EXTENSIONS
    ]
    return sorted(files, key=lambda p: str(p).lower())


def run_command(command: list[str], log: Callable[[str], None]) -> None:
    log("> " + " ".join(f'"{part}"' if " " in part else part for part in command))
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert process.stdout is not None
    for line in process.stdout:
        log(line.rstrip())
    code = process.wait()
    if code:
        raise RuntimeError(f"Command failed with exit code {code}: {' '.join(command)}")


def ffmpeg_path() -> str:
    local = Path(__file__).resolve().parent / "ffmpeg" / "bin" / "ffmpeg.exe"
    if local.exists():
        return str(local)
    found = shutil.which("ffmpeg")
    if found:
        return found
    raise RuntimeError(
        "ffmpeg was not found. Install ffmpeg or run the MOSS-Audio installer first."
    )


def ffprobe_path() -> str | None:
    local = Path(__file__).resolve().parent / "ffmpeg" / "bin" / "ffprobe.exe"
    if local.exists():
        return str(local)
    return shutil.which("ffprobe")


def probe_duration_seconds(path: Path) -> float | None:
    probe = ffprobe_path()
    if not probe:
        return None
    command = [
        probe,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        return float(result.stdout.strip())
    except Exception:
        return None


def convert_to_training_wav(source: Path, target: Path, log: Callable[[str], None]) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    run_command(
        [
            ffmpeg_path(),
            "-y",
            "-i",
            str(source),
            "-vn",
            "-ar",
            "44100",
            "-c:a",
            "pcm_s16le",
            str(target),
        ],
        log,
    )


def split_to_training_wavs(
    source: Path, chunk_seconds: int, work_dir: Path, log: Callable[[str], None]
) -> list[Path]:
    work_dir.mkdir(parents=True, exist_ok=True)
    pattern = work_dir / "chunk_%05d.wav"
    run_command(
        [
            ffmpeg_path(),
            "-y",
            "-i",
            str(source),
            "-vn",
            "-ar",
            "44100",
            "-c:a",
            "pcm_s16le",
            "-f",
            "segment",
            "-segment_time",
            str(chunk_seconds),
            "-reset_timestamps",
            "1",
            str(pattern),
        ],
        log,
    )
    chunks = sorted(work_dir.glob("chunk_*.wav"))
    if not chunks:
        raise RuntimeError("ffmpeg did not create any chunk files.")
    return chunks


def next_output_index(output_dir: Path, requested_start: int | None) -> int:
    if requested_start and requested_start > 0:
        return requested_start

    highest = 0
    for item in output_dir.glob("*.wav"):
        match = re.match(r"^(\d+)\.wav$", item.name)
        if match:
            highest = max(highest, int(match.group(1)))
    return highest + 1


def get_inference() -> MossAudioHFInference:
    model_name_or_path = os.environ.get("MOSS_AUDIO_MODEL_PATH") or read_env_model_id()
    device = os.environ.get("MOSS_AUDIO_DEVICE") or resolve_device()
    return MossAudioHFInference(
        model_name_or_path=model_name_or_path,
        device=device,
        torch_dtype="auto",
    )


def caption_audio(
    inference: MossAudioHFInference,
    audio_path: Path,
    prompt: str,
    max_new_tokens: int,
    log: Callable[[str], None],
) -> str:
    log(f"Captioning {audio_path.name}...")
    answer = inference.generate(
        question=prompt or DEFAULT_PROMPT,
        audio_path=str(audio_path),
        max_new_tokens=max_new_tokens,
        num_beams=1,
        do_sample=False,
    )
    return clean_caption(answer)


def get_whisper_model(model_name: str, log: Callable[[str], None]):
    model_key = model_name or "base"
    if model_key in WHISPER_MODELS:
        return WHISPER_MODELS[model_key]

    try:
        import whisper
    except ImportError as exc:
        raise RuntimeError(
            "Whisper is not installed. Install it in this environment with: "
            "pip install -U openai-whisper"
        ) from exc

    log(f"Loading Whisper model: {model_key}")
    WHISPER_MODELS[model_key] = whisper.load_model(model_key)
    return WHISPER_MODELS[model_key]


def transcribe_lyrics_with_whisper(
    audio_path: Path,
    model_name: str,
    language: str,
    write_timestamps: bool,
    log: Callable[[str], None],
) -> tuple[str, str, str]:
    model = get_whisper_model(model_name, log)
    kwargs = {
        "task": "transcribe",
        "fp16": False,
        "verbose": False,
    }
    if language.strip():
        kwargs["language"] = language.strip()

    log(f"Transcribing lyrics with Whisper: {audio_path.name}")
    result = model.transcribe(str(audio_path), **kwargs)
    segments = result.get("segments") or []

    clean_lines: list[str] = []
    lrc_lines: list[str] = []
    timestamp_lines: list[str] = []

    if segments:
        for segment in segments:
            text = clean_caption(segment.get("text", ""))
            if not text:
                continue
            clean_lines.append(text)
            if write_timestamps:
                start = float(segment.get("start") or 0.0)
                end = float(segment.get("end") or start)
                lrc_lines.append(f"[{format_lrc_time(start)}] {text}")
                timestamp_lines.append(
                    f"{format_range_time(start)} --> {format_range_time(end)} {text}"
                )
    else:
        text = clean_caption(result.get("text", ""))
        if text:
            clean_lines.append(text)

    return "\n".join(clean_lines), "\n".join(lrc_lines), "\n".join(timestamp_lines)


def write_metadata_csv(output_dir: Path, rows: list[dict]) -> None:
    csv_path = output_dir / "metadata.csv"
    by_file: dict[str, dict] = {}

    if csv_path.exists():
        with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                filename = row.get("File", "")
                if filename:
                    by_file[filename] = row

    for row in rows:
        by_file[row["File"]] = {
            "File": row["File"],
            "Artist": row.get("Artist", ""),
            "Title": row.get("Title", ""),
            "BPM": row.get("BPM", ""),
            "Key": row.get("Key", ""),
            "Camelot": row.get("Camelot", ""),
            "Caption": row.get("Caption", ""),
        }

    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        fieldnames = ["File", "Artist", "Title", "BPM", "Key", "Camelot", "Caption"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(by_file[name] for name in sorted(by_file))


def write_dataset_json(
    output_dir: Path,
    dataset_name: str,
    custom_tag: str,
    rows: list[dict],
    log: Callable[[str], None],
) -> None:
    existing_samples: dict[str, dict] = {}
    dataset_path = output_dir / "dataset.json"

    if dataset_path.exists():
        try:
            existing = json.loads(dataset_path.read_text(encoding="utf-8"))
            for sample in existing.get("samples", []):
                filename = sample.get("filename")
                if filename:
                    existing_samples[filename] = sample
        except Exception as exc:
            log(f"Could not read existing dataset.json, rebuilding it: {exc}")

    for row in rows:
        filename = row["File"]
        lyrics = row.get("Lyrics", "[Instrumental]")
        is_instrumental = lyrics.strip().lower() == "[instrumental]"
        existing_samples[filename] = {
            "id": Path(filename).stem,
            "audio_path": str((output_dir / filename).resolve()),
            "filename": filename,
            "caption": row.get("Caption", ""),
            "genre": "",
            "lyrics": lyrics,
            "raw_lyrics": "" if is_instrumental else lyrics,
            "formatted_lyrics": "",
            "bpm": row.get("BPM") or None,
            "keyscale": row.get("Key", ""),
            "timesignature": row.get("TimeSignature", "4"),
            "duration": int(row.get("Duration", 0) or 0),
            "language": "instrumental" if is_instrumental else "unknown",
            "is_instrumental": is_instrumental,
            "custom_tag": custom_tag or "",
            "labeled": bool(row.get("Caption", "")),
            "prompt_override": None,
        }

    samples = [existing_samples[name] for name in sorted(existing_samples)]
    all_instrumental = all(sample.get("is_instrumental", True) for sample in samples)
    payload = {
        "metadata": {
            "name": clean_dataset_name(dataset_name),
            "custom_tag": custom_tag or "",
            "tag_position": "prepend",
            "created_at": datetime.now().isoformat(),
            "num_samples": len(samples),
            "all_instrumental": all_instrumental,
            "genre_ratio": 0,
        },
        "samples": samples,
    }
    dataset_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_sidecars(
    output_dir: Path,
    filename: str,
    caption: str,
    lyrics: str,
    lrc_text: str,
    timestamp_text: str,
    duration: float | None,
    custom_tag: str,
) -> dict:
    stem = Path(filename).stem
    lyrics_text = lyrics.strip() or "[Instrumental]"
    caption_text = clean_caption(caption)

    (output_dir / f"{stem}.txt").write_text(lyrics_text + "\n", encoding="utf-8")
    (output_dir / f"{stem}.lyrics.txt").write_text(lyrics_text + "\n", encoding="utf-8")
    (output_dir / f"{stem}.caption.txt").write_text(caption_text + "\n", encoding="utf-8")
    if lrc_text.strip():
        (output_dir / f"{stem}.lrc").write_text(lrc_text.strip() + "\n", encoding="utf-8")
    if timestamp_text.strip():
        (output_dir / f"{stem}.timestamps.txt").write_text(
            timestamp_text.strip() + "\n", encoding="utf-8"
        )

    json_payload = {
        "caption": caption_text,
        "bpm": None,
        "keyscale": "",
        "timesignature": "4",
        "language": "instrumental"
        if lyrics_text.lower() == "[instrumental]"
        else "unknown",
        "custom_tag": custom_tag or "",
    }
    (output_dir / f"{stem}.json").write_text(
        json.dumps(json_payload, indent=2), encoding="utf-8"
    )

    return {
        "File": filename,
        "Caption": caption_text,
        "BPM": "",
        "Key": "",
        "Camelot": "",
        "Lyrics": lyrics_text,
        "Duration": int(duration or 0),
        "TimeSignature": "4",
        "HasTimestamps": bool(lrc_text.strip() or timestamp_text.strip()),
    }


def prepare_audio_units(
    source: Path,
    chunk_seconds: int,
    force_chunking: bool,
    temp_dir: Path,
    log: Callable[[str], None],
) -> list[Path]:
    duration = probe_duration_seconds(source)
    should_chunk = force_chunking or bool(duration and duration > chunk_seconds)
    if should_chunk:
        log(f"Splitting {source.name} into {chunk_seconds}-second chunks...")
        return split_to_training_wavs(source, chunk_seconds, temp_dir / "chunks", log)

    single = temp_dir / "single.wav"
    log(f"Converting {source.name} to ACE training WAV...")
    convert_to_training_wav(source, single, log)
    return [single]


def process_files(
    files: Iterable[Path],
    output_dir_text: str,
    dataset_name: str,
    custom_tag: str,
    prompt: str,
    lyrics_text: str,
    treat_as_instrumental: bool,
    use_whisper_lyrics: bool,
    whisper_model_name: str,
    whisper_language: str,
    write_whisper_timestamps: bool,
    chunk_seconds: int,
    force_chunking: bool,
    start_index: int,
    max_new_tokens: int,
    progress_log: list[str],
) -> tuple[str, str]:
    def log(message: str) -> None:
        progress_log.append(message)

    output_dir = Path(output_dir_text or DEFAULT_DATASET_DIR).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    start_number = int(start_index or 0)
    current_index = next_output_index(
        output_dir, start_number if start_number > 0 else None
    )
    rows: list[dict] = []

    inference = get_inference()
    source_files = [Path(item) for item in files]
    if not source_files:
        raise RuntimeError("No audio or video files were selected.")

    base_lyrics = "[Instrumental]" if treat_as_instrumental else lyrics_text.strip()

    for source in source_files:
        if not source.exists():
            raise RuntimeError(f"File not found: {source}")
        if source.suffix.lower() not in MEDIA_EXTENSIONS:
            log(f"Skipping unsupported file: {source}")
            continue

        log("")
        log(f"Preparing {source.name}")
        with tempfile.TemporaryDirectory(prefix="moss_ace_") as temp_name:
            temp_dir = Path(temp_name)
            units = prepare_audio_units(
                source=source,
                chunk_seconds=int(chunk_seconds),
                force_chunking=bool(force_chunking),
                temp_dir=temp_dir,
                log=log,
            )

            for unit in units:
                filename = f"{current_index:03d}.wav"
                target = output_dir / filename
                shutil.copy2(unit, target)
                caption = caption_audio(
                    inference=inference,
                    audio_path=target,
                    prompt=prompt,
                    max_new_tokens=int(max_new_tokens),
                    log=log,
                )
                unit_lyrics = base_lyrics
                lrc_text = ""
                timestamp_text = ""
                if use_whisper_lyrics:
                    whisper_lyrics, lrc_text, timestamp_text = transcribe_lyrics_with_whisper(
                        audio_path=target,
                        model_name=whisper_model_name,
                        language=whisper_language,
                        write_timestamps=write_whisper_timestamps,
                        log=log,
                    )
                    if whisper_lyrics.strip():
                        unit_lyrics = whisper_lyrics
                    elif not unit_lyrics.strip():
                        unit_lyrics = "[Instrumental]"
                duration = probe_duration_seconds(target)
                row = write_sidecars(
                    output_dir=output_dir,
                    filename=filename,
                    caption=caption,
                    lyrics=unit_lyrics,
                    lrc_text=lrc_text,
                    timestamp_text=timestamp_text,
                    duration=duration,
                    custom_tag=custom_tag,
                )
                rows.append(row)
                timestamp_note = ", timestamp files" if row["HasTimestamps"] else ""
                log(
                    f"Wrote {filename}, {Path(filename).stem}.caption.txt, "
                    f"lyrics files{timestamp_note}"
                )
                current_index += 1

    if not rows:
        raise RuntimeError("No usable media files were processed.")

    write_metadata_csv(output_dir, rows)
    write_dataset_json(output_dir, dataset_name, custom_tag, rows, log)

    summary_lines = [
        f"Output folder: {output_dir}",
        f"Dataset JSON: {output_dir / 'dataset.json'}",
        f"Metadata CSV: {output_dir / 'metadata.csv'}",
        f"New training audio files: {len(rows)}",
        "",
        "Created files:",
    ]
    for row in rows:
        stem = Path(row["File"]).stem
        summary_lines.append(
            f"- {row['File']}, {stem}.txt, {stem}.lyrics.txt, "
            f"{stem}.caption.txt, {stem}.json"
        )
        if row.get("HasTimestamps"):
            summary_lines.append(f"  timestamps: {stem}.lrc, {stem}.timestamps.txt")

    return "\n".join(summary_lines), "\n".join(progress_log)


def process_single_stream(
    media_file,
    output_dir: str,
    dataset_name: str,
    custom_tag: str,
    prompt: str,
    lyrics_text: str,
    treat_as_instrumental: bool,
    use_whisper_lyrics: bool,
    whisper_model_name: str,
    whisper_language: str,
    write_whisper_timestamps: bool,
    chunk_seconds: int,
    force_chunking: bool,
    start_index: int,
    max_new_tokens: int,
):
    logs: list[str] = []
    try:
        if media_file is None:
            raise RuntimeError("Choose an audio or video file first.")
        path = Path(media_file)
        summary, log_text = process_files(
            files=[path],
            output_dir_text=output_dir,
            dataset_name=dataset_name,
            custom_tag=custom_tag,
            prompt=prompt,
            lyrics_text=lyrics_text,
            treat_as_instrumental=treat_as_instrumental,
            use_whisper_lyrics=use_whisper_lyrics,
            whisper_model_name=whisper_model_name,
            whisper_language=whisper_language,
            write_whisper_timestamps=write_whisper_timestamps,
            chunk_seconds=chunk_seconds,
            force_chunking=force_chunking,
            start_index=start_index,
            max_new_tokens=max_new_tokens,
            progress_log=logs,
        )
        yield summary, log_text
    except Exception as exc:
        logs.append(f"ERROR: {exc}")
        yield "Processing failed.", "\n".join(logs)


def process_folder_stream(
    folder_path: str,
    output_dir: str,
    dataset_name: str,
    custom_tag: str,
    prompt: str,
    lyrics_text: str,
    treat_as_instrumental: bool,
    use_whisper_lyrics: bool,
    whisper_model_name: str,
    whisper_language: str,
    write_whisper_timestamps: bool,
    chunk_seconds: int,
    force_chunking: bool,
    start_index: int,
    max_new_tokens: int,
    recursive: bool,
    max_files: int,
):
    logs: list[str] = []
    try:
        folder = Path(folder_path or "").expanduser().resolve()
        if not folder.exists() or not folder.is_dir():
            raise RuntimeError("Enter a valid folder path.")
        files = media_files_in_folder(folder, bool(recursive))
        if max_files and max_files > 0:
            files = files[: int(max_files)]
        logs.append(f"Found {len(files)} supported media file(s).")
        summary, log_text = process_files(
            files=files,
            output_dir_text=output_dir,
            dataset_name=dataset_name,
            custom_tag=custom_tag,
            prompt=prompt,
            lyrics_text=lyrics_text,
            treat_as_instrumental=treat_as_instrumental,
            use_whisper_lyrics=use_whisper_lyrics,
            whisper_model_name=whisper_model_name,
            whisper_language=whisper_language,
            write_whisper_timestamps=write_whisper_timestamps,
            chunk_seconds=chunk_seconds,
            force_chunking=force_chunking,
            start_index=start_index,
            max_new_tokens=max_new_tokens,
            progress_log=logs,
        )
        yield summary, log_text
    except Exception as exc:
        logs.append(f"ERROR: {exc}")
        yield "Processing failed.", "\n".join(logs)


CSS = """
:root {
  --accent: #40c5bd;
  --panel: #111821;
  --panel-soft: #18222e;
  --text: #f4fbff;
  --muted: #9eb2bf;
}
body, .gradio-container {
  background: #071016 !important;
  color: var(--text) !important;
  font-family: Inter, Arial, sans-serif !important;
}
.main-wrap {
  max-width: 1180px;
  margin: 0 auto;
}
.hero {
  background: linear-gradient(135deg, #121d26, #0b1218);
  border: 1px solid rgba(64, 197, 189, 0.24);
  border-radius: 14px;
  padding: 22px 24px;
  margin-bottom: 16px;
}
.hero h1 {
  margin: 0 0 6px;
  color: #ffffff;
  font-size: 34px;
  letter-spacing: 0;
}
.hero p {
  margin: 4px 0;
  color: var(--muted);
}
.hero a {
  color: #74eee8;
}
.hint {
  color: #d7e3ea;
  background: rgba(64, 197, 189, 0.08);
  border-left: 3px solid var(--accent);
  padding: 10px 12px;
  border-radius: 8px;
}
textarea, input, .gr-text-input, .gr-textbox {
  background: #ffffff !important;
  color: #111820 !important;
}
button.primary {
  background: var(--accent) !important;
  color: #061013 !important;
}
"""


with gr.Blocks(css=CSS, title="MOSS to ACE-Step Dataset Prep") as demo:
    with gr.Column(elem_classes=["main-wrap"]):
        gr.HTML(
            """
            <div class="hero">
              <h1>MOSS to ACE-Step Dataset Prep</h1>
              <p>Caption audio and video, chunk long recordings, and write ACE-Step LoRA-ready files.</p>
              <p><a href="https://mossaudio.getgoingfast.pro">Quick installer</a> | <a href="https://getgoingfast.pro">getgoingfast.pro</a> | <a href="https://cognibuild.ai">cognibuild.ai</a></p>
            </div>
            <div class="hint">
              Best practice: use shorter, clean clips for training. Chunking at 30-60 seconds is usually safer for consumer GPUs and easier to review before ACE-Step preprocessing.
            </div>
            """
        )

        with gr.Row():
            dataset_name = gr.Textbox(
                label="Dataset Name",
                value="moss_ace_dataset",
            )
            custom_tag = gr.Textbox(
                label="ACE Trigger Tag",
                placeholder="Example: myartiststyle",
            )
            output_dir = gr.Textbox(
                label="Output Folder",
                value=str(DEFAULT_DATASET_DIR),
            )

        prompt = gr.Textbox(
            label="Caption Prompt",
            value=DEFAULT_PROMPT,
            lines=3,
        )

        with gr.Row():
            lyrics_text = gr.Textbox(
                label="Lyrics",
                placeholder="Optional manual lyrics. Whisper can fill this per file.",
                lines=4,
            )
            with gr.Column():
                treat_as_instrumental = gr.Checkbox(
                    label="Treat as instrumental when Whisper/manual lyrics are empty",
                    value=True,
                )
                chunk_seconds = gr.Slider(
                    label="Chunk Length Seconds",
                    minimum=15,
                    maximum=240,
                    step=15,
                    value=60,
                )
                force_chunking = gr.Checkbox(
                    label="Always chunk, even short files",
                    value=False,
                )

        with gr.Accordion("Whisper Lyrics / Timestamp Files", open=False):
            gr.Markdown(
                "ACE-Step wants clean lyrics without timestamps for training. "
                "When enabled, Whisper writes clean lyrics to `.lyrics.txt` and "
                "puts timestamped text in separate `.lrc` and `.timestamps.txt` files. "
                "Requires `pip install -U openai-whisper` in this environment."
            )
            with gr.Row():
                use_whisper_lyrics = gr.Checkbox(
                    label="Transcribe lyrics with Whisper",
                    value=False,
                )
                write_whisper_timestamps = gr.Checkbox(
                    label="Also write timestamp files",
                    value=True,
                )
            with gr.Row():
                whisper_model_name = gr.Dropdown(
                    label="Whisper Model",
                    choices=["tiny", "base", "small", "medium", "large"],
                    value="base",
                )
                whisper_language = gr.Textbox(
                    label="Language Code",
                    placeholder="Optional, e.g. en, ja, es. Leave blank to auto-detect.",
                )

        with gr.Row():
            start_index = gr.Number(
                label="Start Number (0 = append after existing files)",
                value=0,
                precision=0,
            )
            max_new_tokens = gr.Slider(
                label="Caption Max Tokens",
                minimum=64,
                maximum=512,
                step=32,
                value=192,
            )

        with gr.Tabs():
            with gr.Tab("Single File"):
                media_file = gr.File(
                    label="Audio or Video File",
                    file_count="single",
                    type="filepath",
                )
                single_button = gr.Button("Create ACE-Step Files", variant="primary")

            with gr.Tab("Folder Batch"):
                folder_path = gr.Textbox(
                    label="Folder Path",
                    placeholder=r"D:\training-audio\my-dataset",
                )
                with gr.Row():
                    recursive = gr.Checkbox(label="Include subfolders", value=True)
                    max_files = gr.Number(
                        label="Max files (0 = all)",
                        value=0,
                        precision=0,
                    )
                folder_button = gr.Button("Process Folder", variant="primary")

        summary = gr.Textbox(label="Output Summary", lines=10)
        logs = gr.Textbox(label="Processing Log", lines=16)

        gr.Markdown(
            """
            ACE-Step output includes numbered WAV files plus matching `.txt`, `.lyrics.txt`, `.caption.txt`, `.json`, `metadata.csv`, and `dataset.json`.
            Whisper timestamp files are saved separately as `.lrc` and `.timestamps.txt` for review; the ACE lyrics files remain timestamp-free.
            Load the output folder in ACE-Step's LoRA Training tab, scan it, review captions, then preprocess.
            """
        )

    shared_inputs = [
        output_dir,
        dataset_name,
        custom_tag,
        prompt,
        lyrics_text,
        treat_as_instrumental,
        use_whisper_lyrics,
        whisper_model_name,
        whisper_language,
        write_whisper_timestamps,
        chunk_seconds,
        force_chunking,
        start_index,
        max_new_tokens,
    ]

    single_button.click(
        process_single_stream,
        inputs=[media_file] + shared_inputs,
        outputs=[summary, logs],
    )
    folder_button.click(
        process_folder_stream,
        inputs=[folder_path] + shared_inputs + [recursive, max_files],
        outputs=[summary, logs],
    )


if __name__ == "__main__":
    demo.launch()
