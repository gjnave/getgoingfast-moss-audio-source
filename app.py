from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from functools import lru_cache
from pathlib import Path

import gradio as gr

from src.hf_inference import MossAudioHFInference, read_env_model_id, resolve_device

TITLE = "MOSS Audio Captioning"
SUBTITLE = "Upload media, transcribe it, and prepare clean caption outputs."
BRAND_LINK = "https://getgoingfast.pro"
CONTACT_MARKDOWN = (
    "[getgoingfast.pro](https://getgoingfast.pro) | "
    "[cognibuild.ai](https://www.cognibuild.ai) | "
    "[patreon.com/cognibuild](https://www.patreon.com/cognibuild) | "
    "[youtube.com/@cognibuild](https://www.youtube.com/@cognibuild)"
)
CONTACT_HTML = (
    '<a href="https://getgoingfast.pro" target="_blank" rel="noopener noreferrer">getgoingfast.pro</a> | '
    '<a href="https://www.cognibuild.ai" target="_blank" rel="noopener noreferrer">cognibuild.ai</a> | '
    '<a href="https://www.patreon.com/cognibuild" target="_blank" rel="noopener noreferrer">patreon.com/cognibuild</a> | '
    '<a href="https://www.youtube.com/@cognibuild" target="_blank" rel="noopener noreferrer">youtube.com/@cognibuild</a>'
)

DEFAULT_QUESTION = "Describe this audio."
DEFAULT_MAX_NEW_TOKENS = 1024
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_P = 1.0
DEFAULT_TOP_K = 50
CHUNK_SECONDS = 60
INLINE_SECONDS_LIMIT = 120
TARGET_SAMPLE_RATE = 16000
ADVISORY_BATCH_FILE_COUNT = 25
YOUTUBE_URL_RE = re.compile(r"^https?://(www\.)?(youtube\.com|youtu\.be)/", re.IGNORECASE)
MEDIA_EXTENSIONS = {
    ".aac",
    ".flac",
    ".m4a",
    ".mov",
    ".mp3",
    ".mp4",
    ".wav",
    ".webm",
}
OUTPUTS_DIR = Path(__file__).resolve().parent / "outputs"

CUSTOM_CSS = """
:root {
  --app-bg: #11161b;
  --app-panel: #171d24;
  --app-panel-2: #1d242d;
  --app-card: #ffffff;
  --app-card-soft: #f4f7fb;
  --app-line: rgba(255, 255, 255, 0.08);
  --app-line-strong: rgba(16, 24, 32, 0.14);
  --app-text: #eef3f7;
  --app-muted: #b6c3cf;
  --app-dark-text: #18202a;
  --app-dark-muted: #4c5a67;
  --app-accent: #59b8b0;
  --app-accent-deep: #24827d;
  --app-shadow: 0 22px 70px rgba(0, 0, 0, 0.28);
  --app-radius: 20px;
}

html, body {
  background: #0d1217 !important;
  min-height: 100vh;
  margin: 0 !important;
}

body, .gradio-container {
  background:
    radial-gradient(circle at top left, rgba(89, 184, 176, 0.10), transparent 22%),
    linear-gradient(180deg, #0d1217 0%, #11161b 100%) !important;
  color: var(--app-text) !important;
  font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif !important;
}

.gradio-container {
  width: calc(100vw - 26px) !important;
  max-width: 1700px !important;
  margin: 0 auto !important;
  padding: 10px 0 18px !important;
}

h1, h2, h3, h4, .brand-title {
  font-family: "Segoe UI Semibold", "Segoe UI", "Helvetica Neue", Arial, sans-serif !important;
  letter-spacing: -0.02em;
  color: var(--app-text) !important;
}

.app-hero-shell {
  position: relative;
  padding: 12px 18px;
  border-radius: 18px;
  background: linear-gradient(180deg, #151b22, #12181f);
  border: 1px solid rgba(255, 255, 255, 0.06);
  box-shadow: var(--app-shadow);
  overflow: hidden;
  margin-bottom: 10px;
}

.app-hero-shell::before {
  content: "";
  position: absolute;
  inset: 0;
  background:
    radial-gradient(circle at 15% 0%, rgba(91, 183, 176, 0.10), transparent 30%),
    radial-gradient(circle at 100% 0%, rgba(200, 155, 97, 0.08), transparent 24%);
  pointer-events: none;
}

.app-hero-main {
  position: relative;
  z-index: 1;
  padding: 2px 0;
}

.kicker {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 5px 10px;
  border-radius: 999px;
  background: rgba(91, 183, 176, 0.10);
  border: 1px solid rgba(91, 183, 176, 0.16);
  color: #a7e4df;
  font-size: 11px;
  font-weight: 800;
  letter-spacing: 0.06em;
  text-transform: uppercase;
}

.brand-title {
  margin: 8px 0 5px;
  font-size: clamp(1.9rem, 2.3vw, 2.9rem);
  line-height: 1.02;
  font-weight: 700;
}

.brand-subtitle {
  max-width: 72ch;
  font-size: 0.95rem;
  line-height: 1.45;
  color: var(--app-muted);
  margin: 0;
}

.contact-line {
  margin: 8px 0 0;
  color: var(--app-muted);
  font-size: 0.88rem;
}

.block, .gr-box, .gr-panel, .gr-group, .gr-form, .gradio-group, .gr-accordion {
  border-radius: var(--app-radius) !important;
}

.gr-box, .gr-panel, .gr-group, .gr-form, .gradio-group, .gr-accordion {
  background: var(--app-panel) !important;
  border: 1px solid var(--app-line) !important;
  box-shadow: 0 10px 26px rgba(0, 0, 0, 0.18) !important;
}

.gradio-container .prose,
.gradio-container .prose p,
.gradio-container .prose li,
.gradio-container .prose strong,
label,
.gr-label {
  color: var(--app-text) !important;
}

.gradio-container .prose a,
.gr-markdown a,
a {
  color: #8edbd5 !important;
}

input, textarea, .gr-textbox input, .gr-textbox textarea, .gr-textbox > label > textarea {
  border-radius: 16px !important;
  background: var(--app-card) !important;
  color: var(--app-dark-text) !important;
  border: 1px solid var(--app-line-strong) !important;
}

textarea::placeholder,
input::placeholder {
  color: #7d8b98 !important;
}

.gradio-container .gr-audio-container,
.gradio-container .gr-file,
.gradio-container .gr-image,
.gradio-container .gr-video,
.gradio-container .gradio-audio,
.gradio-container .gradio-file,
.gradio-container .gradio-textbox,
.gradio-container .gradio-html,
.gradio-container .gradio-checkbox,
.gradio-container .gradio-markdown.output-markdown {
  background: var(--app-card-soft) !important;
  color: var(--app-dark-text) !important;
  border: 1px solid rgba(23, 31, 39, 0.10) !important;
}

.gradio-container .gradio-audio *,
.gradio-container .gradio-file *,
.gradio-container .gradio-textbox label,
.gradio-container .gradio-checkbox label,
.gradio-container .gradio-checkbox span,
.gradio-container .gradio-accordion .label-wrap,
.gradio-container .gradio-accordion .label-wrap * {
  color: var(--app-dark-text) !important;
}

.gradio-container .gradio-accordion {
  background: var(--app-card) !important;
}

.gradio-container .gradio-accordion .label-wrap,
.gradio-container .gradio-accordion .icon-wrap {
  color: var(--app-dark-text) !important;
}

.gradio-container .gradio-checkbox {
  padding: 10px 12px !important;
}

.gradio-container .gradio-row,
.gradio-container .gr-row {
  gap: 12px !important;
}

.gradio-container .gradio-column,
.gradio-container .gr-column {
  gap: 10px !important;
}

.gradio-container .gr-form {
  gap: 10px !important;
}

button.primary, .gr-button-primary, button[variant="primary"] {
  background: linear-gradient(135deg, var(--app-accent), var(--app-accent-deep)) !important;
  color: #081012 !important;
  border: none !important;
  box-shadow: 0 10px 24px rgba(91, 183, 176, 0.22) !important;
}

button.secondary, .gr-button-secondary {
  background: rgba(255,255,255,0.04) !important;
  color: var(--app-text) !important;
  border: 1px solid var(--app-line-strong) !important;
}

footer {
  display: none !important;
}

@media (max-width: 980px) {
  .gradio-container {
    width: calc(100vw - 14px) !important;
  }
  .app-hero-shell { padding: 12px 14px; border-radius: 16px; }
  .brand-title { font-size: 1.9rem; }
}
"""

HERO_MARKDOWN = f"""
<div class="app-hero-shell">
  <section class="app-hero-main">
    <div class="kicker">CogniBuild AI</div>
    <div class="brand-title">{TITLE}</div>
    <p class="brand-subtitle">{SUBTITLE}</p>
    <p class="contact-line">{CONTACT_HTML}</p>
  </section>
</div>
"""


@lru_cache(maxsize=2)
def get_inference(model_name_or_path: str, device: str) -> MossAudioHFInference:
    return MossAudioHFInference(
        model_name_or_path=model_name_or_path,
        device=device,
        torch_dtype="auto",
        enable_time_marker=True,
    )


def append_log(log_lines: list[str], message: str) -> str:
    timestamp = time.strftime("%H:%M:%S")
    log_lines.append(f"[{timestamp}] {message}")
    return "\n".join(log_lines[-400:])


def format_status(
    model_name_or_path: str,
    device: str,
    elapsed_seconds: float,
    duration_seconds: float,
    chunk_count: int,
) -> str:
    return (
        f"Model: `{model_name_or_path}`  \n"
        f"Device: `{device}`  \n"
        f"Audio Length: `{duration_seconds:.1f}s`  \n"
        f"Chunks: `{chunk_count}`  \n"
        f"Elapsed: `{elapsed_seconds:.2f}s`"
    )


def run_command_logged(command: list[str], error_prefix: str, log_lines: list[str]) -> None:
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    output = (result.stdout or "").splitlines()
    for line in output[-30:]:
        if line.strip():
            append_log(log_lines, line.strip())
    if result.returncode != 0:
        detail = "\n".join(output[-12:]).strip()
        if detail:
            raise gr.Error(f"{error_prefix}\n{detail}")
        raise gr.Error(error_prefix)


def resolve_media_path(
    audio_path: str | None,
    video_path: str | None,
    youtube_url: str | None,
) -> str | None:
    if youtube_url and youtube_url.strip():
        return youtube_url.strip()
    if video_path:
        return video_path
    return audio_path


def is_youtube_url(value: str | None) -> bool:
    return bool(value and YOUTUBE_URL_RE.match(value.strip()))


def sanitize_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    return cleaned.strip("-._") or "output"


def validate_single_run_inputs(
    audio_path: str | None,
    video_path: str | None,
    youtube_url: str | None,
    log_lines: list[str],
) -> None:
    source_count = int(bool(audio_path)) + int(bool(video_path)) + int(bool((youtube_url or "").strip()))
    if source_count == 0:
        raise gr.Error("Add an audio file, a video file, or a YouTube URL before running.")
    if source_count > 1:
        append_log(
            log_lines,
            "Multiple source types were provided. The app will use YouTube first, then video, then audio.",
        )


def describe_selected_source(audio_path: str | None, video_path: str | None, youtube_url: str | None) -> str:
    if youtube_url and youtube_url.strip():
        return "YouTube URL"
    if video_path:
        return f"Video: {Path(video_path).name}"
    if audio_path:
        return f"Audio: {Path(audio_path).name}"
    return "Text-only"


def download_youtube_audio(youtube_url: str, temp_dir: str, log_lines: list[str]) -> str:
    append_log(log_lines, f"Downloading YouTube media: {youtube_url}")
    output_template = os.path.join(temp_dir, "youtube.%(ext)s")
    command = [
        sys.executable,
        "-m",
        "yt_dlp",
        "--newline",
        "--no-playlist",
        "-f",
        "bestaudio",
        "-o",
        output_template,
        youtube_url,
    ]
    run_command_logged(
        command,
        "Failed to download audio from YouTube. Please make sure the URL is public and reachable.",
        log_lines,
    )
    for candidate in sorted(Path(temp_dir).glob("youtube.*")):
        if candidate.is_file():
            append_log(log_lines, f"Downloaded source file: {candidate.name}")
            return str(candidate)
    raise gr.Error("YouTube download completed, but no audio file was produced.")


def convert_media_to_wav(media_path: str, output_path: str, log_lines: list[str]) -> None:
    append_log(log_lines, f"Normalizing media to mono {TARGET_SAMPLE_RATE} Hz wav")
    command = [
        "ffmpeg",
        "-y",
        "-i",
        media_path,
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(TARGET_SAMPLE_RATE),
        "-c:a",
        "pcm_s16le",
        output_path,
    ]
    run_command_logged(
        command,
        "Failed to prepare the media. Please make sure the file is valid and decodable.",
        log_lines,
    )


def probe_duration_seconds(media_path: str) -> float:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        media_path,
    ]
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        raise gr.Error(f"Failed to inspect media duration.\n{result.stderr.strip()}")
    try:
        return float(result.stdout.strip())
    except ValueError as exc:
        raise gr.Error("Could not determine media duration.") from exc


def split_audio_chunks(audio_path: str, output_dir: str, log_lines: list[str]) -> list[str]:
    duration_seconds = probe_duration_seconds(audio_path)
    if duration_seconds <= INLINE_SECONDS_LIMIT:
        append_log(log_lines, f"Audio length {duration_seconds:.1f}s, no chunking needed")
        return [audio_path]

    append_log(
        log_lines,
        f"Audio length {duration_seconds:.1f}s, splitting into {CHUNK_SECONDS}-second chunks",
    )
    chunk_pattern = os.path.join(output_dir, "chunk_%03d.wav")
    command = [
        "ffmpeg",
        "-y",
        "-i",
        audio_path,
        "-f",
        "segment",
        "-segment_time",
        str(CHUNK_SECONDS),
        "-ac",
        "1",
        "-ar",
        str(TARGET_SAMPLE_RATE),
        "-c:a",
        "pcm_s16le",
        chunk_pattern,
    ]
    run_command_logged(
        command,
        "Failed to split long audio into chunks for safer processing.",
        log_lines,
    )
    chunks = sorted(str(path) for path in Path(output_dir).glob("chunk_*.wav"))
    if not chunks:
        raise gr.Error("Chunking completed, but no audio chunks were created.")
    append_log(log_lines, f"Created {len(chunks)} chunk(s)")
    return chunks


def prepare_audio_source(media_path: str | None, temp_dir: str, log_lines: list[str]) -> tuple[list[str], float, str]:
    if not media_path:
        return [], 0.0, "text-only"

    source_path = media_path
    source_label = Path(media_path).name if not is_youtube_url(media_path) else "youtube"
    if is_youtube_url(media_path):
        source_path = download_youtube_audio(media_path, temp_dir, log_lines)
        source_label = Path(source_path).stem

    normalized_path = os.path.join(temp_dir, "prepared.wav")
    convert_media_to_wav(source_path, normalized_path, log_lines)
    duration_seconds = probe_duration_seconds(normalized_path)
    chunks = split_audio_chunks(normalized_path, temp_dir, log_lines)
    return chunks, duration_seconds, source_label


def combine_answers(chunk_answers: list[str]) -> str:
    if len(chunk_answers) == 1:
        return chunk_answers[0]
    return "\n\n".join(
        f"Chunk {index + 1}:\n{answer}" for index, answer in enumerate(chunk_answers)
    )


def ensure_output_dir(source_label: str) -> Path:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_dir = OUTPUTS_DIR / f"{sanitize_name(source_label)}-{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def export_chunk_outputs(chunk_paths: list[str], chunk_answers: list[str], output_dir: Path) -> None:
    for index, (chunk_path, chunk_answer) in enumerate(zip(chunk_paths, chunk_answers), start=1):
        chunk_name = f"{index:03d}"
        chunk_target = output_dir / f"{chunk_name}.wav"
        shutil.copyfile(chunk_path, chunk_target)
        (output_dir / f"{chunk_name}.txt").write_text(chunk_answer, encoding="utf-8")


def iter_media_files(folder_path: str) -> list[Path]:
    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        raise gr.Error("Batch folder does not exist or is not a directory.")
    files = [path for path in sorted(folder.iterdir()) if path.is_file() and path.suffix.lower() in MEDIA_EXTENSIONS]
    if not files:
        raise gr.Error("No supported audio or video files were found in that folder.")
    return files


def run_generation(
    inference: MossAudioHFInference,
    chunk_paths: list[str],
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    log_lines: list[str],
) -> list[str]:
    if not chunk_paths:
        append_log(log_lines, "Running text-only generation")
        return [
            inference.generate(
                question=prompt,
                audio_path=None,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )
        ]

    answers: list[str] = []
    total_chunks = len(chunk_paths)
    for index, chunk_path in enumerate(chunk_paths, start=1):
        append_log(log_lines, f"Processing chunk {index}/{total_chunks}: {Path(chunk_path).name}")
        answers.append(
            inference.generate(
                question=prompt,
                audio_path=chunk_path,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )
        )
    return answers


def run_inference_stream(
    audio_path: str | None,
    video_path: str | None,
    youtube_url: str,
    question: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    save_chunks: bool,
):
    log_lines: list[str] = []
    output_text = ""
    status_text = "Preparing run..."
    yield output_text, status_text, append_log(log_lines, "Starting single-file inference")

    validate_single_run_inputs(audio_path, video_path, youtube_url, log_lines)

    prompt = (question or "").strip() or DEFAULT_QUESTION
    model_name_or_path = read_env_model_id()
    device = resolve_device()
    append_log(log_lines, f"Selected source: {describe_selected_source(audio_path, video_path, youtube_url)}")
    append_log(log_lines, f"Loading model on {device}")
    yield output_text, status_text, "\n".join(log_lines)

    try:
        inference = get_inference(model_name_or_path, device)
        media_path = resolve_media_path(audio_path, video_path, youtube_url)
        started_at = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="moss-audio-") as temp_dir:
            chunk_paths, duration_seconds, source_label = prepare_audio_source(media_path, temp_dir, log_lines)
            yield output_text, "Media prepared, running inference...", "\n".join(log_lines)
            chunk_answers = run_generation(
                inference,
                chunk_paths,
                prompt,
                max_new_tokens,
                temperature,
                top_p,
                top_k,
                log_lines,
            )
            output_text = combine_answers(chunk_answers)
            output_dir = None
            if save_chunks and chunk_paths:
                output_dir = ensure_output_dir(source_label)
                export_chunk_outputs(chunk_paths, chunk_answers, output_dir)
                append_log(log_lines, f"Saved chunk files and captions to {output_dir}")
            elapsed_seconds = time.perf_counter() - started_at
            status_text = format_status(
                model_name_or_path,
                device,
                elapsed_seconds,
                duration_seconds if media_path else 0.0,
                len(chunk_paths) if media_path else 0,
            )
            if output_dir is not None:
                status_text += f"  \nSaved Chunk Folder: `{output_dir}`"
            yield output_text, status_text, append_log(log_lines, "Inference complete")
    except Exception as exc:  # pragma: no cover - runtime environment dependent
        raise gr.Error(
            f"Inference failed. Please make sure the uploaded file is readable and the format is supported.\n{exc}"
        ) from exc


def batch_process_stream(
    folder_path: str,
    question: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    save_chunks: bool,
):
    log_lines: list[str] = []
    summary_text = ""
    status_text = "Preparing batch run..."
    yield summary_text, status_text, append_log(log_lines, "Starting batch processing")

    prompt = (question or "").strip() or DEFAULT_QUESTION
    model_name_or_path = read_env_model_id()
    device = resolve_device()
    files = iter_media_files(folder_path)
    append_log(log_lines, f"Found {len(files)} media file(s) in {folder_path}")
    if len(files) > ADVISORY_BATCH_FILE_COUNT:
        append_log(
            log_lines,
            f"Large batch detected. Consider testing with a smaller sample first before running all {len(files)} files.",
        )
    yield summary_text, "Loading model...", "\n".join(log_lines)

    try:
        inference = get_inference(model_name_or_path, device)
        summaries: list[str] = []
        for file_index, media_file in enumerate(files, start=1):
            append_log(log_lines, f"Batch file {file_index}/{len(files)}: {media_file.name}")
            yield summary_text, f"Processing {media_file.name}...", "\n".join(log_lines)
            with tempfile.TemporaryDirectory(prefix="moss-audio-batch-") as temp_dir:
                chunk_paths, duration_seconds, _ = prepare_audio_source(str(media_file), temp_dir, log_lines)
                chunk_answers = run_generation(
                    inference,
                    chunk_paths,
                    prompt,
                    max_new_tokens,
                    temperature,
                    top_p,
                    top_k,
                    log_lines,
                )
                merged_answer = combine_answers(chunk_answers)
                output_txt = media_file.with_suffix(".txt")
                output_txt.write_text(merged_answer, encoding="utf-8")
                append_log(log_lines, f"Wrote caption file {output_txt.name}")
                if save_chunks and chunk_paths:
                    chunk_output_dir = media_file.parent / f"{media_file.stem}_chunks"
                    chunk_output_dir.mkdir(parents=True, exist_ok=True)
                    export_chunk_outputs(chunk_paths, chunk_answers, chunk_output_dir)
                    append_log(log_lines, f"Saved chunk pairs to {chunk_output_dir}")
                summaries.append(
                    f"{media_file.name} -> {output_txt.name} ({duration_seconds:.1f}s, {max(1, len(chunk_paths))} chunk(s))"
                )
                summary_text = "\n".join(summaries)
                yield summary_text, f"Processed {media_file.name}", "\n".join(log_lines)

        status_text = f"Batch complete on `{len(files)}` file(s)."
        yield summary_text, status_text, append_log(log_lines, "Batch processing complete")
    except Exception as exc:  # pragma: no cover - runtime environment dependent
        raise gr.Error(f"Batch processing failed.\n{exc}") from exc


with gr.Blocks(title=TITLE) as demo:
    gr.Markdown(HERO_MARKDOWN)

    with gr.Row():
        with gr.Column(scale=5):
            audio_input = gr.Audio(
                label="Audio",
                sources=["upload", "microphone"],
                type="filepath",
            )
            with gr.Accordion("Optional Video Input (.mp4)", open=False):
                gr.Markdown(
                    "Upload an mp4 when needed. Its audio track is extracted and used for inference."
                )
                video_input = gr.File(
                    label="Video File",
                    file_types=[".mp4"],
                    type="filepath",
                )
            youtube_url_input = gr.Textbox(
                label="YouTube URL",
                placeholder="https://www.youtube.com/watch?v=...",
            )
            question_input = gr.Textbox(
                label="Prompt",
                lines=4,
                value=DEFAULT_QUESTION,
                placeholder="For example: Please transcribe this audio. Describe the sounds in this clip. What emotion does the speaker convey?",
            )
            save_chunks_input = gr.Checkbox(
                label="Save training-ready chunk pairs (.wav + .txt)",
                value=False,
            )

            with gr.Accordion("Advanced Settings", open=False):
                max_new_tokens_input = gr.Slider(
                    minimum=64,
                    maximum=2048,
                    value=DEFAULT_MAX_NEW_TOKENS,
                    step=32,
                    label="Max New Tokens",
                )
                temperature_input = gr.Slider(
                    minimum=0.0,
                    maximum=1.5,
                    value=DEFAULT_TEMPERATURE,
                    step=0.1,
                    label="Temperature",
                )
                top_p_input = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=DEFAULT_TOP_P,
                    step=0.05,
                    label="Top-p",
                )
                top_k_input = gr.Slider(
                    minimum=1,
                    maximum=100,
                    value=DEFAULT_TOP_K,
                    step=1,
                    label="Top-k",
                )

            with gr.Row():
                submit_btn = gr.Button("Generate", variant="primary")
                gr.ClearButton(
                    [
                        audio_input,
                        video_input,
                        youtube_url_input,
                        question_input,
                        save_chunks_input,
                        max_new_tokens_input,
                        temperature_input,
                        top_p_input,
                        top_k_input,
                    ],
                    value="Clear",
                )

        with gr.Column(scale=5):
            output_text = gr.Textbox(label="Output", lines=16)
            status_text = gr.Markdown("Waiting for input.")
            with gr.Accordion("Processing Terminal", open=False):
                process_log = gr.Textbox(
                    label="Live Processing Log",
                    lines=18,
                    max_lines=30,
                    autoscroll=True,
                )

    with gr.Accordion("Batch Processing", open=False):
        gr.Markdown(
            "Start with a small test folder first. Very large folders or very long source media can still push GPU memory hard and lead to long waits. This mode writes captions back to the same folder as `filename.txt`. If chunk export is enabled, the app creates numbered training-ready pairs like `001.wav` plus `001.txt`."
        )
        batch_folder_input = gr.Textbox(
            label="Folder Path",
            placeholder=r"D:\path\to\folder-with-audio-or-video",
        )
        with gr.Row():
            batch_run_btn = gr.Button("Process Folder", variant="primary")
        batch_output_text = gr.Textbox(label="Batch Summary", lines=10)
        batch_status_text = gr.Markdown("Batch processing is idle.")
        with gr.Accordion("Batch Terminal", open=False):
            batch_log = gr.Textbox(
                label="Batch Log",
                lines=18,
                max_lines=30,
                autoscroll=True,
            )

    submit_btn.click(
        fn=run_inference_stream,
        inputs=[
            audio_input,
            video_input,
            youtube_url_input,
            question_input,
            max_new_tokens_input,
            temperature_input,
            top_p_input,
            top_k_input,
            save_chunks_input,
        ],
        outputs=[output_text, status_text, process_log],
    )

    batch_run_btn.click(
        fn=batch_process_stream,
        inputs=[
            batch_folder_input,
            question_input,
            max_new_tokens_input,
            temperature_input,
            top_p_input,
            top_k_input,
            save_chunks_input,
        ],
        outputs=[batch_output_text, batch_status_text, batch_log],
    )


if __name__ == "__main__":
    server_name = os.environ.get("MOSS_AUDIO_SERVER_NAME", "127.0.0.1")
    server_port = int(os.environ.get("MOSS_AUDIO_SERVER_PORT", "7860"))
    demo.queue(max_size=8).launch(
        server_name=server_name,
        server_port=server_port,
        theme=gr.themes.Soft(),
        css=CUSTOM_CSS,
        inbrowser=True,
    )
