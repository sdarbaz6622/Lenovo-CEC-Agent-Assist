import sys
import os
import threading
import queue
import wave
import io
import json
import re
import shutil
import stat
from urllib.parse import urlencode
import numpy as np
from datetime import datetime
from pathlib import Path
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTextEdit, QTabWidget, QFileDialog,
    QMessageBox, QTableWidget, QTableWidgetItem, QHeaderView, QGroupBox,
    QDialog, QLineEdit, QFormLayout, QInputDialog, QDateEdit, QFrame, QSlider,
    QSplitter, QScrollArea, QComboBox, QCheckBox, QGridLayout
)
from PySide6.QtCore import Qt, Signal, QObject, QUrl, QDate, QTimer
from PySide6.QtGui import QColor, QFont, QTextCursor, QDesktopServices
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtWebEngineWidgets import QWebEngineView
 
try:
    import pyaudiowpatch as pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False
 
try:
    from faster_whisper import WhisperModel
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
 
VAULT_DIR = r"C:\Users\sarbaz\OneDrive - Lenovo\Lenovo_CEC_Master_Vault"
CRED_FILE = "users_credentials.txt"
os.makedirs(VAULT_DIR, exist_ok=True)
 
class DualAudioRecorderThread(threading.Thread):
    def __init__(self, filename, samplerate=16000):
        super().__init__()
        self.filename = filename
        self.samplerate = samplerate
        self._stop_event = threading.Event()
        self.ready_event = threading.Event()
        self.error = None
        self.frames_written = 0
        self.sample_rate = samplerate
        self.loopback_queue = queue.Queue()
        self.mic_queue = queue.Queue()
        self.live_audio_queue = queue.Queue()


    @staticmethod
    def _write_aligned_audio(wf, loopback_buffer, mic_buffer, flush=False):
        """Write all audio currently available instead of waiting for both devices.

        WASAPI loopback and microphone callbacks are independent and their callback
        chunk sizes are not guaranteed to match. Using min() here silently delayed
        one channel and could leave most of a recording unwritten when one device
        delivered fewer samples. Pad the shorter side with silence for this block;
        preserve every sample from both input queues.
        """
        loopback_frames = len(loopback_buffer) // 2
        mic_frames = len(mic_buffer) // 2
        frame_count = max(loopback_frames, mic_frames)
        if not frame_count:
            return 0, np.empty((0, 2), dtype=np.int16)

        loopback_count = min(loopback_frames, frame_count)
        mic_count = min(mic_frames, frame_count)
        loopback_audio = np.zeros(frame_count, dtype=np.int16)
        mic_audio = np.zeros(frame_count, dtype=np.int16)
        if loopback_count:
            loopback_audio[:loopback_count] = np.frombuffer(
                loopback_buffer, dtype=np.int16, count=loopback_count
            )
        if mic_count:
            mic_audio[:mic_count] = np.frombuffer(
                mic_buffer, dtype=np.int16, count=mic_count
            )

        stereo_audio = np.column_stack((loopback_audio, mic_audio)).astype(np.int16, copy=False)
        wf.writeframesraw(stereo_audio.tobytes())
        del loopback_buffer[:loopback_count * 2]
        del mic_buffer[:mic_count * 2]
        return frame_count, stereo_audio

    def queue_live_audio(self, samples, pending_bytes):
        pending_bytes.extend(samples.tobytes())
        chunk_size = self.sample_rate * 4
        while len(pending_bytes) >= chunk_size:
            chunk = bytes(pending_bytes[:chunk_size])
            del pending_bytes[:chunk_size]
            self.live_audio_queue.put(
                np.frombuffer(chunk, dtype=np.int16).reshape(-1, 2).copy()
            )

    def run(self):
        if not PYAUDIO_AVAILABLE:
            self.error = "PyAudioWPatch is not installed. Install the project requirements and restart."
            self.ready_event.set()
            return
        p = pyaudio.PyAudio()
        streams = []
        wf = None
        try:
            wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)
            default_speakers = p.get_device_info_by_index(wasapi_info["defaultOutputDevice"])
            loopback_device = default_speakers
            if not default_speakers["isLoopbackDevice"]:
                loopback_device = next(
                    (
                        device for device in p.get_loopback_device_info_generator()
                        if default_speakers["name"] in device["name"]
                    ),
                    None,
                )
            if loopback_device is None:
                raise RuntimeError(f"No WASAPI loopback device found for {default_speakers['name']}.")

            default_mic = p.get_device_info_by_index(wasapi_info["defaultInputDevice"])
            if default_mic["maxInputChannels"] < 1:
                raise RuntimeError(f"No microphone input is available: {default_mic['name']}.")

            loopback_idx = loopback_device["index"]
            mic_idx = default_mic["index"]
            rate = int(loopback_device["defaultSampleRate"])
            self.sample_rate = rate

            output_directory = os.path.dirname(self.filename)
            if output_directory:
                os.makedirs(output_directory, exist_ok=True)
            wf = wave.open(self.filename, 'wb')
            wf.setnchannels(2)
            wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
            wf.setframerate(rate)
 
            def make_callback(audio_queue):
                def callback(in_data, frame_count, time_info, status):
                    if in_data:
                        audio_queue.put(in_data)
                    return (None, pyaudio.paContinue)
                return callback

            loopback_stream = p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=rate,
                input=True,
                input_device_index=loopback_idx,
                stream_callback=make_callback(self.loopback_queue),
                start=False,
            )
            streams.append(loopback_stream)
           
            mic_stream = p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=rate,
                input=True,
                input_device_index=mic_idx,
                stream_callback=make_callback(self.mic_queue),
                start=False,
            )
            streams.append(mic_stream)
 
            loopback_stream.start_stream()
            mic_stream.start_stream()
            self.ready_event.set()
            loopback_buffer = bytearray()
            mic_buffer = bytearray()
            live_pending_bytes = bytearray()
 
            while not self._stop_event.is_set():
                for audio_queue, audio_buffer in (
                    (self.loopback_queue, loopback_buffer),
                    (self.mic_queue, mic_buffer),
                ):
                    while True:
                        try:
                            audio_buffer.extend(audio_queue.get_nowait())
                        except queue.Empty:
                            break
                frames_written, live_samples = self._write_aligned_audio(
                    wf, loopback_buffer, mic_buffer
                )
                self.frames_written += frames_written
                if frames_written:
                    self.queue_live_audio(live_samples, live_pending_bytes)
                self._stop_event.wait(0.01)
 
            for stream in streams:
                if stream.is_active():
                    stream.stop_stream()
            for audio_queue, audio_buffer in (
                (self.loopback_queue, loopback_buffer),
                (self.mic_queue, mic_buffer),
            ):
                while True:
                    try:
                        audio_buffer.extend(audio_queue.get_nowait())
                    except queue.Empty:
                        break
            frames_written, live_samples = self._write_aligned_audio(
                wf, loopback_buffer, mic_buffer, flush=True
            )
            self.frames_written += frames_written
            if frames_written:
                self.queue_live_audio(live_samples, live_pending_bytes)
            if live_pending_bytes:
                self.live_audio_queue.put(
                    np.frombuffer(bytes(live_pending_bytes), dtype=np.int16)
                    .reshape(-1, 2)
                    .copy()
                )
        except Exception as e:
            self.error = str(e)
        finally:
            self.ready_event.set()
            for stream in streams:
                try:
                    if stream.is_active():
                        stream.stop_stream()
                    stream.close()
                except Exception as e:
                    if self.error is None:
                        self.error = f"Could not close an audio input stream: {e}"
            if wf is not None:
                try:
                    wf.close()
                except Exception as e:
                    if self.error is None:
                        self.error = f"Could not finalize the recording file: {e}"
            try:
                p.terminate()
            except Exception as e:
                if self.error is None:
                    self.error = f"Could not close the audio device manager: {e}"
 
    def stop(self):
        self._stop_event.set()


def _retry_removing_readonly_path(function, path, exc_info):
    error = exc_info[1]
    if not isinstance(error, PermissionError):
        raise error
    if os.path.islink(path):
        raise error

    mode = os.stat(path, follow_symlinks=False).st_mode
    os.chmod(path, mode | stat.S_IWRITE)
    function(path)


def _prepare_audio_for_transcription(audio):
    """Convert one decoded channel to finite float32 and gently lift quiet speech."""
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        return audio
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    peak = float(np.max(np.abs(audio)))
    # decode_audio returns float audio in approximately [-1, 1]. Boost low-level
    # channels for recognition, but avoid amplifying near-silence/noise excessively.
    if peak > 1e-5 and peak < 0.35:
        audio = audio * min(0.35 / peak, 6.0)
    return np.clip(audio, -1.0, 1.0).astype(np.float32, copy=False)


def normalize_stereo_wav(path, target_peak=0.72, max_gain=8.0):
    """Normalize each call channel independently, preserving stereo speaker separation.

    The receiver/system-audio loopback is often much quieter than the local mic.
    Independent channel gain improves audibility while limiting clipping. Near-silent
    channels are left alone rather than amplifying noise indefinitely.
    """
    try:
        with wave.open(str(path), 'rb') as wf:
            channels = wf.getnchannels()
            width = wf.getsampwidth()
            rate = wf.getframerate()
            frames = wf.getnframes()
            raw = wf.readframes(frames)
        if channels != 2 or width != 2 or frames == 0:
            return
        samples = np.frombuffer(raw, dtype='<i2').copy().reshape(-1, 2)
        for channel_index in range(2):
            channel = samples[:, channel_index].astype(np.float32)
            peak = float(np.max(np.abs(channel))) if channel.size else 0.0
            if peak >= 80.0:
                gain = min((32767.0 * target_peak) / peak, max_gain)
                channel *= gain
                samples[:, channel_index] = np.clip(channel, -32768, 32767).astype(np.int16)
        temp_path = str(path) + '.normalized.tmp'
        with wave.open(temp_path, 'wb') as wf:
            wf.setnchannels(2)
            wf.setsampwidth(2)
            wf.setframerate(rate)
            wf.writeframes(samples.astype('<i2', copy=False).tobytes())
        os.replace(temp_path, path)
    except (OSError, wave.Error, ValueError):
        # Keep the original recording if normalization is not possible.
        try:
            temp_path = str(path) + '.normalized.tmp'
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except OSError:
            pass


def transcribe_audio_channels(model, channels, beam_size, time_offset=0):
    transcript_segments = []
    for speaker, raw_audio in channels:
        audio = _prepare_audio_for_transcription(raw_audio)
        if audio.size == 0 or float(np.max(np.abs(audio))) < 1e-5:
            continue
        segments, info = model.transcribe(
            audio,
            beam_size=max(1, int(beam_size)),
            task="translate",
            vad_filter=True,
            vad_parameters={
                "threshold": 0.25,
                "min_speech_duration_ms": 120,
                "min_silence_duration_ms": 450,
                "speech_pad_ms": 350,
            },
            condition_on_previous_text=False,
            without_timestamps=False,
            temperature=0.0,
            compression_ratio_threshold=2.4,
            log_prob_threshold=-1.0,
            no_speech_threshold=0.55,
        )
        for segment in segments:
            text = segment.text.strip()
            # Ignore empty or common Whisper silence hallucinations.
            if text and text not in {"...", "Thank you.", "Thanks for watching!"}:
                transcript_segments.append((segment.start + time_offset, speaker, text))

    transcript_segments.sort(key=lambda segment: segment[0])
    return [
        f"[{int(start)}s] {speaker}: {text}"
        for start, speaker, text in transcript_segments
    ]


def transcribe_audio_buffer(model, audio_samples, sample_rate, beam_size, time_offset=0):
    from faster_whisper.audio import decode_audio

    audio_samples = np.asarray(audio_samples, dtype=np.int16)
    if audio_samples.ndim != 2 or audio_samples.shape[1] != 2:
        raise ValueError("Live audio buffer must contain two channels (customer and agent).")
    audio_buffer = io.BytesIO()
    with wave.open(audio_buffer, "wb") as audio_file:
        audio_file.setnchannels(2)
        audio_file.setsampwidth(2)
        audio_file.setframerate(sample_rate)
        audio_file.writeframes(audio_samples.astype('<i2', copy=False).tobytes())
    audio_buffer.seek(0)
    customer_audio, agent_audio = decode_audio(audio_buffer, split_stereo=True)
    return transcribe_audio_channels(
        model,
        [("Customer", customer_audio), ("Agent", agent_audio)],
        beam_size,
        time_offset,
    )


def transcribe_call_audio(model, audio_path, beam_size):
    from faster_whisper.audio import decode_audio

    try:
        with wave.open(audio_path, "rb") as audio_file:
            channels_count = audio_file.getnchannels()
            frame_count = audio_file.getnframes()
            sample_rate = audio_file.getframerate()
        if frame_count == 0 or sample_rate <= 0:
            raise ValueError("The recording is empty or has invalid audio metadata.")
    except (wave.Error, OSError) as exc:
        raise ValueError(f"The recording cannot be read as audio: {exc}") from exc

    if channels_count == 2:
        customer_audio, agent_audio = decode_audio(audio_path, split_stereo=True)
        channels = [("Customer", customer_audio), ("Agent", agent_audio)]
    else:
        channels = [("Speaker (not separated)", decode_audio(audio_path))]
    return transcribe_audio_channels(model, channels, beam_size)


def build_call_outputs(transcript, agent_email, contact_number, call_source):
    """Build case notes, email draft, and compliance evidence from transcript text only.

    This deliberately avoids inventing troubleshooting or resolution details. The
    compliance output is evidence for human review, not an automatic pass/fail verdict.
    """
    def clean_repeated_speech(text):
        """Collapse exact repeated phrases caused by speech-recognition loops."""
        text = re.sub(r"\s+", " ", str(text or "")).strip()
        if not text:
            return text
        # First collapse repeated clauses separated by punctuation.
        for _ in range(5):
            previous = text
            text = re.sub(r"(.{12,180}?)(?:[,;.!?]\s*\1){1,}", r"\1", text, flags=re.I)
            words = text.split()
            lowered = [w.casefold().strip(".,!?;:") for w in words]
            changed = False
            max_n = min(24, len(words) // 2)
            for n in range(max_n, 3, -1):
                out = []
                i = 0
                while i < len(words):
                    if i + 2 * n <= len(words) and lowered[i:i+n] == lowered[i+n:i+2*n]:
                        out.extend(words[i:i+n])
                        i += 2 * n
                        changed = True
                    else:
                        out.append(words[i])
                        i += 1
                words = out
                lowered = [w.casefold().strip(".,!?;:") for w in words]
            text = " ".join(words)
            if text == previous or not changed:
                break
        return text.strip()

    records = []
    for line in transcript.splitlines():
        line = line.strip()
        if not line:
            continue
        match = re.match(r'^\[(\d+)s\]\s+([^:]+):\s*(.*)$', line)
        if match:
            timestamp, speaker, utterance = match.groups()
            utterance = clean_repeated_speech(utterance)
            cleaned_line = f"[{timestamp}s] {speaker.strip()}: {utterance}"
            records.append((int(timestamp), speaker.strip(), utterance, cleaned_line))
        else:
            utterance = clean_repeated_speech(line)
            records.append((None, "Speaker not separated", utterance, utterance))
    if not records:
        raise ValueError("Cannot generate case notes from an empty transcript.")

    # Preserve conversation order and remove only exact repeated utterances.
    unique_records = []
    seen = set()
    for record in records:
        identity = (record[1].casefold(), re.sub(r'\s+', ' ', record[2]).casefold())
        if identity[1] and identity not in seen:
            seen.add(identity)
            unique_records.append(record)

    def excerpts(selected_records, limit=8):
        if not selected_records:
            return "Not established from the transcript."
        return "\n".join(
            f'- [{r[0]}s] {r[1]}: "{r[2]}"' if r[0] is not None
            else f'- {r[1]}: "{r[2]}"'
            for r in selected_records[:limit]
        )

    def concise_statements(selected_records, limit=6, fallback="Not established from the call."):
        if not selected_records:
            return fallback
        statements = []
        for record in selected_records[:limit]:
            statement = re.sub(r"\s+", " ", record[2]).strip(" \t\r\n.,;:-")
            statement = re.sub(r"\b(?:sir|madam)\b[, ]*", "", statement, flags=re.I)
            statement = re.sub(
                r"^(?:sir|madam|okay|ok|so|well|yes|yeah|right|now)[, ]+",
                "",
                statement,
                flags=re.I,
            ).strip()
            statement = re.sub(
                r"^(?:as far as i understand|i understand that|to summarize[, ]*)",
                "",
                statement,
                flags=re.I,
            ).strip()
            statement = re.sub(
                r"^(?:i|we)\s+(?:(?:have|had)\s+)?",
                "",
                statement,
                flags=re.I,
            ).strip()
            statement = re.sub(r",\s*(?=[.!?]|$)", "", statement).strip(" \t\r\n.,;:-")
            if re.search(r"\b(?:cables?|connectors?)\b", statement, re.I) and re.search(
                r"\b(removed|disconnected|reconnected|connected|secured|fixed|checked|tested)\b",
                statement,
                re.I,
            ):
                statement = "Cable connections were handled; verify the exact steps."
            if statement and statement.casefold() not in {
                item.casefold() for item in statements
            }:
                statements.append(statement)
        return "\n".join(f"- {statement}" for statement in statements) or fallback

    def summary_phrase(selected_records, limit=3, fallback="the reason for the call was not clear"):
        if not selected_records:
            return fallback
        statements = []
        for record in selected_records[:limit]:
            statement = re.sub(r"\s+", " ", record[2]).strip(" \t\r\n.,;:-")
            statement = re.sub(r"\b(?:sir|madam)\b[, ]*", "", statement, flags=re.I)
            statement = re.sub(
                r"^(?:sir|madam|okay|ok|so|well|yes|right)[, ]+",
                "",
                statement,
                flags=re.I,
            ).strip()
            statement = re.sub(
                r"^(?:as far as i understand|i understand that|to summarize[, ]*)",
                "",
                statement,
                flags=re.I,
            ).strip()
            statement = re.sub(
                r"^(?:i|we)\s+(?:(?:have|had)\s+)?",
                "",
                statement,
                flags=re.I,
            ).strip()
            statement = re.sub(r",\s*(?=[.!?]|$)", "", statement).strip(" \t\r\n.,;:-")
            statement = re.sub(
                r"\byour ticket has been created\b",
                "a support ticket has been created",
                statement,
                flags=re.I,
            )
            statement = re.sub(
                r"\byour system will be replaced in (\d+) business days\b",
                r"a system replacement is expected within \1 business days",
                statement,
                flags=re.I,
            )
            statement = re.sub(
                r"\b(?:there is )?no display in your monitor\b",
                "a monitor display issue",
                statement,
                flags=re.I,
            )
            if statement and statement.casefold() not in {
                item.casefold() for item in statements
            }:
                statements.append(statement)
        return " and ".join(statements) if statements else fallback

    def split_statements(selected_records):
        statements = []
        for record in selected_records:
            parts = re.split(r"(?<=[.!?])\s+|;\s+", record[2].strip())
            for part in parts:
                text = part.strip()
                if text:
                    statements.append((record[0], record[1], text, text))
        return statements

    customer_records = [r for r in unique_records if r[1].casefold() == "customer"]
    agent_records = [r for r in unique_records if r[1].casefold() == "agent"]
    other_records = [r for r in unique_records if r[1].casefold() not in {"customer", "agent"}]
    statement_records = split_statements(unique_records)
    issue_pattern = re.compile(
        r'\b(issue|problem|error|fault|broken|not working|doesn.t work|unable to|'
        r"can.t|cannot|slow|crash|failed|failure|damage|overheat|battery|screen|"
        r'keyboard|display|wifi|wi-fi|blue screen|boot|power|charging|network)\b', re.I)
    issue_symptom_pattern = re.compile(
        r'\b(no|not|unable|cannot|can.t|doesn.t|does not|won.t|will not|'
        r'problem\w*|issues?|errors?|faults?|broken|failed|failure|flicker\w*|crash\w*|'
        r'drain\w*|overheat\w*|disconnect\w*|unstable|replacement|replace\w*)\b',
        re.I,
    )
    action_pattern = re.compile(
        r'\b(check\w*|click\w*|press\w*|hold|open\w*|close\w*|restart\w*|reboot\w*|'
        r'install\w*|uninstall\w*|update\w*|run|launch\w*|connect\w*|disconnect\w*|'
        r'select\w*|try|test\w*|verif\w*|confirm\w*|wait|guide\w*|follow\w*|'
        r'reset|replac\w*|troubleshoot\w*|navigate\w*|go to|share screen)\b', re.I)
    completed_action_pattern = re.compile(
        r'\b(?:i|we)\s+(?:(?:have|had)\s+)?'
        r'(?:checked|tested|changed|set|connected|disconnected|restarted|rebooted|'
        r'installed|uninstalled|updated|reset|replaced|captured|took|ran|opened|closed)\b',
        re.I,
    )
    completed_cable_action_pattern = re.compile(
        r'\b(?:cables?|connectors?)\b.{0,40}\b'
        r'(?:removed|disconnected|reconnected|connected|secured|fixed|checked|tested)\b|'
        r'\b(?:removed|disconnected|reconnected|connected|secured|fixed|checked|tested)\b'
        r'.{0,40}\b(?:cables?|connectors?)\b',
        re.I,
    )
    outcome_pattern = re.compile(
        r'\b(working now|works now|issue is gone|problem is gone|back to normal|'
        r'completed successfully|still not working|not resolved|persists|'
        r'(?:issue|problem|device|monitor|keyboard)\s+(?:is|was|has been)\s+'
        r'(?:resolved|fixed|working|operational)|'
        r'ticket (?:has been |was )?(?:created|raised|traced|updated)|'
        r'case (?:has been |was )?(?:created|raised|traced|updated)|'
        r'(?:transferred|escalated) to\b|'
        r'(?:replacement|repair|service)\s+(?:has been |was )?(?:approved|scheduled|arranged)|'
        r'will be (?:replaced|repaired|shipped)|within \d+ business days?)\b',
        re.I,
    )

    # Filter routine greetings and acknowledgements before suggesting issue details.
    # When the recording has no speaker separation, never pretend to know which
    # person spoke; use only technically relevant statements and label them for review.
    routine_pattern = re.compile(
        r'^(?:good morning|good afternoon|good evening|hello|hi|hey|thank you|'
        r'thank you very much|thanks|okay|ok|yes|yes yes|no problem|how are you|'
        r'you are welcome|welcome|bye|goodbye|have a nice day)[.!?,\s]*$',
        re.I,
    )

    def is_routine(utterance):
        cleaned = re.sub(r'^[\W_]+|[\W_]+$', '', utterance).strip()
        return bool(routine_pattern.match(cleaned)) or len(cleaned) < 5

    issue_pool = (
        [r for r in statement_records if r[1].casefold() == "customer"]
        if customer_records
        else [r for r in statement_records if r[1].casefold() not in {"customer", "agent"}]
    )
    issue_records = [
        r for r in issue_pool
        if issue_pattern.search(r[2])
        and issue_symptom_pattern.search(r[2])
        and not re.search(r'\bto summarize\b', r[2], re.I)
        and not is_routine(r[2])
        and not r[2].rstrip().endswith("?")
    ]
    if not issue_records and customer_records:
        # Customer descriptions sometimes omit the technical keywords used by
        # issue_pattern, so retain their non-greeting statements as a fallback.
        issue_records = [r for r in customer_records if not is_routine(r[2])]
    if not issue_records and not customer_records:
        issue_records = [
            r for r in statement_records
            if r[1].casefold() not in {"customer", "agent"}
            and issue_pattern.search(r[2])
            and not is_routine(r[2])
        ]

    system_hold_pattern = re.compile(
        r'\b(person you are (?:speaking|talking) (?:to|with) has|'
        r'put your call on hold|kept your call on hold)\b', re.I
    )
    future_action_pattern = re.compile(r'\b(will|shall|would)\b', re.I)

    def is_performed_action(record):
        text = record[2].strip()
        if system_hold_pattern.search(text) or is_routine(text):
            return False
        if future_action_pattern.search(text):
            return False
        if record[1].casefold() not in {"customer", "agent"}:
            return bool(
                completed_action_pattern.search(text)
                or completed_cable_action_pattern.search(text)
            )
        return bool(action_pattern.search(text))

    action_records = [
        r for r in statement_records
        if r[1].casefold() == "agent" and is_performed_action(r)
    ]
    if not action_records and not agent_records:
        # With an unseparated/mono transcript, show only possible action statements.
        # Do not label them as confirmed agent actions.
        action_records = [
            r for r in statement_records
            if r[1].casefold() not in {"customer", "agent"}
            and is_performed_action(r)
        ]
    outcome_records = [
        r for r in statement_records
        if outcome_pattern.search(r[2]) and not is_routine(r[2])
        and not re.search(r'\bto summarize\b', r[2], re.I)
        and not r[2].rstrip().endswith("?")
    ]

    issue_text = concise_statements(
        issue_records[:1],
        limit=1,
        fallback="The reported issue was not clear from the call; confirm with the customer.",
    )
    troubleshooting_text = concise_statements(
        action_records,
        limit=8,
        fallback="No troubleshooting step was clearly captured; confirm before submitting.",
    )
    if not customer_records and issue_records:
        issue_text = "Speaker not separated; verify these reported symptoms:\n" + issue_text
    if other_records and not agent_records and action_records:
        troubleshooting_text = (
            "Speaker not separated; verify these steps were performed:\n"
            + troubleshooting_text
        )
    outcome_text = concise_statements(
        outcome_records[-5:],
        limit=5,
        fallback="No final resolution or customer update was stated during the call.",
    )

    identity_pattern = re.compile(
        r'\b(my name is|this is|i am|i\'m|calling from|dealer|customer)\b', re.I
    )
    caller_pool = customer_records if customer_records else other_records
    caller_records = [
        r for r in caller_pool
        if identity_pattern.search(r[2]) and not is_routine(r[2])
    ]
    duration_pattern = re.compile(
        r'\b(since (?:yesterday|today|last (?:night|week|month|year)|\d+)|'
        r'for (?:about |around |the past |last |the last )?\d+\s*'
        r'(?:hours?|days?|weeks?|months?|years?)|started|began|yesterday|'
        r'from tomorrow|last (?:night|week|month|year)|\d+\s*'
        r'(?:hours?|days?|weeks?|months?|years?))\b',
        re.I,
    )
    duration_pool = customer_records if customer_records else other_records
    issue_duration_records = [
        r for r in duration_pool
        if duration_pattern.search(r[2]) and not r[2].rstrip().endswith("?")
    ]
    further_action_pattern = re.compile(
        r'\b(will (?:be )?(?:replac\w*|ship\w*|send\w*|call back|follow[- ]?up)|'
        r'need(?:s)? to (?:replac\w*|ship\w*|send\w*|call|follow[- ]?up)|'
        r'(?:please )?(?:re)?connect\b|connect again|'
        r'follow[- ]?up|call back|callback|escalat\w*|schedule\w*|book\w*|'
        r'raise (?:a |an )?(?:case|ticket|work order|wo)|create (?:a |an )?(?:case|ticket)|'
        r'arrange\w*|pending|further action|next step|service order|work order|WO)\b',
        re.I
    )
    action_followup_pool = agent_records if agent_records else other_records
    further_action_records = [
        r for r in statement_records
        if (
            r[1].casefold() == "agent"
            if agent_records
            else r[1].casefold() not in {"customer", "agent"}
        )
        if further_action_pattern.search(r[2])
        and not is_routine(r[2])
        and not r[2].rstrip().endswith("?")
    ]
    rescue_mentions = [r for r in unique_records if re.search(r'\brescue\s*lens\b', r[2], re.I)]
    availability_records = [r for r in unique_records if re.search(
        r'\b(available now|good time|convenient time|free now|availability|is now a good time|'
        r'Mon(?:day)?|Tue(?:sday)?|Wed(?:nesday)?|Thu(?:rsday)?|Fri(?:day)?|weekdays|'
        r'10\s*(?:AM|a\.m\.))\b', r[2], re.I)]
    gplus_records = [r for r in unique_records if re.search(r'\bG\s*\+\s*(?:code)?\b', r[2], re.I)]
    alternate_contact_records = [r for r in unique_records if re.search(
        r'\b(alternate|alternative|secondary|another)\b.{0,35}\b(number|contact|phone)\b', r[2], re.I)]
    mobile_allowed_records = [r for r in unique_records if re.search(
        r'\b(mobile|cell phone)\b.{0,80}\b(allow(?:ed)?|permission|yes|no|can|cannot|may)\b|'
        r'\b(allow(?:ed)?|permission|yes|no|can|cannot|may)\b.{0,80}\b(mobile|cell phone)\b',
        r[2], re.I)]
    work_order_records = [r for r in unique_records if re.search(
        r'\b(WO|work order)\b', r[2], re.I
    )]
    prt_records = [r for r in unique_records if re.search(
        r'\b(PRT|recommend(?:ation|ed)?|part replacement)\b', r[2], re.I
    )]

    additional_issue_marker = re.compile(
        r'\b(also|another issue|in addition|as well|separate issue|different issue)\b',
        re.I,
    )
    additional_issue_records = [
        r for r in customer_records
        if issue_pattern.search(r[2]) and additional_issue_marker.search(r[2])
    ]
    customer_name = "Customer"
    name_match = next(
        (
            match for record in caller_records
            if record[1].casefold() == "customer"
            for match in [
                re.search(
                    r'\b(?:my name is|this is|i am|i\'m)\s+([A-Z][\w\'-]+(?:\s+[A-Z][\w\'-]+){0,2})',
                    record[2],
                    re.I,
                )
            ]
            if match
        ),
        None,
    )
    if name_match:
        customer_name = name_match.group(1)

    email_issue_records = [
        record for record in issue_records
        if not re.search(r'\bto summarize\b', record[2], re.I)
    ]
    issue_email_text = summary_phrase(
        email_issue_records[:1],
        limit=1,
        fallback="the reason for the call was not clear",
    )
    if (
        re.search(r"\bAC\b", issue_email_text, re.I)
        and re.search(r"\bkeyboard\b", issue_email_text, re.I)
        and re.search(r"\bnot working\b", issue_email_text, re.I)
    ):
        issue_email_text = "an AC-related issue and a nonworking keyboard"
    def email_action_summaries(selected_records):
        summaries = []
        cable_pattern = re.compile(
            r'\b(cables?|connectors?)\b.*\b(removed|disconnected|reconnected|'
            r'connected|secured|fixed|checked|tested)\b|'
            r'\b(removed|disconnected|reconnected|connected|secured|fixed|'
            r'checked|tested)\b.*\b(cables?|connectors?)\b',
            re.I,
        )
        for record in selected_records[:4]:
            text = record[2]
            if cable_pattern.search(text):
                summary = (
                    "Cable connections were handled during the call; "
                    "please verify the exact steps before sending."
                )
            elif re.search(r'\bpower cable\b', text, re.I) and re.search(
                r'\b(checked|tested|changed|set|connected|disconnected|reconnected)\b',
                text,
                re.I,
            ):
                summary = "The power connection was checked or adjusted during the call."
            else:
                summary = summary_phrase([record], limit=1)
                summary = summary[0].upper() + summary[1:] if summary else summary
            if summary and summary.casefold() not in {
                item.casefold() for item in summaries
            }:
                summaries.append(summary)
        return summaries

    email_action_summaries = email_action_summaries(action_records)
    troubleshooting_email_text = (
        "\n".join(f"- {item}" for item in email_action_summaries)
        if email_action_summaries
        else "The transcript does not clearly confirm any completed troubleshooting steps. "
        "Please verify the actions taken before sending."
    )
    resolution_email_text = summary_phrase(
        outcome_records[-3:],
        limit=3,
        fallback=(
            "No confirmed resolution or customer update was clear from the transcript. "
            "Please verify before sending."
        ),
    )
    next_action_email_text = (
        "The transcript indicates a connection step was requested. "
        "Please verify the exact next action before sending."
        if not further_action_records
        and any(
            re.search(r"\b(?:please )?(?:re)?connect\b", r[2], re.I)
            for r in statement_records
        )
        else (
            summary_phrase(
                further_action_records[-3:],
                limit=3,
                fallback=(
                    "The next action was not clear from the transcript. "
                    "Please verify the agreed follow-up before sending."
                ),
            )
        )
    )
    if further_action_records:
        if any(
            re.search(r"\b(?:please )?(?:re)?connect\b", record[2], re.I)
            for record in further_action_records
        ):
            next_action_email_text = (
                "The transcript indicates that a connection step was requested. "
                "Please verify what needs to be connected and the agreed follow-up "
                "before sending."
            )
        elif any(
            re.search(r"\bfollow[- ]?up\b|\bcall back\b|\bcallback\b", record[2], re.I)
            for record in further_action_records
        ):
            follow_up_text = summary_phrase(further_action_records[-3:], limit=3)
            follow_up_text = re.sub(
                r"^will follow[- ]?up\b",
                "Lenovo will follow up",
                follow_up_text,
                flags=re.I,
            )
            next_action_email_text = follow_up_text.rstrip(".!?") + "."
            if not agent_records:
                next_action_email_text += " Please verify the follow-up before sending."
        else:
            next_action_email_text = (
                "The agreed follow-up mentioned during the call was: "
                + summary_phrase(further_action_records[-3:], limit=3)
                + (
                    ". Please verify the follow-up before sending."
                    if not agent_records
                    else "."
                )
            )
    if customer_records:
        caller_details = []
        if name_match:
            caller_details.append(f"Name: {name_match.group(1)}")
        company_records = [
            r for r in caller_records
            if re.search(r"\b(dealer|company|business|organization|organisation)\b", r[2], re.I)
        ]
        if company_records:
            caller_details.extend(
                concise_statements(company_records, limit=2).splitlines()
            )
        caller_text = "\n".join(caller_details) or (
            "Customer is speaking; name or organization was not stated clearly."
        )
    elif caller_records:
        caller_text = (
            "Caller identity was not distinguishable from the agent in this recording; "
            "enter the caller type and name manually."
        )
    else:
        caller_text = "Not identified in the transcript; confirm whether the caller is IT, the customer, or a dealer."
    duration_text = concise_statements(
        issue_duration_records,
        limit=3,
        fallback="Not stated during the call.",
    )
    further_action_text = concise_statements(
        further_action_records,
        limit=5,
        fallback="No further action was agreed during the call.",
    )
    if work_order_records:
        prt_text = (
            concise_statements(prt_records, limit=3)
            if prt_records
            else "WO mentioned, but no PRT recommendation was stated; confirm in the work order."
        )
    else:
        prt_text = "WO case not identified in the transcript; complete only when applicable."
    if availability_records:
        availability_text = concise_statements(availability_records, limit=3)
    else:
        availability_text = (
            "Mon-Fri 10 AM to 6 PM (standard window provided; "
            "customer-specific availability was not stated in the transcript)."
        )

    case_template = (
        "LENOVO CASE TEMPLATE\n\n"
        f"Agent: {agent_email}\n"
        f"Call Source Type: {call_source}\n\n"
        f"****IT / Customer / Dealer:****\n{caller_text}\n\n"
        f"****Incoming Caller Number:**** {contact_number}\n\n"
        "****Alternate Contact Number:****\n"
        f"{concise_statements(alternate_contact_records, 3)}\n\n"
        "****Issue Description & Duration:****\n"
        f"{issue_text}\n\n"
        "Issue duration explicitly stated:\n"
        f"{duration_text}\n\n"
        "****Troubleshooting Performed:****\n"
        f"{troubleshooting_text}\n\n"
        "=================================================\n\n"
        "****Any Additional Issue Identified:**** Apart from reported issue\n"
        f"{concise_statements(additional_issue_records, limit=3, fallback='No separate additional issue was identified during the call.')}\n\n"
        "****Resolution Provided: What is the final resolution provided ****\n"
        f"{outcome_text}\n\n"
        "****Further Action Required:****\n"
        f"{further_action_text}\n\n"
        "****Copy-Paste PRT Recommendation (For WO cases only):****\n"
        f"{prt_text}\n\n"
        "****Rescue Lens Used: Video session used ?****\n"
        + ("Mentioned during the call; verify whether a video session was used.\n"
           + concise_statements(rescue_mentions, 2)
           if rescue_mentions else "Not stated during the call; verify manually.")
        + "\n\nSDI:\n"
        "Customer alternative contact number:\n" + concise_statements(alternate_contact_records, 3) + "\n"
        "Customer availability:\n" + availability_text + "\n"
        "G+ Code: " + concise_statements(gplus_records, 2) + "\n\n"
        "Mobile phone allowed: Yes/no\n"
        + (concise_statements(mobile_allowed_records, 3) if mobile_allowed_records else
           "Not stated during the call; confirm with the customer.")
        + "\n\n"
    )

    # Prefer a specific technical detail over generic statements such as
    # "Sir, I have a problem." Never use an opening greeting as the subject.
    generic_issue_pattern = re.compile(
        r'^(?:sir[, ]*)?(?:i have a problem|i have an issue|there is a problem|'
        r'hello|hi|good morning|good afternoon|good evening|thank you|okay|ok)[.! ,]*$',
        re.I,
    )
    specific_subject_candidates = [
        r for r in issue_records
        if not is_routine(r[2])
        and not generic_issue_pattern.match(r[2].strip())
        and re.search(
            r'\b(mouse|left click|click|not selecting|not working|not matching|battery|'
            r'screen|keyboard|display|wifi|wi-fi|power|charging|laptop|'
            r'manufactur(?:e|ing)|warranty|support|serial number|damage|'
            r'problem|issue|system|replacement|replace|bundle)\b',
            r[2], re.I
        )
    ]
    subject_record = next(iter(specific_subject_candidates), None)
    if subject_record:
        subject_phrase = summary_phrase([subject_record], limit=1)
        if (
            re.search(r"\bAC\b", subject_phrase, re.I)
            and re.search(r"\bkeyboard\b", subject_phrase, re.I)
        ):
            subject_phrase = "AC and keyboard concerns"
        subject_phrase = re.sub(
            r'^(?:sir[, ]*|hello[, ]*|hi[, ]*|good morning[, ]*|good afternoon[, ]*)',
            '', subject_phrase, flags=re.I
        ).strip()
        subject_phrase = subject_phrase[:65].strip(" .,:;!?-")
    else:
        subject_phrase = ""
    email_subject = (
        f"Call follow-up: {subject_phrase}"
        if subject_phrase else "Call follow-up: Technical issue and eligibility to be confirmed"
    )
    email_draft = (
        f"Subject: {email_subject}\n\n"
        f"Dear {customer_name},\n\n"
        "Thank you for contacting Lenovo Technical Support.\n\n"
        "As discussed during our call, please find below a summary of the issues "
        "reported and the troubleshooting performed.\n\n"
        "**1. Reason for Contact**\n\n"
        f"You contacted Lenovo Technical Support with the following reported concern: "
        f"{issue_email_text}."
        + (
            " Please verify the issue details before sending."
            if not customer_records
            else ""
        )
        + "\n\n**2. Troubleshooting Performed**\n\n"
        f"{troubleshooting_email_text}\n\n"
        "**3. Resolution / Update Provided**\n\n"
        f"{resolution_email_text.rstrip('.!?')}.\n\n"
        "**4. Further Action Required**\n\n"
        f"{next_action_email_text}\n\n"
        "Thank you for choosing Lenovo. We appreciate your time and cooperation.\n\n"
        "Regards,\n"
        "Lenovo Technical Support"
    )

    verification_pattern = re.compile(
        r'\b(correct|right|confirm\w*|verif\w*|match\w*|spell\w*|letter by letter|'
        r'repeat\w*|read back|is that|did i get|am i right)\b|'
        r'\b[A-Z]\s+from\s+[A-Z][a-z]+\b', re.I
    )
    compliance_rules = [
        (
            "Customer details verified",
            re.compile(r'\b(name|customer details?|phone|telephone|contact number|serial number|S\s*N)\b', re.I),
            verification_pattern,
        ),
        (
            "Email ID phonetically verified",
            re.compile(r'\b(e-?mail|email address|email id|e mail)\b', re.I),
            re.compile(
                r'\b(correct|right|confirm\w*|verif\w*|spell\w*|letter by letter|'
                r'repeat\w*|read back|dictat\w*|at the rate|dot com)\b|'
                r'\b[A-Z]\s+from\s+[A-Z][a-z]+\b', re.I
            ),
        ),
        (
            "Customer availability checked",
            re.compile(
                r'\b(are you available|is this a good time|good time to talk|'
                r'convenient time|are you free|free now|can we talk|'
                r'when can (?:i|we) call|best time to call|'
                r'what time.{0,30}(?:call|available)|availability)\b', re.I
            ),
            None,
        ),
        (
            "Rescue Lens disclaimer given",
            re.compile(
                r'\b(rescue\s*lens|riskulence|risk\s*lens|remote assistance|'
                r'remote access|video session|screen share|share screen)\b', re.I
            ),
            re.compile(
                r'\b(permission|consent|agree|allow|disclaimer|authori[sz]\w*|'
                r'before we (?:start|get started)|i would like to inform|'
                r'i need to inform|let me explain|explain\w*|'
                r'personal (?:information|files)|private (?:information|files))\b', re.I
            ),
        ),
        (
            "Faulty-part collection disclaimer given",
            re.compile(r'\b(faulty|defective|replacement)\b.{0,40}\bpart\b|\bpart\b.{0,40}\b(faulty|defective)\b', re.I),
            re.compile(
                r'\b(collect\w*|collection|pick up|pickup|return\w*|courier|'
                r'old part|faulty part|replacement part)\b', re.I
            ),
        ),
        (
            "APOS warranty upselling checked",
            re.compile(
                r'\b(warranty|protection plan|A\s*P\s*O\s*S|premium support|'
                r'extended coverage)\b', re.I
            ),
            re.compile(
                r'\b(offer\w*|option|quote|quotation|plan|price|purchase|'
                r'interested|declin\w*|accept\w*|would you like|available)\b', re.I
            ),
        ),
    ]
    compliance_evidence = {}
    for label, topic_pattern, evidence_pattern in compliance_rules:
        matched_contexts = []
        if label == "Rescue Lens disclaimer given":
            topic_records = [r for r in unique_records if topic_pattern.search(r[2])]
            disclaimer_records = [
                r for r in unique_records if evidence_pattern.search(r[2])
            ]
            for topic_record in topic_records:
                for disclaimer_record in disclaimer_records:
                    close_in_time = (
                        topic_record[0] is None
                        or disclaimer_record[0] is None
                        or abs(topic_record[0] - disclaimer_record[0]) <= 180
                    )
                    if close_in_time:
                        matched_contexts.extend((topic_record, disclaimer_record))
        elif label in {
            "Customer details verified",
            "Email ID phonetically verified",
        }:
            topic_records = [r for r in unique_records if topic_pattern.search(r[2])]
            verification_records = [
                r for r in unique_records if evidence_pattern.search(r[2])
            ]
            for topic_record in topic_records:
                for verification_record in verification_records:
                    close_in_time = (
                        topic_record[0] is None
                        or verification_record[0] is None
                        or abs(topic_record[0] - verification_record[0]) <= 120
                    )
                    if close_in_time:
                        matched_contexts.extend((topic_record, verification_record))
        else:
            for index, record in enumerate(unique_records):
                context_records = unique_records[
                    max(0, index - 1):min(len(unique_records), index + 2)
                ]
                context_text = " ".join(context_record[2] for context_record in context_records)
                if topic_pattern.search(context_text) and (
                    evidence_pattern is None or evidence_pattern.search(context_text)
                ):
                    matched_contexts.extend(context_records)
        # Preserve order and remove repeated context records from overlapping windows.
        matched_records = []
        matched_seen = set()
        for record in matched_contexts:
            if record[3] not in matched_seen:
                matched_seen.add(record[3])
                matched_records.append(record)
        if not agent_records and matched_records:
            attribution_note = "Speaker not separated; matching transcript evidence:\n"
        else:
            attribution_note = "Matching transcript evidence:\n"
        compliance_evidence[label] = (
            attribution_note + excerpts(matched_records, limit=3)
            if matched_records else None
        )

    return case_template, email_draft, compliance_evidence

class LoginDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Lenovo CEC — Secure Agent Authentication")
        self.setFixedSize(900, 570)
        self.setStyleSheet("""
            QDialog { background: #07101F; color: #EAF1FF; font-family: 'Segoe UI'; }
            QFrame#loginHero {
                background: qlineargradient(x1:0,y1:0,x2:1,y2:1,
                    stop:0 #10264A, stop:0.55 #112C55, stop:1 #2B1652);
                border: 1px solid #2D4C7C; border-radius: 20px;
            }
            QLabel#heroBrand {
                background: #E2231A; color: white; font-size: 22pt; font-weight: 900;
                border-radius: 7px; padding: 7px 12px;
            }
            QLabel#heroTitle { color: white; font-size: 23pt; font-weight: 800; }
            QLabel#heroSub { color: #B5C8E8; font-size: 10pt; }
            QLabel#featureLine { color: #D4E3FC; font-size: 10pt; padding: 4px 0; }
            QFrame#loginCard {
                background: #0F1A2C; border: 1px solid #2B3F5D; border-radius: 18px;
            }
            QLabel#formTitle { color: white; font-size: 18pt; font-weight: 800; }
            QLabel#formSub { color: #8FA8CB; font-size: 9pt; }
            QLabel#fieldLabel { color: #C4D3E9; font-weight: 700; }
            QLineEdit {
                background: #08111F; color: white; border: 1px solid #344B6B;
                border-radius: 9px; padding: 12px; min-height: 24px; font-size: 10pt;
            }
            QLineEdit:focus { border: 1px solid #5598F2; }
            QPushButton#loginButton {
                background: #E2231A; color: white; border: 1px solid #FF5148;
                border-radius: 9px; padding: 13px; font-size: 11pt; font-weight: 800;
            }
            QPushButton#loginButton:hover { background: #FF342B; }
            QLabel#secureNote { color: #7F98BA; font-size: 8pt; }
        """)

        root = QHBoxLayout(self)
        root.setContentsMargins(18, 18, 18, 18)
        root.setSpacing(16)

        hero = QFrame()
        hero.setObjectName("loginHero")
        hero_layout = QVBoxLayout(hero)
        hero_layout.setContentsMargins(26, 28, 26, 24)
        hero_layout.setSpacing(14)

        brand = QLabel("Lenovo")
        brand.setObjectName("heroBrand")
        brand.setAlignment(Qt.AlignCenter)
        brand.setMaximumWidth(150)
        hero_layout.addWidget(brand, 0, Qt.AlignLeft)

        hero_title = QLabel("CEC SmartAssist AI")
        hero_title.setObjectName("heroTitle")
        hero_title.setWordWrap(True)
        hero_layout.addWidget(hero_title)

        hero_sub = QLabel("ENTERPRISE OPERATIONS SUITE")
        hero_sub.setObjectName("heroSub")
        hero_layout.addWidget(hero_sub)

        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setStyleSheet("color: #34547F; background: #34547F; max-height: 1px;")
        hero_layout.addWidget(separator)

        for line in (
            "◉  Live call quality monitoring",
            "▣  AI-assisted offline call audits",
            "▤  Case templates and customer emails",
            "♛  Secure recording vault for authorised users",
        ):
            item = QLabel(line)
            item.setObjectName("featureLine")
            item.setWordWrap(True)
            hero_layout.addWidget(item)

        hero_layout.addStretch(1)
        hero_footer = QLabel("SMARTER SUPPORT. BETTER EXPERIENCES.")
        hero_footer.setObjectName("heroSub")
        hero_layout.addWidget(hero_footer)
        root.addWidget(hero, 5)

        card = QFrame()
        card.setObjectName("loginCard")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(26, 30, 26, 24)
        card_layout.setSpacing(13)

        form_title = QLabel("Welcome back")
        form_title.setObjectName("formTitle")
        card_layout.addWidget(form_title)

        form_sub = QLabel("Sign in with your Lenovo work account to continue.")
        form_sub.setObjectName("formSub")
        form_sub.setWordWrap(True)
        card_layout.addWidget(form_sub)

        form_layout = QFormLayout()
        form_layout.setVerticalSpacing(8)
        form_layout.setHorizontalSpacing(0)
        self.txt_email = QLineEdit()
        self.txt_email.setPlaceholderText("agent.name@lenovo.com")
        self.txt_email.setMinimumHeight(46)
        self.txt_pass = QLineEdit()
        self.txt_pass.setEchoMode(QLineEdit.Password)
        self.txt_pass.setPlaceholderText("Enter your password")
        self.txt_pass.setMinimumHeight(46)

        email_label = QLabel("LENOVO EMAIL ID")
        email_label.setObjectName("fieldLabel")
        password_label = QLabel("PASSWORD")
        password_label.setObjectName("fieldLabel")
        form_layout.addRow(email_label)
        form_layout.addRow(self.txt_email)
        form_layout.addRow(password_label)
        form_layout.addRow(self.txt_pass)
        card_layout.addLayout(form_layout)

        btn_login = QPushButton("Authenticate  →  Launch Workspace")
        btn_login.setObjectName("loginButton")
        btn_login.setMinimumHeight(48)
        btn_login.clicked.connect(self.validate_login)
        self.txt_pass.returnPressed.connect(self.validate_login)
        card_layout.addSpacing(6)
        card_layout.addWidget(btn_login)

        secure_note = QLabel("🔒  Secure workspace  •  Access is subject to your assigned role")
        secure_note.setObjectName("secureNote")
        secure_note.setWordWrap(True)
        card_layout.addWidget(secure_note)
        card_layout.addStretch(1)

        root.addWidget(card, 5)

        self.user_email = ""
        self.is_master = False

    def validate_login(self):
        email = self.txt_email.text().strip().lower()
        password = self.txt_pass.text().strip()
 
        if not email.endswith("@lenovo.com"):
            QMessageBox.warning(self, "Access Denied", "Invalid domain. You must use an official '@lenovo.com' email address.")
            return
       
        if not password:
            QMessageBox.warning(self, "Password Required", "Please enter your password.")
            return
 
        MASTER_EMAIL = "sarbaz@lenovo.com"
        MASTER_SECRET_PIN = "554658"
 
        if email == MASTER_EMAIL:
            pin_dialog, ok = QInputDialog.getText(self, "Master Verification", "Enter Master Admin Security PIN:", QLineEdit.Password)
            if not ok or pin_dialog != MASTER_SECRET_PIN:
                QMessageBox.critical(self, "Access Blocked", "Incorrect Master PIN! Unauthorized attempt to access Master Admin account.")
                return
            self.is_master = True
        else:
            self.is_master = False
 
        creds = {}
        if os.path.exists(CRED_FILE):
            with open(CRED_FILE, "r") as f:
                for line in f:
                    if ":" in line:
                        stored_email, stored_pass = line.strip().split(":", 1)
                        creds[stored_email.lower()] = stored_pass
 
        if email in creds:
            if creds[email] != password and email != MASTER_EMAIL:
                QMessageBox.warning(self, "Authentication Failed", "Incorrect password for this email ID.")
                return
        elif email != MASTER_EMAIL:
            with open(CRED_FILE, "a") as f:
                f.write(f"{email}:{password}\n")
 
        self.user_email = email
        self.accept()
 
class WorkerSignals(QObject):
    finished = Signal(str, list)
    failed = Signal(str)
    live_update = Signal(str)
    status = Signal(str)
 
class CECSmartAssistApp(QMainWindow):
    def __init__(self, user_email, is_master):
        super().__init__()
        self.user_email = user_email
        self.is_master = is_master
        role_title = "Master Admin Dashboard (All Calls)" if self.is_master else f"Agent Workspace ({user_email})"
        self.setWindowTitle(f"Lenovo CEC SmartAssist AI — {role_title}")
        self.resize(1600, 1000)
        self.setMinimumSize(1100, 700)
       
        self.is_recording = False
        self.has_transcript = False
        self.recorder = None
        self.live_transcription_thread = None
        self.transcription_lock = threading.Lock()
        self.playback_dialog = None
        self.playback_slider = None
        self.playback_time_label = None
        self.playback_button = None
        self.playback_duration_ms = 0
        self.audio_file_path = ""
        self.agent_call_folder = ""
        self.sidebar_collapsed = False
        self.call_source_type = "Incoming"
        self.current_customer_number = "Unknown_Customer"
        self.signals = WorkerSignals()
        self.signals.finished.connect(self.handle_transcription_results)
        self.signals.failed.connect(self.handle_processing_error)
        self.signals.live_update.connect(self.update_live_transcript)
        self.signals.status.connect(self.update_status_bar)
        self.model_ready_event = threading.Event()
        self.model_error = None
       
        self.media_player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.audio_output.setVolume(1.0)
        self.media_player.setAudioOutput(self.audio_output)
        self.audio_output.setVolume(1.0)
        self.media_player.errorOccurred.connect(self.handle_playback_error)
        self.media_player.positionChanged.connect(self.update_playback_position)
        self.media_player.durationChanged.connect(self.update_playback_duration)
        self.media_player.playbackStateChanged.connect(self.update_playback_button)
 
        self.model = None
        if WHISPER_AVAILABLE:
            threading.Thread(target=self.init_whisper_model, daemon=True).start()
        else:
            self.model_error = "faster-whisper is not installed."
            self.model_ready_event.set()
 
        self.apply_stylesheet()
 
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.layout = QVBoxLayout(self.main_widget)
        self.layout.setContentsMargins(14, 12, 14, 10)
        self.layout.setSpacing(12)

        # Branded header: presentation-only; existing application workflows remain unchanged.
        self.header = QFrame()
        self.header.setObjectName("appHeader")
        header_layout = QHBoxLayout(self.header)
        header_layout.setContentsMargins(18, 12, 18, 12)
        header_layout.setSpacing(14)

        self.lbl_brand = QLabel("Lenovo")
        self.lbl_brand.setObjectName("brandLogo")
        self.lbl_brand.setAlignment(Qt.AlignCenter)
        self.lbl_brand.setMinimumWidth(112)

        brand_text_layout = QVBoxLayout()
        brand_title = QLabel("CEC SmartAssist AI")
        brand_title.setObjectName("brandTitle")
        brand_subtitle = QLabel("ENTERPRISE OPERATIONS SUITE  •  CALL QUALITY  •  CASE DOCUMENTATION")
        brand_subtitle.setObjectName("brandSubtitle")
        brand_text_layout.addWidget(brand_title)
        brand_text_layout.addWidget(brand_subtitle)

        header_layout.addWidget(self.lbl_brand)
        header_layout.addLayout(brand_text_layout)
        header_layout.addStretch(1)

        self.lbl_header_role = QLabel(
            f"MASTER ADMIN  •  {self.user_email}" if self.is_master
            else f"AGENT WORKSPACE  •  {self.user_email}"
        )
        self.lbl_header_role.setObjectName("userBadge")
        self.lbl_header_role.setToolTip("Current authenticated application session")
        self.btn_toggle_sidebar = QPushButton("☰ Close Workspace")
        self.btn_toggle_sidebar.setObjectName("sidebarToggleButton")
        self.btn_toggle_sidebar.setToolTip("Close or reopen the workspace navigation panel")
        self.btn_toggle_sidebar.clicked.connect(self.toggle_sidebar)
        header_layout.addWidget(self.btn_toggle_sidebar)
        header_layout.addWidget(self.lbl_header_role)
        self.layout.addWidget(self.header)

        self.body = QFrame()
        self.body.setObjectName("appBody")
        body_layout = QHBoxLayout(self.body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(12)

        self.sidebar = QFrame()
        self.sidebar.setObjectName("sideRail")
        self.sidebar.setMinimumWidth(205)
        self.sidebar.setMaximumWidth(225)
        side_layout = QVBoxLayout(self.sidebar)
        side_layout.setContentsMargins(12, 14, 12, 12)
        side_layout.setSpacing(8)

        side_heading_layout = QHBoxLayout()
        side_heading = QLabel("WORKSPACE")
        side_heading.setObjectName("sideHeading")
        side_heading_layout.addWidget(side_heading)
        side_heading_layout.addStretch(1)
        self.btn_close_sidebar = QPushButton("×")
        self.btn_close_sidebar.setObjectName("sidebarToggleButton")
        self.btn_close_sidebar.setFixedSize(30, 28)
        self.btn_close_sidebar.setToolTip("Close the workspace navigation panel")
        self.btn_close_sidebar.clicked.connect(self.toggle_sidebar)
        side_heading_layout.addWidget(self.btn_close_sidebar)
        side_layout.addLayout(side_heading_layout)

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.setMovable(False)

        self.side_nav_buttons = []
        side_nav_items = [
            ("◉   Live Call Quality", 0),
            ("▣   Offline Audit", 1),
            ("▤   Case Documentation", 2),
            ("✦   Agent Assist", 3),
        ]
        if self.is_master:
            side_nav_items.append(("♛   Master Vault", 4))

        for label, index in side_nav_items:
            nav_btn = QPushButton(label)
            nav_btn.setObjectName("sideNavButton")
            nav_btn.setCheckable(True)
            nav_btn.clicked.connect(lambda checked=False, i=index: self.tabs.setCurrentIndex(i))
            side_layout.addWidget(nav_btn)
            self.side_nav_buttons.append((nav_btn, index))

        side_layout.addSpacing(12)
        divider = QFrame()
        divider.setFrameShape(QFrame.HLine)
        divider.setStyleSheet("color: #263B59; background: #263B59; max-height: 1px;")
        side_layout.addWidget(divider)

        side_note_title = QLabel("SMARTASSIST AI")
        side_note_title.setObjectName("sideHeading")
        side_layout.addWidget(side_note_title)
        side_note = QLabel("Monitor calls, review audits and prepare consistent case documentation.")
        side_note.setObjectName("sideNote")
        side_note.setWordWrap(True)
        side_layout.addWidget(side_note)
        side_layout.addStretch(1)

        self.lbl_connection = QLabel("●  APPLICATION READY\\nLocal workspace active")
        self.lbl_connection.setObjectName("connectionCard")
        self.lbl_connection.setWordWrap(True)
        side_layout.addWidget(self.lbl_connection)

        body_layout.addWidget(self.sidebar)
        body_layout.addWidget(self.tabs, 1)
        self.layout.addWidget(self.body, 1)
        self.tabs.currentChanged.connect(self.sync_sidebar_selection)
       
        self.init_live_tab()
        self.init_audit_tab()
        self.init_output_tab()
        self.init_agent_assist_tab()
       
        if self.is_master:
            self.init_master_vault_tab()
 
        self.statusBar().showMessage(f"Logged in: {self.user_email} | Role: {'Master Admin' if self.is_master else 'Agent'}")
 
    def sync_sidebar_selection(self, current_index):
        for button, index in getattr(self, "side_nav_buttons", []):
            button.setChecked(index == current_index)

    def toggle_sidebar(self):
        """Collapse or restore the left navigation rail."""
        self.sidebar_collapsed = not self.sidebar_collapsed
        self.sidebar.setVisible(not self.sidebar_collapsed)
        self.btn_toggle_sidebar.setText(
            "☰ Open Workspace" if self.sidebar_collapsed else "☰ Close Workspace"
        )
        self.btn_toggle_sidebar.setToolTip(
            "Open the workspace navigation panel"
            if self.sidebar_collapsed
            else "Close the workspace navigation panel"
        )

    def apply_stylesheet(self):
        self.setStyleSheet("""
            * { font-family: "Segoe UI"; font-size: 9pt; }
            QMainWindow { background: #070D18; color: #E7EEF9; }
            QWidget { color: #E7EEF9; }
            QFrame#appHeader {
                background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                    stop:0 #101D35, stop:0.55 #13294A, stop:1 #251B48);
                border: 1px solid #293F64; border-radius: 13px;
            }
            QFrame#appBody { background: transparent; }
            QFrame#agentAssistHero {
                background: qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #142B4A,stop:1 #291B4A);
                border: 1px solid #36557D; border-radius: 12px;
            }
            QFrame#sideRail {
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                    stop:0 #0D1A30, stop:1 #0A1424);
                border: 1px solid #243956; border-radius: 12px;
            }
            QLabel#sideHeading { color: #6F8EB9; font-size: 8pt; font-weight: 900; letter-spacing: 1px; padding: 5px 6px; }
            QLabel#sideNote { color: #9CB0CE; font-size: 8pt; padding: 4px 6px; }
            QLabel#connectionCard {
                color: #55E2B2; background: #0D2929; border: 1px solid #1E6559;
                border-radius: 9px; padding: 10px; font-size: 8pt; font-weight: 700;
            }
            QPushButton#sideNavButton {
                text-align: left; background: transparent; color: #B5C5DD;
                border: 1px solid transparent; border-radius: 8px; padding: 12px 9px; font-weight: 700;
            }
            QPushButton#sideNavButton:hover { background: #182A43; border-color: #2D4769; color: white; }
            QPushButton#sideNavButton:checked {
                background: qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #E2231A,stop:1 #9C2147);
                color: white; border: 1px solid #FF5A52;
            }
            QLabel#brandLogo {
                background: #E2231A; color: white; font-size: 21pt; font-weight: 900;
                border-radius: 7px; padding: 5px 12px;
            }
            QLabel#brandTitle { color: white; font-size: 19pt; font-weight: 800; }
            QLabel#brandSubtitle { color: #9DB4D8; font-size: 8pt; font-weight: 700; letter-spacing: 1px; }
            QLabel#userBadge {
                background: #182B49; color: #C7D9F7; border: 1px solid #35527D;
                border-radius: 10px; padding: 10px 14px; font-size: 8pt; font-weight: 700;
            }
            QTabWidget::pane { border: 1px solid #26364F; background: #0A1424; border-radius: 11px; top: -1px; }
            QTabBar::tab {
                background: #172236; color: #A9B9D2; padding: 11px 13px; margin-right: 4px;
                border: 1px solid #26364F; border-bottom: 3px solid transparent;
                border-top-left-radius: 8px; border-top-right-radius: 8px; font-weight: 700; min-width: 105px;
            }
            QTabBar::tab:hover { background: #20314C; color: white; }
            QTabBar::tab:selected {
                background: #182944; color: white; border: 1px solid #3C557A; border-bottom: 3px solid #E2231A;
            }
            QGroupBox {
                border: 1px solid #263951; border-radius: 11px; margin-top: 15px;
                padding: 16px 9px 9px 9px; background: #101B2C; color: #F2F6FC; font-weight: 700;
            }
            QGroupBox::title {
                subcontrol-origin: margin; left: 12px; padding: 3px 7px; color: #FF625B;
                background: #101B2C; border-radius: 5px; font-weight: 800;
            }
            QFrame#metricCard {
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 #14243B,stop:1 #0D1829);
                border: 1px solid #2C4261; border-top: 3px solid #3997FF; border-radius: 10px;
            }
            QFrame#metricCard[accent="#12C997"] { border-top: 3px solid #12C997; }
            QFrame#metricCard[accent="#A78BFA"] { border-top: 3px solid #A78BFA; }
            QFrame#metricCard[accent="#F5A524"] { border-top: 3px solid #F5A524; }
            QLabel#metricTitle { color: #93A9C8; font-size: 7pt; font-weight: 800; letter-spacing: 1px; }
            QLabel#metricValue { color: white; font-size: 18pt; font-weight: 900; }
            QLabel#metricDetail { color: #8EA5C5; font-size: 7pt; }
            QLabel#statusPill {
                background: #102C2A; color: #6FE3C0; border: 1px solid #1D6658;
                border-radius: 8px; padding: 8px 10px; font-weight: 700;
            }
            QLabel#subtleBadge {
                color: #8FA9CE; background: #14243A; border: 1px solid #2A4262;
                border-radius: 7px; padding: 4px 7px; font-size: 7pt; font-weight: 700;
            }
            QLabel#mutedText { color: #8298B8; font-size: 8pt; }
            QPushButton {
                background: #22334B; color: #F4F7FC; border: 1px solid #344B6A;
                border-radius: 7px; padding: 8px 10px; font-weight: 700;
            }
            QPushButton:hover { background: #304967; border-color: #5A7CA8; }
            QPushButton:pressed { background: #17263C; }
            QPushButton#primaryButton { background: #E2231A; color: white; border: 1px solid #FF5148; }
            QPushButton#primaryButton:hover { background: #FF342B; }
            QPushButton#sidebarToggleButton {
                background: #1A2B44; color: #EAF1FF; border: 1px solid #42618A;
                border-radius: 7px; padding: 8px 11px; font-weight: 800;
            }
            QPushButton#sidebarToggleButton:hover { background: #2B4568; }
            QTextEdit, QTableWidget, QListWidget, QDateEdit, QLineEdit {
                background: #080F1B; alternate-background-color: #101B2C; color: #E7EEF9;
                border: 1px solid #2A3B55; border-radius: 7px; padding: 6px;
                selection-background-color: #264D82; selection-color: white;
            }
            QTextEdit:focus, QLineEdit:focus, QDateEdit:focus { border: 1px solid #4D8DDF; }
            QTableWidget { gridline-color: #202F45; background: #0A1220; alternate-background-color: #101B2C; }
            QTableWidget::item { padding: 7px 6px; border-bottom: 1px solid #1D2A3D; }\n            QTableWidget QTableCornerButton::section { background: #1A2940; border: none; }
            QTableWidget::item:selected { background: #203E65; color: white; }
            QHeaderView::section {
                background: #1A2940; color: #F4F7FC; padding: 7px 6px; border: none;
                border-right: 1px solid #2B3D56; border-bottom: 2px solid #E2231A; font-weight: 800;
            }
            QScrollBar:vertical { background: #0B1422; width: 11px; margin: 2px; border-radius: 5px; }
            QScrollBar::handle:vertical { background: #354B69; min-height: 26px; border-radius: 5px; }
            QScrollBar::handle:vertical:hover { background: #4D6990; }
            QScrollBar:horizontal { background: #0B1422; height: 11px; margin: 2px; border-radius: 5px; }
            QScrollBar::handle:horizontal { background: #354B69; min-width: 26px; border-radius: 5px; }
            QStatusBar { background: #0B1422; color: #8FA9CE; font-weight: 600; border-top: 1px solid #26364F; padding: 4px; }
            QToolTip { background: #17263C; color: white; border: 1px solid #4D6688; padding: 5px; }
        """)

    def init_whisper_model(self):
        try:
            # small is noticeably more accurate than base for noisy phone audio;
            # int8 keeps CPU memory use practical on standard workstations.
            self.model = WhisperModel("small", device="cpu", compute_type="int8")
            self.signals.status.emit("AI Engine Online | Ready")
        except Exception as e:
            self.model_error = str(e)
            self.signals.status.emit(f"AI Engine Notice: {str(e)}")
        finally:
            self.model_ready_event.set()
 
    def update_status_bar(self, message):
        self.statusBar().showMessage(message)
 
    def init_live_tab(self):
        live_widget = QWidget()
        layout = QVBoxLayout(live_widget)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        controls_group = QGroupBox("GENESYS WORKSPACE  /  LIVE CALL OPERATIONS")
        controls_layout = QHBoxLayout(controls_group)
        controls_layout.setContentsMargins(12, 18, 12, 12)
        controls_layout.setSpacing(8)

        self.btn_record = QPushButton("🎙  Start Recording Call")
        self.btn_record.setObjectName("primaryButton")
        self.btn_record.clicked.connect(self.toggle_recording)
        self.btn_download_audio = QPushButton("⬇  Download Recording")
        self.btn_download_audio.clicked.connect(self.download_current_recording)
        self.btn_reset = QPushButton("↻  Reset Workspace")
        self.btn_reset.clicked.connect(self.reset_application)
        self.lbl_timer = QLabel(f"●  {self.user_email}  |  IDLE")
        self.lbl_timer.setObjectName("statusPill")
        self.lbl_timer.setMinimumWidth(210)

        controls_layout.addWidget(self.btn_record)
        controls_layout.addWidget(self.btn_download_audio)
        controls_layout.addWidget(self.btn_reset)
        controls_layout.addStretch(1)
        controls_layout.addWidget(self.lbl_timer)
        layout.addWidget(controls_group)

        metrics_row = QHBoxLayout()
        metrics_row.setSpacing(8)

        def make_metric(title, value, detail, accent):
            card = QFrame()
            card.setObjectName("metricCard")
            card.setProperty("accent", accent)
            box = QVBoxLayout(card)
            box.setContentsMargins(12, 10, 12, 10)
            box.setSpacing(3)
            title_lbl = QLabel(title.upper())
            title_lbl.setObjectName("metricTitle")
            value_lbl = QLabel(value)
            value_lbl.setObjectName("metricValue")
            detail_lbl = QLabel(detail)
            detail_lbl.setObjectName("metricDetail")
            detail_lbl.setWordWrap(True)
            box.addWidget(title_lbl)
            box.addWidget(value_lbl)
            box.addWidget(detail_lbl)
            return card, value_lbl, detail_lbl

        self.metric_calls_card, self.metric_calls_value, self.metric_calls_detail = make_metric(
            "Saved Recordings", "0", "Recordings in vault", "#3997FF")
        self.metric_quality_card, self.metric_quality_value, self.metric_quality_detail = make_metric(
            "Compliance", "0 / 6", "Checklist items passed", "#12C997")
        self.metric_ai_card, self.metric_ai_value, self.metric_ai_detail = make_metric(
            "AI Engine", "Loading", "Local transcription engine", "#A78BFA")
        self.metric_status_card, self.metric_status_value, self.metric_status_detail = make_metric(
            "Workspace", "IDLE", "Ready for a call", "#F5A524")
        for card in (self.metric_calls_card, self.metric_quality_card,
                     self.metric_ai_card, self.metric_status_card):
            metrics_row.addWidget(card, 1)
        layout.addLayout(metrics_row)

        live_trans_group = QGroupBox("🎙  LIVE TRANSCRIPTION  /  MULTILINGUAL → ENGLISH")
        live_trans_layout = QVBoxLayout(live_trans_group)
        transcript_header = QHBoxLayout()
        self.lbl_transcript_state = QLabel("●  WAITING FOR AUDIO")
        self.lbl_transcript_state.setObjectName("subtleBadge")
        transcript_header.addWidget(self.lbl_transcript_state)
        transcript_header.addStretch(1)
        live_trans_layout.addLayout(transcript_header)

        self.txt_live_transcript = QTextEdit()
        self.txt_live_transcript.setObjectName("transcriptPanel")
        self.txt_live_transcript.setPlaceholderText(
            "Live transcript will appear here after you start recording and the AI engine begins processing audio."
        )
        live_trans_layout.addWidget(self.txt_live_transcript, 1)
        transcript_actions = QHBoxLayout()
        btn_copy_live = QPushButton("▣  Copy Transcript")
        btn_copy_live.clicked.connect(lambda: self.copy_to_clipboard(self.txt_live_transcript))
        btn_clear_live = QPushButton("Clear View")
        btn_clear_live.clicked.connect(self.txt_live_transcript.clear)
        transcript_actions.addWidget(btn_copy_live)
        transcript_actions.addWidget(btn_clear_live)
        transcript_actions.addStretch(1)
        live_trans_layout.addLayout(transcript_actions)

        checklist_group = QGroupBox("✓  REAL-TIME QUALITY COMPLIANCE")
        checklist_layout = QVBoxLayout(checklist_group)
        checklist_hint = QLabel(
            "Automatic transcript check: green means matching evidence was found; red means it was not found. Verify evidence before case submission."
        )
        checklist_hint.setObjectName("mutedText")
        checklist_layout.addWidget(checklist_hint)

        self.checklist_table = QTableWidget(6, 3)
        self.checklist_table.setHorizontalHeaderLabels(
            ["Compliance Check", "Status", "Transcript Evidence"])
        # Keep the checks compact; full transcript evidence remains available on hover.
        self.checklist_table.setColumnCount(3)
        self.checklist_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Interactive)
        self.checklist_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Fixed)
        self.checklist_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.checklist_table.setColumnWidth(0, 230)
        self.checklist_table.setColumnWidth(1, 115)
        self.checklist_table.horizontalHeader().setMinimumSectionSize(90)
        self.checklist_table.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.checklist_table.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.checklist_table.setAlternatingRowColors(True)
        self.checklist_table.verticalHeader().setDefaultSectionSize(32)
        self.checklist_table.verticalHeader().setFixedWidth(28)
        self.checklist_table.verticalHeader().setSectionResizeMode(QHeaderView.Fixed)
        self.checklist_table.setWordWrap(False)
        self.checklist_table.setTextElideMode(Qt.ElideRight)
        parameters = [
            "Customer details verified",
            "Email ID phonetically verified",
            "Customer availability checked",
            "Rescue Lens disclaimer given",
            "Faulty-part collection disclaimer given",
            "APOS warranty upselling checked"
        ]
        for i, param in enumerate(parameters):
            self.checklist_table.setItem(i, 0, QTableWidgetItem(param))
            item = QTableWidgetItem("🟡 PENDING")
            item.setForeground(QColor("#F5A524"))
            self.checklist_table.setItem(i, 1, item)
            self.checklist_table.setItem(i, 2, QTableWidgetItem("Awaiting call conversation..."))
        checklist_layout.addWidget(self.checklist_table, 1)

        # Give each primary workspace its own full-size tab. Side-by-side panels
        # were too short on typical laptop displays, hiding the lower compliance rows.
        self.txt_live_transcript.setMinimumHeight(260)
        self.txt_live_transcript.setSizePolicy(
            self.txt_live_transcript.sizePolicy().horizontalPolicy(),
            self.txt_live_transcript.sizePolicy().verticalPolicy()
        )
        live_trans_group.setMinimumHeight(360)
        checklist_group.setMinimumHeight(295)
        self.checklist_table.setMinimumHeight(230)
        self.checklist_table.setColumnWidth(0, 300)
        self.checklist_table.setColumnWidth(1, 150)
        self.checklist_table.verticalHeader().setDefaultSectionSize(32)
        self.checklist_table.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        work_tabs = QTabWidget()
        work_tabs.setDocumentMode(True)
        work_tabs.addTab(live_trans_group, "🎙  Live Transcription")
        work_tabs.addTab(checklist_group, "✓  Quality Compliance")
        work_tabs.setMinimumHeight(330)
        self.live_work_tabs = work_tabs

        recent_group = QGroupBox("◷  RECENT CALLS  /  LOCAL VAULT ACTIVITY")
        recent_layout = QVBoxLayout(recent_group)
        recent_top = QHBoxLayout()
        recent_caption = QLabel("Latest recordings available in the configured vault folder")
        recent_caption.setObjectName("mutedText")
        recent_top.addWidget(recent_caption)
        recent_top.addStretch(1)
        self.btn_recent_refresh = QPushButton("↻  Refresh")
        self.btn_recent_refresh.clicked.connect(self.refresh_recent_calls)
        recent_top.addWidget(self.btn_recent_refresh)
        recent_layout.addLayout(recent_top)

        self.recent_calls_table = QTableWidget(0, 6)
        self.recent_calls_table.setHorizontalHeaderLabels(
            ["Agent", "Contact Number", "Date & Time", "Duration", "Call Type", "Playback"])
        self.recent_calls_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.recent_calls_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.recent_calls_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.recent_calls_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.recent_calls_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self.recent_calls_table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeToContents)
        self.recent_calls_table.setAlternatingRowColors(True)
        self.recent_calls_table.verticalHeader().setDefaultSectionSize(27)
        self.recent_calls_table.setMaximumHeight(125)
        recent_layout.addWidget(self.recent_calls_table)
        # Recent calls can be reduced to give the transcript and compliance panels
        # more room, without removing the recent-call functionality.
        vertical_splitter = QSplitter(Qt.Vertical)
        vertical_splitter.setChildrenCollapsible(False)
        vertical_splitter.addWidget(work_tabs)
        vertical_splitter.addWidget(recent_group)
        vertical_splitter.setStretchFactor(0, 10)
        vertical_splitter.setStretchFactor(1, 1)
        vertical_splitter.setSizes([780, 90])
        layout.addWidget(vertical_splitter, 1)
        self.live_vertical_splitter = vertical_splitter

        self.tabs.addTab(live_widget, "🎙  Live Call Quality")
        self.refresh_dashboard_metrics()
        self.refresh_recent_calls()
        self.metrics_timer = QTimer(self)
        self.metrics_timer.timeout.connect(self.refresh_dashboard_metrics)
        self.metrics_timer.start(2500)

    def refresh_recent_calls(self):
        if not hasattr(self, "recent_calls_table"):
            return
        self.recent_calls_table.setRowCount(0)
        records = []
        try:
            if os.path.isdir(VAULT_DIR):
                current_agent_folder = self.user_email.replace("@", "_at_").replace(".", "_")
                agent_folders = (
                    [current_agent_folder] if not self.is_master
                    else os.listdir(VAULT_DIR)
                )
                for agent_folder in agent_folders:
                    agent_path = os.path.join(VAULT_DIR, agent_folder)
                    if not os.path.isdir(agent_path):
                        continue
                    agent_email = agent_folder.replace("_at_", "@").replace("_", ".")
                    for call_folder in os.listdir(agent_path):
                        call_path = os.path.join(agent_path, call_folder)
                        audio_path = os.path.join(call_path, "call_recording.wav")
                        if not os.path.isdir(call_path) or not os.path.exists(audio_path):
                            continue
                        try:
                            with wave.open(audio_path, "rb") as recorded_audio:
                                if recorded_audio.getnframes() == 0:
                                    continue
                        except (wave.Error, OSError):
                            continue
                        parts = call_folder.split("_")
                        contact = parts[1] if len(parts) > 1 else "Unknown"
                        date_text = "N/A"
                        if len(parts) >= 4:
                            try:
                                date_text = datetime.strptime(
                                    f"{parts[2]} {parts[3]}", "%Y%m%d %H%M%S"
                                ).strftime("%d-%b-%Y %H:%M")
                            except ValueError:
                                pass
                        duration = "00:00"
                        try:
                            with wave.open(audio_path, "rb") as wf:
                                seconds = int(wf.getnframes() / max(1, wf.getframerate()))
                            duration = f"{seconds // 60:02d}:{seconds % 60:02d}"
                        except Exception:
                            pass
                        call_type = "Incoming"
                        report = os.path.join(call_path, "case_report.txt")
                        if os.path.exists(report):
                            try:
                                if "Offline" in Path(report).read_text(encoding="utf-8", errors="ignore"):
                                    call_type = "Offline"
                            except OSError:
                                pass
                        records.append((os.path.getmtime(audio_path), agent_email, contact,
                                        date_text, duration, call_type, audio_path))
            records.sort(key=lambda x: x[0], reverse=True)
            records = records[:8]
            self.recent_calls_table.setRowCount(len(records))
            for row, record in enumerate(records):
                _, agent, contact, date_text, duration, call_type, audio_path = record
                for col, value in enumerate((agent, contact, date_text, duration, call_type)):
                    item = QTableWidgetItem(str(value))
                    if col == 4:
                        item.setForeground(QColor("#22D3A6" if call_type == "Incoming" else "#F5A524"))
                    self.recent_calls_table.setItem(row, col, item)
                play_button = QPushButton("▶ Play")
                play_button.clicked.connect(
                    lambda checked=False, path=audio_path: self.play_specific_audio(path)
                )
                self.recent_calls_table.setCellWidget(row, 5, play_button)
        except Exception as exc:
            self.statusBar().showMessage(f"Recent calls could not be refreshed: {exc}")

    def refresh_dashboard_metrics(self):
        # Metrics are derived from the current local vault and checklist.
        try:
            recordings = 0
            if os.path.isdir(VAULT_DIR):
                for root, dirs, files in os.walk(VAULT_DIR):
                    if "call_recording.wav" in files:
                        recordings += 1
            self.metric_calls_value.setText(str(recordings))
            self.metric_calls_detail.setText("Recordings currently found in vault")

            passed = 0
            for row in range(self.checklist_table.rowCount()):
                item = self.checklist_table.item(row, 1)
                if item and "PASS" in item.text():
                    passed += 1
            self.metric_quality_value.setText(f"{passed} / {self.checklist_table.rowCount()}")
            self.metric_quality_detail.setText(
                f"{round(passed / max(1, self.checklist_table.rowCount()) * 100)}% checklist passed"
            )

            if self.model is not None:
                self.metric_ai_value.setText("ONLINE")
                self.metric_ai_detail.setText("Whisper model loaded")
            elif WHISPER_AVAILABLE:
                self.metric_ai_value.setText("STARTING")
                self.metric_ai_detail.setText("Loading local Whisper model")
            else:
                self.metric_ai_value.setText("OFFLINE")
                self.metric_ai_detail.setText("Install faster-whisper to enable AI")

            if self.is_recording:
                self.metric_status_value.setText("RECORDING")
                self.metric_status_detail.setText("Live audio capture active")
                self.lbl_transcript_state.setText(
                    "●  LIVE ENGLISH TRANSCRIPTION" if self.model is not None
                    else "●  RECORDING · AI ENGINE LOADING"
                )
                self.lbl_live_footer.setText("●  Recording active  •  Confirm required customer consent and follow company policy")
            else:
                self.metric_status_value.setText("IDLE")
                self.metric_status_detail.setText("Ready for a call")
                self.lbl_transcript_state.setText("●  WAITING FOR AUDIO")
                self.lbl_live_footer.setText("●  Secure workspace ready  •  Local processing when the AI engine is available")
        except (AttributeError, RuntimeError):
            # Widgets may not all exist while the UI is being constructed.
            return

    def init_audit_tab(self):
        audit_widget = QWidget()
        layout = QVBoxLayout(audit_widget)
        upload_group = QGroupBox("📂 Offline Call Audit")
        upload_layout = QVBoxLayout()
        self.btn_upload = QPushButton("Upload Call Recording (.wav / .mp3)")
        self.btn_upload.setStyleSheet("background-color: #27272A; color: white;")
        self.btn_upload.clicked.connect(self.upload_offline_call)
        self.lbl_uploaded_file = QLabel("No file selected.")
        upload_layout.addWidget(self.btn_upload)
        upload_layout.addWidget(self.lbl_uploaded_file)
        upload_group.setLayout(upload_layout)
        layout.addWidget(upload_group)
       
        transcript_group = QGroupBox("📜 Smart Transcript (English Translated)")
        transcript_layout = QVBoxLayout()
        self.txt_transcript = QTextEdit()
        transcript_layout.addWidget(self.txt_transcript)
        transcript_group.setLayout(transcript_layout)
        layout.addWidget(transcript_group)
        self.tabs.addTab(audit_widget, "📂 Offline Audit")
 
    def init_output_tab(self):
        output_widget = QWidget()
        layout = QVBoxLayout(output_widget)
       
        case_group = QGroupBox("📋 Lenovo Case Template & SDI Generator")
        case_vbox = QVBoxLayout()
        self.txt_case_template = QTextEdit()
        self.txt_case_template.setMinimumSize(400, 520)
        btn_copy_case = QPushButton("📋 Copy Case Template & SDI")
        btn_copy_case.setStyleSheet("background-color: #1E1E24; color: #F3F4F6;")
        btn_copy_case.clicked.connect(lambda: self.copy_to_clipboard(self.txt_case_template))
        case_vbox.addWidget(self.txt_case_template)
        case_vbox.addWidget(btn_copy_case)
        case_group.setLayout(case_vbox)
       
        email_group = QGroupBox("✉️ Customer Summary Email Draft")
        email_vbox = QVBoxLayout()
        self.txt_email_draft = QTextEdit()
        self.txt_email_draft.setMinimumSize(400, 520)
        btn_copy_email = QPushButton("📋 Copy Email Draft")
        btn_copy_email.setStyleSheet("background-color: #1E1E24; color: #F3F4F6;")
        btn_copy_email.clicked.connect(lambda: self.copy_to_clipboard(self.txt_email_draft))
        email_vbox.addWidget(self.txt_email_draft)
        email_vbox.addWidget(btn_copy_email)
        email_group.setLayout(email_vbox)

        outputs_splitter = QSplitter(Qt.Horizontal)
        outputs_splitter.setChildrenCollapsible(False)
        outputs_splitter.addWidget(case_group)
        outputs_splitter.addWidget(email_group)
        outputs_splitter.setStretchFactor(0, 1)
        outputs_splitter.setStretchFactor(1, 1)
        outputs_splitter.setSizes([700, 700])
        layout.addWidget(outputs_splitter, 1)
        btn_export = QPushButton("💾 Export Report")
        btn_export.setStyleSheet("background-color: #E2231A; color: white;")
        btn_export.clicked.connect(self.export_data)
        layout.addWidget(btn_export)
        self.tabs.addTab(output_widget, "📋 Case Documentation")
 
    def init_agent_assist_tab(self):
        """Agent Assist: serial lookup workflow, contact validation, issue evidence,
        guided checks, and a live mirror of the existing call transcript.

        Lenovo product/warranty/repair fields must be confirmed from the official
        support page until the approved support integration is configured.
        """
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(14, 14, 14, 14)
        outer.setSpacing(10)

        intro = QFrame()
        intro.setObjectName("agentAssistHero")
        intro_layout = QVBoxLayout(intro)
        intro_layout.setContentsMargins(18, 14, 18, 14)
        title = QLabel("AGENT ASSIST  |  LIVE IB CALL WORKFLOW")
        title.setStyleSheet("font-size: 16pt; font-weight: 900; color: white;")
        subtitle = QLabel(
            "Verify the machine first, capture customer details, document the issue, "
            "then follow guided checks while the live transcript remains visible."
        )
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet("color: #B5C8E8;")
        intro_layout.addWidget(title)
        intro_layout.addWidget(subtitle)
        outer.addWidget(intro)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(2, 2, 8, 2)
        content_layout.setSpacing(12)
        scroll.setWidget(content)
        outer.addWidget(scroll, 1)

        # Stage 1: Lenovo Support lookup and human confirmation of returned data.
        machine_group = QGroupBox("1  |  SERIAL NUMBER & LENOVO SUPPORT LOOKUP")
        machine_layout = QGridLayout(machine_group)
        machine_layout.setHorizontalSpacing(10)
        machine_layout.setVerticalSpacing(8)
        self.assist_serial = QLineEdit()
        self.assist_serial.setPlaceholderText("Enter laptop serial number")
        self.assist_serial.setMaxLength(40)
        self.assist_serial.textChanged.connect(self.invalidate_assist_machine)
        self.btn_assist_open_support = QPushButton("Search Lenovo Support")
        self.btn_assist_open_support.setObjectName("primaryButton")
        self.btn_assist_open_support.clicked.connect(self.open_assist_support)
        self.btn_assist_open_support.setToolTip(
            "Search Lenovo Support with this serial number and retrieve Product Home, warranty and repair-status details."
        )
        machine_layout.addWidget(QLabel("Serial number"), 0, 0)
        machine_layout.addWidget(self.assist_serial, 0, 1, 1, 2)
        machine_layout.addWidget(self.btn_assist_open_support, 0, 3)
        self.btn_assist_confirm_machine = QPushButton("Confirm serial number and continue")
        self.btn_assist_confirm_machine.setObjectName("primaryButton")
        self.btn_assist_confirm_machine.clicked.connect(self.confirm_assist_machine)
        machine_layout.addWidget(self.btn_assist_confirm_machine, 1, 0, 1, 4)
        self.assist_product = QLineEdit()
        self.assist_product.setPlaceholderText("Product / model")
        self.assist_mtm = QLineEdit()
        self.assist_mtm.setPlaceholderText("MTM / machine type model")
        self.assist_configuration = QTextEdit()
        self.assist_configuration.setPlaceholderText("Configuration / product description")
        self.assist_configuration.setMaximumHeight(86)
        self.assist_warranty_start = QLineEdit()
        self.assist_warranty_start.setPlaceholderText("Warranty start date")
        self.assist_warranty_end = QLineEdit()
        self.assist_warranty_end.setPlaceholderText("Warranty end date")
        self.assist_last_ticket = QTextEdit()
        self.assist_last_ticket.setPlaceholderText(
            "Last repair ticket ID, status, dates and issue summary"
        )
        self.assist_last_ticket.setMaximumHeight(58)
        machine_layout.addWidget(QLabel("Machine description / product"), 2, 0)
        machine_layout.addWidget(self.assist_product, 2, 1)
        machine_layout.addWidget(QLabel("MTM"), 2, 2)
        machine_layout.addWidget(self.assist_mtm, 2, 3)
        machine_layout.addWidget(QLabel("Configuration"), 3, 0)
        machine_layout.addWidget(self.assist_configuration, 3, 1, 1, 3)
        machine_layout.addWidget(QLabel("Warranty start"), 4, 0)
        machine_layout.addWidget(self.assist_warranty_start, 4, 1)
        machine_layout.addWidget(QLabel("Warranty end"), 4, 2)
        machine_layout.addWidget(self.assist_warranty_end, 4, 3)
        machine_layout.addWidget(QLabel("Last repair ticket"), 5, 0)
        machine_layout.addWidget(self.assist_last_ticket, 5, 1, 1, 3)
        self.assist_machine_status = QLabel(
            "Enter the serial number and retrieve these details from Lenovo Support. Approved automatic support-site integration is pending."
        )
        self.assist_machine_status.setWordWrap(True)
        self.assist_machine_status.setObjectName("mutedText")
        machine_layout.addWidget(self.assist_machine_status, 6, 0, 1, 4)
        content_layout.addWidget(machine_group)

        # Stage 2: Contact details.
        contact_group = QGroupBox("2  |  CUSTOMER CONTACT DETAILS")
        contact_layout = QGridLayout(contact_group)
        self.assist_mobile = QLineEdit()
        self.assist_mobile.setPlaceholderText("10-digit Indian mobile number (without +91)")
        self.assist_email = QLineEdit()
        self.assist_email.setPlaceholderText("customer@example.com")
        self.assist_email_status = QLabel("Email domain has not been checked.")
        self.assist_email_status.setWordWrap(True)
        self.assist_mobile.textChanged.connect(self.invalidate_assist_contact)
        self.assist_email.textChanged.connect(self.invalidate_assist_contact)
        self.btn_assist_validate_contact = QPushButton("Validate contact details")
        self.btn_assist_validate_contact.clicked.connect(self.validate_assist_contact)
        contact_layout.addWidget(QLabel("Mobile (+91)"), 0, 0)
        contact_layout.addWidget(self.assist_mobile, 0, 1)
        contact_layout.addWidget(QLabel("Email ID"), 1, 0)
        contact_layout.addWidget(self.assist_email, 1, 1)
        contact_layout.addWidget(self.btn_assist_validate_contact, 2, 0, 1, 2)
        contact_layout.addWidget(self.assist_email_status, 3, 0, 1, 2)
        content_layout.addWidget(contact_group)

        # Stage 3: issue capture and attachments.
        issue_group = QGroupBox("3  |  CUSTOMER ISSUE & EVIDENCE")
        issue_layout = QVBoxLayout(issue_group)
        self.assist_issue = QTextEdit()
        self.assist_issue.setPlaceholderText(
            "Capture the customer's exact issue, symptoms, error codes, when it started, "
            "recent changes and impact. Avoid paraphrasing away important details."
        )
        self.assist_issue.setMinimumHeight(95)
        self.assist_issue.textChanged.connect(self.invalidate_assist_issue)
        issue_layout.addWidget(self.assist_issue)
        issue_buttons = QHBoxLayout()
        self.btn_assist_upload = QPushButton("Attach snapshots / evidence (max 200 MB per file)")
        self.btn_assist_upload.clicked.connect(self.upload_assist_evidence)
        self.btn_assist_prepare_steps = QPushButton("Prepare troubleshooting steps")
        self.btn_assist_prepare_steps.setObjectName("primaryButton")
        self.btn_assist_prepare_steps.clicked.connect(self.prepare_assist_steps)
        issue_buttons.addWidget(self.btn_assist_upload)
        issue_buttons.addWidget(self.btn_assist_prepare_steps)
        issue_layout.addLayout(issue_buttons)
        self.assist_evidence_list = QLabel("No evidence attached.")
        self.assist_evidence_list.setWordWrap(True)
        self.assist_evidence_list.setObjectName("mutedText")
        issue_layout.addWidget(self.assist_evidence_list)
        self.assist_evidence_paths = []
        content_layout.addWidget(issue_group)

        # Stage 4: one-step-at-a-time Yes/No guidance.
        troubleshoot_group = QGroupBox("4  |  GUIDED TROUBLESHOOTING — YES / NO")
        troubleshoot_layout = QVBoxLayout(troubleshoot_group)
        self.assist_step_counter = QLabel("No troubleshooting sequence started.")
        self.assist_step_counter.setObjectName("subtleBadge")
        self.assist_step_text = QLabel(
            "Confirm the machine details and capture the issue before preparing a troubleshooting path."
        )
        self.assist_step_text.setWordWrap(True)
        self.assist_step_text.setMinimumHeight(58)
        self.assist_step_text.setStyleSheet("font-size: 11pt; font-weight: 700; padding: 8px;")
        troubleshoot_layout.addWidget(self.assist_step_counter)
        troubleshoot_layout.addWidget(self.assist_step_text)
        answer_buttons = QHBoxLayout()
        self.btn_assist_yes = QPushButton("YES — Step succeeded / symptom gone")
        self.btn_assist_no = QPushButton("NO — Step failed / symptom remains")
        self.btn_assist_yes.clicked.connect(lambda: self.record_assist_step(True))
        self.btn_assist_no.clicked.connect(lambda: self.record_assist_step(False))
        self.btn_assist_howto = QPushButton("HOW TO — Show this step from start to finish")
        self.btn_assist_howto.clicked.connect(self.show_assist_howto)
        answer_buttons.addWidget(self.btn_assist_yes)
        answer_buttons.addWidget(self.btn_assist_no)
        answer_buttons.addWidget(self.btn_assist_howto)
        troubleshoot_layout.addLayout(answer_buttons)
        self.assist_step_history = QTextEdit()
        self.assist_step_history.setReadOnly(True)
        self.assist_step_history.setPlaceholderText("Completed steps and Yes/No results will appear here.")
        self.assist_step_history.setMaximumHeight(150)
        troubleshoot_layout.addWidget(self.assist_step_history)
        self.assist_step_observation = QLineEdit()
        self.assist_step_observation.setPlaceholderText(
            "Optional observation or exact Lenovo diagnostic error code for this step"
        )
        troubleshoot_layout.addWidget(self.assist_step_observation)
        content_layout.addWidget(troubleshoot_group)

        # Stage 5: evidence-based outcome.
        outcome_group = QGroupBox("5  |  ROOT CAUSE & HARDWARE DISPOSITION")
        outcome_layout = QGridLayout(outcome_group)
        self.assist_root_cause = QTextEdit()
        self.assist_root_cause.setPlaceholderText("Confirmed root cause and evidence. If not confirmed, state that it remains undetermined.")
        self.assist_root_cause.setMaximumHeight(75)
        self.assist_failed_hardware = QLineEdit()
        self.assist_failed_hardware.setPlaceholderText("Suggested or confirmed faulty part")
        self.assist_suggested_resolution = QTextEdit()
        self.assist_suggested_resolution.setPlaceholderText(
            "Suggested resolution based on the recorded troubleshooting and diagnostic evidence"
        )
        self.assist_suggested_resolution.setMaximumHeight(70)
        self.assist_outcome = QComboBox()
        self.assist_outcome.addItems([
            "Select final outcome", "Resolved during call", "Unresolved — further troubleshooting", 
            "Hardware failure confirmed", "Escalation / repair required", "Insufficient evidence to confirm root cause"
        ])
        self.assist_final_notes = QTextEdit()
        self.assist_final_notes.setPlaceholderText("Customer update, next action, escalation details and other relevant findings")
        self.assist_final_notes.setMaximumHeight(70)
        self.btn_assist_generate_summary = QPushButton("Generate call summary")
        self.btn_assist_generate_summary.setObjectName("primaryButton")
        self.btn_assist_generate_summary.clicked.connect(self.generate_assist_summary)
        self.assist_summary = QTextEdit()
        self.assist_summary.setReadOnly(True)
        self.assist_summary.setPlaceholderText("Final Agent Assist summary will appear here.")
        self.assist_summary.setMinimumHeight(150)
        outcome_layout.addWidget(QLabel("Root cause / evidence"), 0, 0)
        outcome_layout.addWidget(self.assist_root_cause, 0, 1)
        outcome_layout.addWidget(QLabel("Suggested / faulty part"), 1, 0)
        outcome_layout.addWidget(self.assist_failed_hardware, 1, 1)
        outcome_layout.addWidget(QLabel("Suggested resolution"), 2, 0)
        outcome_layout.addWidget(self.assist_suggested_resolution, 2, 1)
        outcome_layout.addWidget(QLabel("Outcome"), 3, 0)
        outcome_layout.addWidget(self.assist_outcome, 3, 1)
        outcome_layout.addWidget(QLabel("Final notes"), 4, 0)
        outcome_layout.addWidget(self.assist_final_notes, 4, 1)
        outcome_layout.addWidget(self.btn_assist_generate_summary, 5, 0, 1, 2)
        outcome_layout.addWidget(self.assist_summary, 6, 0, 1, 2)
        content_layout.addWidget(outcome_group)

        # Live transcript view uses the existing audio/transcription pipeline.
        transcript_group = QGroupBox("LIVE CALL TRANSCRIPT — CONNECTED TO EXISTING RECORDER")
        transcript_layout = QVBoxLayout(transcript_group)
        self.assist_live_transcript = QTextEdit()
        self.assist_live_transcript.setReadOnly(True)
        self.assist_live_transcript.setPlaceholderText(
            "Start recording in Live Call Quality. The existing live transcript will be mirrored here."
        )
        self.assist_live_transcript.setMinimumHeight(170)
        transcript_layout.addWidget(self.assist_live_transcript)
        transcript_controls = QHBoxLayout()
        btn_assist_copy_transcript = QPushButton("Copy transcript")
        btn_assist_copy_transcript.clicked.connect(lambda: self.copy_to_clipboard(self.assist_live_transcript))
        btn_assist_refresh_transcript = QPushButton("Refresh from live transcript")
        btn_assist_refresh_transcript.clicked.connect(self.refresh_assist_transcript)
        transcript_controls.addWidget(btn_assist_copy_transcript)
        transcript_controls.addWidget(btn_assist_refresh_transcript)
        transcript_controls.addStretch(1)
        transcript_layout.addLayout(transcript_controls)
        content_layout.addWidget(transcript_group)
        content_layout.addStretch(1)

        self.tabs.addTab(page, "✦ Agent Assist")
        self.assist_steps = []
        self.assist_step_index = 0
        self.assist_step_results = []
        self.assist_machine_verified = False
        self.assist_contact_verified = False
        self.assist_diagnostic_component = "the affected component"
        self.assist_symptom_category = "general hardware or software"
        self.assist_support_product_url = ""
        self.assist_support_urls = {}
        self.assist_support_stage = ""
        self.assist_support_search_started = False
        self.assist_support_results_clicked = False
        self.assist_machine_lookup_complete = False
        self.assist_support_repair_capture_attempts = 0
        self.assist_support_warranty_capture_attempts = 0
        self.assist_support_spec_capture_attempts = 0
        self.btn_assist_yes.setEnabled(False)
        self.btn_assist_no.setEnabled(False)
        self.btn_assist_howto.setEnabled(False)

    def invalidate_assist_machine(self):
        was_verified = self.assist_machine_verified
        self.assist_machine_verified = False
        self.assist_machine_lookup_complete = False
        self.assist_support_product_url = ""
        for field_name in (
            "assist_product", "assist_mtm", "assist_configuration",
            "assist_warranty_start", "assist_warranty_end", "assist_last_ticket",
        ):
            field = getattr(self, field_name, None)
            if field is not None:
                field.clear()
        self.reset_assist_troubleshooting()
        self.assist_evidence_paths.clear()
        self.assist_evidence_list.setText("No evidence attached.")
        if hasattr(self, "assist_machine_status"):
            if was_verified:
                message = "Serial number changed. Retrieve and confirm the machine details again."
            elif self.assist_serial.text().strip():
                message = "Retrieve the machine details from Lenovo Support, then confirm the serial number."
            else:
                message = "Enter the serial number and retrieve the machine details from Lenovo Support."
            self.assist_machine_status.setText(message)
            self.assist_machine_status.setStyleSheet("color: #F5A524; font-weight: 700;")

    def invalidate_assist_issue(self):
        if self.assist_steps or self.assist_step_results or self.assist_summary.toPlainText():
            self.reset_assist_troubleshooting()
            self.assist_step_counter.setText("Issue details changed; prepare a new troubleshooting sequence.")
            self.assist_step_text.setText("The previous troubleshooting results were cleared because the issue changed.")

    def reset_assist_troubleshooting(self):
        self.assist_steps = []
        self.assist_step_index = 0
        self.assist_step_results = []
        self.assist_step_history.clear()
        self.assist_step_observation.clear()
        self.assist_step_counter.setText("No troubleshooting sequence started.")
        self.assist_step_text.setText(
            "Confirm machine and contact details, then capture the issue before preparing a path."
        )
        self.assist_root_cause.clear()
        self.assist_failed_hardware.clear()
        self.assist_suggested_resolution.clear()
        self.assist_outcome.setCurrentIndex(0)
        self.assist_final_notes.clear()
        self.assist_summary.clear()
        self.btn_assist_yes.setEnabled(False)
        self.btn_assist_no.setEnabled(False)

    def invalidate_assist_contact(self):
        self.assist_contact_verified = False
        self.assist_summary.clear()
        warning_style = "border: 2px solid #F5A524; background: #2A2111; color: white;"
        self.assist_mobile.setStyleSheet(warning_style)
        self.assist_email.setStyleSheet(warning_style)
        if hasattr(self, "assist_email_status"):
            self.assist_email_status.setText("Contact details changed. Validate the mobile number and email domain again.")
            self.assist_email_status.setStyleSheet("color: #F5A524; font-weight: 700;")

    def open_assist_support(self):
        serial = self.assist_serial.text().strip().upper()
        if not serial:
            QMessageBox.warning(self, "Serial number required", "Enter the laptop serial number first.")
            self.assist_serial.setFocus()
            return
        self.assist_machine_verified = False
        self.assist_machine_lookup_complete = False
        self.invalidate_assist_machine()
        self.assist_support_serial = serial
        self.assist_support_stage = "search"
        self.assist_support_search_started = False
        self.assist_support_results_clicked = False
        self.assist_support_product_capture_attempts = 0
        self.assist_support_repair_capture_attempts = 0
        self.assist_support_warranty_capture_attempts = 0
        self.assist_support_spec_capture_attempts = 0
        self.assist_support_processed_url = ""
        self.assist_support_urls = {}
        self.assist_support_product_url = ""
        self.assist_machine_status.setText(
            f"Searching Lenovo PC Support for serial {serial}…"
        )
        self.assist_machine_status.setStyleSheet("color: #F5A524; font-weight: 700;")

        self.assist_support_dialog = QDialog(self)
        self.assist_support_dialog.setWindowTitle(f"Lenovo PC Support — {serial}")
        self.assist_support_dialog.resize(1200, 850)
        support_layout = QVBoxLayout(self.assist_support_dialog)
        self.assist_support_browser_status = QLabel(
            "Opening Lenovo PC Support and submitting the serial number…"
        )
        self.assist_support_browser_status.setWordWrap(True)
        support_layout.addWidget(self.assist_support_browser_status)
        self.assist_support_view = QWebEngineView(self.assist_support_dialog)
        support_layout.addWidget(self.assist_support_view, 1)
        self.assist_support_view.loadFinished.connect(self.assist_support_page_loaded)
        self.assist_support_view.urlChanged.connect(self.assist_support_url_changed)
        self.assist_support_poll_timer = QTimer(self.assist_support_dialog)
        self.assist_support_poll_timer.setInterval(1000)
        self.assist_support_poll_timer.timeout.connect(self.process_assist_support_page)
        self.assist_support_dialog.finished.connect(self.assist_support_poll_timer.stop)
        self.assist_support_dialog.show()
        self.assist_support_poll_timer.start()
        search_query = urlencode(
            {
                "query": serial,
                "SearchType": "Customer search",
                "searchLocation": "Homepage",
            }
        )
        self.assist_support_search_started = True
        self.assist_support_view.load(
            QUrl(f"https://pcsupport.lenovo.com/in/en/redesign-search?{search_query}")
        )
        self.statusBar().showMessage(f"Searching Lenovo PC Support for serial {serial}.")

    def assist_support_page_loaded(self, succeeded):
        if not succeeded:
            self.assist_support_browser_status.setText(
                "Lenovo Support did not finish loading this page. Check connectivity or complete the lookup in the browser below."
            )
            return
        QTimer.singleShot(900, self.process_assist_support_page)

    def assist_support_url_changed(self, _url):
        QTimer.singleShot(1000, self.process_assist_support_page)

    def process_assist_support_page(self):
        if not hasattr(self, "assist_support_view") or not self.assist_support_view:
            return
        current_url = self.assist_support_view.url().toString()
        parsed_url = QUrl(current_url)
        host = parsed_url.host().lower()
        if host and host != "pcsupport.lenovo.com" and not host.endswith(".pcsupport.lenovo.com"):
            self.assist_support_browser_status.setText(
                "The portal opened an external page. Return to Lenovo PC Support to continue this lookup."
            )
            return

        path = parsed_url.path().lower().rstrip("/")
        serial = self.assist_support_serial.lower()
        if "/products/" in path and path.endswith("/" + serial):
            if self.assist_support_stage not in {
                "product", "opening_spec_info", "reading_spec_info",
                "loading_warranty", "warranty", "loading_repair",
                "repair_capture_pending", "complete",
            }:
                self.assist_support_stage = "product"
                self.assist_support_product_capture_attempts += 1
                self.assist_support_browser_status.setText(
                    "Product Home found. Reading the model description and MTM…"
                )
                self.assist_support_view.page().runJavaScript(
                    self._assist_product_capture_script(),
                    self.handle_assist_product_capture,
                )
            return

        if path.endswith("/warranty") and self.assist_support_stage not in {
            "warranty", "warranty_capture_pending", "loading_repair",
            "repair_capture_pending", "complete"
        }:
            self.assist_support_stage = "warranty_capture_pending"
            self.assist_support_browser_status.setText("Reading warranty start and end dates…")
            self.assist_support_view.page().runJavaScript(
                self._assist_page_text_script(),
                self.handle_assist_warranty_capture,
            )
            return

        if path.endswith("/repair") and self.assist_support_stage not in {
            "repair_capture_pending", "complete"
        }:
            self.assist_support_stage = "repair_capture_pending"
            self.assist_support_browser_status.setText("Reading repair status and case history…")
            self.assist_support_view.page().runJavaScript(
                self._assist_page_text_script(),
                self.handle_assist_repair_capture,
            )
            return

        if "/redesign-search" in path and self.assist_support_stage == "search":
            if not self.assist_support_results_clicked:
                self.assist_support_results_clicked = True
                script = f"""
                    (() => {{
                        const query = new URLSearchParams(location.search).get('query') || '';
                        const candidates = [...document.querySelectorAll(
                            'button, [role="button"], a'
                        )];
                        const match = candidates.find(item =>
                            /^Product Details\\b/i.test(item.innerText.trim()) &&
                            query.toLowerCase() === {json.dumps(self.assist_support_serial.lower())}
                        );
                        if (!match) return false;
                        match.click();
                        return true;
                    }})()
                """
                self.assist_support_view.page().runJavaScript(
                    script, self.handle_assist_support_result_selection
                )

    def handle_assist_support_result_selection(self, selected):
        if not selected:
            self.assist_support_results_clicked = False
            self.assist_support_browser_status.setText(
                "Waiting for Lenovo's matching Product Details result to load…"
            )
        else:
            self.assist_support_browser_status.setText(
                "Opened the matching serial result. Reading Product Home…"
            )

    def handle_assist_support_search_started(self, result):
        if result and result != "Serial submitted.":
            self.assist_support_browser_status.setText(str(result))

    @staticmethod
    def _assist_product_capture_script():
        return r"""
            (() => {
                const main = document.querySelector('.new-product-info') ||
                    document.querySelector('[aria-label="Product Information"]') ||
                    document.querySelector('main') || document.body;
                const leafText = value => [...main.querySelectorAll('*')]
                    .find(node => node.children.length === 0 &&
                        node.textContent.trim().toLowerCase() === value.toLowerCase());
                const fieldValue = label => {
                    const node = leafText(label);
                    if (!node) return '';
                    let branch = node;
                    for (let depth = 0; depth < 4 && branch.parentElement; depth += 1) {
                        const parent = branch.parentElement;
                        const sibling = [...parent.children].find(item =>
                            item !== branch && item.innerText.trim()
                        );
                        if (sibling) return sibling.innerText.trim();
                        const lines = parent.innerText.split('\n')
                            .map(line => line.trim())
                            .filter(line => line && line.toLowerCase() !== label.toLowerCase());
                        if (lines.length) return lines[0];
                        branch = parent;
                    }
                    return '';
                };
                const headings = [...main.querySelectorAll('h2')]
                    .map(node => node.innerText.trim())
                    .filter(text => text && !/product information/i.test(text));
                const links = [...document.querySelectorAll('a')];
                const supportLink = expression => {
                    const link = links.find(item => expression.test(item.innerText.trim()));
                    return link ? link.href : '';
                };
                return {
                    description: headings[0] || '',
                    serial: fieldValue('Serial Number'),
                    mtm: fieldValue('Machine Type Model'),
                    warrantyUrl: supportLink(/Warranty\s*&\s*Services/i),
                    repairUrl: supportLink(/Repair Status|Check Repair Status/i),
                    productUrl: location.href
                };
            })()
        """

    @staticmethod
    def _assist_page_text_script():
        return "(() => (document.querySelector('main') || document.body).innerText || '')()"

    @staticmethod
    def _assist_spec_info_script():
        return r"""
            (() => {
                const text = document.body.innerText || '';
                const start = text.lastIndexOf('Spec Info');
                return start < 0 ? '' : text.slice(start, start + 2200).trim();
            })()
        """

    def open_assist_spec_info(self):
        script = r"""
            (() => {
                const controls = [...document.querySelectorAll(
                    'button, [role="button"]'
                )];
                const button = controls.find(item =>
                    /View Spec Info/i.test(
                        `${item.getAttribute('aria-label') || ''} ${item.innerText || ''}`
                    )
                );
                if (!button) return false;
                button.click();
                return true;
            })()
        """
        self.assist_support_stage = "opening_spec_info"
        self.assist_support_view.page().runJavaScript(
            script, self.handle_assist_spec_info_opened
        )

    def handle_assist_spec_info_opened(self, opened):
        if opened:
            self.assist_support_stage = "reading_spec_info"
            QTimer.singleShot(
                700,
                lambda: self.assist_support_view.page().runJavaScript(
                    self._assist_spec_info_script(),
                    self.handle_assist_spec_info_capture,
                ),
            )
        else:
            self.assist_configuration.setPlainText(
                "Lenovo Spec Info was not available on Product Home."
            )
            self.continue_assist_support_lookup()

    def handle_assist_spec_info_capture(self, configuration):
        configuration = str(configuration or "").strip()
        if configuration and configuration.lower().startswith("spec info"):
            self.assist_configuration.setPlainText(configuration[:2200])
        else:
            self.assist_support_spec_capture_attempts += 1
            if self.assist_support_spec_capture_attempts < 4:
                QTimer.singleShot(
                    1000,
                    lambda: self.assist_support_view.page().runJavaScript(
                        self._assist_spec_info_script(),
                        self.handle_assist_spec_info_capture,
                    ),
                )
                return
            self.assist_configuration.setPlainText(
                "Lenovo Spec Info did not load automatically; review Product Home."
            )
        self.continue_assist_support_lookup()

    def continue_assist_support_lookup(self):
        product_url = self.assist_support_product_url.rstrip("/")
        warranty_url = self.assist_support_urls.get("warranty") or f"{product_url}/warranty"
        self.assist_support_urls["warranty"] = warranty_url
        self.assist_support_urls["repair"] = (
            self.assist_support_urls.get("repair") or f"{product_url}/repair"
        )
        self.assist_support_stage = "loading_warranty"
        self.assist_support_browser_status.setText(
            "Product details captured. Loading Warranty & Services, then Repair Status…"
        )
        self.assist_support_view.load(QUrl(warranty_url))

    def handle_assist_product_capture(self, data):
        if not isinstance(data, dict):
            self.assist_support_browser_status.setText(
                "Product Home opened, but its machine information could not be read automatically. Review the portal and enter the fields manually."
            )
            if self.assist_support_product_capture_attempts < 4:
                self.assist_support_stage = "search"
                QTimer.singleShot(1500, self.process_assist_support_page)
            return
        expected_serial = self.assist_support_serial.upper()
        if data.get("serial", "").strip().upper() != expected_serial:
            self.assist_support_browser_status.setText(
                "The Product Home serial did not match the entered serial. Review the portal; no machine details were accepted."
            )
            if self.assist_support_product_capture_attempts < 4:
                self.assist_support_stage = "search"
                QTimer.singleShot(1500, self.process_assist_support_page)
            return
        description = data.get("description", "").strip()
        mtm = data.get("mtm", "").strip()
        configuration = data.get("configuration", "").strip()
        self.assist_product.setText(description)
        self.assist_mtm.setText(mtm)
        self.assist_configuration.setPlainText(configuration)
        self.assist_support_product_url = data.get("productUrl", "")
        self.assist_support_urls = {
            "warranty": data.get("warrantyUrl", ""),
            "repair": data.get("repairUrl", ""),
        }
        self.assist_support_browser_status.setText(
            f"Product Home captured: {description or 'description unavailable'}; MTM {mtm or 'not displayed'}. "
            "Reading Lenovo Spec Info…"
        )
        self.open_assist_spec_info()

    def handle_assist_warranty_capture(self, page_text):
        text = str(page_text or "")
        start_match = re.search(
            r"\bStart Date\s*:?\s*((?:\d{4}[-/]\d{1,2}[-/]\d{1,2})|(?:[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4}))",
            text,
            re.IGNORECASE,
        )
        end_match = re.search(
            r"\bEnd Date\s*:?\s*((?:\d{4}[-/]\d{1,2}[-/]\d{1,2})|(?:[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4}))",
            text,
            re.IGNORECASE,
        )
        self.assist_warranty_start.setText(start_match.group(1) if start_match else "")
        self.assist_warranty_end.setText(end_match.group(1) if end_match else "")
        if not (start_match and end_match):
            self.assist_support_warranty_capture_attempts += 1
            if self.assist_support_warranty_capture_attempts < 4:
                self.assist_support_stage = "loading_warranty"
                self.assist_support_browser_status.setText(
                    "Waiting for Lenovo Warranty & Services to finish loading…"
                )
                QTimer.singleShot(1200, self.process_assist_support_page)
                return
            self.assist_support_browser_status.setText(
                "Warranty page loaded, but one or both dates were not displayed. Review Warranty History in the portal."
            )
        else:
            self.assist_support_browser_status.setText(
                f"Warranty dates captured ({start_match.group(1)} to {end_match.group(1)}). Opening Repair Status…"
            )
        repair_url = self.assist_support_urls.get("repair", "")
        if repair_url:
            self.assist_support_stage = "loading_repair"
            self.assist_support_view.load(QUrl(repair_url))
        else:
            self.assist_support_stage = "complete"

    def handle_assist_repair_capture(self, page_text):
        lines = [
            re.sub(r"[ \t]+", " ", line).strip()
            for line in str(page_text or "").splitlines()
        ]
        lines = [line for line in lines if line]
        repair_lines = []
        for index, line in enumerate(lines):
            if re.search(r"Repair\s*Ticket\s*:?\s*\d+", line, re.IGNORECASE):
                repair_lines.extend(lines[index:index + 18])
                break
        for index, line in enumerate(lines):
            if "Other Repair Tickets" in line:
                repair_lines.extend(lines[index:index + 24])
                break
        if repair_lines:
            self.assist_last_ticket.setPlainText("\n".join(dict.fromkeys(repair_lines))[:4000])
        elif re.search(r"service busy|try again later", str(page_text), re.IGNORECASE):
            self.assist_support_repair_capture_attempts += 1
            if self.assist_support_repair_capture_attempts < 4:
                self.assist_support_stage = "loading_repair"
                self.assist_support_browser_status.setText(
                    "Lenovo Repair Status is temporarily busy; retrying the page…"
                )
                QTimer.singleShot(1500, self.process_assist_support_page)
                return
            self.assist_last_ticket.setPlainText(
                "Lenovo Repair Status did not return ticket details after retries. Review the open portal page."
            )
        else:
            self.assist_last_ticket.setPlainText(
                "No repair ticket details were displayed by Lenovo Support."
            )
        self.assist_support_stage = "complete"
        self.assist_machine_lookup_complete = True
        self.assist_support_browser_status.setText(
            "Portal lookup sequence finished. Review each captured field; fill any missing details from the open Lenovo pages before confirming."
        )
        self.assist_machine_status.setText(
            "Product Home, Spec Info, warranty and repair details were checked automatically. Review the fields and correct any values if needed, then confirm to continue."
        )
        self.assist_machine_status.setStyleSheet("color: #55E2B2; font-weight: 700;")

    def confirm_assist_machine(self):
        serial = self.assist_serial.text().strip().upper()
        if not serial:
            QMessageBox.warning(self, "Serial number required", "Enter a serial number.")
            self.assist_serial.setFocus()
            return
        machine_fields = {
            "Product / model": self.assist_product.text().strip(),
            "MTM": self.assist_mtm.text().strip(),
            "Configuration": self.assist_configuration.toPlainText().strip(),
            "Warranty start": self.assist_warranty_start.text().strip(),
            "Warranty end": self.assist_warranty_end.text().strip(),
            "Repair status / case history": self.assist_last_ticket.toPlainText().strip(),
        }
        missing = [label for label, value in machine_fields.items() if not value]
        if missing:
            QMessageBox.warning(
                self,
                "Support details incomplete",
                "Run Search Lenovo Support and wait for the portal details, or enter the missing details manually. Use 'Not displayed' or 'Not applicable' when the portal has no value:\n"
                + "\n".join(missing),
            )
            return
        self.assist_machine_verified = True
        self.assist_machine_status.setText(
            f"Serial number {serial} and the displayed Lenovo Support details confirmed."
        )
        self.assist_machine_status.setStyleSheet("color: #55E2B2; font-weight: 700;")
        self.statusBar().showMessage("Agent Assist: serial number recorded.")

    def validate_assist_contact(self):
        import re
        mobile_raw = self.assist_mobile.text().strip()
        digits = re.sub(r"\D", "", mobile_raw)
        if digits.startswith("91") and len(digits) == 12:
            digits = digits[2:]
        if len(digits) != 10 or not digits.isdigit() or digits[0] not in "6789":
            self.assist_mobile.setStyleSheet("border: 2px solid #FF5148; background: #2A1118; color: white;")
            QMessageBox.warning(
                self, "Invalid mobile number",
                "Enter a valid 10-digit Indian mobile number. The application will format it as +91XXXXXXXXXX."
            )
            self.assist_contact_verified = False
            return
        self.assist_mobile.setText("+91" + digits)
        self.assist_mobile.setStyleSheet("border: 1px solid #22C997; background: #0D2929; color: white;")

        email = self.assist_email.text().strip()
        email_pattern = re.compile(r"^[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)+$")
        if not email_pattern.match(email):
            self.assist_email.setStyleSheet("border: 2px solid #FF5148; background: #2A1118; color: white;")
            self.assist_email_status.setText("INVALID: Email format is incomplete or malformed. Check the spelling and domain.")
            self.assist_email_status.setStyleSheet("color: #FF7777; font-weight: 800;")
            self.assist_contact_verified = False
            QMessageBox.warning(self, "Invalid email ID", "Enter a complete email address, including a valid-looking domain such as name@example.com.")
            return

        domain = email.rsplit("@", 1)[1].lower().rstrip(".")
        suspicious_domains = {
            "gmial.com": "gmail.com", "gmai.com": "gmail.com", "gmail.con": "gmail.com",
            "yaho.com": "yahoo.com", "yahoo.con": "yahoo.com", "outlok.com": "outlook.com",
            "hotnail.com": "hotmail.com", "lenovo.co": "lenovo.com", "lenov.com": "lenovo.com",
        }
        if domain in suspicious_domains:
            self.assist_email.setStyleSheet("border: 2px solid #FF5148; background: #2A1118; color: white;")
            self.assist_email_status.setText(f"CHECK DOMAIN: '{domain}' may be a typo. Did you mean '{suspicious_domains[domain]}'? Verify with the customer.")
            self.assist_email_status.setStyleSheet("color: #FF7777; font-weight: 800;")
            self.assist_contact_verified = False
            QMessageBox.warning(self, "Email domain needs review", self.assist_email_status.text())
            return

        try:
            import dns.resolver
            import dns.exception
        except ImportError:
            self.assist_email.setStyleSheet("border: 2px solid #F5A524; background: #2A2111; color: white;")
            self.assist_email_status.setText(
                "UNVERIFIED: DNS domain verification is unavailable because dnspython is not installed."
            )
            self.assist_email_status.setStyleSheet("color: #F5A524; font-weight: 800;")
            self.assist_contact_verified = False
            QMessageBox.warning(self, "Email domain not verified", self.assist_email_status.text())
            return

        try:
            answers = dns.resolver.resolve(domain, "MX", lifetime=3)
            if not list(answers):
                self.assist_email.setStyleSheet("border: 2px solid #F5A524; background: #2A2111; color: white;")
                self.assist_email_status.setText(
                    f"UNVERIFIED: '{domain}' returned no MX records. Confirm the domain and email address with the customer."
                )
                self.assist_email_status.setStyleSheet("color: #F5A524; font-weight: 800;")
                self.assist_contact_verified = False
                QMessageBox.warning(self, "Email domain not verified", self.assist_email_status.text())
                return
        except dns.resolver.NoAnswer:
            self.assist_email.setStyleSheet("border: 2px solid #F5A524; background: #2A2111; color: white;")
            self.assist_email_status.setText(
                f"UNVERIFIED: '{domain}' has no MX record. Confirm the domain and email address with the customer."
            )
            self.assist_email_status.setStyleSheet("color: #F5A524; font-weight: 800;")
            self.assist_contact_verified = False
            QMessageBox.warning(self, "Email domain not verified", self.assist_email_status.text())
            return
        except dns.resolver.NXDOMAIN:
            self.assist_email.setStyleSheet("border: 2px solid #FF5148; background: #2A1118; color: white;")
            self.assist_email_status.setText(
                f"INVALID DOMAIN: '{domain}' does not resolve. Check the spelling with the customer."
            )
            self.assist_email_status.setStyleSheet("color: #FF7777; font-weight: 800;")
            self.assist_contact_verified = False
            QMessageBox.warning(self, "Email domain not found", self.assist_email_status.text())
            return
        except (dns.exception.DNSException, OSError) as exc:
            self.assist_email.setStyleSheet("border: 2px solid #F5A524; background: #2A2111; color: white;")
            self.assist_email_status.setText(
                f"UNVERIFIED: Could not check mail records for '{domain}' ({exc}). Check the address and retry."
            )
            self.assist_email_status.setStyleSheet("color: #F5A524; font-weight: 800;")
            self.assist_contact_verified = False
            QMessageBox.warning(self, "Email domain not verified", self.assist_email_status.text())
            return

        self.assist_email.setStyleSheet("border: 1px solid #22C997; background: #0D2929; color: white;")
        self.assist_email_status.setText(
            f"Email syntax valid; domain '{domain}' has mail (MX) records. Still confirm spelling with the customer."
        )
        self.assist_email_status.setStyleSheet("color: #55E2B2; font-weight: 700;")
        self.assist_contact_verified = True
        self.statusBar().showMessage("Agent Assist: mobile format and email syntax checked; review any domain warning.")

    def upload_assist_evidence(self):
        if not self.assist_machine_verified:
            QMessageBox.warning(self, "Confirm machine first", "Verify and confirm the serial number details before attaching call evidence.")
            return
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select snapshots or supporting evidence", "",
            "Evidence files (*.png *.jpg *.jpeg *.webp *.bmp *.pdf *.txt *.log);;All files (*.*)"
        )
        if not paths:
            return
        max_bytes = 200 * 1024 * 1024
        accepted, rejected = [], []
        serial = re.sub(r"[^A-Za-z0-9_-]", "_", self.assist_serial.text().strip()) or "unknown_serial"
        agent_folder = self.user_email.replace("@", "_at_").replace(".", "_")
        evidence_dir = os.path.join(VAULT_DIR, agent_folder, "AgentAssistEvidence", serial)
        try:
            os.makedirs(evidence_dir, exist_ok=True)
        except OSError as exc:
            QMessageBox.warning(self, "Evidence folder unavailable", str(exc))
            return
        for path in paths:
            try:
                size = os.path.getsize(path)
                if size > max_bytes:
                    rejected.append(f"{os.path.basename(path)} ({size / (1024*1024):.1f} MB exceeds 200 MB)")
                    continue
                destination = os.path.join(evidence_dir, os.path.basename(path))
                stem, ext = os.path.splitext(destination)
                counter = 1
                while os.path.exists(destination):
                    destination = f"{stem}_{counter}{ext}"
                    counter += 1
                shutil.copy2(path, destination)
                accepted.append(destination)
            except OSError as exc:
                rejected.append(f"{os.path.basename(path)} (copy failed: {exc})")
        self.assist_evidence_paths.extend(accepted)
        names = [os.path.basename(p) for p in self.assist_evidence_paths]
        self.assist_evidence_list.setText(
            ("Attached: " + ", ".join(names) if names else "No evidence attached.")
            + ("\nRejected: " + "; ".join(rejected) if rejected else "")
        )
        if rejected:
            QMessageBox.warning(self, "Some files were not attached", "\n".join(rejected))

    def prepare_assist_steps(self):
        if not self.assist_machine_verified:
            QMessageBox.warning(self, "Machine verification required", "Complete and confirm the serial-number lookup first.")
            return
        if not self.assist_contact_verified:
            QMessageBox.warning(self, "Contact validation required", "Validate the customer's mobile number and email before continuing.")
            return
        issue = self.assist_issue.toPlainText().strip()
        if len(issue) < 8:
            QMessageBox.warning(self, "Issue details required", "Capture the customer's exact issue and symptoms before generating troubleshooting steps.")
            return

        text = issue.lower()
        # Keep the suggestions non-invasive; use model-specific Lenovo service
        # documentation for any procedure that depends on the exact machine.
        common = [
            "Confirm and repeat the customer's exact symptom, any error code, and when it started. Did you capture it accurately?",
            "Check whether the issue reproduces after disconnecting non-essential external peripherals, where safe and appropriate. Did the symptom change?",
        ]
        if any(k in text for k in ("power", "not turn on", "won't turn on", "wont turn on", "no power", "charging", "battery")):
            self.assist_symptom_category = "power or battery"
            self.assist_diagnostic_component = (
                "battery" if any(k in text for k in ("battery", "charging"))
                else "power adapter / system power path"
            )
            steps = common + [
                "Confirm the wall outlet works with a known-good device and inspect the approved power adapter/cable for visible damage. Is power delivery normal?",
                "With the laptop disconnected from peripherals, follow the approved model-specific power-drain/reset procedure if available. Does the laptop power on?",
                "If available for this model, run Lenovo diagnostics for the battery or power subsystem and record the exact result/code. Did the diagnostic pass?",
            ]
        elif any(k in text for k in ("display", "screen", "black screen", "no display", "flicker")):
            self.assist_symptom_category = "display"
            self.assist_diagnostic_component = "display panel / graphics subsystem"
            steps = common + [
                "Check display brightness and whether the laptop shows power indicators or startup sounds. Is the display visible after these checks?",
                "If available, connect a known-good compatible external display using an approved port/cable. Does the external display work?",
                "If available for this model, run Lenovo diagnostics for the display/graphics subsystem and record the exact result/code. Did the diagnostic pass?",
            ]
        elif any(k in text for k in ("wifi", "wi-fi", "wireless", "network", "internet", "bluetooth")):
            self.assist_symptom_category = (
                "wireless network" if any(k in text for k in ("wifi", "wi-fi", "wireless", "network", "internet"))
                else "Bluetooth"
            )
            self.assist_diagnostic_component = (
                "WLAN adapter" if any(k in text for k in ("wifi", "wi-fi", "wireless"))
                else "Bluetooth adapter"
            )
            steps = common + [
                "Confirm Airplane mode is off and the relevant wireless adapter is enabled. Is the adapter enabled?",
                "Check whether other devices can connect to the same network and whether this laptop can see other networks. Can it connect to a known-good network?",
                "Check Lenovo Vantage or the supported operating-system diagnostics for the wireless adapter and record the exact result. Did the diagnostic pass?",
            ]
        elif any(k in text for k in ("overheat", "overheating", "hot", "fan", "thermal", "temperature")):
            self.assist_symptom_category = "thermal or fan"
            self.assist_diagnostic_component = "fan / thermal subsystem"
            steps = common + [
                "Check that vents are unobstructed and the laptop is on a hard, flat surface. Is airflow unobstructed?",
                "Ask whether the fan is audible and whether the heat occurs only under a particular workload; stop use and escalate immediately for burning smell, smoke or battery swelling. Is there a safety concern?",
                "Use Lenovo Vantage or the model-supported diagnostic to check fan/thermal status and record any exact code. Did the diagnostic pass?",
            ]
        elif any(k in text for k in ("usb", "dock", "thunderbolt", "type-c", "type c", "port")):
            self.assist_symptom_category = "USB, port or dock"
            self.assist_diagnostic_component = "USB-C / USB port or dock (not isolated)"
            steps = common + [
                "Test the device and cable separately on a known-good compatible port/device, then test a known-good supported device on the affected port. Does the issue follow the laptop port?",
                "If using a dock, confirm its exact model, supported power adapter, firmware status and direct-connect behavior using Lenovo's dock support page. Does direct connection work?",
                "Check Lenovo Vantage or model-supported diagnostics for the relevant port/controller and record the exact result/code. Did the diagnostic pass?",
            ]
        elif any(k in text for k in ("ssd", "storage drive", "disk error", "drive not detected", "storage failure")):
            self.assist_symptom_category = "storage"
            self.assist_diagnostic_component = "storage drive"
            steps = common + [
                "Record the exact storage or boot error and check whether the drive is detected by the model-supported firmware/diagnostic screen. Is the drive detected?",
                "Protect customer data; do not initialize, format, reset or reinstall the operating system. Is important data backed up or otherwise protected?",
                "Run Lenovo diagnostics for the storage drive and record the exact failure ID/result. Did the diagnostic pass?",
            ]
        elif any(k in text for k in ("slow", "performance", "hang", "freeze", "freezing", "crash")):
            self.assist_symptom_category = "performance, hang or crash"
            self.assist_diagnostic_component = (
                "memory" if any(k in text for k in ("memory", "ram"))
                else "storage drive" if any(k in text for k in ("disk", "ssd", "storage"))
                else "system performance (component not isolated)"
            )
            steps = common + [
                "Check Task Manager for sustained CPU, memory or disk saturation and record the process/resource involved. Is there a clear resource bottleneck?",
                "Confirm available storage space and whether pending updates or a recent software change correlate with the symptom. Is there a confirmed software/update cause?",
                "If indicated, run Lenovo Vantage or the model-supported pre-boot diagnostic for memory/storage and record the exact result/code. Did the diagnostic pass?",
            ]
        elif any(k in text for k in ("keyboard", "touchpad", "mouse", "key", "click")):
            self.assist_symptom_category = "keyboard or pointing device"
            self.assist_diagnostic_component = (
                "keyboard" if any(k in text for k in ("keyboard", "key"))
                else "touchpad" if "touchpad" in text
                else "pointing device"
            )
            steps = common + [
                "Check whether the affected input device works in the applicable pre-boot diagnostic or BIOS environment, if the model supports it and policy permits. Does it work there?",
                "Test an external known-good input device and check the relevant device status/driver in the operating system. Does the external device work normally?",
                "If available for this model, run Lenovo diagnostics for the affected input device and record the exact result/code. Did the diagnostic pass?",
            ]
        elif any(k in text for k in ("audio", "sound", "speaker", "microphone", "mic")):
            self.assist_symptom_category = "audio"
            self.assist_diagnostic_component = "audio device"
            steps = common + [
                "Check mute, volume, the selected input/output device and whether the issue occurs with a known-good headset. Does audio work with the alternate device?",
                "Check the audio device status in the operating system and apply only approved driver updates if indicated. Does the device work after the check?",
                "Run Lenovo Vantage or model-supported diagnostics for the audio device and record the exact result/code. Did the diagnostic pass?",
            ]
        elif any(k in text for k in ("camera", "webcam", "camera privacy")):
            self.assist_symptom_category = "camera"
            self.assist_diagnostic_component = "camera"
            steps = common + [
                "Check the physical/electronic camera privacy control and the operating-system camera permission. Is the camera available?",
                "Check whether an approved video application can use the camera and whether another application is holding it. Does the camera work?",
                "Run Lenovo Vantage or model-supported diagnostics for the camera and record the exact result/code. Did the diagnostic pass?",
            ]
        elif any(k in text for k in ("boot", "startup", "start up", "operating system", "windows")):
            self.assist_symptom_category = "boot or operating system"
            self.assist_diagnostic_component = "boot/storage subsystem (not isolated)"
            steps = common + [
                "Record any exact startup error and check whether the system reaches the Lenovo logo or built-in diagnostics. Is the failure reproducible?",
                "Use only the approved startup-repair options for the installed operating system; do not reset or reinstall without customer approval. Does startup complete?",
                "If available, run Lenovo diagnostics for the storage drive and record the exact result/code. Did the diagnostic pass?",
            ]
        else:
            self.assist_symptom_category = "general hardware or software"
            self.assist_diagnostic_component = "the affected component"
            steps = common + [
                "Check the exact error message/code and reproduce the issue once while noting the conditions. Is the symptom reproducible?",
                "Run Lenovo Vantage or model-supported diagnostics for the component indicated by the symptoms and record the exact result/code. Did the diagnostic pass?",
            ]
        steps.append(
            "Review all recorded results. Is the issue resolved and verified with the customer? If not, stop before invasive actions and follow the approved Lenovo escalation/repair workflow."
        )
        self.assist_root_cause.clear()
        self.assist_failed_hardware.clear()
        self.assist_suggested_resolution.clear()
        self.assist_outcome.setCurrentIndex(0)
        self.assist_final_notes.clear()
        self.assist_summary.clear()
        self.assist_steps = steps
        self.assist_step_index = 0
        self.assist_step_results = []
        self.assist_step_history.clear()
        self.show_assist_step()
        self.statusBar().showMessage("Agent Assist: a general triage path is ready. Follow approved model-specific Lenovo procedures.")

    def show_assist_step(self):
        if not self.assist_steps or self.assist_step_index >= len(self.assist_steps):
            self.assist_step_counter.setText("Troubleshooting sequence completed")
            self.assist_step_text.setText("All listed steps have been answered. Review the evidence before recording a root cause or hardware failure.")
            self.btn_assist_yes.setEnabled(False)
            self.btn_assist_no.setEnabled(False)
            self.btn_assist_howto.setEnabled(False)
            return
        self.assist_step_counter.setText(f"STEP {self.assist_step_index + 1} OF {len(self.assist_steps)}")
        self.assist_step_text.setText(self.assist_steps[self.assist_step_index])
        self.btn_assist_yes.setEnabled(True)
        self.btn_assist_no.setEnabled(True)
        self.btn_assist_howto.setEnabled(True)

    def show_assist_howto(self):
        if not self.assist_steps or self.assist_step_index >= len(self.assist_steps):
            return
        step = self.assist_steps[self.assist_step_index]
        issue = self.assist_issue.toPlainText().strip()
        issue_guides = {
            "power or battery": [
                "Confirm the outlet with another device; inspect only the external adapter and cable for damage.",
                "Check charging/power indicators and whether the symptom changes with a known-good, compatible Lenovo adapter if available.",
                "Use only the exact model's approved power-reset instructions; then run Lenovo battery/power diagnostics if available.",
            ],
            "display": [
                "Check brightness, display output mode and whether startup indicators/sounds are present.",
                "If supported, test with a known-good external display. Record whether the external display works while the built-in panel does not.",
                "Run the available Lenovo display/graphics diagnostic and save its exact result/code.",
            ],
            "wireless network": [
                "Check Airplane mode, wireless enablement and whether other devices connect to the same network.",
                "Compare this PC on another known-good network; record whether it can see networks and whether only one network is affected.",
                "Check adapter status and Lenovo-supported driver/diagnostic results; do not change router/security settings without authorization.",
            ],
            "Bluetooth": [
                "Confirm Bluetooth is enabled and test with a second supported peripheral.",
                "Remove/re-pair only with customer approval and record whether the PC detects other Bluetooth devices.",
                "Check adapter status and Lenovo-supported driver/diagnostic results.",
            ],
            "thermal or fan": [
                "Check unobstructed vents and use on a hard, flat surface; note workload, fan noise and temperature behavior.",
                "Stop the session and escalate for smoke, burning odor or battery swelling; do not open the chassis.",
                "Run model-supported Lenovo fan/thermal diagnostics and record the exact result.",
            ],
            "USB, port or dock": [
                "Isolate the accessory, cable, port and dock by testing known-good supported combinations.",
                "Compare direct connection with dock connection; record the dock model and whether the problem follows the PC port.",
                "Check Lenovo support for the exact dock/PC firmware and run the supported port/controller diagnostic.",
            ],
            "storage": [
                "Record the exact drive/boot error and whether the drive is detected by supported firmware/diagnostics.",
                "Protect customer data. Do not format, initialize, reset, or reinstall the OS during this triage.",
                "Run Lenovo storage diagnostics and retain its exact failure ID/result for authorized repair.",
            ],
            "performance, hang or crash": [
                "Record when the issue occurs, recent changes and any stop/error code; check Task Manager resource usage.",
                "Check free storage and pending updates without uninstalling software or changing security settings without approval.",
                "Run Lenovo memory/storage diagnostics when symptoms point to those components; save exact failure IDs.",
            ],
            "keyboard or pointing device": [
                "Check external-device behavior and approved keyboard/touchpad settings; test only model-supported pre-boot diagnostics.",
                "Compare a known-good external keyboard/mouse with the built-in device and record the difference.",
                "Run the available Lenovo input-device diagnostic and retain the exact result.",
            ],
            "audio": [
                "Check mute/volume, selected input/output and privacy controls; compare with a known-good headset.",
                "Test recording/playback in an approved app and record which direction (input/output) fails.",
                "Check device status and run Lenovo-supported audio diagnostics.",
            ],
            "camera": [
                "Check the physical/electronic camera shutter and OS/app permissions.",
                "Test with an approved camera app and confirm whether another app is using the camera.",
                "Check device status and run Lenovo-supported camera diagnostics.",
            ],
            "boot or operating system": [
                "Record the exact boot screen/error and identify whether the Lenovo logo, firmware diagnostics, or OS appears.",
                "Protect user data. Do not reset/reinstall or select destructive recovery options without approval.",
                "Run model-supported storage diagnostics and use only approved OS startup repair; escalate if data or drive health is at risk.",
            ],
        }
        issue_guide = issue_guides.get(
            self.assist_symptom_category,
            [
                "Reproduce once and record exact conditions, messages, and any error code.",
                "Use the relevant Lenovo Vantage or model-supported diagnostic and save its exact result.",
                "Avoid invasive or destructive changes; escalate to the approved Lenovo TSS/repair process if unresolved.",
            ],
        )
        howto = [
            f"Current TSS step\n{step}",
            "",
            "Before you begin",
            "1. Confirm the customer is ready and explain what this check will do.",
            "2. Confirm the exact symptom and any error code; use the confirmed model shown in Product Home.",
            "3. Save open work before any restart. Do not ask the customer to open the computer, disconnect internal parts, change BIOS/security settings, or reinstall/reset the OS as an initial check.",
            "",
            "Perform the check",
            "1. Follow the instruction above only if it is safe and applicable to this exact model and operating system.",
            "2. If the step references Lenovo diagnostics, open Lenovo Vantage or the model-specific Lenovo Support diagnostics page. Select the test matching the symptom; if an available pre-boot test is required, explain that the customer must save work first and follow Lenovo's on-screen instructions.",
            "3. Record what the customer sees, including the exact diagnostic name, result, error code, and whether the original symptom changed.",
            "",
            f"Issue-specific TSS path: {self.assist_symptom_category}",
            *[f"- {instruction}" for instruction in issue_guide],
            "",
            "Record the result",
            "Enter the observation or exact diagnostic code in the note field, then select YES only when the question in the current step is true. Select NO when it is false or the symptom remains.",
            "",
            "Issue captured for this call",
            issue or "No issue description captured.",
            "",
            "The generated sequence is a safe triage aid, not a substitute for Lenovo's model-specific service manual, approved TSS procedure, or technician authorization. Stop and escalate if the symptom indicates a safety risk or the proposed action is not supported for this model.",
        ]
        dialog = QDialog(self)
        dialog.setWindowTitle("How to perform this troubleshooting step")
        dialog.resize(760, 620)
        layout = QVBoxLayout(dialog)
        instructions = QTextEdit()
        instructions.setReadOnly(True)
        instructions.setPlainText("\n".join(howto))
        layout.addWidget(instructions, 1)
        controls = QHBoxLayout()
        if self.assist_support_product_url:
            btn_lenovo_guide = QPushButton("Open this model's Lenovo troubleshooting page")
            guide_url = self.assist_support_product_url.rstrip("/") + "/diagnostics-troubleshooting"
            btn_lenovo_guide.clicked.connect(
                lambda: QDesktopServices.openUrl(QUrl(guide_url))
            )
            controls.addWidget(btn_lenovo_guide)
        controls.addStretch(1)
        close_button = QPushButton("Close")
        close_button.clicked.connect(dialog.accept)
        controls.addWidget(close_button)
        layout.addLayout(controls)
        dialog.exec()

    def record_assist_step(self, answer):
        if not self.assist_steps or self.assist_step_index >= len(self.assist_steps):
            return
        result = "YES" if answer else "NO"
        step = self.assist_steps[self.assist_step_index]
        observation = self.assist_step_observation.text().strip()
        self.assist_step_results.append((step, result, observation))
        history_entry = f"{len(self.assist_step_results)}. {step}\n   Result: {result}"
        if observation:
            history_entry += f"\n   Observation / code: {observation}"
        self.assist_step_history.append(history_entry + "\n")
        self.assist_step_observation.clear()
        # A successful resolution check ends the flow; do not ask the agent to keep
        # performing irrelevant steps after the customer confirms recovery.
        is_resolution_check = "issue resolved and verified" in step.lower()
        self.assist_step_index += 1
        if is_resolution_check and answer:
            self.assist_steps = self.assist_steps[:self.assist_step_index]
        self.update_assist_resolution_suggestion(
            resolved=bool(is_resolution_check and answer),
            failed_diagnostic=bool("diagnostic" in step.lower() and not answer),
            troubleshooting_complete=self.assist_step_index >= len(self.assist_steps),
            observation=observation,
        )
        self.show_assist_step()

    def update_assist_resolution_suggestion(
        self, resolved=False, failed_diagnostic=False, troubleshooting_complete=False, observation=""
    ):
        if resolved:
            self.assist_root_cause.setPlainText(
                "Issue resolved and verified during the call; the underlying cause was not isolated."
            )
            self.assist_failed_hardware.setText("Not indicated")
            self.assist_suggested_resolution.setPlainText(
                "No part replacement indicated. Confirm normal operation with the customer and document the verified recovery."
            )
            self.assist_outcome.setCurrentText("Resolved during call")
            self.assist_final_notes.setPlainText(
                "Customer confirmed the symptom is resolved. No hardware failure was established."
            )
        elif failed_diagnostic:
            evidence = f" Recorded diagnostic evidence/code: {observation}." if observation else (
                " Capture the exact diagnostic error code before case submission."
            )
            self.assist_root_cause.setPlainText(
                f"Suspected {self.assist_diagnostic_component} failure: a Lenovo/model-supported "
                f"diagnostic did not pass.{evidence} Technician confirmation is required."
            )
            self.assist_failed_hardware.setText(
                f"{self.assist_diagnostic_component} (suspected; not confirmed)"
            )
            self.assist_suggested_resolution.setPlainText(
                f"Use the recorded diagnostic ID/code to confirm the {self.assist_diagnostic_component} "
                "against Lenovo's model-specific service guidance. If confirmed, arrange the approved "
                "warranty/service route and authorized repair or FRU replacement. Do not replace parts "
                "based only on this triage suggestion."
            )
            self.assist_outcome.setCurrentText("Escalation / repair required")
            self.assist_final_notes.setPlainText(
                "Escalate through the approved Lenovo repair workflow with the recorded diagnostic "
                "result/code. Do not represent the suspected component as confirmed until validated."
            )
        elif troubleshooting_complete:
            diagnostic_failure = any(
                "diagnostic" in step.lower() and result == "NO"
                for step, result, _ in self.assist_step_results
            )
            if not diagnostic_failure:
                completed = len(self.assist_step_results)
                self.assist_root_cause.setPlainText(
                    f"Undetermined from available evidence after {completed} troubleshooting "
                    "response(s). Review diagnostic results and customer observations."
                )
                self.assist_failed_hardware.setText("Not confirmed")
                self.assist_suggested_resolution.setPlainText(
                    f"No hardware fault is confirmed for the {self.assist_symptom_category} symptom. "
                    f"Continue with Lenovo's model-specific diagnostics for {self.assist_diagnostic_component}, "
                    "capture the exact test result and escalate if the issue remains."
                )
                self.assist_outcome.setCurrentText("Unresolved — further troubleshooting")
                self.assist_final_notes.setPlainText(
                    "Continue with approved Lenovo/model-specific diagnostics. Root cause and failed "
                    "hardware are not confirmed."
                )

    def refresh_assist_transcript(self):
        if hasattr(self, "txt_live_transcript"):
            self.assist_live_transcript.setPlainText(self.txt_live_transcript.toPlainText())
            self.assist_live_transcript.moveCursor(QTextCursor.End)

    def generate_assist_summary(self):
        if not self.assist_machine_verified:
            QMessageBox.warning(self, "Machine not verified", "Confirm machine details before generating the final summary.")
            return
        if not self.assist_contact_verified:
            QMessageBox.warning(
                self, "Contact details not verified",
                "Validate the customer's 10-digit Indian mobile number and email domain before generating the summary."
            )
            return
        if not self.assist_steps or self.assist_step_index < len(self.assist_steps):
            QMessageBox.warning(
                self, "Troubleshooting not complete",
                "Prepare and complete the guided troubleshooting steps before recording the final root cause and outcome."
            )
            return
        serial = self.assist_serial.text().strip().upper()
        mobile = self.assist_mobile.text().strip()
        email = self.assist_email.text().strip()
        outcome = self.assist_outcome.currentText()
        root_cause = self.assist_root_cause.toPlainText().strip() or "Not confirmed from available evidence."
        hardware = self.assist_failed_hardware.text().strip() or "Not confirmed."
        if outcome == "Select final outcome":
            QMessageBox.warning(self, "Select an outcome", "Choose the final outcome before generating the summary.")
            return
        hardware_lower = hardware.lower()
        hardware_is_unconfirmed = (
            hardware_lower in {"not confirmed", "not confirmed.", "not indicated"}
            or "suspected" in hardware_lower
        )
        if outcome != "Hardware failure confirmed" and not hardware_is_unconfirmed:
            QMessageBox.warning(
                self, "Confirm hardware evidence",
                "You entered a failed hardware component but the outcome is not 'Hardware failure confirmed'. "
                "Verify the diagnostic evidence or change the outcome."
            )
            return
        if outcome == "Hardware failure confirmed" and hardware_is_unconfirmed:
            QMessageBox.warning(self, "Hardware name required", "Enter the hardware name and diagnostic evidence supporting the confirmed failure.")
            return
        if outcome == "Hardware failure confirmed" and not any(
            "diagnostic" in step.lower() and answer == "NO" and observation
            for step, answer, observation in self.assist_step_results
        ):
            QMessageBox.warning(
                self, "Diagnostic evidence required",
                "Record a failed Lenovo diagnostic step and its exact error code or result before confirming a hardware failure."
            )
            return
        step_text = "\n".join(
            f"- {i}. {step}\n  Agent response: {answer}"
            + (f"\n  Observation / diagnostic code: {observation}" if observation else "")
            for i, (step, answer, observation) in enumerate(self.assist_step_results, 1)
        ) or "No troubleshooting steps recorded."
        transcript = self.assist_live_transcript.toPlainText().strip() or "No live transcript available."
        source_note = (
            "Source note: Lenovo PC Support lookup was performed. Verify each retrieved or manually entered value against the portal before case submission.\n\n"
            if self.assist_machine_lookup_complete
            else "Source note: Device details were entered manually; verify all values against Lenovo PC Support.\n\n"
        )
        summary = (
            "LENOVO CEC — AGENT ASSIST CALL SUMMARY\n"
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Agent: {self.user_email}\n"
            f"Serial number: {serial}\n"
            f"Machine description / product: {self.assist_product.text().strip() or 'Not retrieved'}\n"
            f"MTM: {self.assist_mtm.text().strip() or 'Not retrieved'}\n"
            f"Configuration: {self.assist_configuration.toPlainText().strip() or 'Not retrieved'}\n"
            f"Warranty start: {self.assist_warranty_start.text().strip() or 'Not retrieved'}\n"
            f"Warranty end: {self.assist_warranty_end.text().strip() or 'Not retrieved'}\n"
            f"Last repair ticket: {self.assist_last_ticket.toPlainText().strip() or 'Not retrieved'}\n"
            + source_note
            + f"Customer mobile: {mobile or 'Not captured'}\n"
            f"Customer email: {email or 'Not captured'}\n\n"
            f"Customer issue: {self.assist_issue.toPlainText().strip() or 'Not captured'}\n"
            f"Evidence files: {', '.join(os.path.basename(p) for p in self.assist_evidence_paths) or 'None attached'}\n\n"
            "TROUBLESHOOTING RESULTS\n" + step_text + "\n\n"
            f"Root cause: {root_cause}\n"
            f"Suggested / faulty hardware: {hardware}\n"
            f"Suggested resolution: {self.assist_suggested_resolution.toPlainText().strip() or 'Not captured'}\n"
            f"Final outcome: {outcome}\n"
            f"Next action / notes: {self.assist_final_notes.toPlainText().strip() or 'Not captured'}\n\n"
            "LIVE TRANSCRIPT (for agent review)\n" + transcript + "\n"
        )
        self.assist_summary.setPlainText(summary)
        self.assist_summary.moveCursor(QTextCursor.Start)
        self.statusBar().showMessage("Agent Assist summary generated. Review it before copying into the official case system.")

    def init_master_vault_tab(self):
        vault_widget = QWidget()
        layout = QVBoxLayout(vault_widget)
       
        search_group = QGroupBox("🔍 Search & Filter Master Vault")
        search_layout = QHBoxLayout()
       
        self.date_picker = QDateEdit()
        self.date_picker.setCalendarPopup(True)
        self.date_picker.setDate(QDate.currentDate())
        self.date_picker.setStyleSheet("padding: 6px; background: #0D0D0F; border: 1px solid #3F3F46; border-radius: 6px; color: white;")
       
        self.chk_use_date = QPushButton("📅 Filter by Date [OFF]")
        self.chk_use_date.setCheckable(True)
        self.chk_use_date.setStyleSheet("background-color: #27272A; color: #9CA3AF; padding: 6px 10px;")
        self.chk_use_date.clicked.connect(self.toggle_date_filter_state)
       
        self.txt_search_agent = QLineEdit()
        self.txt_search_agent.setPlaceholderText("agent@lenovo.com")
        self.txt_search_agent.setStyleSheet("padding: 6px; background: #0D0D0F; border: 1px solid #3F3F46; border-radius: 6px; color: white;")
       
        self.txt_search_phone = QLineEdit()
        self.txt_search_phone.setPlaceholderText("Contact Number")
        self.txt_search_phone.setStyleSheet("padding: 6px; background: #0D0D0F; border: 1px solid #3F3F46; border-radius: 6px; color: white;")
       
        btn_search = QPushButton("🔍 Search")
        btn_search.setStyleSheet("background-color: #E2231A; color: white; padding: 6px 14px;")
        btn_search.clicked.connect(self.filter_vault_records)
       
        btn_clear = QPushButton("Reset")
        btn_clear.setStyleSheet("background-color: #27272A; color: white; padding: 6px 14px;")
        btn_clear.clicked.connect(self.refresh_vault_list)
       
        search_layout.addWidget(self.chk_use_date)
        search_layout.addWidget(self.date_picker)
        search_layout.addWidget(QLabel("Agent Email:"))
        search_layout.addWidget(self.txt_search_agent)
        search_layout.addWidget(QLabel("Contact No:"))
        search_layout.addWidget(self.txt_search_phone)
        search_layout.addWidget(btn_search)
        search_layout.addWidget(btn_clear)
        search_group.setLayout(search_layout)
        layout.addWidget(search_group)
 
        group = QGroupBox("👑 Master Admin Vault — All Agent Recordings (Synced via OneDrive)")
        vbox = QVBoxLayout()
       
        self.vault_table = QTableWidget(0, 8)
        self.vault_table.setHorizontalHeaderLabels([
            "Agent Name", "Contact Number", "Date & Time", "Call Duration",
            "Call Type", "▶️ Play", "📥 Download", "🗑 Delete"
        ])
        self.vault_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.vault_table.setAlternatingRowColors(True)
        self.vault_table.verticalHeader().setDefaultSectionSize(38)
        self.vault_table.setWordWrap(False)
        self.populate_vault_table()
       
        btn_layout = QHBoxLayout()
        btn_refresh = QPushButton("🔄 Refresh Vault")
        btn_refresh.setStyleSheet("background-color: #27272A; color: white;")
        btn_refresh.clicked.connect(self.refresh_vault_list)
       
        btn_layout.addWidget(btn_refresh)
        btn_layout.addStretch()
       
        vbox.addWidget(self.vault_table)
        vbox.addLayout(btn_layout)
        group.setLayout(vbox)
        layout.addWidget(group)
       
        self.tabs.addTab(vault_widget, "👑 Master Vault (All Calls)")
 
    def toggle_date_filter_state(self):
        if self.chk_use_date.isChecked():
            self.chk_use_date.setText("📅 Filter by Date [ON]")
            self.chk_use_date.setStyleSheet("background-color: #10B981; color: white; padding: 6px 10px;")
        else:
            self.chk_use_date.setText("📅 Filter by Date [OFF]")
            self.chk_use_date.setStyleSheet("background-color: #27272A; color: #9CA3AF; padding: 6px 10px;")
 
    def filter_vault_records(self):
        filter_date = self.date_picker.date().toString("yyyyMMdd") if self.chk_use_date.isChecked() else ""
        filter_agent = self.txt_search_agent.text().strip().lower()
        filter_phone = self.txt_search_phone.text().strip()
        self.populate_vault_table(filter_date, filter_agent, filter_phone)
        self.statusBar().showMessage("Vault filtered successfully.")
 
    def refresh_vault_list(self):
        self.chk_use_date.setChecked(False)
        self.toggle_date_filter_state()
        self.txt_search_agent.clear()
        self.txt_search_phone.clear()
        self.populate_vault_table()
        self.statusBar().showMessage("Master Vault refreshed successfully.")
 
    def populate_vault_table(self, f_date="", f_agent="", f_phone=""):
        self.vault_table.setRowCount(0)
        if not os.path.exists(VAULT_DIR):
            return
 
        row_data = []
        for agent_folder in sorted(os.listdir(VAULT_DIR), reverse=True):
            agent_email_mapped = agent_folder.replace("_at_", "@").replace("_", ".")
            if f_agent and f_agent not in agent_email_mapped.lower() and f_agent not in agent_folder.lower():
                continue
 
            agent_path = os.path.join(VAULT_DIR, agent_folder)
            if os.path.isdir(agent_path):
                for call_folder in sorted(os.listdir(agent_path), reverse=True):
                    call_path = os.path.join(agent_path, call_folder)
                    if os.path.isdir(call_path):
                        audio_path = os.path.join(call_path, "call_recording.wav")
                        if not os.path.isfile(audio_path):
                            continue
                        try:
                            with wave.open(audio_path, "rb") as recorded_audio:
                                if recorded_audio.getnframes() == 0:
                                    continue
                        except (wave.Error, OSError):
                            continue
                       
                        parts = call_folder.split("_")
                        cust_num = parts[1] if len(parts) > 1 else "Unknown"
                       
                        if f_phone and f_phone not in cust_num:
                            continue
 
                        dt_str = "N/A"
                        date_raw_check = ""
                        if len(parts) >= 4:
                            date_raw_check = parts[2]
                            raw_dt = f"{parts[2]} {parts[3]}"
                            try:
                                parsed_dt = datetime.strptime(raw_dt, "%Y%m%d %H%M%S")
                                dt_str = parsed_dt.strftime("%d-%b-%Y %H:%M:%S")
                            except:
                                dt_str = raw_dt
 
                        if f_date and f_date not in date_raw_check:
                            continue
 
                        duration_str = "00:00"
                        if os.path.exists(audio_path):
                            try:
                                with wave.open(audio_path, 'rb') as wf:
                                    frames = wf.getnframes()
                                    rate = wf.getframerate()
                                    seconds = int(frames / rate if rate > 0 else 0)
                                    mins, secs = divmod(seconds, 60)
                                    duration_str = f"{mins:02d}:{secs:02d}"
                            except:
                                pass
 
                        call_type = "Incoming"
                        report_path = os.path.join(call_path, "case_report.txt")
                        if os.path.exists(report_path):
                            with open(report_path, "r", encoding="utf-8") as rf:
                                content = rf.read()
                                if "Offline" in content:
                                    call_type = "Offline"
 
                        row_data.append((agent_email_mapped, cust_num, dt_str, duration_str, call_type, audio_path))
 
        self.vault_table.setRowCount(len(row_data))
        for row_idx, (agent, cust, dt, duration, ctype, a_path) in enumerate(row_data):
            self.vault_table.setItem(row_idx, 0, QTableWidgetItem(agent))
            self.vault_table.setItem(row_idx, 1, QTableWidgetItem(cust))
            self.vault_table.setItem(row_idx, 2, QTableWidgetItem(dt))
            self.vault_table.setItem(row_idx, 3, QTableWidgetItem(duration))
           
            type_item = QTableWidgetItem(ctype)
            type_item.setForeground(QColor("#10B981" if ctype == "Incoming" else "#F59E0B"))
            self.vault_table.setItem(row_idx, 4, type_item)
           
            btn_p = QPushButton("▶️ Play")
            btn_p.setStyleSheet("background-color: #10B981; color: white; padding: 4px;")
            btn_p.clicked.connect(lambda checked=False, path=a_path: self.play_specific_audio(path))
            self.vault_table.setCellWidget(row_idx, 5, btn_p)
           
            btn_d = QPushButton("📥 Download")
            btn_d.setStyleSheet("background-color: #E2231A; color: white; padding: 4px;")
            btn_d.clicked.connect(lambda checked=False, path=a_path, c=cust: self.download_specific_audio(path, c))
            self.vault_table.setCellWidget(row_idx, 6, btn_d)

            if self.is_master:
                btn_delete = QPushButton("🗑 Delete")
                btn_delete.setStyleSheet("background-color: #991B1B; color: white; padding: 4px;")
                btn_delete.clicked.connect(
                    lambda checked=False, path=a_path: self.delete_vault_call(path)
                )
                self.vault_table.setCellWidget(row_idx, 7, btn_delete)

    def play_specific_audio(self, path):
        if not os.path.isfile(path):
            QMessageBox.warning(self, "Playback Error", f"Audio file not found:\n{path}")
            return
        try:
            # Reject incomplete/invalid WAV files before passing them to Qt.
            if path.lower().endswith(".wav"):
                with wave.open(path, "rb") as wf:
                    frames = wf.getnframes()
                    rate = wf.getframerate()
                    channels = wf.getnchannels()
                    if frames <= 0 or rate <= 0:
                        raise ValueError("The WAV file contains no playable audio frames.")
                    duration = frames / rate
            else:
                duration = None
            self.show_playback_controls()
            self.media_player.stop()
            self.media_player.setSource(QUrl.fromLocalFile(os.path.abspath(path)))
            self.media_player.play()
            if duration is None:
                self.statusBar().showMessage(f"Loading audio: {os.path.basename(path)}")
            else:
                self.statusBar().showMessage(
                    f"Playing {os.path.basename(path)} | {duration:.1f}s | {channels} channel(s)"
                )
        except Exception as exc:
            QMessageBox.warning(self, "Playback Error", f"Could not open this recording:\n{exc}")

    def handle_playback_error(self, error, error_string=""):
        details = error_string or self.media_player.errorString() or str(error)
        self.statusBar().showMessage(f"Playback error: {details}")
        if details:
            QMessageBox.warning(self, "Playback Error", f"Qt could not play this audio file:\n{details}")

    def show_playback_controls(self):
        if self.playback_dialog is None:
            dialog = QDialog(self)
            dialog.setWindowTitle("Call Recording Playback")
            dialog.setWindowFlags(dialog.windowFlags() | Qt.Tool)
            dialog.setFixedSize(440, 145)
            layout = QVBoxLayout(dialog)
            layout.setContentsMargins(14, 12, 14, 10)

            self.playback_time_label = QLabel("00:00 / 00:00")
            layout.addWidget(self.playback_time_label)
            self.playback_slider = QSlider(Qt.Horizontal)
            self.playback_slider.setRange(0, 0)
            self.playback_slider.sliderMoved.connect(self.media_player.setPosition)
            layout.addWidget(self.playback_slider)

            controls = QHBoxLayout()
            back_button = QPushButton("⏪ 10 sec")
            back_button.clicked.connect(
                lambda: self.media_player.setPosition(
                    max(0, self.media_player.position() - 10000)
                )
            )
            self.playback_button = QPushButton("▶ Play")
            self.playback_button.clicked.connect(self.toggle_audio_playback)
            forward_button = QPushButton("10 sec ⏩")
            forward_button.clicked.connect(
                lambda: self.media_player.setPosition(
                    min(
                        self.media_player.duration(),
                        self.media_player.position() + 10000,
                    )
                )
            )
            end_button = QPushButton("⏹ End Playback")
            end_button.clicked.connect(self.end_audio_playback)
            controls.addWidget(back_button)
            controls.addWidget(self.playback_button)
            controls.addWidget(forward_button)
            controls.addWidget(end_button)
            layout.addLayout(controls)
            dialog.finished.connect(self.end_audio_playback)
            self.playback_dialog = dialog

        self.playback_dialog.show()
        self.playback_dialog.raise_()
        self.playback_dialog.activateWindow()

    def format_playback_time(self, milliseconds):
        seconds = max(0, milliseconds // 1000)
        return f"{seconds // 60:02d}:{seconds % 60:02d}"

    def update_playback_position(self, position):
        if self.playback_slider is not None and not self.playback_slider.isSliderDown():
            self.playback_slider.setValue(position)
        if self.playback_time_label is not None:
            self.playback_time_label.setText(
                f"{self.format_playback_time(position)} / "
                f"{self.format_playback_time(self.playback_duration_ms)}"
            )

    def update_playback_duration(self, duration):
        self.playback_duration_ms = duration
        if self.playback_slider is not None:
            self.playback_slider.setRange(0, max(0, duration))
        self.update_playback_position(self.media_player.position())

    def update_playback_button(self, state):
        if self.playback_button is not None:
            self.playback_button.setText(
                "⏸ Pause" if state == QMediaPlayer.PlaybackState.PlayingState else "▶ Play"
            )

    def toggle_audio_playback(self):
        if self.media_player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.media_player.pause()
        else:
            self.media_player.play()

    def end_audio_playback(self):
        self.media_player.stop()
        self.media_player.setPosition(0)

    def delete_vault_call(self, audio_path):
        if not self.is_master:
            QMessageBox.warning(self, "Permission Denied", "Only master access can delete call recordings.")
            return

        vault_root = Path(VAULT_DIR).resolve()
        requested_recording_path = Path(audio_path)
        requested_call_folder = requested_recording_path.parent
        recording_path = requested_recording_path.resolve()
        call_folder = recording_path.parent
        if (
            requested_recording_path.name != "call_recording.wav"
            or recording_path.name != "call_recording.wav"
            or not requested_call_folder.name.startswith("Call_")
            or requested_call_folder.is_symlink()
            or call_folder.parent.parent != vault_root
            or not recording_path.is_file()
        ):
            QMessageBox.warning(self, "Delete Failed", "The selected recording is outside the managed call vault.")
            return

        answer = QMessageBox.question(
            self,
            "Delete Call Folder",
            "Permanently delete this call folder, including its recording and case report?\n\n"
            f"{call_folder.name}\n\n"
            "If this folder is OneDrive-synced, OneDrive will sync the deletion when online.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return

        try:
            if self.media_player.source().toLocalFile() == str(recording_path):
                self.media_player.stop()
                self.media_player.setSource(QUrl())
            shutil.rmtree(call_folder, onerror=_retry_removing_readonly_path)
        except OSError as exc:
            QMessageBox.warning(
                self,
                "Delete Failed",
                "Windows still denied access after retrying read-only items.\n\n"
                "Close the recording or any app using files in this call folder, wait for "
                "OneDrive to finish syncing, then try again. Check that you have permission "
                "to delete the folder. Some files may already have been removed; refresh the "
                "vault before retrying.\n\n"
                f"Details: {exc}",
            )
            return

        self.populate_vault_table()
        self.refresh_recent_calls()
        self.statusBar().showMessage(
            "Call folder deleted from the local vault. OneDrive will sync the deletion when available."
        )
 
    def download_specific_audio(self, path, cust_num):
        if os.path.exists(path):
            save_path, _ = QFileDialog.getSaveFileName(self, "Download Recording", f"Call_{cust_num}_recording.wav", "Audio Files (*.wav)")
            if save_path:
                shutil.copy(path, save_path)
                QMessageBox.information(self, "Success", "Recording downloaded successfully!")
        else:
            QMessageBox.warning(self, "Error", "Audio file not found.")
 
    def download_current_recording(self):
        target_file = self.audio_file_path if (self.audio_file_path and os.path.exists(self.audio_file_path)) else "live_call_recording.wav"
        if os.path.exists(target_file):
            save_path, _ = QFileDialog.getSaveFileName(self, "Download Call Recording", "Lenovo_Call_Recording.wav", "Audio Files (*.wav)")
            if save_path:
                shutil.copy(target_file, save_path)
                QMessageBox.information(self, "Success", "Call recording downloaded successfully!")
        else:
            QMessageBox.warning(self, "Notice", "Please record or complete an audio session first.")
 
    def copy_to_clipboard(self, text_widget):
        QApplication.clipboard().setText(text_widget.toPlainText())
        QMessageBox.information(self, "Success", "Copied to clipboard!")
 
    def reset_application(self):
        self.is_recording = False
        self.has_transcript = False
        self.call_source_type = "Incoming"
        self.current_customer_number = "Unknown_Customer"
        self.btn_record.setText("Start Recording Call")
        self.btn_record.setStyleSheet("background-color: #E2231A; color: white;")
        self.lbl_timer.setText(f"Agent: {self.user_email} | Status: Idle")
        self.txt_live_transcript.clear()
        self.txt_transcript.clear()
        self.txt_case_template.clear()
        self.txt_email_draft.clear()
        self.reset_compliance_checklist()
        self.statusBar().showMessage("Workspace reset successfully.")
 
    def toggle_recording(self):
        if not self.is_recording:
            self.has_transcript = False
            cust_num, ok = QInputDialog.getText(self, "Customer Details", "Enter Customer Contact Number:")
            if not ok or not cust_num.strip():
                cust_num = "Unknown_Customer"
            else:
                cust_num = cust_num.strip()
 
            self.current_customer_number = cust_num
            self.call_source_type = "Incoming"
            self.agent_call_folder = ""
            self.txt_live_transcript.clear()
            self.txt_transcript.clear()
            self.txt_case_template.clear()
            self.txt_email_draft.clear()
            self.reset_compliance_checklist()

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_agent_email = self.user_email.replace("@", "_at_").replace(".", "_")
            self.agent_call_folder = os.path.join(VAULT_DIR, safe_agent_email, f"Call_{cust_num}_{timestamp}")
            self.audio_file_path = os.path.join(self.agent_call_folder, "call_recording.wav")

            self.recorder = DualAudioRecorderThread(self.audio_file_path)
            self.recorder.start()
            if not self.recorder.ready_event.wait(timeout=10):
                self.recorder.stop()
                self.recorder.join()
                QMessageBox.critical(self, "Recording Failed", "Audio devices did not start within 10 seconds.")
                self.statusBar().showMessage("Recording failed to start.")
                return
            if self.recorder.error:
                self.recorder.join()
                QMessageBox.critical(self, "Recording Failed", self.recorder.error)
                self.statusBar().showMessage(f"Recording failed: {self.recorder.error}")
                return

            self.is_recording = True
            self.btn_record.setText("Stop & Process Recording")
            self.btn_record.setStyleSheet("background-color: #DC2626; color: white;")
            self.lbl_timer.setText(f"Agent: {self.user_email} | Status: Recording Live ({cust_num})...")
            self.statusBar().showMessage(
                "Recording both call-audio loopback and microphone; live English translation is starting."
            )
            self.live_transcription_thread = threading.Thread(
                target=self.poll_live_transcription, daemon=True
            )
            self.live_transcription_thread.start()
        else:
            if self.recorder:
                self.recorder.stop()
                self.recorder.join()
           
            self.is_recording = False
            self.btn_record.setText("Start Recording Call")
            self.btn_record.setStyleSheet("background-color: #E2231A; color: white;")
            self.lbl_timer.setText(f"Agent: {self.user_email} | Status: Processing Final AI Audit...")

            if self.recorder and self.recorder.error:
                self.handle_processing_error(f"Recording failed: {self.recorder.error}")
                return
            if self.audio_file_path and os.path.isfile(self.audio_file_path):
                # Lift quiet loopback/customer audio independently while retaining
                # the two channels needed for speaker separation.
                normalize_stereo_wav(self.audio_file_path)
            if not self.recorder or self.recorder.frames_written == 0:
                self.handle_processing_error("No audio was captured. Check the microphone and call-audio devices, then try again.")
                return
            try:
                with wave.open(self.audio_file_path, "rb") as recorded:
                    captured_seconds = recorded.getnframes() / max(1, recorded.getframerate())
                    captured_frames = recorded.getnframes()
                if captured_seconds < 1:
                    self.handle_processing_error(
                        f"The recording is only {captured_seconds:.2f} seconds long. "
                        "The audio device stopped supplying samples; check Windows microphone/privacy settings and the selected playback device."
                    )
                    return
                self.statusBar().showMessage(
                    f"Recording saved: {captured_seconds:.1f} seconds ({captured_frames:,} frames). Starting transcription..."
                )
            except (wave.Error, OSError) as exc:
                self.handle_processing_error(f"The saved WAV file is incomplete or unreadable: {exc}")
                return

            self.refresh_recent_calls()
            if self.is_master:
                self.populate_vault_table()
            threading.Thread(target=self.run_local_transcription, args=(self.audio_file_path,)).start()

    def poll_live_transcription(self):
        recorder = self.recorder
        if recorder is None:
            return
        while self.is_recording and not self.model_ready_event.wait(timeout=0.25):
            pass
        if not self.is_recording:
            return
        if self.model is None:
            self.signals.status.emit(
                f"Live translation is unavailable: {self.model_error}"
            )
            return

        sample_rate = recorder.sample_rate
        rolling_audio = np.empty((0, 2), dtype=np.int16)
        rolling_window_frames = sample_rate * 12
        total_frames = 0
        next_update_frame = sample_rate * 6
        live_segments = []

        while self.is_recording:
            try:
                new_audio = recorder.live_audio_queue.get(timeout=0.25)
            except queue.Empty:
                continue

            total_frames += len(new_audio)
            rolling_audio = np.concatenate((rolling_audio, new_audio), axis=0)
            if len(rolling_audio) > rolling_window_frames:
                rolling_audio = rolling_audio[-rolling_window_frames:]

            if self.model is None or total_frames < next_update_frame:
                continue

            window_start = max(0, total_frames - len(rolling_audio)) / sample_rate
            try:
                with self.transcription_lock:
                    live_lines = transcribe_audio_buffer(
                        self.model,
                        rolling_audio,
                        sample_rate,
                        beam_size=1,
                        time_offset=window_start,
                    )
                if not self.is_recording:
                    break

                if live_lines:
                    stable_lines = []
                    for line in live_segments:
                        header = line.partition("] ")[0].lstrip("[")
                        try:
                            start_seconds = int(header.rstrip("s"))
                        except ValueError:
                            continue
                        if start_seconds < window_start:
                            stable_lines.append(line)
                    live_segments = sorted(
                        stable_lines + live_lines,
                        key=lambda line: int(line.partition("] ")[0].lstrip("[").rstrip("s"))
                        if line.partition("] ")[0].lstrip("[").rstrip("s").isdigit()
                        else 0,
                    )
                    self.signals.live_update.emit("\n".join(live_segments))
                    self.signals.status.emit(
                        "Live transcript updated in English from both audio channels."
                    )
            except Exception as e:
                self.signals.status.emit(f"Live transcription failed: {e}")
            finally:
                next_update_frame = total_frames + sample_rate * 4

    def update_live_transcript(self, text):
        self.txt_live_transcript.setPlainText(text)
        self.txt_live_transcript.moveCursor(QTextCursor.End)
        if hasattr(self, "assist_live_transcript"):
            self.assist_live_transcript.setPlainText(text)
            self.assist_live_transcript.moveCursor(QTextCursor.End)
        if text.strip():
            self.has_transcript = True
            _, _, evidence = build_call_outputs(
                text,
                self.user_email,
                self.current_customer_number,
                self.call_source_type,
            )
            self.populate_compliance_checklist(evidence)
 
    def upload_offline_call(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Audio Recording", "", "Audio Files (*.wav *.mp3 *.m4a)")
        if file_name:
            self.has_transcript = False
            self.reset_compliance_checklist()
            self.txt_transcript.clear()
            self.txt_case_template.clear()
            self.txt_email_draft.clear()
            cust_num, ok = QInputDialog.getText(self, "Customer Details", "Enter Customer Contact Number for this Audit:")
            if not ok or not cust_num.strip():
                cust_num = "Offline_Customer"
            else:
                cust_num = cust_num.strip()
 
            self.current_customer_number = cust_num
            self.call_source_type = "Offline"
            self.agent_call_folder = ""
            self.audio_file_path = file_name
            self.lbl_uploaded_file.setText(f"Loaded: {os.path.basename(file_name)} ({cust_num})")
            self.statusBar().showMessage("Processing uploaded offline audio...")
            threading.Thread(target=self.run_local_transcription, args=(file_name,)).start()
 
    def run_local_transcription(self, audio_path):
        try:
            if not self.model_ready_event.wait(timeout=180):
                raise RuntimeError("The speech-to-text model is still loading after 3 minutes.")
            if self.model is None:
                raise RuntimeError(
                    f"The speech-to-text model did not become available: {self.model_error}"
                )
            if not os.path.isfile(audio_path):
                raise FileNotFoundError(f"Recording not found: {audio_path}")
            with self.transcription_lock:
                transcript_lines = transcribe_call_audio(self.model, audio_path, beam_size=5)
            if not transcript_lines:
                raise RuntimeError("No speech was detected. No case notes or email draft were generated.")
        except Exception as e:
            self.signals.failed.emit(str(e))
            return

        self.signals.finished.emit("\n".join(transcript_lines), transcript_lines)

    def reset_compliance_checklist(self):
        for row in range(self.checklist_table.rowCount()):
            item = QTableWidgetItem("🟡 PENDING")
            item.setForeground(QColor("#F59E0B"))
            self.checklist_table.setItem(row, 1, item)
            self.checklist_table.setItem(
                row, 2, QTableWidgetItem("Requires manual confirmation from the call.")
            )

    def populate_compliance_checklist(self, evidence_by_parameter):
        for row in range(self.checklist_table.rowCount()):
            evidence = evidence_by_parameter.get(
                self.checklist_table.item(row, 0).text()
            )
            status = QTableWidgetItem("🟢 PASS" if evidence else "🔴 FAIL")
            status.setForeground(QColor("#10B981" if evidence else "#EF4444"))
            self.checklist_table.setItem(row, 1, status)
            evidence_item = QTableWidgetItem(
                (evidence or "No matching evidence found in the transcript.")
                .replace("\n", " ")
            )
            evidence_item.setToolTip(
                evidence or "No matching evidence found in the transcript."
            )
            self.checklist_table.setItem(row, 2, evidence_item)
            self.checklist_table.setRowHeight(row, 32)

    def handle_processing_error(self, message):
        self.is_recording = False
        self.has_transcript = False
        self.txt_transcript.setPlainText(f"Recording/transcription failed: {message}")
        self.txt_live_transcript.setPlainText(f"Recording/transcription failed: {message}")
        if hasattr(self, "assist_live_transcript"):
            self.assist_live_transcript.setPlainText(f"Recording/transcription failed: {message}")
        self.txt_case_template.clear()
        self.txt_email_draft.clear()
        self.reset_compliance_checklist()
        self.lbl_timer.setText(f"Agent: {self.user_email} | Status: Recording/Transcription Failed")
        self.statusBar().showMessage(message)
        QMessageBox.warning(self, "Recording/Transcription Failed", message)
 
    def handle_transcription_results(self, transcript, segments_list):
        self.txt_transcript.setPlainText(transcript)
        self.txt_live_transcript.setPlainText(transcript)
        if hasattr(self, "assist_live_transcript"):
            self.assist_live_transcript.setPlainText(transcript)
        try:
            case_template, email_draft, compliance_evidence = build_call_outputs(
                transcript,
                self.user_email,
                self.current_customer_number,
                self.call_source_type,
            )
        except ValueError as e:
            self.handle_processing_error(str(e))
            return

        self.has_transcript = True
        self.txt_case_template.setPlainText(case_template)
        self.txt_email_draft.setPlainText(email_draft)
        self.populate_compliance_checklist(
            compliance_evidence,
        )

        safe_agent_email = self.user_email.replace("@", "_at_").replace(".", "_")
        report_saved = False
        try:
            if not self.agent_call_folder:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                self.agent_call_folder = os.path.join(
                    VAULT_DIR, safe_agent_email,
                    f"Call_{self.current_customer_number}_{timestamp}",
                )
            os.makedirs(self.agent_call_folder, exist_ok=True)
            permanent_audio_path = os.path.join(self.agent_call_folder, "call_recording.wav")
            if os.path.exists(self.audio_file_path) and self.audio_file_path != permanent_audio_path:
                shutil.copy(self.audio_file_path, permanent_audio_path)
                self.audio_file_path = permanent_audio_path

            with open(
                os.path.join(self.agent_call_folder, "case_report.txt"),
                "w",
                encoding="utf-8",
            ) as report_file:
                report_file.write(case_template + "\n\n" + email_draft)
            report_saved = True
        except OSError as e:
            self.statusBar().showMessage(f"Could not save the transcript: {e}")
            QMessageBox.warning(
                self,
                "Save Failed",
                "The case template and email draft were generated and remain available "
                f"in the workspace, but could not be saved to the vault:\n{e}",
            )

        if self.is_master:
            self.populate_vault_table()
        self.refresh_recent_calls()
        self.lbl_timer.setText(f"Agent: {self.user_email} | Status: Analyzed")
        if report_saved:
            self.statusBar().showMessage("Transcript-based case notes and email draft saved.")
 
    def export_data(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Report", "Lenovo_Case_Summary.txt", "Text Files (*.txt)")
        if file_path:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(
                    self.txt_case_template.toPlainText()
                    + "\n\n--- CUSTOMER EMAIL DRAFT ---\n\n"
                    + self.txt_email_draft.toPlainText()
                )
            QMessageBox.information(self, "Success", "Report exported successfully!")
 
if __name__ == "__main__":
    app = QApplication(sys.argv)
   
    login = LoginDialog()
    if login.exec() == QDialog.Accepted:
        window = CECSmartAssistApp(login.user_email, login.is_master)
        window.showMaximized()
        sys.exit(app.exec())