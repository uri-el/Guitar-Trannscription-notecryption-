# Notecryption

A desktop app that transcribes guitar audio into sheet music and guitar tabs.
Record live or load an audio file → get standard notation + TAB + MIDI output.

---

## Features

- **Live recording** with microphone level monitoring
- **File upload** — WAV, MP3, FLAC, etc.
- **Automatic transcription** using [Basic Pitch](https://github.com/spotify/basic-pitch) (local) or Klangio (cloud)
- **Score view** — scrollable sheet music rendered with OpenSheetMusicDisplay
- **Tab view** — guitar tablature with string/fret positions
- **Waveform view** — visual audio waveform
- **Playback** — synthesized guitar playback with moving playhead
- **Export** — MusicXML, MIDI, PDF (score or tab)
- Optional **source separation** via Demucs (isolates guitar before transcription)

---

## Requirements

- Python **3.10 – 3.11** (recommended; TensorFlow 2.15 does not support 3.12+)
- Windows / macOS / Linux

---

## Installation

### 1. Clone the repo

```bash
git clone https://github.com/yourname/notecryption.git
cd notecryption
```

### 2. Create a virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note:** `basic-pitch[tf]` pulls in TensorFlow. If you hit version conflicts,
> install with the pinned constraints file:
> ```bash
> pip install -r requirements.txt -c constraints.txt
> ```

### 4. Install PyQt6 WebEngine (if not pulled in automatically)

```bash
pip install PyQt6-WebEngine
```

### 5. (Optional) Source separation with Demucs

Demucs is used to isolate the guitar track before transcription.
It is installed via `requirements.txt` but requires **ffmpeg** on your PATH:

- **Windows**: download from https://ffmpeg.org/download.html and add to PATH
- **macOS**: `brew install ffmpeg`
- **Linux**: `sudo apt install ffmpeg`

---

## Running the App

```bash
python scripts/run_ui.py
```

### CLI (no GUI)

```bash
python scripts/run_pipeline.py --in audio.wav --out output/ --backend basic_pitch
```

| Flag | Description |
|---|---|
| `--in` | Input audio file path |
| `--out` | Output directory |
| `--backend` | `basic_pitch` (default) or `klangio` |
| `--use-llm` | Enable LLM note refinement |
| `--outputs` | Comma-separated: `musicxml,midi` |

---

## Project Structure

```
notecryption/
├── app/
│   ├── audio/          # Preprocessing, source separation
│   ├── cloud/          # Klangio API client
│   ├── export/         # MusicXML and MIDI export
│   ├── live/           # Real-time audio recording
│   ├── llm/            # LLM-based post-processing
│   ├── models/         # NoteEvent data model
│   ├── pipeline/       # Main transcription pipeline
│   ├── postprocess/    # Note cleanup and quantization
│   ├── transcription/  # Basic Pitch, guitar mapper, articulation
│   └── ui/             # Qt6 GUI (main window, views, player)
├── scripts/
│   ├── run_ui.py       # Launch GUI
│   └── run_pipeline.py # CLI entry point
├── requirements.txt
├── constraints.txt     # Pinned ML dependency versions
└── README.md
```

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'basic_pitch'`**
→ Run `pip install basic-pitch[tf]`

**`ImportError` for PyQt6 WebEngine**
→ Run `pip install PyQt6-WebEngine`

**TensorFlow version conflicts**
→ Use `pip install -r requirements.txt -c constraints.txt`

**No audio devices found**
→ Make sure your microphone is connected and not in use by another app.
→ On Linux, you may need `sudo apt install portaudio19-dev` before installing `sounddevice`.

**Demucs not working**
→ Ensure `ffmpeg` is installed and available in your system PATH.
