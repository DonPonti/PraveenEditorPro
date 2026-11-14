# Praveen Editor v1.4 — Audio Mix Studio

A local Windows desktop video editor built with Python, PySide6 and FFmpeg.

## What's new in v1.4

### Audio Mix tab
- **Original clip audio:** include or mute the audio recorded by video clips.
- **Clip volume:** set camera/dialogue audio level from 0× to 2×.
- **Voice-over / narration:** add one external audio file (WAV, MP3, M4A, AAC, FLAC or OGG; video containers with an audio stream are also accepted).
- **Voice-over timing:** choose when the narration starts on the finished timeline.
- **Voice-over volume:** set narration level from 0× to 2×.
- **Music volume:** independently set background music from 0× to 2×.
- **Automatic music ducking:** FFmpeg side-chain compression lowers the music while narration or camera dialogue is present.
- All audio processing and final rendering happen locally on your computer.

### Audio behavior / current limits
- Background music is still required by the current export workflow.
- The voice-over track is one continuous file per project. Multiple independently placed narration clips, keyframe volume automation and waveform editing are future upgrades.
- Clip audio follows normal clip speed changes. Speed-ramped clips currently use normal-speed source audio rather than a matching audio speed ramp.
- Audio from clips is aligned to the clip's timeline position; still photos have no source audio.
- Ducking is automatic compression, not AI speech detection. It reacts to the selected dialogue/voice-over audio signal.

## Windows setup

Open PowerShell in this folder:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python app.py
```

FFmpeg and FFprobe must be available on PATH. If `ffmpeg -version` works in PowerShell, that part is ready.

For local automatic captions (optional):

```powershell
python -m pip install faster-whisper
```

The first Whisper run downloads model weights. This is separate from Gemini and does not upload your footage.

## AI Director — story-led automatic editing

The AI Creator tab now includes an **AI Director** workflow. Add your source clips/photos,
choose music in Settings, enter a creative brief, and select a target duration and editing
direction. AI Director samples representative frames locally, sends those still frames plus
your brief to Gemini, validates the returned shot plan, and builds an editable timeline in
the existing editor. Review the timeline, adjust individual clips, then preview or export.

Examples of briefs:
- "Make a 30-second cinematic travel film. Start with the strongest landscape, build a
  feeling of discovery, and finish on the most memorable view. Use restrained cuts."
- "Create a fast 20-second product teaser. Open on the clearest product shot, show details,
  build energy, and end with a clean product hero shot."
- "Make an emotional family-memory montage. Keep the pacing gentle and avoid flashy effects."

**Privacy:** original video and audio files are not uploaded. The app sends a limited set of
sampled JPEG frames and the written brief to Gemini for visual shot selection. Rendering and
the original media stay on your PC. The AI plan is a first cut, not a guarantee of perfect
story interpretation; review it before publishing.

The AI Director uses the existing Gemini API key field and the `google-genai` package.
The model configured in `ai_director.py` is `gemini-2.5-flash`.

## Gemini Creator Assistant

The AI Creator tab uses the Google Gen AI Python SDK. Set an API key in the app or as `GEMINI_API_KEY` / `GOOGLE_API_KEY`. Gemini text generation uses API access and its own usage limits; a Gemini app subscription does not automatically include API quota. Video/audio editing and rendering remain local.

## Launch

Double-click `Run Editor.bat`, or run `python app.py` from PowerShell.

## Compatibility

Existing project JSON files remain loadable. New audio fields use safe defaults when opening older projects.
