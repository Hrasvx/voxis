<div align="center">

# VOXIS

### Audio, made spatial.

Voxis transforms sound into a living 3D network of spectral peaks, acoustic
relationships, and persistent history.

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![OpenGL](https://img.shields.io/badge/OpenGL-3.3%2B-5586A4?style=for-the-badge&logo=opengl&logoColor=white)](https://www.opengl.org/)
[![License](https://img.shields.io/badge/License-MIT-8A63FF?style=for-the-badge)](LICENSE)
[![Stars](https://img.shields.io/github/stars/Hrasvx/voxis?style=for-the-badge&logo=github&color=FFE779)](https://github.com/Hrasvx/voxis/stargazers)

**[Features](#features) · [Install](#requirements) · [Usage](#preview-usage) · [Export](#mp4-export) · [Star history](#star-history)**

</div>

---

## Example

<div align="center">

<img src="assets/voxis-output.gif" alt="Voxis visualization" width="100%">


</div>

---

Voxis turns the audio track from a video or audio file into a standalone,
3D point mapping network. It's only usage is to look cool lol

## Architecture and visual mapping

1. **Desktop and playback:** PySide6 supplies the responsive interface,
   `QMediaPlayer` audio playback, volume, transport controls.
2. **Decode and analysis:** a dedicated PyAV worker decodes only the selected
   audio stream into overlapping STFT windows. Each window detects up to forty
   meaningful local peaks with magnitude, prominence, SNR, bandwidth, phase.
3. **Context and phrases:** MFCC-like coefficients, delta coefficients,
   centroid, bandwidth, contrast energy, flatness, rolloff, RMS, onset,
   dominant pitch, and harmonicity are standardized and embedded into a stable
   global 3D PCA space. Silence, onset, and spectral-change detection divide the
   recording into tracking segments.
4. **3D mapping:** selected peaks are emitted on a deterministic visual cadence
   instead of once per STFT hop. Log frequency shapes the vertical axis;
   centroid, timbre, stereo balance, bandwidth, and peak volume shape the
   horizontal axis; pitch, loudness, rolloff, harmonicity, and timbre shape
   depth.
5. **Typed point networks:** temporal tracking, intra-frame harmonic webs, and
   cross-time similarity links are built separately. A bounded spatial hash and
   strict per-type/total degree limits prevent all-pairs graph soup and duplicate
   edges.
6. **Preview synchronization:** the audio is pre-analyzed with progress before
   transport is enabled. A seek deterministically replays cached features so tracking
   and retained history match uninterrupted playback.
6. **Deterministic export:** export analyzes a finite feature cache, resets the
   configured random seed, advances the same point simulation at exact
   `frame_index / FPS` timestamps, and renders with a standalone ModernGL
   context. Raw RGB frames are streamed to FFmpeg, which takes audio only from
   the imported source.
7. **GPU rendering:** reusable ModernGL point/line buffers render up to 25,000
   visible nodes and 100,000 visible lines. White point cores, colored halos,
   neutral-gray typed edges, a faint 3D acoustic grid, fog, auto-fit camera
   framing, and model-space three-axis tumble keep the structure legible and
   volumetric.
8. **Overlays and post effects:** strongest-node labels and boxes
   remain visible throughout retained history and are shared by preview and
   offline export. Their percentage, maximum count, box size, text size, and
   opacity floor are adjustable. Logarithmic frequency legends, trails,
   restrained bloom, fine grain, flicker, vignette, and fog remain configurable.

## Features

- Imports MP4, MOV, MKV, AVI, WebM, MP3, WAV, FLAC, AAC, OGG, and M4A when their
  codecs are available.
- Ignores video imagery completely.
- Audio-synchronized play, pause, stop, replay, and seeking.
- Perspective GPU rendering with multi-axis object tumble, fog, point scaling,
  camera drift, mouse rotation, and wheel zoom.
- Adaptive 0–40 peak creation per STFT frame with true silence gating.
- Acoustic clouds, branches, fans, long-lived history, typed edges,
  bounded approximate-neighbor searches, and adaptive preview density.
- Auto-fit framing, manual orbit/pan/zoom, reset, and slow non-identical X/Y/Z
  rotation with deterministic modulation.
- Full-screen abstract preview.
- Deterministic offline MP4 export at custom even dimensions, 24/30/60 FPS, and
  custom bitrate.
- Export progress, cancellation, partial-file cleanup, disk-space validation,
  FFmpeg validation, and clear codec/GPU errors.
- Persistent live settings and six visual presets.

## Presets

- **Full Spectral Rainbow (default):** violet/magenta through red, yellow, green,
  cyan, and near-white using logarithmic frequency.
- **Birdsong Spectral:** a bright high-frequency cyan/white birdsong palette.
- **Scientific Neon:** magenta, violet, cyan, green, yellow, and white.
- **Archival Gold:** amber, gold, orange, and white with stronger grain.
- **Monochrome Laboratory:** white/gray with a pale green monitor cast.
- **Deep Space:** dim blue, violet, white, and red with stronger fog and slower
  rotation.

## Requirements

- Python 3.11 or newer
- OpenGL 3.3 Core or newer
- FFmpeg with `libx264` and AAC encoding
- A Qt Multimedia backend capable of playing the source audio codec

## Alpine Linux installation

This is the installation path for Alpine/musl:

```sh
cd /home/user/audio-reactive-3d

doas apk add \
  py3-pyside6 py3-numpy py3-pytest py3-pip py3-setuptools \
  python3-dev build-base pkgconf \
  ffmpeg ffmpeg-dev \
  mesa-dri-gallium mesa-dev libx11-dev \
  qt6-qtmultimedia qt6-qtmultimedia-gstreamer

python3 -m venv --system-site-packages .venv
. .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install av moderngl glcontext
python -m pip install -e . --no-deps
```

## Other Linux distributions

Create a virtual environment first:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[test]"
```

Ubuntu/Debian multimedia packages:

```sh
sudo apt install ffmpeg libgl1-mesa-dri libegl1 \
  gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
  gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly
```

Fedora:

```sh
sudo dnf install ffmpeg mesa-dri-drivers \
  gstreamer1-plugins-base gstreamer1-plugins-good \
  gstreamer1-plugins-bad-free gstreamer1-plugins-ugly-free
```

Arch:

```sh
sudo pacman -S ffmpeg mesa gstreamer gst-plugins-base \
  gst-plugins-good gst-plugins-bad gst-plugins-ugly
```

On macOS install FFmpeg with `brew install ffmpeg`. On Windows, install FFmpeg
and place its `bin` directory on `PATH`.

## Run

```sh
cd /home/user/audio-reactive-3d
. .venv/bin/activate
voxis
```

Or run without the console script:

```sh
PYTHONPATH=src python -m voxis
```

## Preview usage

1. Select **Import file** and choose video or audio.
2. Use **Preview** to restart deterministically from zero, or use Play/Pause.
3. Wait for the one-time analysis progress to finish, then preview or seek.
   Seeking reconstructs the correct retained historical state from the cache.
4. Adjust any live parameter. Left-drag rotates, right/middle-drag pans, and the
   wheel zooms.
5. Use **Reset camera**, **Reset visualization**, F11, or double-click the
   preview as needed.

The visualization occupies the application window; there is deliberately no
source-video panel.

## MP4 export

1. Import a source with audio.
2. Select a preset and settings.
3. Choose **Export MP4**.
4. Select the output path, custom resolution, 24/30/60 FPS, and bitrate.
5. Watch analysis/render progress or choose **Cancel export**.

The default export is 1920×1080, 60 FPS, H.264 at 16 Mbps with stereo AAC.
Dimensions are normalized to even values for broadly compatible `yuv420p`.

## Settings

The UI updates all controls immediately:

- FFT/hop size, adaptive peak limits, prominence, noise/silence thresholds,
  smoothing, phrase sensitivity, and band influence;
- core/halo size, white-core brightness, fresh-node size/glow boost and settle
  time, visible count, active/history durations, persistent history, spread,
  and brightness;
- independent temporal/intra/similarity edge strength and caps, total degree,
  line count, thickness, brightness, opacity, and lifetime;
- X/Y/Z rotation, auto-fit occupancy/smoothing, new-node camera follow strength,
  responsiveness, speed, hold/return timing, distance limits, FOV, and fog;
- palette reversal, persistent scientific label percentage/count/box/text/
  opacity controls, preview/export frequency legends, and medium/high quality
  limits;
- glow/bloom/grain/flicker/trails/scanlines/chromatic separation/background;
- random seed, preview FPS limit, and analysis look-ahead.

Factory defaults are documented in `config/default.json`.

Higher **Sensitivity** generally admits more audible activity and therefore
more points. Higher **Peak prominence** or **Spectral noise floor** rejects more
peaks and therefore produces fewer points. Increasing **Node interval** also
reduces point creation. For strict ceilings, use **Maximum peaks / frame**,
**Visible point limit**, **Max lines per point**, and **Visible line limit**.

## Error handling and logs

Rotating diagnostic logs are stored at:

```text
~/.local/state/voxis/application.log
```

## Tests

```sh
pytest
```

Tests cover:

- FFT frequency, multi-peak creation, silence, spectral-shape, onset, stereo,
  and timestamp features;
- media-clock interpolation, seek generations, stale-frame dropping;
- temporal peak tracking, phrase-boundary gating, time/phrase-invariant
  positioning, typed similarity/harmonic webs, duplicate prevention, count and
  degree caps;
- deterministic point/network output and seeking reconstruction;
- FFmpeg mapping that excludes source video and includes source audio;
- absence of any runtime video decode/widget path and export dimension/FPS
  normalization.

A real integration smoke export using the supplied 20.295-second MOV produced a
20.300-second H.264/AAC MP4. Two identical runs produced byte-identical files,
proving fixed-timestep deterministic output on the tested machine.

## Star history

<div align="center">

<a href="https://www.star-history.com/Hrasvx/voxis">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=Hrasvx/voxis&type=Date&theme=dark">
    <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=Hrasvx/voxis&type=Date">
    <img alt="Voxis star history graph" src="https://api.star-history.com/svg?repos=Hrasvx/voxis&type=Date" width="700">
  </picture>
</a>

If Voxis is useful to you, consider leaving a star. It helps other people find
my project on GitHub :) .

</div>

---

<div align="center">

Inspired by Lucio Arese's project

</div>
