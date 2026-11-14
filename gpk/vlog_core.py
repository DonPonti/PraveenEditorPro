"""
vlog_core: the engine behind the Workout Vlog Editor
====================================================

Three separate pieces, so the desktop app can mix automatic and manual editing:

    auto_edit(...)      analyses footage + music and PLANS an edit  -> Project
    Project             the plan: a list of timeline Items + settings (saved as JSON)
    render_project(...) turns ANY Project (auto-made or hand-edited) into an MP4

Nothing in here prints or exits: errors raise EditorError and progress goes through
an optional callback cb(message, fraction_or_None), so a GUI can use it safely.
"""

import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field, asdict
from functools import lru_cache
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

VIDEO_EXT = {".mp4", ".mov", ".mkv", ".m4v", ".avi", ".webm"}
PHOTO_EXT = {".jpg", ".jpeg", ".png", ".webp"}
MUSIC_EXT = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg"}

STYLES = {
    "cinema": {
        "grade": "curves=m='0/0.035 0.5/0.5 1/0.97',eq=contrast=1.14:saturation=1.08,"
                 "colorbalance=rs=.05:gs=0:bs=.04:rm=-.02:bm=.05:rh=.06:gh=.01:bh=-.05,"
                 "vignette=PI/5",
        "bloom": 0.20,
    },
    "gritty": {
        "grade": "eq=contrast=1.18:saturation=0.85:brightness=-0.03:gamma=0.95,vignette=PI/5",
        "bloom": 0.0,
    },
    "clean": {
        "grade": "eq=contrast=1.08:saturation=1.15:brightness=0.02,vignette=PI/7",
        "bloom": 0.08,
    },
    "warm": {
        "grade": "eq=contrast=1.12:saturation=1.1,"
                 "colorbalance=rs=.06:gs=.01:bs=-.06:rm=.05:bm=-.04,vignette=PI/6",
        "bloom": 0.15,
    },
    "teal": {
        "grade": "eq=contrast=1.15:saturation=1.2,"
                 "colorbalance=rs=.07:bs=-.05:rh=.05:bh=-.03:gs=-.02,"
                 "colorbalance=bs=.05:bm=.04,vignette=PI/6",
        "bloom": 0.12,
    },
}

MOVE_CHOICES = ["none", "push", "pull", "punch", "shake"]
SECTION_NAMES = ["Intro", "Build", "Peak", "Cooldown"]

MOVES = {
    0: ["push", "pull"],
    1: ["push", "punch", "pull"],
    2: ["punch", "shake", "punch", "push"],
    3: ["pull", "push"],
}

Callback = Callable[[str, Optional[float]], None]


class EditorError(Exception):
    """Something the user can fix (missing file, bad input, ffmpeg failure...)."""


# ----------------------------------------------------------------------------
# Project model
# ----------------------------------------------------------------------------
@dataclass
class Item:
    """One clip/photo on the timeline."""
    source: str
    kind: str = "video"            # "video" | "photo"
    src_start: float = 0.0         # seconds into the source (video only)
    duration: float = 1.0          # length on the timeline, in seconds
    speed: float = 1.0             # <1 slow-mo, >1 faster (video only)
    section: int = 1               # 0 intro, 1 build, 2 peak, 3 cooldown (cosmetic + sfx)
    hero: bool = False             # the "drop" shot: impact + riser sound
    move: Optional[str] = None     # None | push | pull | punch | shake
    flash: bool = False
    glitch: bool = False
    whip: bool = False
    ramp: bool = False             # slow->fast speed ramp
    dip_in: bool = False
    dip_out: bool = False
    reframe: str = ""              # "" = project default | smart | fit | fill
    focus_x: float = 0.5           # where the subject is horizontally (0 left .. 1 right)


@dataclass
class Project:
    music: str = ""
    music_offset: float = 0.0
    style: str = "cinema"
    lut: str = ""
    effects: str = "hype"          # off | subtle | hype
    title: str = ""
    subtitle: str = ""
    outro: str = ""
    format: str = "vertical"       # vertical | horizontal
    fps: int = 30
    sfx: bool = True
    reframe: str = "smart"         # how landscape clips become vertical: smart | fit | fill
    move_strength: float = 1.0     # camera move (zoom) strength, 0 = none
    items: list = field(default_factory=list)
    captions: list = field(default_factory=list)  # timed subtitle cues: {start, end, text}
    caption_style: str = "bold"  # bold | clean | minimal
    # Audio mix: music remains the required bed; dialogue and voice-over are optional.
    voiceover: str = ""
    voiceover_offset: float = 0.0
    voiceover_gain: float = 1.0
    music_gain: float = 0.90
    use_clip_audio: bool = True
    clip_audio_gain: float = 1.0
    duck_music: bool = True

    def total_length(self):
        return sum(i.duration for i in self.items)

    def to_json(self):
        return json.dumps(asdict(self), indent=2)

    def save(self, path):
        Path(path).write_text(self.to_json(), encoding="utf-8")

    @staticmethod
    def from_dict(d):
        d = dict(d)
        items = [Item(**i) for i in d.pop("items", [])]
        known = {k for k in Project.__dataclass_fields__}
        return Project(items=items, **{k: v for k, v in d.items() if k in known})

    @staticmethod
    def load(path):
        return Project.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------
def _noop(msg, frac=None):
    pass


def _setup_bundled_ffmpeg():
    """If ffmpeg(.exe)/ffprobe(.exe) sit next to the app (or inside the packaged exe),
    put that folder first on PATH so no separate ffmpeg install is needed."""
    here = [Path(getattr(sys, "_MEIPASS", "."))]
    here.append(Path(sys.executable).parent if getattr(sys, "frozen", False)
                else Path(__file__).resolve().parent)
    for base in here:
        for d in (base, base / "ffmpeg"):
            if (d / "ffmpeg.exe").exists() or (d / "ffmpeg").exists():
                os.environ["PATH"] = str(d) + os.pathsep + os.environ.get("PATH", "")
                return


_setup_bundled_ffmpeg()
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)    # hides console flashes on Windows


def run(cmd):
    res = subprocess.run(cmd, capture_output=True, text=True, creationflags=_NO_WINDOW)
    if res.returncode != 0:
        raise EditorError("ffmpeg failed:\n" + " ".join(map(str, cmd[:6])) + " ...\n"
                          + res.stderr[-1500:])
    return res


def check_ffmpeg():
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise EditorError("ffmpeg was not found. Install it and restart the app "
                          "(Windows: winget install Gyan.FFmpeg).")


@lru_cache(maxsize=512)
def media_duration(path):
    path = str(path)
    if Path(path).suffix.lower() in PHOTO_EXT:
        return 0.0
    if not Path(path).exists():
        raise EditorError(f"File not found: {path}")
    out = run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
               "-of", "json", path]).stdout
    return float(json.loads(out)["format"]["duration"])


def media_kind(path):
    ext = Path(path).suffix.lower()
    if ext in VIDEO_EXT:
        return "video"
    if ext in PHOTO_EXT:
        return "photo"
    return None


def scan_paths(paths):
    """Expand folders, keep only video/photo files."""
    out = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            out += [str(f) for f in sorted(p.rglob("*")) if media_kind(f)]
        elif media_kind(p):
            out.append(str(p))
    return out


def thumbnail(path, t=0.0, width=240):
    """BGR numpy image (or None). Unicode-path safe on Windows."""
    try:
        if media_kind(path) == "photo":
            data = np.fromfile(str(path), dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        else:
            cap = cv2.VideoCapture(str(path))
            cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t) * 1000)
            ok, img = cap.read()
            cap.release()
            if not ok:
                return None
        if img is None:
            return None
        h, w = img.shape[:2]
        return cv2.resize(img, (width, max(1, int(width * h / w))))
    except Exception:
        return None


@lru_cache(maxsize=512)
def media_size(path):
    """(width, height) as displayed (phone rotation metadata taken into account)."""
    path = str(path)
    try:
        out = run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                   "stream=width,height:stream_tags=rotate:stream_side_data=rotation",
                   "-of", "json", path]).stdout
        st = json.loads(out)["streams"][0]
        w, h = int(st["width"]), int(st["height"])
        rot = 0
        try:
            rot = int((st.get("tags") or {}).get("rotate", 0))
        except (TypeError, ValueError):
            pass
        for sd in st.get("side_data_list") or []:
            if "rotation" in sd:
                rot = int(float(sd["rotation"]))
        if abs(rot) % 180 == 90:
            w, h = h, w
        return w, h
    except Exception:
        img = thumbnail(path, 0.0, 640)
        if img is None:
            return 1920, 1080
        return img.shape[1], img.shape[0]


def effective_reframe(mode, sw, sh, W, H):
    """Landscape clips in a vertical video need reframing; vertical clips just fill."""
    if H <= W:
        return "fill"
    if sw / max(sh, 1) <= (W / H) * 1.25:
        return "fill"
    return mode if mode in ("smart", "fit", "fill") else "smart"


def comp_filter(mode, W, H, sw, sh, fx):
    """ffmpeg filter chain that turns the source into a W x H frame."""
    fx = min(max(float(fx), 0.0), 1.0)
    cover = (f"scale={W}:{H}:force_original_aspect_ratio=increase,"
             f"crop={W}:{H}:x='min(max(iw*{fx:.3f}-ow/2,0),iw-ow)':y='(ih-oh)/2',setsar=1")
    if mode == "fill":
        return cover
    w8, h8 = max(2, W // 8 // 2 * 2), max(2, H // 8 // 2 * 2)
    bg = (f"scale={w8}:{h8}:force_original_aspect_ratio=increase,crop={w8}:{h8},"
          f"gblur=sigma=5,scale={W}:{H},eq=brightness=-0.12:saturation=0.9")
    if mode == "fit":
        fg = f"scale={W}:-2"
    else:                                   # smart: a 4:5 window that follows the subject
        cw = min(sw, int(sh * 0.8)) // 2 * 2
        fg = f"crop={cw}:{sh}:x='min(max(iw*{fx:.3f}-ow/2,0),iw-ow)':y=0,scale={W}:-2"
    return (f"split[a][b];[a]{bg}[bg];[b]{fg}[fg];"
            f"[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1")


def compose_frame(img, mode, fx, W, H):
    """Same reframing as comp_filter, done on one BGR image (for the in-app preview)."""
    h, w = img.shape[:2]
    mode = effective_reframe(mode, w, h, W, H)
    fx = min(max(float(fx), 0.0), 1.0)

    def cover(im, tw, th, f):
        s = max(tw / im.shape[1], th / im.shape[0])
        r = cv2.resize(im, (int(np.ceil(im.shape[1] * s)), int(np.ceil(im.shape[0] * s))))
        x = int(np.clip(f * r.shape[1] - tw / 2, 0, r.shape[1] - tw))
        y = (r.shape[0] - th) // 2
        return r[y:y + th, x:x + tw]

    if mode == "fill":
        return cover(img, W, H, fx)
    bg = cover(img, W, H, 0.5)
    small = cv2.resize(bg, (max(2, W // 8), max(2, H // 8)))
    bg = cv2.resize(cv2.GaussianBlur(small, (0, 0), 3), (W, H))
    bg = (bg * 0.85).astype(np.uint8)
    if mode == "fit":
        fg = img
    else:
        cw = min(w, int(h * 0.8))
        x0 = int(np.clip(fx * w - cw / 2, 0, w - cw))
        fg = img[:, x0:x0 + cw]
    fh = int(fg.shape[0] * W / fg.shape[1])
    fg = cv2.resize(fg, (W, max(1, fh)))
    y = (H - fg.shape[0]) // 2
    out = bg.copy()
    out[max(0, y):max(0, y) + min(fg.shape[0], H)] = fg[max(0, -y):max(0, -y) + min(fg.shape[0], H)]
    return out


def estimate_focus_x(path, start, duration, samples=6):
    """Where the action is horizontally (0..1), from motion. 0.5 if unsure."""
    try:
        cap = cv2.VideoCapture(str(path))
        xs = []
        for k in range(samples):
            t = start + duration * (k + 0.5) / samples
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok1, f1 = cap.read()
            for _ in range(4):                 # step forward a few real frames
                cap.grab()
            ok2, f2 = cap.read()
            if not (ok1 and ok2):
                continue
            g1 = cv2.GaussianBlur(cv2.cvtColor(cv2.resize(f1, (160, 90)), cv2.COLOR_BGR2GRAY), (5, 5), 0)
            g2 = cv2.GaussianBlur(cv2.cvtColor(cv2.resize(f2, (160, 90)), cv2.COLOR_BGR2GRAY), (5, 5), 0)
            mask = cv2.absdiff(g1, g2) > 18
            total = mask.sum()
            if total > 0.002 * mask.size:
                cols = mask.sum(axis=0)
                xs.append(float((cols * np.arange(160)).sum() / total / 160))
        cap.release()
        if not xs:
            return 0.5
        fx = float(np.median(xs))
        return float(min(max(0.7 * fx + 0.3 * 0.5, 0.2), 0.8))
    except Exception:
        return 0.5


def find_font():
    for p in ["C:/Windows/Fonts/arialbd.ttf",
              "/System/Library/Fonts/Helvetica.ttc",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"]:
        if Path(p).exists():
            return p
    return None


def ff_path(p):
    return str(p).replace("\\", "/").replace(":", "\\:")


def esc(text):
    return (text.replace("\\", "\\\\").replace(":", "\\:")
                .replace("'", "\u2019").replace("%", "\\%"))


# ----------------------------------------------------------------------------
# Analysis (used by the automatic editor)
# ----------------------------------------------------------------------------
@dataclass
class Clip:
    path: str
    duration: float
    windows: list = field(default_factory=list)      # (start, score)


def analyse_video(path, window, sample_every=0.5):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    dur = n / fps if fps and n else media_duration(path)

    samples, prev, t = [], None, 0.0
    while t < dur - 0.05:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if not ok:
            break
        small = cv2.resize(frame, (320, int(320 * h / max(w, 1))))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        sharp = cv2.Laplacian(gray, cv2.CV_64F).var()
        motion = float(np.mean(cv2.absdiff(gray, prev))) if prev is not None else 0.0
        exposure_ok = 1.0 if 35 < gray.mean() < 215 else 0.4
        samples.append((t, sharp, motion, exposure_ok))
        prev = gray
        t += sample_every
    cap.release()

    clip = Clip(str(path), dur)
    if not samples:
        return clip
    arr = np.array(samples)
    sharp_n = arr[:, 1] / (arr[:, 1].max() + 1e-6)
    motion_n = np.clip(arr[:, 2] / 25.0, 0, 1)
    motion_n = 1 - np.abs(motion_n - 0.45) * 1.4
    score = (0.45 * sharp_n + 0.35 * np.clip(motion_n, 0, 1)) * arr[:, 3]
    k = max(1, int(window / sample_every))
    for i in range(0, max(1, len(score) - k + 1)):
        s = arr[i, 0]
        if s + window > dur:
            continue
        clip.windows.append((float(s), float(score[i:i + k].mean())))
    return clip


def analyse_music(music, target_len, fallback_bpm=120, cb=_noop):
    """Return (beats, drop_time or None, music_len). Times are absolute in the song."""
    music_len = media_duration(music)
    try:
        import librosa
        y, sr = librosa.load(str(music), sr=22050, mono=True)
        tempo, frames = librosa.beat.beat_track(y=y, sr=sr, trim=False)
        beats = list(map(float, librosa.frames_to_time(frames, sr=sr)))
        tempo = float(np.atleast_1d(tempo)[0])
        if len(beats) < 8:
            raise ValueError("too few beats")
        period = float(np.median(np.diff(beats)))
        while beats[0] - period >= 0:
            beats.insert(0, beats[0] - period)
        while beats[-1] + period <= music_len:
            beats.append(beats[-1] + period)
        cb(f"Tempo ~{tempo:.0f} BPM", None)

        hop = 512
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop)[0]
        fps_e = sr / hop
        k = max(1, int(fps_e * 1.0))
        sm = np.convolve(rms, np.ones(k) / k, mode="same")
        win = int(fps_e * 2.0)
        cs = np.concatenate([[0], np.cumsum(sm)])
        drop, best = None, 0.0
        lo = int(max(win, 0.25 * target_len * fps_e))
        hi = int(min(len(sm) - win, (music_len - 0.45 * target_len) * fps_e))
        for i in range(lo, hi):
            rise = (cs[i + win] - cs[i]) / win - (cs[i] - cs[i - win]) / win
            if rise > best:
                best, drop = rise, i / fps_e
        if drop is not None:
            cb(f"Drop detected at {drop:.1f}s", None)
        return beats, drop, music_len
    except Exception as e:
        cb(f"Music analysis fallback ({e}); using {fallback_bpm} BPM", None)
        return list(np.arange(0, music_len, 60.0 / fallback_bpm)), None, music_len


def choose_offset(beats, drop, music_len, target_len, mode="auto"):
    if mode != "auto":
        return float(mode)
    if drop is None or music_len <= target_len + 1:
        return 0.0
    desired = min(max(0.0, drop - 0.40 * target_len), music_len - target_len)
    earlier = [b for b in beats if b <= desired + 1e-3]
    return max(earlier) if earlier else 0.0


@dataclass
class Cut:
    start: float
    end: float
    speed: float
    section: int
    hero: bool = False


def plan_cuts(beats, target_len, drop_rel):
    beats = [b for b in beats if b <= target_len]
    n = len(beats)
    if n < 8:
        raise EditorError("The music is too short (or has too few beats) for this length.")
    if drop_rel is None:
        drop_idx = int(n * 0.40)
    else:
        drop_idx = int(np.argmin([abs(b - drop_rel) for b in beats]))
    drop_idx = max(max(4, int(0.25 * n)), min(drop_idx, int(0.65 * n)))
    peak_end = min(max(drop_idx + 4, int(0.88 * n)), n - 1)
    intro_end = max(4, int(drop_idx * 0.40))

    cuts = []

    def fill(i, end, bpc, speed, section):
        while i + bpc <= end:
            cuts.append(Cut(beats[i], beats[i + bpc], speed, section))
            i += bpc
        return i

    i = fill(0, intro_end, 4, 1.0, 0)
    i = fill(i, drop_idx, 2, 1.0, 1)
    if i < drop_idx:
        cuts.append(Cut(beats[i], beats[drop_idx], 1.0, 1))
    i = drop_idx
    if i + 2 <= peak_end:
        cuts.append(Cut(beats[i], beats[i + 2], 0.6, 2, hero=True))
        i += 2
    i = fill(i, peak_end, 1, 1.0, 2)
    i = fill(i, n - 1, 4, 0.8, 3)
    if n - 1 - i >= 2:
        cuts.append(Cut(beats[i], beats[n - 1], 0.8, 3))
    if cuts:
        cuts[0].start = 0.0
    return cuts


def pick_effects(cut, index, pre_drop, level, rng):
    fx = dict(move=None, flash=False, glitch=False, whip=False, ramp=False,
              dip_in=False, dip_out=False)
    if level == "off":
        return fx
    moves = MOVES[cut.section]
    if level == "subtle":
        moves = [m for m in moves if m in ("push", "pull")] or ["push"]
    fx["move"] = "push" if cut.hero else rng.choice(moves)
    if cut.section == 3:
        fx["dip_in"] = fx["dip_out"] = True
    if level != "hype":
        return fx
    if cut.hero:
        fx["flash"] = True
    elif cut.section == 2:
        fx["flash"] = index % 4 == 0
        fx["glitch"] = index % 3 == 1
        fx["ramp"] = index % 5 == 2
    elif cut.section == 1:
        fx["whip"] = index % 3 == 0
        fx["flash"] = index % 5 == 0
        if pre_drop:
            fx["dip_out"] = True
            fx["flash"] = False
    return fx


def auto_edit(media, music, length=30.0, style="cinema", effects="hype", seed=None,
              music_start="auto", title="", subtitle="", outro="", fmt="vertical",
              fps=30, sfx=True, lut="", reframe="smart", move_strength=1.0,
              cb: Callback = _noop):
    """Analyse footage + music and return a ready-to-render (and editable) Project."""
    check_ffmpeg()
    media = scan_paths(media)
    videos_p = [m for m in media if media_kind(m) == "video"]
    photos = [m for m in media if media_kind(m) == "photo"]
    if not videos_p:
        raise EditorError("Add at least one video clip (photos are used as extras).")
    if not music or not Path(music).exists():
        raise EditorError("Choose a music file first.")

    rng = random.Random(seed)
    target = min(length, media_duration(music))

    videos = []
    for i, p in enumerate(videos_p):
        cb(f"Analysing {Path(p).name}", 0.05 + 0.45 * i / len(videos_p))
        c = analyse_video(p, 1.0)
        if c.windows:
            videos.append(c)
    if not videos:
        raise EditorError("None of the video clips could be analysed.")

    cb("Analysing music", 0.55)
    beats, drop, music_len = analyse_music(music, target, cb=cb)
    offset = choose_offset(beats, drop, music_len, target, music_start)
    beats = [b - offset for b in beats if b >= offset - 1e-3]
    drop_rel = (drop - offset) if drop is not None else None

    cb("Planning the edit", 0.75)
    cuts = plan_cuts(beats, target, drop_rel)

    # ---- assign footage to cuts
    pool = sorted(((c, s, sc) for c in videos for s, sc in c.windows), key=lambda x: -x[2])
    used, assigned = {}, [None] * len(cuts)
    photo_q = photos[:]
    rng.shuffle(photo_q)
    order = [i for i, c in enumerate(cuts) if c.hero] + [i for i, c in enumerate(cuts) if not c.hero]
    for i in order:
        cut = cuts[i]
        length_src = (cut.end - cut.start) * cut.speed
        if photo_q and cut.section in (0, 3) and i % 3 == 1:
            assigned[i] = ("photo", photo_q.pop(), 0.0)
            continue
        best = None
        for c, start, score in pool:
            if start + length_src > c.duration:
                continue
            if any(abs(start - u) < max(length_src, 1.0) for u in used.get(c.path, [])):
                continue
            val = score + 0.15 / (1 + len(used.get(c.path, [])))
            if best is None or val > best[0]:
                best = (val, c, start)
            if cut.hero:
                break
        if best is None:
            c = rng.choice(videos)
            start = rng.choice(c.windows)[0]
            best = (0, c, min(start, max(0, c.duration - length_src)))
        _, c, start = best
        used.setdefault(c.path, []).append(start)
        assigned[i] = ("video", c.path, start)

    items = []
    for i, (cut, (kind, src, start)) in enumerate(zip(cuts, assigned)):
        pre_drop = i + 1 < len(cuts) and cuts[i + 1].hero
        fx = pick_effects(cut, i, pre_drop, effects, rng)
        items.append(Item(source=src, kind=kind, src_start=round(start, 3),
                          duration=round(cut.end - cut.start, 4), speed=cut.speed,
                          section=cut.section, hero=cut.hero, **fx))

    cb("Finding the subject in each clip", 0.9)
    for it in items:
        if it.kind == "video":
            it.focus_x = round(estimate_focus_x(it.source, it.src_start,
                                                it.duration * max(it.speed, 0.05)), 3)

    cb("Edit planned", 1.0)
    return Project(music=str(music), music_offset=round(offset, 3), style=style, lut=lut,
                   effects=effects, title=title, subtitle=subtitle, outro=outro,
                   format=fmt, fps=fps, sfx=sfx, reframe=reframe,
                   move_strength=move_strength, items=items)


# ----------------------------------------------------------------------------
# Rendering
# ----------------------------------------------------------------------------
def move_filter(move, frames, size, fps, strength=1.0):
    """Gentle camera moves. `strength` scales how much it zooms (0 = none)."""
    W, H = size
    k = max(0.0, float(strength))
    if k == 0:
        return None
    cx, cy = "iw/2-(iw/zoom/2)", "ih/2-(ih/zoom/2)"
    N = max(frames, 2)
    x, y = cx, cy
    if move == "push":
        z = f"1+{0.08 * k:.4f}*on/{N}"
    elif move == "pull":
        z = f"1+{0.08 * k:.4f}-{0.08 * k:.4f}*on/{N}"
    elif move == "punch":
        z = f"1+{0.12 * k:.4f}*pow(0.86,on)"
    elif move == "shake":
        z = f"{1 + 0.06 * k:.4f}"
        x = f"{cx}+sin(on*1.9)*{W * 0.005 * k:.1f}"
        y = f"{cy}+cos(on*2.3)*{H * 0.003 * k:.1f}"
    else:
        return None
    return f"zoompan=z='{z}':x='{x}':y='{y}':d=1:s={W}x{H}:fps={fps}"


def render_item(item: Item, size, fps, grade, tmp: Path, idx, preview,
                reframe="smart", move_strength=1.0):
    W, H = size
    out_len = max(0.2, item.duration)
    out = tmp / f"shot_{idx:04d}.mp4"
    frames = max(int(out_len * fps), 2)
    source = item.source
    if not Path(source).exists():
        raise EditorError(f"Source file is missing: {source}")

    sw, sh = media_size(source)
    mode = effective_reframe(item.reframe or reframe, sw, sh, W, H)
    comp = comp_filter(mode, W, H, sw, sh, item.focus_x)
    big_w, big_h = int(W * 1.5) // 2 * 2, int(H * 1.5) // 2 * 2

    post = []
    if item.glitch:
        post.append("rgbashift=rh=-10:bh=10:rv=4:bv=-4:enable='lt(t,0.12)'")
    if item.whip:
        post.append("gblur=sigma=28:enable='lt(t,0.1)'")
    post.append(grade)
    if item.flash:
        post.append("fade=t=in:st=0:d=0.14:color=white")
    if item.dip_in:
        post.append("fade=t=in:st=0:d=0.15")
    if item.dip_out:
        post.append(f"fade=t=out:st={max(out_len - 0.22, 0):.3f}:d=0.22")
    post.append("format=yuv420p")
    post = ",".join(post)

    move = item.move if item.move in ("push", "pull", "punch", "shake") else None
    k = max(0.0, float(move_strength))
    if item.kind == "photo":
        z = (f"min(zoom+{0.0007 * k:.5f},1.15)" if move in (None, "push", "punch", "shake")
             else f"if(eq(on,0),1.15,max(zoom-{0.0007 * k:.5f},1.0))")
        vf = (f"{comp},scale={big_w}:{big_h},"
              f"zoompan=z='{z}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
              f"d={frames}:s={W}x{H}:fps={fps},{post}")
        pre, take = ["-loop", "1"], []
    else:
        speed = item.speed if item.speed > 0.05 else 1.0
        mv = move_filter(move, frames, size, fps, move_strength) if move else None
        if item.ramp:
            speed_f = f"setpts='(T*(1.6-0.6*T/{out_len:.4f}))/TB'"
            src_len = out_len
        else:
            speed_f = f"setpts=PTS*{1 / speed:.4f}"
            src_len = out_len * speed
        vf = f"{speed_f},fps={fps},tpad=stop_mode=clone:stop_duration=0.5,{comp}"
        if mv:
            vf += f",scale={big_w}:{big_h},{mv}"
        vf += f",{post}"
        sdur = media_duration(source)
        start = max(0.0, min(item.src_start, max(0.0, sdur - src_len - 0.05)))
        pre, take = ["-ss", f"{start:.3f}"], ["-t", f"{src_len + 0.1:.3f}"]

    crf, preset = ("26", "ultrafast") if preview else ("14", "veryfast")
    run(["ffmpeg", "-y", "-loglevel", "error"] + pre + ["-i", source] + take +
        ["-vf", vf, "-r", str(fps), "-frames:v", str(frames),
         "-c:v", "libx264", "-preset", preset, "-crf", crf, "-an", str(out)])
    return out


def make_sfx(tmp: Path):
    specs = {
        "impact": "aevalsrc='0.9*sin(2*PI*(45+90*exp(-t*18))*t)*exp(-t*3.2)"
                  "+0.25*random(0)*exp(-t*9)':d=1.6:s=48000",
        "whoosh": "anoisesrc=d=0.7:c=pink:a=0.5:r=48000,highpass=f=500,lowpass=f=6000,"
                  "afade=t=in:d=0.35,afade=t=out:st=0.35:d=0.35",
        "riser": "aevalsrc='0.5*sin(2*PI*(200*t+300*t*t))*pow(t/2,2)':d=2:s=48000",
    }
    files = {}
    for name, src in specs.items():
        f = tmp / f"{name}.wav"
        run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", src, str(f)])
        files[name] = f
    return files


SFX_VOL = {"impact": 1.0, "riser": 0.55, "whoosh": 0.55}


@lru_cache(maxsize=256)
def has_audio_stream(path):
    """Return whether a media file has an audio stream (without decoding it)."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0",
             "-show_entries", "stream=codec_type", "-of", "json", str(path)],
            capture_output=True, text=True, timeout=15
        )
        if result.returncode != 0:
            return False
        return bool(json.loads(result.stdout or "{}").get("streams"))
    except Exception:
        return False


def _atempo_chain(speed):
    """Build legal FFmpeg atempo stages for any positive playback speed."""
    speed = max(0.05, float(speed))
    parts = []
    while speed > 2.0:
        parts.append("atempo=2.0")
        speed /= 2.0
    while speed < 0.5:
        parts.append("atempo=0.5")
        speed /= 0.5
    parts.append(f"atempo={speed:.5f}")
    return ",".join(parts)


def _finish(joined, project: Project, events, size, total_len, out, preview, tmp):
    W, H = size
    style = STYLES.get(project.style, STYLES["cinema"])
    font = find_font()
    fontarg = f"fontfile='{ff_path(font)}':" if font else ""
    big, small = int(H * 0.06), int(H * 0.026)
    fx_on = project.effects != "off"

    graph = []
    bloom = 0 if preview or not fx_on else style["bloom"]
    if bloom > 0:
        graph.append(f"[0:v]split[a][b];[b]scale={W // 4}:{H // 4},gblur=sigma=6,"
                     f"scale={W}:{H}[g];[a][g]blend=all_mode=screen:all_opacity={bloom}[bl]")
        src = "[bl]"
    else:
        src = "[0:v]"

    vf = []
    if not preview:
        if fx_on:
            vf.append("noise=alls=3:allf=t")
        vf.append("unsharp=5:5:0.5:5:5:0.0")
    vf += ["fade=t=in:st=0:d=0.6", f"fade=t=out:st={max(total_len - 1.0, 0):.2f}:d=1.0"]

    ty = int(H * 0.30)
    if project.title:
        vf.append(
            f"drawtext={fontarg}text='{esc(project.title.upper())}':fontsize={big}:"
            f"fontcolor=white:borderw=4:bordercolor=black@0.55:shadowx=3:shadowy=3:"
            f"x=(w-text_w)/2:y={ty}-text_h/2:"
            f"alpha='if(lt(t,0.4),0,if(lt(t,1.0),(t-0.4)/0.6,if(lt(t,3.4),1,max(0,1-(t-3.4)/0.6))))'")
        vf.append(f"drawbox=x=(iw-160)/2:y={ty + int(big * 0.65)}:w=160:h=3:"
                  f"color=white@0.85:t=fill:enable='between(t,0.7,3.4)'")
    if project.subtitle:
        vf.append(
            f"drawtext={fontarg}text='{esc(project.subtitle)}':fontsize={small}:"
            f"fontcolor=white@0.95:borderw=3:bordercolor=black@0.55:"
            f"x=(w-text_w)/2:y={ty + int(big * 0.95)}:"
            f"alpha='if(lt(t,0.8),0,if(lt(t,1.3),(t-0.8)/0.5,if(lt(t,3.4),1,max(0,1-(t-3.4)/0.6))))'")
    if project.outro:
        t0 = max(total_len - 4.5, 0)
        vf.append(
            f"drawtext={fontarg}text='{esc(project.outro)}':fontsize={int(small * 1.2)}:"
            f"fontcolor=white:borderw=3:bordercolor=black@0.55:"
            f"x=(w-text_w)/2:y=h*0.66:alpha='if(lt(t,{t0:.2f}),0,min(1,(t-{t0:.2f})/0.6))'")

    # Burn generated speech captions into the final video.
    caption_style = getattr(project, "caption_style", "bold")
    cap_size = int(H * (0.050 if caption_style == "bold" else 0.038))
    cap_y = "h-text_h-(h*0.13)"
    for cue in getattr(project, "captions", []) or []:
        try:
            start = max(0.0, float(cue.get("start", 0)))
            end = min(total_len, float(cue.get("end", 0)))
            text = _caption_escape(cue.get("text", ""))
            if not text or end <= start:
                continue
            if caption_style == "minimal":
                box = "box=0"
                border = "borderw=2:bordercolor=black@0.8"
                color = "fontcolor=white"
            elif caption_style == "clean":
                box = "box=1:boxcolor=black@0.48:boxborderw=12"
                border = "borderw=1:bordercolor=black@0.6"
                color = "fontcolor=white"
            else:
                box = "box=1:boxcolor=black@0.68:boxborderw=16"
                border = "borderw=3:bordercolor=black"
                color = "fontcolor=yellow"
            vf.append(
                f"drawtext={fontarg}text='{text}':fontsize={cap_size}:{color}:{border}:"
                f"{box}:x=(w-text_w)/2:y={cap_y}:"
                f"enable='between(t,{start:.3f},{end:.3f})'")
        except (TypeError, ValueError):
            continue
    graph.append(f"{src}{','.join(vf)}[v]")

    fmt = "aformat=sample_fmts=fltp:channel_layouts=stereo:sample_rates=48000"
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(joined),
           "-ss", f"{project.music_offset:.3f}", "-i", str(project.music)]
    graph.append(f"[1:a]aresample=48000,atrim=duration={total_len:.3f},asetpts=PTS-STARTPTS,"
                 f"volume={max(0.0, float(project.music_gain)):.3f},{fmt}[m]")

    # Add an optional narration/voice-over track. Its offset is measured from the
    # beginning of the finished timeline, not from the source file.
    dialogue_labels = []
    next_input = 2
    voiceover = getattr(project, "voiceover", "") or ""
    if voiceover:
        if not Path(voiceover).exists():
            raise EditorError(f"Voice-over file is missing: {voiceover}")
        if not has_audio_stream(voiceover):
            raise EditorError("The selected voice-over file has no readable audio stream.")
        cmd += ["-i", str(voiceover)]
        delay = int(max(0.0, float(project.voiceover_offset)) * 1000)
        graph.append(
            f"[{next_input}:a]aresample=48000,{fmt},asetpts=PTS-STARTPTS,"
            f"volume={max(0.0, float(project.voiceover_gain)):.3f},"
            f"adelay={delay}|{delay},apad,atrim=duration={total_len:.3f}[vo]"
        )
        dialogue_labels.append("[vo]")
        next_input += 1

    # Preserve source-camera audio for video clips, aligned to each clip's
    # timeline position. Photos and silent clips are simply skipped.
    if getattr(project, "use_clip_audio", True):
        timeline_pos = 0.0
        clip_number = 0
        for item in project.items:
            duration = max(0.2, float(item.duration))
            if item.kind == "video" and Path(item.source).exists() and has_audio_stream(item.source):
                speed = max(0.05, float(item.speed)) if not item.ramp else 1.0
                source_duration = duration * speed
                source_duration = max(0.05, source_duration)
                source_start = max(0.0, float(item.src_start))
                cmd += ["-ss", f"{source_start:.3f}", "-t", f"{source_duration + 0.05:.3f}", "-i", item.source]
                tempo = _atempo_chain(speed)
                delay = int(timeline_pos * 1000)
                label = f"clipaudio{clip_number}"
                graph.append(
                    f"[{next_input}:a]aresample=48000,{fmt},asetpts=PTS-STARTPTS,"
                    f"{tempo},apad,atrim=duration={duration:.3f},"
                    f"volume={max(0.0, float(project.clip_audio_gain)):.3f},"
                    f"adelay={delay}|{delay}[{label}]"
                )
                dialogue_labels.append(f"[{label}]")
                next_input += 1
                clip_number += 1
            timeline_pos += duration

    # Duck the music under narration and/or camera dialogue, while leaving
    # dialogue at its selected level. The compressor is driven by the dialogue bus.
    music_label = "[m]"
    final_dialogue_labels = dialogue_labels
    if dialogue_labels and getattr(project, "duck_music", True):
        if len(dialogue_labels) == 1:
            graph.append(f"{dialogue_labels[0]}asplit=2[dialogue_side][dialogue_mix]")
        else:
            graph.append(f"{''.join(dialogue_labels)}amix=inputs={len(dialogue_labels)}:"
                         f"duration=longest:normalize=0,asplit=2[dialogue_side][dialogue_mix]")
        graph.append("[m][dialogue_side]sidechaincompress=threshold=0.025:ratio=8:"
                     "attack=15:release=350:makeup=1[ducked_music]")
        music_label = "[ducked_music]"
        final_dialogue_labels = ["[dialogue_mix]"]

    labels = [music_label] + final_dialogue_labels
    if project.sfx and events and fx_on:
        files = make_sfx(tmp)
        for k, (t, kind) in enumerate(events):
            cmd += ["-i", str(files[kind])]
            graph.append(f"[{next_input}:a]aresample=48000,adelay={int(t * 1000)}:all=1,"
                         f"volume={SFX_VOL[kind]},{fmt}[s{k}]")
            labels.append(f"[s{k}]")
            next_input += 1
    graph.append(f"{''.join(labels)}amix=inputs={len(labels)}:duration=first:normalize=0,"
                 f"loudnorm=I=-14:TP=-1.5:LRA=11[a]")

    crf, preset = ("26", "veryfast") if preview else ("18", "medium")
    cmd += ["-filter_complex", ";".join(graph), "-map", "[v]", "-map", "[a]",
            "-t", f"{total_len:.3f}", "-c:v", "libx264", "-preset", preset, "-crf", crf,
            "-maxrate", "16M", "-bufsize", "32M", "-profile:v", "high", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "256k", "-ar", "48000", "-movflags", "+faststart", str(out)]
    run(cmd)



def transcribe_project(project: Project, model_size="tiny", cb: Callback = _noop):
    """Transcribe speech from timeline video clips with local faster-whisper.

    Subtitle timestamps are converted from source-clip time to sequence time.
    Audio/video never leaves the computer. The optional faster-whisper package
    and its model weights are downloaded separately on first use.
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise EditorError(
            "Local captioning needs faster-whisper. Install it with:\n"
            "python -m pip install faster-whisper"
        ) from exc

    video_items = [(idx, item) for idx, item in enumerate(project.items)
                   if item.kind == "video" and Path(item.source).exists()]
    if not video_items:
        raise EditorError("Add at least one video clip with spoken audio before generating captions.")

    cb(f"Loading local Whisper '{model_size}' model (first run may download it)…", 0.02)
    try:
        model = WhisperModel(model_size, device="cpu", compute_type="int8")
    except Exception as exc:
        raise EditorError(f"Could not load Whisper model: {exc}") from exc

    cues = []
    timeline_cursor = 0.0
    total = len(project.items)
    for item_index, item in enumerate(project.items):
        if item.kind != "video" or not Path(item.source).exists():
            timeline_cursor += item.duration
            continue
        cb(f"Transcribing clip {item_index + 1} of {total}: {Path(item.source).name}",
           0.05 + 0.85 * (item_index / max(total, 1)))
        try:
            segments, _info = model.transcribe(
                item.source, beam_size=3, word_timestamps=True, vad_filter=True
            )
            words = []
            source_end = item.src_start + item.duration * max(item.speed, 0.05)
            for segment in segments:
                if not segment.words:
                    continue
                for word in segment.words:
                    if word.start is None or word.end is None:
                        continue
                    if word.end <= item.src_start or word.start >= source_end:
                        continue
                    a = max(float(word.start), item.src_start)
                    b = min(float(word.end), source_end)
                    speed = max(float(item.speed), 0.05)
                    local_a = (a - item.src_start) / speed
                    local_b = (b - item.src_start) / speed
                    words.append((local_a, local_b, word.word.strip()))
            # Keep captions readable: group up to 5 words or about 2.2 seconds.
            group = []
            for wa, wb, wt in words:
                if not wt:
                    continue
                if group and (len(group) >= 5 or wb - group[0][0] > 2.2):
                    cues.append({"start": round(timeline_cursor + group[0][0], 3),
                                 "end": round(timeline_cursor + group[-1][1], 3),
                                 "text": " ".join(x[2] for x in group)})
                    group = []
                group.append((wa, wb, wt))
            if group:
                cues.append({"start": round(timeline_cursor + group[0][0], 3),
                             "end": round(timeline_cursor + group[-1][1], 3),
                             "text": " ".join(x[2] for x in group)})
        except Exception as exc:
            # A silent/no-audio clip should not prevent captioning the remaining clips.
            cb(f"Skipped {Path(item.source).name}: {exc}", None)
        timeline_cursor += item.duration

    if not cues:
        raise EditorError("No speech was detected. Check that your video clips contain audible dialogue.")
    project.captions = cues
    cb(f"Generated {len(cues)} caption cues", 1.0)
    return cues


def _caption_escape(value):
    return (str(value).replace("\\", "\\\\").replace(":", "\\:")
            .replace("'", "’").replace(",", "\\,").replace("%", "\\%"))


def _grade_for(project: Project):
    style = STYLES.get(project.style, STYLES["cinema"])
    grade = style["grade"]
    if project.lut:
        if not Path(project.lut).exists():
            raise EditorError(f"LUT file not found: {project.lut}")
        grade = f"lut3d='{ff_path(project.lut)}',vignette=PI/6"
    return grade


def render_clip_preview(project: Project, index, out, cb: Callback = _noop):
    """Render ONE timeline clip (half-res, no audio) so you can see its framing and effects."""
    check_ffmpeg()
    item = project.items[index]
    full = (1080, 1920) if project.format == "vertical" else (1920, 1080)
    size = (full[0] // 2, full[1] // 2)
    cb("Rendering clip preview", 0.3)
    with tempfile.TemporaryDirectory() as t:
        f = render_item(item, size, project.fps, _grade_for(project), Path(t), 0, True,
                        project.reframe, project.move_strength)
        shutil.copy(f, out)
    cb("Done", 1.0)
    return str(out)


def render_project(project: Project, out, preview=False, cb: Callback = _noop):
    """Render any Project (automatic or hand-edited) to an MP4 file."""
    check_ffmpeg()
    if not project.items:
        raise EditorError("The timeline is empty.")
    if not project.music or not Path(project.music).exists():
        raise EditorError("Choose a music file first.")

    full = (1080, 1920) if project.format == "vertical" else (1920, 1080)
    size = (full[0] // 2, full[1] // 2) if preview else full
    grade = _grade_for(project)

    n = len(project.items)
    events, cursor, parts = [], 0.0, []
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        for i, item in enumerate(project.items):
            cb(f"Rendering clip {i + 1}/{n}", 0.02 + 0.78 * i / n)
            parts.append(render_item(item, size, project.fps, grade, tmp, i, preview,
                                     project.reframe, project.move_strength))
            if item.hero:
                events.append((cursor, "impact"))
                events.append((max(cursor - 2.0, 0.0), "riser"))
            elif item.flash or item.whip:
                events.append((cursor, "whoosh"))
            cursor += max(0.2, item.duration)
        cb("Joining clips", 0.82)
        lst = tmp / "list.txt"
        lst.write_text("".join(f"file '{p.as_posix()}'\n" for p in parts))
        joined = tmp / "joined.mp4"
        run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
             "-i", str(lst), "-c", "copy", str(joined)])
        cb("Grade, titles, sound design, final encode", 0.88)
        _finish(joined, project, events[:14], size, cursor, out, preview, tmp)
    cb("Done", 1.0)
    return str(out)


# ----------------------------------------------------------------------------
# Gemini creator assistant (optional cloud AI; editing/rendering stays local)
# ----------------------------------------------------------------------------
def gemini_creator_pack(topic, fmt="vertical", content_type="Workout vlog",
                        tone="Energetic", api_key=None, model="gemini-3.8-flash"):
    """Generate a practical creator pack. Requires optional `google-genai` package + API key."""
    key = (api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or "").strip()
    if not key:
        raise EditorError(
            "Gemini API key not found. Create a key in Google AI Studio, then paste it here "
            "or set GEMINI_API_KEY in Windows environment variables."
        )
    try:
        from google import genai
    except ImportError as exc:
        raise EditorError("Gemini SDK is not installed. Run: pip install -U google-genai") from exc

    shape = "9:16 vertical short-form video" if fmt == "vertical" else "16:9 horizontal YouTube video"
    prompt = f"""
You are a professional short-form video producer and social-media editor.
Create a creator pack for this video.
Topic/context: {topic}
Content type: {content_type}
Format: {shape}
Tone: {tone}

Return ONLY valid JSON with these keys:
{{
  "title": "short compelling title (max 60 characters)",
  "hook": "spoken/on-screen opening hook (max 12 words)",
  "subtitle": "short supporting line",
  "outro": "short call to action",
  "caption": "natural platform caption, not spammy",
  "hashtags": ["8 relevant hashtags, no duplicates"],
  "shot_list": ["6 practical shots in a useful filming order"],
  "edit_notes": ["5 concise editing suggestions for pacing, framing, and sound"]
}}
Avoid fake claims, engagement bait, irrelevant hashtags, and generic motivational clichés.
For workout content, do not invent medical or fitness claims. Keep advice safe and realistic.
"""
    try:
        client = genai.Client(api_key=key)
        response = client.models.generate_content(model=model, contents=prompt)
        raw = (response.text or "").strip()
        raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        data = json.loads(raw)
        required = ("title", "hook", "subtitle", "outro", "caption", "hashtags", "shot_list", "edit_notes")
        if not all(k in data for k in required):
            raise ValueError("Gemini response is missing required fields.")
        return data
    except EditorError:
        raise
    except Exception as exc:
        raise EditorError(f"Gemini request failed: {type(exc).__name__}: {exc}") from exc

