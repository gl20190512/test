"""歌声合成：用离线神经 TTS 逐字发音，再用 WORLD 声码器把音高、时长改到乐谱上。"""
import functools
import os

import numpy as np
import pyworld as pw
import sherpa_onnx

SR = 44100
FRAME_MS = 5.0
MODEL_DIR = os.environ.get("TTS_MODEL_DIR", "vits-melo-tts-zh_en")


@functools.lru_cache(maxsize=1)
def _tts():
    d = MODEL_DIR.rstrip("/") + "/"
    cfg = sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                model=d + "model.onnx", lexicon=d + "lexicon.txt",
                tokens=d + "tokens.txt", dict_dir=d + "dict"),
            num_threads=4))
    return sherpa_onnx.OfflineTts(cfg)


def _trim(x, thresh=0.01):
    idx = np.where(np.abs(x) > thresh * np.max(np.abs(x)))[0]
    return x[idx[0]:idx[-1] + 1] if len(idx) else x


@functools.lru_cache(maxsize=None)
def _analyse(char):
    """发音一个字并做 WORLD 分析，结果缓存复用。"""
    audio = _tts().generate(char + "。", sid=0, speed=0.75)
    x = np.asarray(audio.samples, dtype=np.float64)
    assert audio.sample_rate == SR, audio.sample_rate
    x = _trim(x)
    f0, t = pw.harvest(x, SR, f0_floor=70, f0_ceil=800, frame_period=FRAME_MS)
    sp = pw.cheaptrick(x, f0, t, SR)
    ap = pw.d4c(x, f0, t, SR)
    return f0, sp, ap


def midi_hz(m):
    return 440.0 * 2 ** ((m - 69) / 12.0)


def sing_note(char, midi, dur, prev_midi=None):
    """合成一个音符，返回长度约为 dur 秒的波形。"""
    f0, sp, ap = _analyse(char)
    n = len(f0)
    voiced = np.where(f0 > 0)[0]
    if len(voiced) == 0:
        v0, v1 = 0, n
    else:
        v0, v1 = voiced[0], voiced[-1] + 1
    # 辅音部分保持原速；元音部分拉伸填满音符时值，尾部留一点点收音
    total = max(int(dur * 1000 / FRAME_MS), v0 + 8)
    tail = min(n - v1, 6)
    body = max(total - v0 - tail, 4)
    src = np.concatenate([
        np.arange(v0),
        np.linspace(v0, v1 - 1, body),
        np.linspace(v1, n - 1, tail) if tail > 0 else np.array([]),
    ])
    lo = np.floor(src).astype(int)
    hi = np.minimum(lo + 1, n - 1)
    w = (src - lo)[:, None]
    sp2 = sp[lo] * (1 - w) + sp[hi] * w
    ap2 = ap[lo] * (1 - w) + ap[hi] * w
    uv = f0[np.rint(src).astype(int)] > 0

    # 目标音高：起音滑音 + 渐强颤音
    m = len(src)
    tsec = np.arange(m) * FRAME_MS / 1000.0
    target = np.full(m, float(midi))
    if prev_midi is not None and prev_midi != midi:
        glide = np.clip((tsec - v0 * FRAME_MS / 1000) / 0.06, 0, 1)
        target = prev_midi + (midi - prev_midi) * (0.5 - 0.5 * np.cos(np.pi * glide))
    vib_depth = 0.25 * np.clip((tsec - 0.3) / 0.4, 0, 1)
    target = target + vib_depth * np.sin(2 * np.pi * 5.4 * tsec)
    f0_new = np.where(uv, midi_hz(target), 0.0)

    y = pw.synthesize(f0_new, np.ascontiguousarray(sp2), np.ascontiguousarray(ap2), SR, FRAME_MS)
    fade = min(len(y), int(0.04 * SR))
    y[-fade:] *= np.linspace(1, 0, fade)
    y[:64] *= np.linspace(0, 1, 64)
    return y


def render_vocal(notes_with_time, total_sec):
    """notes_with_time: [(start_sec, char, midi, dur_sec, prev_midi)] -> 单声道人声轨。"""
    out = np.zeros(int(total_sec * SR) + SR)
    for start, char, midi, dur, prev in notes_with_time:
        y = sing_note(char, midi, dur * 0.97, prev)
        # 辅音略微提前，让元音落在拍点上
        f0, _, _ = _analyse(char)
        voiced = np.where(f0 > 0)[0]
        lead = (voiced[0] if len(voiced) else 0) * FRAME_MS / 1000.0
        i = max(int((start - min(lead, 0.08)) * SR), 0)
        out[i:i + len(y)] += y[:len(out) - i]
    return out
