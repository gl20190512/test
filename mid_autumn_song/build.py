"""生成《月圆人团圆》MV：伴奏 + 合成人声 + 卡拉OK字幕画面，输出 AVI。

用法：TTS_MODEL_DIR=<vits-melo-tts-zh_en 目录> python3 build.py [输出.avi]
"""
import math
import os
import subprocess
import sys

import numpy as np
import soundfile as sf
from PIL import Image, ImageDraw, ImageFilter, ImageFont

import sing
import song

SR = sing.SR
W, H, FPS = 1280, 720, 25
FONT = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"
HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------- 乐谱展开
def timeline():
    """展开为歌词行：[(section, start_sec, [(start_sec, char, midi, dur_sec)])]，以及和弦表。"""
    sections, total_beats = song.arrangement()
    lines, chords = [], []
    prev = None
    for name, body, chs, start in sections:
        if body is None:
            for i, c in enumerate(chs):
                chords.append(((start + 4 * i) * song.BEAT, 4 * song.BEAT, c))
            continue
        for li, line in enumerate(body):
            b = start + 8 * li
            for bi, c in enumerate(chs[li]):
                chords.append(((b + 4 * bi) * song.BEAT, 4 * song.BEAT, c))
            notes = []
            for ch, midi, beats in line:
                if midi is not None:
                    notes.append((b * song.BEAT, ch, midi, beats * song.BEAT, prev))
                    prev = midi
                b += beats
            lines.append((name, (start + 8 * li) * song.BEAT, notes))
    return lines, chords, total_beats * song.BEAT


# ---------------------------------------------------------------- 伴奏
def chord_notes(root, kind):
    third = 3 if kind == "min" else 4
    return [root, root + third, root + 7]


def pluck(freq, dur, bright=0.5):
    """Karplus-Strong 拨弦，近似古筝音色。"""
    n = int(dur * SR)
    period = max(int(SR / freq), 2)
    buf = np.random.uniform(-1, 1, period)
    out = np.empty(n)
    decay = 0.996
    for i in range(n):
        j = i % period
        out[i] = buf[j]
        buf[j] = decay * (bright * buf[j] + (1 - bright) * buf[(j + 1) % period])
    return out


def pad(freqs, dur):
    t = np.arange(int(dur * SR)) / SR
    y = sum(np.sin(2 * np.pi * f * t) + 0.3 * np.sin(4 * np.pi * f * t + 0.5) for f in freqs)
    env = np.minimum(1, t / 0.4) * np.minimum(1, (dur - t) / 0.5).clip(0)
    lfo = 1 + 0.08 * np.sin(2 * np.pi * 0.3 * t)
    return y * env * lfo / len(freqs)


def accompaniment(chords, total):
    out = np.zeros(int(total * SR) + 3 * SR)
    rng = np.random.default_rng(15)
    cache = {}

    def add(sig, at, gain):
        i = int(at * SR)
        seg = sig[: len(out) - i]
        out[i:i + len(seg)] += gain * seg

    for start, dur, (root, kind) in chords:
        notes = chord_notes(root, kind)
        add(pad([sing.midi_hz(n) for n in notes], dur + 0.3), start, 0.10)
        # 低音
        add(pad([sing.midi_hz(root - 12)], dur), start, 0.16)
        # 古筝琶音：八分音符上行下行
        arp = [notes[0], notes[1], notes[2], notes[0] + 12, notes[2] + 12, notes[0] + 12, notes[2], notes[1]]
        for k, n in enumerate(arp):
            m = n + 12
            if m not in cache:
                cache[m] = pluck(sing.midi_hz(m), 1.6, 0.5)
            add(cache[m], start + k * song.BEAT / 2 + rng.uniform(0, 0.008), 0.22 if k % 2 == 0 else 0.15)
    return out


def reverb(x, seconds=1.8, wet=0.22):
    rng = np.random.default_rng(3)
    n = int(seconds * SR)
    ir = rng.standard_normal(n) * np.exp(-np.arange(n) / SR * 6.9 / seconds)
    ir[: int(0.02 * SR)] = 0
    ir /= np.sqrt(np.sum(ir ** 2))
    from scipy.signal import fftconvolve
    return x + wet * fftconvolve(x, ir)[: len(x)]


def mix(lines, chords, total, wav_path):
    notes = [n for _, _, ns in lines for n in ns]
    print(f"合成人声：{len(notes)} 个音符…", flush=True)
    vocal = sing.render_vocal(notes, total + 2)
    vocal /= np.max(np.abs(vocal)) + 1e-9
    print("合成伴奏…", flush=True)
    acc = accompaniment(chords, total)
    acc /= np.max(np.abs(acc)) + 1e-9
    n = max(len(vocal), len(acc))
    vocal = np.pad(vocal, (0, n - len(vocal)))
    acc = np.pad(acc, (0, n - len(acc)))
    left = reverb(0.62 * vocal + 0.45 * acc)
    # 伴奏轻微立体声展开
    right = reverb(0.62 * vocal + 0.45 * np.roll(acc, int(0.012 * SR)))
    st = np.stack([left, right], axis=1)
    st = st[: int((total + 2.5) * SR)]
    fade = int(2.5 * SR)
    st[-fade:] *= np.linspace(1, 0, fade)[:, None]
    st *= 0.9 / np.max(np.abs(st))
    sf.write(wav_path, st, SR)
    return len(st) / SR


# ---------------------------------------------------------------- 画面
def background():
    img = Image.new("RGB", (W, H))
    top, bot = np.array([8, 12, 40]), np.array([40, 28, 70])
    grad = np.linspace(0, 1, H)[:, None, None]
    arr = (top * (1 - grad) + bot * grad).astype(np.uint8)
    arr = np.broadcast_to(arr, (H, W, 3)).copy()
    img = Image.fromarray(arr)
    # 月亮与光晕
    glow = Image.new("L", (W, H), 0)
    gd = ImageDraw.Draw(glow)
    cx, cy, r = 930, 210, 120
    for k in range(40, 0, -1):
        gd.ellipse([cx - r - k * 5, cy - r - k * 5, cx + r + k * 5, cy + r + k * 5], fill=int(120 * (1 - k / 40) ** 2))
    glow = glow.filter(ImageFilter.GaussianBlur(20))
    img = Image.composite(Image.new("RGB", (W, H), (255, 225, 150)), img, glow)
    d = ImageDraw.Draw(img)
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(255, 238, 180))
    rng = np.random.default_rng(8)
    for _ in range(9):  # 月面纹理
        a, rr = rng.uniform(0, 2 * math.pi), rng.uniform(0, r * 0.7)
        s = rng.uniform(10, 28)
        x, y = cx + rr * math.cos(a), cy + rr * math.sin(a)
        d.ellipse([x - s, y - s, x + s, y + s], fill=(244, 222, 160))
    # 远山剪影
    pts = [(0, H)]
    for x in range(0, W + 40, 40):
        pts.append((x, 520 + 40 * math.sin(x / 170) + 25 * math.sin(x / 67 + 1)))
    pts.append((W, H))
    d.polygon(pts, fill=(18, 16, 38))
    # 桂花枝
    d.line([(0, 90), (120, 120), (260, 110), (360, 150)], fill=(60, 40, 30), width=6)
    d.line([(120, 120), (170, 190)], fill=(60, 40, 30), width=4)
    for _ in range(70):
        x = rng.uniform(20, 360)
        y = 100 + (x / 360) * 50 + rng.uniform(-35, 45)
        s = rng.uniform(3, 6)
        d.ellipse([x - s, y - s, x + s, y + s], fill=(255, 200, 60))
    return img


def lantern(d, x, y, sway):
    x += sway
    d.line([(x - sway, y - 60), (x, y - 38)], fill=(120, 90, 60), width=2)
    d.ellipse([x - 30, y - 38, x + 30, y + 30], fill=(210, 40, 40))
    d.rectangle([x - 16, y - 44, x + 16, y - 36], fill=(230, 180, 60))
    d.rectangle([x - 16, y + 26, x + 16, y + 34], fill=(230, 180, 60))
    for k in range(-2, 3):
        d.line([(x + k * 3, y + 34), (x + k * 3, y + 58)], fill=(240, 190, 70), width=1)


def draw_line(d, font, notes, cy, t, active):
    text = "".join(c for _, c, _, _, _ in notes)
    w = d.textlength(text, font=font)
    x0 = (W - w) / 2
    for s, c, _, dur, _ in notes:
        cw = d.textlength(c, font=font)
        if not active:
            fill = (170, 170, 200)
        elif t >= s + dur:
            fill = (255, 205, 80)
        elif t >= s:
            fill = (255, 240, 170)
        else:
            fill = (235, 235, 245)
        d.text((x0 + 3, cy + 3), c, font=font, fill=(0, 0, 0), anchor="lm")
        d.text((x0, cy), c, font=font, fill=fill, anchor="lm")
        x0 += cw


def render_video(lines, duration, wav_path, out_path):
    bg = background()
    f_big = ImageFont.truetype(FONT, 56)
    f_small = ImageFont.truetype(FONT, 40)
    f_title = ImageFont.truetype(FONT, 84)
    f_sub = ImageFont.truetype(FONT, 34)
    rng = np.random.default_rng(1)
    stars = [(rng.uniform(0, W), rng.uniform(0, 450), rng.uniform(1, 2.5), rng.uniform(0, 6.28)) for _ in range(160)]
    stars = [s for s in stars if math.hypot(s[0] - 930, s[1] - 210) > 135]  # 不画在月亮上
    first_vocal = lines[0][1]
    last_end = max(s + d for _, _, ns in lines for s, _, _, d, _ in ns)

    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-i", wav_path,
           "-c:v", "mpeg4", "-vtag", "XVID", "-q:v", "3", "-pix_fmt", "yuv420p",
           "-c:a", "libmp3lame", "-b:a", "192k", "-shortest", out_path]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    nframes = int(duration * FPS)
    for fi in range(nframes):
        t = fi / FPS
        img = bg.copy()
        d = ImageDraw.Draw(img)
        for x, y, s, ph in stars:
            b = int(150 + 105 * math.sin(t * 2 + ph))
            d.ellipse([x - s, y - s, x + s, y + s], fill=(b, b, min(255, b + 20)))
        lantern(d, 520, 110, 8 * math.sin(t * 1.3))
        lantern(d, 640, 150, 8 * math.sin(t * 1.3 + 1))
        # 飘落的桂花
        for k in range(18):
            px = (k * 97 + 40 * math.sin(t * 0.7 + k)) % W
            py = ((t * (30 + k * 3) + k * 53) % (H + 40)) - 20
            d.ellipse([px - 3, py - 3, px + 3, py + 3], fill=(255, 205, 80))

        if t < first_vocal - 0.5 or t > last_end + 1.5:
            a = 1.0
            d.text((W / 2 + 4, 330 + 4), song.TITLE, font=f_title, fill=(0, 0, 0), anchor="mm")
            d.text((W / 2, 330), song.TITLE, font=f_title, fill=(255, 220, 120), anchor="mm")
            d.text((W / 2, 420), "—— " + song.SUBTITLE + " · 祝您中秋快乐，阖家团圆 ——",
                   font=f_sub, fill=(240, 230, 210), anchor="mm")
        else:
            cur = 0
            for i, (_, st, _) in enumerate(lines):
                if st - 0.6 <= t:
                    cur = i
            sec = lines[cur][0]
            d.text((40, 680), sec, font=f_sub, fill=(200, 190, 230), anchor="lm")
            draw_line(d, f_big, lines[cur][2], 560, t, True)
            if cur + 1 < len(lines):
                draw_line(d, f_small, lines[cur + 1][2], 630, t, False)
        proc.stdin.write(img.tobytes())
        if fi % (FPS * 10) == 0:
            print(f"  画面 {t:5.1f}s / {duration:.1f}s", flush=True)
    proc.stdin.close()
    if proc.wait() != 0:
        raise SystemExit("ffmpeg 编码失败")


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "月圆人团圆_中秋祝福歌.avi")
    lines, chords, total = timeline()
    wav = os.path.splitext(out)[0] + ".wav"
    dur = mix(lines, chords, total, wav)
    print(f"音频完成：{dur:.1f}s，开始渲染画面…", flush=True)
    render_video(lines, dur, wav, out)
    os.remove(wav)
    print("完成：", out)


if __name__ == "__main__":
    main()
