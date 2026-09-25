# 月圆人团圆 —— 中秋祝福歌

成品：`月圆人团圆_中秋祝福歌.avi`（1280×720，XviD 视频 + MP3 立体声，约 2 分 20 秒），画面为月夜、灯笼、桂花与卡拉OK式逐字歌词。

## 歌词

**主歌一**
八月十五月儿圆，桂花飘香满庭院。
一盏清茶一块饼，思念飞过万重山。

**副歌**
月亮圆，人团圆，千里共婵娟。
愿你平安又喜乐，岁岁有今天。

**主歌二**
灯笼高挂照人间，笑声阵阵暖心田。
遥望明月寄祝福，天涯海角共团圆。

**副歌 ×2**

C 大调五声音阶，4/4 拍，84 BPM；伴奏为古筝风格琶音 + 柔和铺底 + 低音。

## 关于演唱

歌声是 **AI 合成的女声**，不是真人录音：用离线神经网络语音模型
[vits-melo-tts-zh_en](https://github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models) 逐字发音，
再用 WORLD 声码器（pyworld）把每个字的音高、时长改到乐谱上，并加入滑音和颤音。
想要真人演唱，可以请歌手按下面的乐谱（`song.py`）录一条人声，替换掉合成人声。

## 重新生成

```bash
apt-get install -y ffmpeg fonts-wqy-zenhei
pip install numpy scipy pillow soundfile pyworld sherpa-onnx
curl -LO https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-melo-tts-zh_en.tar.bz2
tar xjf vits-melo-tts-zh_en.tar.bz2
TTS_MODEL_DIR=vits-melo-tts-zh_en python3 build.py
```

- `song.py`：歌词、旋律、和弦与曲式
- `sing.py`：歌声合成
- `build.py`：伴奏、混音、画面渲染，并输出 AVI
