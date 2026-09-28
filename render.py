# -*- coding: utf-8 -*-
"""
通用音MAD高品质音频合成器 (Universal Otomad Audio Renderer)
支持任意 MIDI 文件解析、素材自适应基频标定、智能音轨识别与交互处理、专属鼓组生成与多格式输出 (WAV / MP3 / M4A)。
"""

from __future__ import annotations

import argparse
import bisect
import os
import re
import sys
import time
from typing import Dict, List, Optional, Set, Tuple

import mido
import numpy as np
from scipy.signal import butter, sosfilt
import soxr
import soundfile as sf
import av

SR = 44100


# ==============================================================================
# 1. 速度映射表 (TempoMap) - 支持多变速 MIDI 文件毫秒级精准对齐
# ==============================================================================
class TempoMap:
    """根据 MIDI 中的全部 set_tempo 事件构建 Tick -> 秒 的单调递增折线映射。"""

    def __init__(self, midi_file: mido.MidiFile):
        self.ticks_per_beat = midi_file.ticks_per_beat
        events: List[Tuple[int, int]] = []

        for track in midi_file.tracks:
            curr = 0
            for msg in track:
                curr += msg.time
                if msg.type == 'set_tempo':
                    events.append((curr, msg.tempo))

        events.sort(key=lambda x: x[0])
        clean_events: List[Tuple[int, int]] = []
        last_tick = -1
        for tick, tempo in events:
            if tick == last_tick and clean_events:
                clean_events[-1] = (tick, tempo)
            else:
                clean_events.append((tick, tempo))
                last_tick = tick

        if not clean_events or clean_events[0][0] != 0:
            clean_events.insert(0, (0, 500000))  # 默认 120 BPM

        self.breakpoints: List[Tuple[int, float, int]] = []
        curr_time = 0.0
        prev_tick = 0
        prev_tempo = clean_events[0][1]

        for tick, tempo in clean_events:
            dt = tick - prev_tick
            curr_time += dt * (prev_tempo / 1_000_000.0) / self.ticks_per_beat
            self.breakpoints.append((tick, curr_time, tempo))
            prev_tick = tick
            prev_tempo = tempo

    def tick_to_seconds(self, tick: int) -> float:
        ticks = [b[0] for b in self.breakpoints]
        idx = bisect.bisect_right(ticks, tick) - 1
        bp_tick, bp_time, tempo = self.breakpoints[idx]
        return bp_time + (tick - bp_tick) * (tempo / 1_000_000.0) / self.ticks_per_beat


# ==============================================================================
# 2. 音高与素材工具 (Pitch & Sample Utilities)
# ==============================================================================
def parse_pitch(pitch_str: str) -> float | str:
    """解析音高输入：支持 float (70.0), 简记名称 (Bb4, A#4, C4), 或 'auto'。"""
    s = str(pitch_str).strip()
    if s.lower() == 'auto':
        return 'auto'
    try:
        return float(s)
    except ValueError:
        pass

    m = re.match(r'^([A-Ga-g])([#b♯♭]?)(-?\d+)$', s)
    if not m:
        raise ValueError(f"无法识别的音高格式: {pitch_str} (示例: 70.0, Bb4, A#4, auto)")
    note_name, acc, octv = m.group(1).upper(), m.group(2), int(m.group(3))
    base_map = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
    semitone = base_map[note_name]
    if acc in ('#', '♯'):
        semitone += 1
    elif acc in ('b', '♭'):
        semitone -= 1
    return 12.0 * (octv + 1) + semitone


def detect_root_pitch(sample: np.ndarray, sr: int) -> float:
    """自相关基频估算，返回最接近的标准 MIDI 音号。"""
    start = int(0.04 * sr)
    end = min(len(sample), int(0.45 * sr))
    if end - start < 1024:
        return 70.0

    seg = sample[start:end]
    corr = np.correlate(seg, seg, mode='full')[len(seg) - 1:]
    min_lag = int(sr / 1200)  # 最高 1200 Hz
    max_lag = int(sr / 65)    # 最低 65 Hz (C2)
    if max_lag >= len(corr):
        max_lag = len(corr) - 1
    if min_lag >= max_lag:
        return 70.0

    lag = min_lag + int(np.argmax(corr[min_lag:max_lag]))
    if min_lag < lag < max_lag - 1:
        y1, y2, y3 = corr[lag - 1], corr[lag], corr[lag + 1]
        denom = y1 - 2 * y2 + y3
        delta = 0.5 * (y1 - y3) / denom if denom != 0 else 0.0
        refined_lag = lag + delta
    else:
        refined_lag = float(lag)

    f0 = sr / refined_lag
    midi = 69.0 + 12.0 * np.log2(f0 / 440.0)
    # 如果极接近整数半音 (误差在 0.15 半音内)，吸附到标准音高
    if abs(midi - round(midi)) < 0.15:
        return float(round(midi))
    return float(midi)


def load_and_prep_sample(sample_path: str, sr: int = SR) -> Tuple[np.ndarray, np.ndarray]:
    """加载音频素材，精准保留辅音 Transient 爆破音头，并生成无缝 Pad 延音循环体。"""
    if not os.path.exists(sample_path):
        raise FileNotFoundError(f"找不到素材文件: {sample_path}")

    y, file_sr = sf.read(sample_path)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if file_sr != sr:
        y = soxr.resample(y, file_sr, sr, quality='VHQ')

    # 精准定位非静音起始点 (保留辅音爆破如 'B')
    peak = np.max(np.abs(y))
    idx = np.where(np.abs(y) > 0.005 * peak)[0][0]
    zc = np.where(np.diff(np.signbit(y[:idx + 1])))[0]
    start = zc[-1] if len(zc) > 0 else idx

    end = np.where(np.abs(y) > 0.002 * peak)[0][-1]
    sample_clean = y[start:end].astype(np.float32)

    # 寻找最佳元音延音无缝循环节
    s0 = int(0.10 * sr)
    target_len = int(50 * (sr / 466.16))
    best_L = target_len
    best_err = 1e9
    search_range = range(max(100, target_len - 150), min(len(sample_clean) - s0 - 400, target_len + 150))
    for L in search_range:
        diff = np.mean((sample_clean[s0:s0 + 300] - sample_clean[s0 + L:s0 + L + 300]) ** 2)
        if diff < best_err:
            best_err = diff
            best_L = L

    xfade = min(128, best_L // 4)
    loop_core = sample_clean[s0:s0 + best_L].copy()
    w = np.linspace(0, 1, xfade)
    loop_core[:xfade] = loop_core[:xfade] * w + sample_clean[s0 + best_L - xfade:s0 + best_L] * (1 - w)
    needed_rep = int(5.5 * sr / len(loop_core)) + 2
    extended_pad = np.concatenate([sample_clean[:s0], np.tile(loop_core, needed_rep)])

    return sample_clean, extended_pad


# ==============================================================================
# 3. 专属音MAD鼓组发生器 (Drum Generator)
# ==============================================================================
def build_drum_kit(sample_clean: np.ndarray, sr: int = SR) -> Dict[str, np.ndarray]:
    """基于单一音频素材，生成打击乐三大件：重低音底鼓、爆破军鼓、清脆踩镲。"""
    # 1. Kick: 辅音瞬态降速到 ~55Hz + 600Hz低通 + 扎实冲击包络
    rate_kick = 0.35
    kick_raw = soxr.resample(sample_clean[:int(0.20 * sr)], int(sr * rate_kick), sr, quality='VHQ')
    sos_kick = butter(2, 600 / (sr / 2), btype='low', output='sos')
    kick = sosfilt(sos_kick, kick_raw)
    n_k = min(len(kick), int(0.22 * sr))
    t_k = np.arange(n_k) / sr
    kick = kick[:n_k] * np.exp(-12.0 * t_k)
    kick = kick / (np.max(np.abs(kick)) + 1e-8)

    # 2. Snare: 原声清脆瞬态 + 紧致衰减 + 切除泥泞超低频
    sos_snare = butter(2, 160 / (sr / 2), btype='high', output='sos')
    snare_raw = sosfilt(sos_snare, sample_clean[:int(0.20 * sr)])
    t_s = np.arange(len(snare_raw)) / sr
    snare = snare_raw * np.exp(-8.5 * t_s)
    snare = snare / (np.max(np.abs(snare)) + 1e-8)

    # 3. Hi-Hat: 辅音瞬态提速 (rate=2.4) + 3500Hz高通 + 极速衰减
    rate_hat = 2.4
    hat_raw = soxr.resample(sample_clean[:int(0.06 * sr)], int(sr * rate_hat), sr, quality='VHQ')
    sos_hat = butter(2, 3500 / (sr / 2), btype='high', output='sos')
    hat = sosfilt(sos_hat, hat_raw)
    n_h = min(len(hat), int(0.05 * sr))
    t_h = np.arange(n_h) / sr
    hat = hat[:n_h] * np.exp(-38.0 * t_h)
    hat = hat / (np.max(np.abs(hat)) + 1e-8)

    return {"kick": kick.astype(np.float32), "snare": snare.astype(np.float32), "hat": hat.astype(np.float32)}


# ==============================================================================
# 4. 音轨智能分类与人机交互 (Track Classification & Interactive Prompting)
# ==============================================================================
ROLE_CONFIGS = {
    "lead":   dict(name="主旋律 (Lead)",   gain=1.15, pan=0.00,  rel=0.030, decay=0.0,  pad=False, lowpass=None),
    "piano":  dict(name="键盘和弦 (Piano)", gain=0.85, pan=-0.10, rel=0.025, decay=0.25, pad=False, lowpass=None),
    "guitar": dict(name="拨弦吉他 (Guitar)", gain=0.75, pan=0.25,  rel=0.025, decay=0.40, pad=False, lowpass=None),
    "arp":    dict(name="琶音跳跃 (Arp)",   gain=0.65, pan=-0.30, rel=0.015, decay=0.0,  pad=False, lowpass=None),
    "bass":   dict(name="低音贝斯 (Bass)",  gain=0.90, pan=0.00,  rel=0.035, decay=0.0,  pad=False, lowpass=850),
    "pad":    dict(name="氛围铺底 (Pad)",   gain=0.40, pan=0.45,  rel=0.150, decay=0.0,  pad=True,  lowpass=None),
    "drum":   dict(name="打击乐 (Drum)",    gain=1.00, pan=0.00,  rel=0.020, decay=0.0,  is_drum=True),
    "fx":     dict(name="音效点缀 (FX)",    gain=0.55, pan=0.30,  rel=0.030, decay=0.0,  pad=False, lowpass=None),
    "mute":   dict(name="静音跳过 (Mute)",  gain=0.00, pan=0.00,  rel=0.000, decay=0.0,  mute=True),
}


def _match_kw(text: str, keywords: List[str]) -> bool:
    pattern = r'\b(' + '|'.join(re.escape(k) for k in keywords) + r')\b'
    return bool(re.search(pattern, text, re.IGNORECASE))


def classify_track_rule(name: str, channels: Set[int], programs: Set[int]) -> Optional[Tuple[str, str]]:
    """根据通道、常规名称、GM音色号进行高确信度判定。"""
    # 1. 打击乐通道 (Channel 9 / 通道10)
    if 9 in channels or _match_kw(name, ['drum', 'drums', 'percussion', 'kit', 'beat']):
        return 'drum', '打击乐通道/关键词'

    # 2. 常见音色名称关键词匹配
    if _match_kw(name, ['bass', 'sub', '808']):
        return 'bass', '低音关键词'
    if _match_kw(name, ['lead', 'vocal', 'vox', 'melody', 'solo', 'melodine', 'sing']):
        return 'lead', '主旋律关键词'
    if _match_kw(name, ['piano', 'keys', 'ep', 'rhodes', 'keyboard', 'clav']):
        return 'piano', '键盘关键词'
    if _match_kw(name, ['pad', 'string', 'strings', 'choir', 'ensemble', 'ambient']):
        return 'pad', '铺底/弦乐关键词'
    if _match_kw(name, ['guitar', 'gtr', 'pluck', 'harp', 'pizz']):
        return 'guitar', '吉他/拨弦关键词'
    if _match_kw(name, ['arp', 'seq', 'arpeggio', 'beep', 'bell', 'chirp']):
        return 'arp', '琶音/音阶关键词'
    if _match_kw(name, ['fx', 'effect', 'sfx', 'rev', 'noise', 'sweep', 'drop', 'riser']):
        return 'fx', '音效关键词'

    # 3. GM 标准音色程序号匹配 (若有非0程序号)
    non_zero_progs = [p for p in programs if p != 0]
    if non_zero_progs:
        p = non_zero_progs[0]
        if 32 <= p <= 39:
            return 'bass', f'GM 音色 {p} (Bass)'
        if (40 <= p <= 55) or (88 <= p <= 95):
            return 'pad', f'GM 音色 {p} (Pad/Strings)'
        if 0 <= p <= 7:
            return 'piano', f'GM 音色 {p} (Piano)'
        if 24 <= p <= 31:
            return 'guitar', f'GM 音色 {p} (Guitar)'
        if 56 <= p <= 87:
            return 'lead', f'GM 音色 {p} (Lead/Wind)'
        if (96 <= p <= 103) or (120 <= p <= 127):
            return 'fx', f'GM 音色 {p} (FX)'
        if 112 <= p <= 119:
            return 'drum', f'GM 音色 {p} (Percussive)'

    return None


def prompt_user_for_track(track_idx: int, track_name: str, notes: List[Tuple[float, float, int, int]], auto_mode: bool) -> str:
    """遇到罕见/未识别音轨时，向用户呈现音符统计信息并询问处理方式。"""
    pitches = [n[2] for n in notes]
    min_p, max_p = min(pitches), max(pitches)
    med_p = float(np.median(pitches))

    # 智能推断推荐默认值
    if med_p < 48:
        rec_key, rec_idx = "bass", 4
    elif (max_p - min_p > 24) or len(notes) > 400:
        rec_key, rec_idx = "piano", 2
    elif med_p > 72 and len(notes) > 150:
        rec_key, rec_idx = "arp", 7
    else:
        rec_key, rec_idx = "lead", 1

    if auto_mode or not sys.stdin.isatty():
        print(f"[*] 音轨 [{track_idx}] '{track_name}' (音符数: {len(notes)}, 中位音高: {med_p:.0f}) -> 自动归类为: {ROLE_CONFIGS[rec_key]['name']}")
        return rec_key

    # 交互式询问
    print("\n" + "=" * 72)
    print(f"[?] 检测到未识别/罕见音轨: Track {track_idx} ('{track_name}')")
    print(f"    音符总数: {len(notes)} | 音高范围: {min_p} ~ {max_p} (中位音高: {med_p:.0f})")
    print("    请选择该音轨在音MAD中的角色定位:")
    print("      [1] Lead   - 主旋律 (响亮饱满、置中定位、全频人声)")
    print("      [2] Piano  - 键盘和弦 (和声厚重、自然衰减、微展宽)")
    print("      [3] Guitar - 拨弦吉他 (节奏扫弦/拨弦、明亮紧凑)")
    print("      [4] Bass   - 低音贝斯 (850Hz低通滤波、浑厚打底)")
    print("      [5] Pad    - 氛围铺底 (无缝平滑延音、大声场空间)")
    print("      [6] Drum   - 节奏打击乐 (按音高拆解底鼓/军鼓/镲片)")
    print("      [7] Arp    - 琶音跳跃 (16分音符快速切音、清爽灵动)")
    print("      [8] FX     - 音效点缀 (立体声反转/过渡音效)")
    print("      [0] Mute   - 静音跳过 (忽略此轨)")
    print(f"    默认推荐: [{rec_idx}] {ROLE_CONFIGS[rec_key]['name']}")

    choice_map = {
        '1': 'lead', '2': 'piano', '3': 'guitar', '4': 'bass',
        '5': 'pad', '6': 'drum', '7': 'arp', '8': 'fx', '0': 'mute'
    }

    while True:
        try:
            val = input(f"    请输入对应数字 [0-8, 直接回车默认 {rec_idx}]: ").strip()
            if not val:
                print(f"    -> 已采纳推荐: {ROLE_CONFIGS[rec_key]['name']}\n")
                return rec_key
            if val in choice_map:
                chosen = choice_map[val]
                print(f"    -> 已设置为: {ROLE_CONFIGS[chosen]['name']}\n")
                return chosen
            print("    [!] 输入有误，请输入 0 到 8 之间的数字。")
        except (EOFError, KeyboardInterrupt):
            print(f"\n    -> 使用默认推荐: {ROLE_CONFIGS[rec_key]['name']}\n")
            return rec_key


# ==============================================================================
# 5. 音频编码与输出 (WAV / MP3 / M4A)
# ==============================================================================
def export_audio(output_path: str, audio: np.ndarray, sr: int = SR, fmt: Optional[str] = None):
    """支持无损 WAV、MP3、以及通过 PyAV 硬件/原生编码的 AAC M4A 格式输出。"""
    if fmt is None:
        ext = os.path.splitext(output_path)[1].lower().lstrip('.')
        fmt = ext if ext in ['wav', 'mp3', 'm4a'] else 'wav'

    fmt = fmt.lower()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    if fmt == 'wav':
        sf.write(output_path, audio, sr, subtype='PCM_16')
    elif fmt == 'mp3':
        try:
            sf.write(output_path, audio, sr, format='MP3')
        except Exception:
            _export_via_pyav(output_path, audio, sr, codec='mp3', container_fmt='mp3', bitrate=320000)
    elif fmt == 'm4a':
        _export_via_pyav(output_path, audio, sr, codec='aac', container_fmt='ipod', bitrate=256000)
    else:
        raise ValueError(f"不支持的音频导出格式: {fmt} (可选: wav, mp3, m4a)")


def _export_via_pyav(path: str, audio: np.ndarray, sr: int, codec: str, container_fmt: str, bitrate: int):
    container = av.open(path, mode='w', format=container_fmt)
    stream = container.add_stream(codec, rate=sr)
    stream.bit_rate = bitrate
    stream.layout = 'stereo'

    frame_size = 1024
    total_samples = len(audio)
    for i in range(0, total_samples, frame_size):
        chunk = audio[i:i + frame_size]
        if len(chunk) < frame_size:
            chunk = np.pad(chunk, ((0, frame_size - len(chunk)), (0, 0)))
        chunk_planar = np.ascontiguousarray(chunk.T, dtype=np.float32)
        frame = av.AudioFrame.from_ndarray(chunk_planar, format='fltp', layout='stereo')
        frame.rate = sr
        for packet in stream.encode(frame):
            container.mux(packet)

    for packet in stream.encode():
        container.mux(packet)
    container.close()


# ==============================================================================
# 6. 主渲染引擎 (Core Synthesis Engine)
# ==============================================================================
def render_otomad(
    midi_path: str,
    sample_path: str,
    output_path: str,
    output_format: str = "wav",
    root_pitch_arg: str | float = "auto",
    bpm_override: Optional[float] = None,
    auto_mode: bool = False,
    gain_db: float = 0.0,
):
    print("\n" + "=" * 72)
    print("  通用音MAD高品质音频合成器 (Universal Otomad Engine)")
    print("=" * 72)

    # 1. 加载并准备素材
    print(f"[*] 步骤 1/7: 加载音效素材: {os.path.basename(sample_path)}")
    sample_clean, extended_pad = load_and_prep_sample(sample_path, SR)

    # 标定根音基频
    if str(root_pitch_arg).lower() == 'auto':
        root_pitch = detect_root_pitch(sample_clean, SR)
        print(f"    -> 自动探测素材基频: MIDI {root_pitch:.2f}")
    else:
        root_pitch = float(root_pitch_arg)
        print(f"    -> 使用指定素材基频: MIDI {root_pitch:.2f}")

    # 2. 制造专属音MAD打击乐组
    print("[*] 步骤 2/7: 锻造基于素材声学 Transient 的专属鼓组 (Kick / Snare / Hi-Hat)...")
    drum_kit = build_drum_kit(sample_clean, SR)

    # 3. 预渲染全音域高品质变调音色缓存
    print("[*] 步骤 3/7: 预渲染 64-bit Sinc (VHQ) 变调音色库...")
    t_start_cache = time.time()
    pitch_cache: Dict[int, np.ndarray] = {}
    pitch_cache_pad: Dict[int, np.ndarray] = {}
    for p in range(21, 109):
        rate = 2.0 ** ((p - root_pitch) / 12.0)
        pitch_cache[p] = soxr.resample(sample_clean, int(SR * rate), SR, quality='VHQ')
        if 36 <= p <= 96:
            pitch_cache_pad[p] = soxr.resample(extended_pad, int(SR * rate), SR, quality='VHQ')

    sos_bass = butter(2, 850 / (SR / 2), btype='low', output='sos')
    for p in range(24, 72):
        pitch_cache[f"bass_{p}"] = sosfilt(sos_bass, pitch_cache[p])
    print(f"    -> 88 个钢琴键变调缓存就绪，耗时 {time.time() - t_start_cache:.2f} 秒")

    # 4. 解析 MIDI 与构建多变速时间轴
    print(f"[*] 步骤 4/7: 解析 MIDI 文件: {os.path.basename(midi_path)}")
    mid = mido.MidiFile(midi_path)
    tempo_map = TempoMap(mid)

    # 速度缩放因子
    speed_factor = 1.0
    if bpm_override is not None and bpm_override > 0:
        # 以第一节有效 BPM 计算缩放
        orig_bpm = mido.tempo2bpm(tempo_map.breakpoints[0][2])
        speed_factor = bpm_override / orig_bpm
        print(f"    -> 变速模式开启: 基础 BPM {orig_bpm:.1f} -> 目标 BPM {bpm_override:.1f} (缩放 {speed_factor:.2f}x)")

    # 收集各轨道音符事件
    track_notes: Dict[int, List[Tuple[float, float, int, int]]] = {}
    track_info: Dict[int, Tuple[str, Set[int], Set[int]]] = {}
    max_sec = 0.0

    for t_idx, track in enumerate(mid.tracks):
        curr_tick = 0
        active: Dict[int, Tuple[float, int]] = {}
        notes: List[Tuple[float, float, int, int]] = []
        channels = set()
        programs = set()

        for msg in track:
            curr_tick += msg.time
            if hasattr(msg, 'channel'):
                channels.add(msg.channel)
            if msg.type == 'program_change':
                programs.add(msg.program)

            t_sec = tempo_map.tick_to_seconds(curr_tick) / speed_factor

            if msg.type == 'note_on' and msg.velocity > 0:
                active[msg.note] = (t_sec, msg.velocity)
            elif (msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0)) and msg.note in active:
                st, vel = active.pop(msg.note)
                dur = max(0.02, t_sec - st)
                notes.append((st, dur, msg.note, vel))
                max_sec = max(max_sec, st + dur)

        if notes:
            track_notes[t_idx] = notes
            track_info[t_idx] = (track.name.strip() or f"Track {t_idx}", channels, programs)

    print(f"    -> MIDI 解析完成: 共 {len(track_notes)} 条含音符轨道，总时长约 {max_sec:.1f} 秒")

    # 5. 音轨识别与处理策略决策
    print("[*] 步骤 5/7: 智能分析并分配各轨道角色...")
    track_roles: Dict[int, str] = {}
    for t_idx, (t_name, channels, programs) in track_info.items():
        rule_res = classify_track_rule(t_name, channels, programs)
        if rule_res is not None:
            role, reason = rule_res
            track_roles[t_idx] = role
            print(f"    [√] 自动匹配 Track {t_idx:2d} ('{t_name}'): {ROLE_CONFIGS[role]['name']} ({reason})")
        else:
            role = prompt_user_for_track(t_idx, t_name, track_notes[t_idx], auto_mode)
            track_roles[t_idx] = role

    # 6. 立体声总线渲染与动态合成
    total_sec = max_sec + 3.0
    total_samples = int(total_sec * SR)
    master = np.zeros((total_samples, 2), dtype=np.float32)

    def get_pan_gains(pan: float) -> Tuple[float, float]:
        theta = (np.clip(pan, -1.0, 1.0) + 1.0) * 0.25 * np.pi
        return float(np.cos(theta)), float(np.sin(theta))

    print(f"[*] 步骤 6/7: 全轨道并行采样与高保真声场混音 (预分配 {total_samples} 采样点)...")
    pad_toggle = 0
    total_notes_rendered = 0

    for t_idx, notes in track_notes.items():
        role = track_roles[t_idx]
        cfg = ROLE_CONFIGS[role]
        if cfg.get("mute", False):
            continue

        base_gain = cfg["gain"]
        base_pan = cfg["pan"]
        rel_sec = cfg["rel"]
        decay = cfg["decay"]
        is_drum = cfg.get("is_drum", False)
        is_pad = cfg.get("pad", False)
        lowpass = cfg.get("lowpass", None)

        for st, dur, pitch, vel in notes:
            start_sample = int(st * SR)
            vel_scale = (vel / 127.0) ** 1.15

            if is_drum:
                # 依据打击乐音高智能路由
                if pitch <= 36 or pitch in (25, 26, 27, 28, 29):
                    note_audio = drum_kit["kick"]
                    pan = 0.0
                    g = base_gain * 1.05 * vel_scale
                elif pitch in (37, 38, 39, 40, 41):
                    note_audio = drum_kit["snare"]
                    pan = 0.0
                    g = base_gain * 0.90 * vel_scale
                else:
                    note_audio = drum_kit["hat"]
                    pan = 0.20
                    g = base_gain * 0.55 * vel_scale

                audio_len = len(note_audio)
                if start_sample + audio_len > total_samples:
                    audio_len = total_samples - start_sample
                gl, gr = get_pan_gains(pan)
                master[start_sample:start_sample + audio_len, 0] += note_audio[:audio_len] * (g * gl)
                master[start_sample:start_sample + audio_len, 1] += note_audio[:audio_len] * (g * gr)
                total_notes_rendered += 1

            elif is_pad:
                raw = pitch_cache_pad.get(pitch)
                if raw is None:
                    raw = pitch_cache.get(pitch, sample_clean)

                needed_len = int((dur + rel_sec) * SR)
                if len(raw) < needed_len:
                    pad_audio = np.pad(raw, (0, needed_len - len(raw)), mode='edge')
                else:
                    pad_audio = raw[:needed_len].copy()

                atk_n = int(cfg.get("atk", 0.035) * SR)
                if len(pad_audio) > atk_n:
                    pad_audio[:atk_n] *= np.linspace(0, 1, atk_n)
                rel_n = int(rel_sec * SR)
                if len(pad_audio) > rel_n:
                    pad_audio[-rel_n:] *= np.linspace(1, 0, rel_n)

                pad_toggle += 1
                pan = -0.45 if (pad_toggle % 2 == 0) else 0.45
                gl, gr = get_pan_gains(pan)
                g = base_gain * vel_scale

                audio_len = len(pad_audio)
                if start_sample + audio_len > total_samples:
                    audio_len = total_samples - start_sample
                master[start_sample:start_sample + audio_len, 0] += pad_audio[:audio_len] * (g * gl)
                master[start_sample:start_sample + audio_len, 1] += pad_audio[:audio_len] * (g * gr)
                total_notes_rendered += 1

            else:
                cache_key = f"bass_{pitch}" if (lowpass and f"bass_{pitch}" in pitch_cache) else pitch
                raw = pitch_cache.get(cache_key, sample_clean)
                needed_len = int((dur + rel_sec) * SR)

                if len(raw) >= needed_len:
                    note_audio = raw[:needed_len].copy()
                    rel_n = int(rel_sec * SR)
                    note_audio[-rel_n:] *= np.linspace(1, 0, rel_n)
                else:
                    note_audio = raw.copy()
                    if decay > 0:
                        t_d = np.arange(len(note_audio)) / SR
                        note_audio *= np.exp(-decay * t_d)

                gl, gr = get_pan_gains(base_pan)
                g = base_gain * vel_scale
                audio_len = len(note_audio)
                if start_sample + audio_len > total_samples:
                    audio_len = total_samples - start_sample
                master[start_sample:start_sample + audio_len, 0] += note_audio[:audio_len] * (g * gl)
                master[start_sample:start_sample + audio_len, 1] += note_audio[:audio_len] * (g * gr)
                total_notes_rendered += 1

    print(f"    -> 成功渲染并混合全部 {total_notes_rendered} 个音符事件！")

    # 7. 母带精修 (Mastering) 与格式输出
    print("[*] 步骤 7/7: 母带处理 (25Hz高通滤波、动态软限幅、-0.3dBFS 标准化)...")
    sos_hp = butter(2, 25 / (SR / 2), btype='high', output='sos')
    master[:, 0] = sosfilt(sos_hp, master[:, 0])
    master[:, 1] = sosfilt(sos_hp, master[:, 1])

    # 用户额外增益
    if gain_db != 0.0:
        master *= (10 ** (gain_db / 20.0))

    threshold = 0.75
    over = np.abs(master) > threshold
    master[over] = np.sign(master[over]) * (
        threshold + (1.0 - threshold) * np.tanh((np.abs(master[over]) - threshold) / (1.0 - threshold))
    )

    ceiling = 10 ** (-0.3 / 20.0)
    master = master * (ceiling / (np.max(np.abs(master)) + 1e-8))

    final_rms = float(np.sqrt(np.mean(master ** 2)))
    final_db = 20 * np.log10(final_rms + 1e-8)
    print(f"    -> 最终母带峰值: -0.30 dBFS | RMS 响度: {final_db:.1f} dBFS")

    print(f"[*] 导出目标音频 [{output_format.upper()}]: {output_path}")
    export_audio(output_path, master, SR, fmt=output_format)
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[√] 音频生成成功！文件大小: {file_size_mb:.2f} MB")
    print("=" * 72 + "\n")


# ==============================================================================
# 7. 命令行接口 (CLI Entrypoint)
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="通用音MAD高品质音频合成器 (Universal Otomad Audio Renderer)",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""示例用法:
  python render.py Dataerror.mid
  python render.py melodiniq.mid -f mp3
  python render.py melodiniq.mid -f m4a -o my_song.m4a
  python render.py Dataerror.mid -s 冰冰冰.mp3 -r 70.0 -f m4a
  python render.py melodiniq.mid -y  # 全自动模式 (不弹出交互式询问)
        """
    )
    parser.add_argument("midi", nargs="?", default=None, help="输入 MIDI 文件路径 (.mid)")
    parser.add_argument("-m", "--midi", dest="midi_flag", default=None, help="输入 MIDI 文件路径 (与位置参数等价)")
    parser.add_argument("-s", "--sample", default="冰冰冰.mp3", help="素材音频文件路径 (默认: 冰冰冰.mp3)")
    parser.add_argument("-o", "--output", default=None, help="输出文件路径 (默认依据 MIDI 与素材名自动命名)")
    parser.add_argument("-f", "--format", choices=["wav", "mp3", "m4a"], default=None, help="输出格式: wav (默认无损), mp3, m4a")
    parser.add_argument("-r", "--root-pitch", default="auto", help="素材基频音高 (如 70.0, Bb4, 或 auto 自动检测, 默认: auto)")
    parser.add_argument("-b", "--bpm", type=float, default=None, help="自定义播放 BPM 速度 (默认使用 MIDI 速度映射表)")
    parser.add_argument("-y", "--auto", "--non-interactive", dest="auto_mode", action="store_true", help="非交互模式: 遇到罕见音轨依据音高自动推荐处理，不暂停询问")
    parser.add_argument("-g", "--gain", type=float, default=0.0, help="总线增益补偿 (dB, 默认: 0.0)")

    args = parser.parse_args()

    # 确定 MIDI 文件路径
    midi_file = args.midi or args.midi_flag
    if midi_file is None:
        # 自动查找当前目录下存在的 mid 文件
        candidates = [f for f in os.listdir('.') if f.lower().endswith('.mid')]
        if "Dataerror.mid" in candidates:
            midi_file = "Dataerror.mid"
        elif candidates:
            midi_file = candidates[0]
        else:
            parser.error("未指定输入 MIDI 文件，且当前目录下未找到 .mid 文件。请使用 `python render.py <file.mid>`。")

    if not os.path.exists(midi_file):
        sys.exit(f"错误: 找不到 MIDI 文件: {midi_file}")

    # 确定素材路径
    sample_file = args.sample
    if not os.path.exists(sample_file):
        sys.exit(f"错误: 找不到素材文件: {sample_file}")

    # 确定格式与输出路径
    fmt = args.format
    out_file = args.output
    if out_file is not None:
        ext = os.path.splitext(out_file)[1].lower().lstrip('.')
        if fmt is None and ext in ['wav', 'mp3', 'm4a']:
            fmt = ext
    if fmt is None:
        fmt = "wav"

    if out_file is None:
        midi_stem = os.path.splitext(os.path.basename(midi_file))[0]
        sample_stem = os.path.splitext(os.path.basename(sample_file))[0]
        out_file = f"{midi_stem}_{sample_stem}.{fmt}"

    # 解析根音
    try:
        root_pitch_val = parse_pitch(args.root_pitch)
    except ValueError as e:
        sys.exit(f"错误: {e}")

    # 执行渲染
    render_otomad(
        midi_path=midi_file,
        sample_path=sample_file,
        output_path=out_file,
        output_format=fmt,
        root_pitch_arg=root_pitch_val,
        bpm_override=args.bpm,
        auto_mode=args.auto_mode,
        gain_db=args.gain,
    )


if __name__ == "__main__":
    main()
