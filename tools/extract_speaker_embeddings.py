#!/usr/bin/env python
"""提取语音片段（一句一条的 wav）的说话人 embedding，保存成 pkl。

用法:
    python -m tools.extract_speaker_embeddings --segments data/segments/阿甘正传
    python -m tools.extract_speaker_embeddings --segments data/segments/阿甘正传 \
        --ckpt ckp-tag/mect_b2_vc2.pt -o outputs/阿甘正传_speaker.pkl

输出 pkl 结构：
    {"ckpt": ..., "embedding_dim": 256, "sampling_rate": 16000,
     "items": [{"file": "0001_xxx.wav", "index": 1, "text": "...", "embedding": np.ndarray}, ...]}
"""

import argparse
import glob
import json
import logging
import os
import pickle
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
# antspeaker 是 vendored 的第三方包（内部用 `from antspeaker.xxx import ...` 绝对导入），
# 直接把包所在目录加进 sys.path，而不是当成 third_party.xxx 的子模块
sys.path.insert(0, os.path.join(ROOT, "third_party", "AntSpeaker"))

import soundfile as sf
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
from tqdm import tqdm
import antspeaker.models  # noqa: F401  必须先 import，才会把 backbone/model 注册进 registry
from antspeaker.utils.registry import create_model

from classifier.utils import default_device

logger = logging.getLogger("speaker_emb")


def build_speaker_model(ckpt_path, device):
    """加载 antspeaker 模型和它的 fbank 配置，返回 (model, feat_config)。"""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = create_model(ckpt["config"]["model"])
    model, _ = model.load_model(ckpt_path)
    return model.to(device).eval(), ckpt["config"]["feature"]


def extract_embedding(path, model, feat, device):
    """单条音频 -> (embedding ndarray[D], 原采样率)。"""
    waveform, sr = sf.read(path, dtype="float32")
    if waveform.ndim > 1:  # 多声道取均值，模型只吃单声道
        waveform = waveform.mean(axis=1)
    wav = torch.from_numpy(waveform).unsqueeze(0)
    if sr != feat["sampling_rate"]:
        wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=feat["sampling_rate"])(wav)
    fbank = kaldi.fbank(
        wav * (1 << 15),
        num_mel_bins=feat.get("num_mel_bins", 80),
        frame_length=feat.get("frame_length", 25),
        frame_shift=feat.get("frame_shift", 10),
        dither=feat.get("dither", 0.0),
        energy_floor=0.0,
        window_type="hamming",
        sample_frequency=feat["sampling_rate"],
    )
    fbank = fbank.unsqueeze(0).transpose(-1, -2).to(device)
    with torch.no_grad():
        embedding = model(fbank).cpu().numpy().squeeze(0)
    return embedding, sr


def load_manifest(segments_dir):
    """segments.jsonl -> {文件名: 该条记录}，没有就返回空 dict。"""
    path = os.path.join(segments_dir, "segments.jsonl")
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        return {rec["file"]: rec for rec in (json.loads(l) for l in f if l.strip())}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--segments", required=True, help="语音片段目录（tools.split_audio_by_srt 的输出）")
    parser.add_argument("--ckpt", default="ckp-tag/mect_b2_vc2.pt", help="说话人模型权重")
    parser.add_argument("-o", "--output", default=None, help="默认 <片段目录>/speaker_embeddings.pkl")
    parser.add_argument("--pattern", default="*.wav", help="片段文件名匹配（默认 *.wav）")
    parser.add_argument("--device", default=default_device(), help="默认 cuda > mps > cpu")
    parser.add_argument("-v", "--verbose", action="store_true", help="每条打印一行结果")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if not os.path.isdir(args.segments):
        logger.error("找不到片段目录: %s", args.segments)
        return 1
    if not os.path.isfile(args.ckpt):
        logger.error("找不到模型权重: %s", args.ckpt)
        return 1

    files = sorted(glob.glob(os.path.join(args.segments, args.pattern)))
    if not files:
        logger.error("%s 下没有匹配 %s 的文件", args.segments, args.pattern)
        return 1

    manifest = load_manifest(args.segments)
    output = args.output or os.path.join(args.segments, "speaker_embeddings.pkl")
    logger.info("ckpt=%s device=%s 片段 %d 条", args.ckpt, args.device, len(files))

    model, feat = build_speaker_model(args.ckpt, torch.device(args.device))
    logger.info("sampling_rate=%d num_mel_bins=%d", feat["sampling_rate"], feat.get("num_mel_bins", 80))

    items, failed = [], 0
    device = torch.device(args.device)
    for path in tqdm(files, desc="speaker embedding", unit="条", dynamic_ncols=True):
        name = os.path.basename(path)
        try:
            embedding, sr = extract_embedding(path, model, feat, device)
        except Exception as e:  # 单条坏了不影响整批
            failed += 1
            tqdm.write(f"失败 {name}: {e}")
            continue
        rec = manifest.get(name, {})
        items.append({
            "file": name,
            "index": rec.get("index"),
            "text": rec.get("text"),
            "start": rec.get("start"),
            "end": rec.get("end"),
            "sr": sr,
            "embedding": embedding,
        })
        if args.verbose:
            tqdm.write(f"{name} -> {sr} Hz")

    with open(output, "wb") as f:
        pickle.dump({
            "ckpt": args.ckpt,
            "embedding_dim": int(items[0]["embedding"].shape[0]) if items else 0,
            "sampling_rate": feat["sampling_rate"],
            "items": items,
        }, f, protocol=4)

    logger.info("完成：%d 条，失败 %d，embedding_dim=%s -> %s",
                len(items), failed, items[0]["embedding"].shape[0] if items else 0, output)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
