"""项目统一配置：路径、图像预处理、分类器推理和 VLM 服务默认值。

命令行参数仍可逐项覆盖；这里只放跨脚本共用的默认值。
"""

import os

# ---- 路径 ----
DATA_DIR = "data"
OUTPUT_DIR = "outputs"
ALL_IMAGES_JSONL = os.path.join(DATA_DIR, "all_images.jsonl")
SFT_JSONL = os.path.join(OUTPUT_DIR, "subtitle_sft.jsonl")
EVAL_RESULTS = os.path.join(OUTPUT_DIR, "eval_results.jsonl")
EVAL_VIDEO_RESULTS = os.path.join(OUTPUT_DIR, "eval_video.jsonl")
VLM_OUTPUT_SUFFIX = "_vlm.jsonl"
WEIGHTS_SUFFIX = "_weights.pth"

# ---- 图像预处理（ImageNet 归一化，与训练一致）----
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
RESIZE_RATIO = 1.14

# ---- 分类器推理 ----
CLASSIFIER_BATCH_SIZE = 64
VIDEO_INTERVAL = 60.0        # 视频抽帧间隔（秒）
CLASSIFIER_VIDEO_BATCH_SIZE = 64
DECODE_WORKERS = 1           # DataLoader 解码进程数
PREFETCH_FACTOR = 2          # 每个解码进程预取批数

# ---- 已部署的 VLM（OpenAI 兼容接口）----
VLM_BASE_URL = os.environ.get("VLM_BASE_URL", "http://127.0.0.1:8080")
VLM_API_KEY = os.environ.get("AIO_API_KEY", "dummy")
VLM_WORKERS = 16
VLM_TIMEOUT = 120.0
VLM_RETRIES = 3
VLM_BATCH_SIZE = 16          # 解码批大小（4K 帧别开太大）
VLM_MAX_SIDE = 896           # 上传前长边缩放

# ---- 提示词 ----
SFT_PROMPT = (
    "这是一帧电影画面。请判断画面中是否有字幕：有字幕则提取中文字幕和英文字幕，"
    "没有则 has_subtitle 为 false。只输出 JSON，格式为 "
    '{"has_subtitle": true, "subtitle_cn": "...", "subtitle_en": "..."}'
)
