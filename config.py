"""项目统一配置：路径、阈值、模型服务地址。

各脚本 argparse 的 default 都从这里取，命令行传参依然可以覆盖。
"""

import os

# ==================== 字幕分类器 / SFT 数据 ====================
DATA_DIR = "data"
OUTPUT_DIR = "outputs"
ALL_IMAGES_JSONL = os.path.join(DATA_DIR, "all_images.jsonl")
SFT_JSONL = os.path.join(OUTPUT_DIR, "subtitle_sft.jsonl")
EVAL_RESULTS = os.path.join(OUTPUT_DIR, "eval_results.jsonl")
EVAL_VIDEO_RESULTS = os.path.join(OUTPUT_DIR, "eval_video.jsonl")
VLM_OUTPUT_SUFFIX = "_vlm.jsonl"
WEIGHTS_SUFFIX = "_weights.pth"

# 图像预处理（ImageNet 归一化，与训练一致）
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
RESIZE_RATIO = 1.14

# 分类器推理
CLASSIFIER_BATCH_SIZE = 64
VIDEO_INTERVAL = 60.0        # 视频抽帧间隔（秒）
CLASSIFIER_VIDEO_BATCH_SIZE = 64
DECODE_WORKERS = 1           # DataLoader 解码进程数
PREFETCH_FACTOR = 2          # 每个解码进程预取批数

# 已部署的 SFT VLM（字幕提取，OpenAI 兼容接口）
VLM_BASE_URL = os.environ.get("VLM_BASE_URL", "http://127.0.0.1:8080")
VLM_API_KEY = os.environ.get("AIO_API_KEY", "dummy")
VLM_WORKERS = 16
VLM_TIMEOUT = 120.0
VLM_RETRIES = 3
VLM_BATCH_SIZE = 16          # 解码批大小（4K 帧别开太大）
VLM_MAX_SIDE = 896           # 上传前长边缩放

# SFT 提示词
SFT_PROMPT = (
    "这是一帧电影画面。请判断画面中是否有字幕：有字幕则提取中文字幕和英文字幕，"
    "没有则 has_subtitle 为 false。只输出 JSON，格式为 "
    '{"has_subtitle": true, "subtitle_cn": "...", "subtitle_en": "..."}'
)

# ==================== 视频理解流水线 ====================
# 输入（本地剧集，按需替换）
VIDEO_PATH = "data/tlou/最后生还者第03集.mp4"
SRT_PATH = "data/tlou/ep3.srt"
SCENES_PKL = "data/tlou/tlou_scenes.pkl"  # scene_det.py 产出的镜头列表

# 输出
SCENES_DIR = "output/tlou_s1e3_scenes"      # cut_videos.py 切出的无台词片段
CAPTIONS_DIR = "output/tlou_s1e3_captions"  # 每片段的画面描述
SRT_JSON = "output/tlou_s1e3_srt.json"      # 台词 + 场景片段（含 caption）的完整导出

# 主要人物参考图：值为「单个图片路径」或「路径列表」；换剧/换集时替换
CHARACTER_REFS = {
    "乔尔": "test/jole.jpg",
    "艾莉": "test/ellie.jpg",
    "比尔": "test/bill.jpg",
    "弗兰克": "test/frank.jpg",
}

# 人脸检测/识别（YuNet + SFace，模型缺失时自动下载）
FACE_MODEL_DIR = "ckp-tag"  # 仓库里已有这两个 onnx
FACE_MODEL_URLS = {
    "detector": "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "recognizer": "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
}
FACE_DETECT_SCORE = 0.7      # YuNet 检测置信度阈值
FACE_MIN_FACE_PX = 40        # 脸太小的直接丢
FACE_MATCH_THRESH = 0.363    # SFace 余弦相似度阈值
FACE_FRAMES_PER_CLIP = 8     # 每个片段抽几帧做人脸比对
FACE_MIN_HIT_FRAMES = 2      # 至少命中几帧才算角色出现
FACE_WORKERS = 4             # 并发处理的片段数
FACES_JSON = "output/tlou_s1e3_faces.json"
FACE_CROPS_DIR = "output/tlou_s1e3_face_crops"

# srt / 场景切分
NO_LINE_TIME_THRES = 5.0              # 超过该时长的字幕间隔算无台词场景
SCENE_IOF_THRES = 0.8                 # 镜头与无台词场景的重叠度阈值
CONTENT_DETECTOR_THRESHOLD = 27.0     # PySceneDetect 阈值，越小越敏感

# ==================== 本地 vLLM（无台词场景画面描述）====================
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://11.216.82.146:8080/v1")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "dummy")  # 本地服务不校验
VLLM_MODEL = os.getenv("VLLM_MODEL", "qwen35")
VLLM_VIDEO_FPS = None         # None=不发该参数、由服务端决定；服务端支持后设 2
VLLM_ENABLE_THINKING = False  # 思考会吃掉 max_tokens，content 可能为空

# ==================== 阿里云 DashScope ====================
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DASHSCOPE_MODEL = "qwen-plus"  # 需要环境变量 DASHSCOPE_API_KEY
