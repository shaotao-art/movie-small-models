import os
from openai import OpenAI

import config

# 本地 vLLM 服务地址与模型名，见 config.py，可用环境变量覆盖
DEFAULT_BASE_URL = config.VLLM_BASE_URL
DEFAULT_MODEL = config.VLLM_MODEL
# 视频默认采样帧率，见 infer_vllm 中的说明
DEFAULT_VIDEO_FPS = config.VLLM_VIDEO_FPS


def make_vllm_text_msg(content: str) -> dict:
    return [{"role": "user", "content": [{"type": "text", "text": content}]}]


def make_vllm_image_msg(image_url: str, text: str) -> dict:
    if os.path.isfile(image_url):
        image_url = "file://" + os.path.abspath(image_url)
    else:
        raise ValueError(f"Image file does not exist: {image_url}")
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        }
    ]


def make_vllm_video_msg(video_url: str, text: str) -> dict:
    if os.path.isfile(video_url):
        video_url = "file://" + os.path.abspath(video_url)
    else:
        raise ValueError(f"Video file does not exist: {video_url}")
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {"type": "video_url", "video_url": {"url": video_url}},
            ],
        }
    ]


def get_vllm_client(
    base_url: str = None, api_key: str = None, timeout: float = None
) -> OpenAI:
    """构造指向本地 vLLM 的 OpenAI 兼容客户端

    base_url 默认取 config.VLLM_BASE_URL（已含 VLLM_BASE_URL 环境变量覆盖），
    api_key 默认取 config.VLLM_API_KEY。
    """
    base_url = base_url or DEFAULT_BASE_URL
    api_key = api_key or config.VLLM_API_KEY
    kwargs = {"timeout": timeout} if timeout is not None else {}
    client = OpenAI(api_key=api_key, base_url=base_url, **kwargs)
    return client


def infer_vllm(
    client: OpenAI,
    messages: list[dict],
    model: str = None,
    max_tokens: int = 2048,
    fps: int = DEFAULT_VIDEO_FPS,
    enable_thinking: bool = config.VLLM_ENABLE_THINKING,
):
    """调用 vLLM 对话接口，含视频时按 fps 采样帧

    默认关闭思考过程（config.VLLM_ENABLE_THINKING）：思考会占用 max_tokens，
    长思考时 content 可能为空。

    When vLLM is launched with `--media-io-kwargs '{"video": {"num_frames": -1}}'`,
    video frame sampling can be configured via `extra_body` (e.g., by setting `fps`).
    This feature is currently supported only in vLLM.

    With `do_sample_frames=True`, you can customize the `fps` value to set your desired video sampling rate.
    """
    model = model or DEFAULT_MODEL
    has_video = any(
        any(
            c.get("type") == "video_url"
            for c in message.get("content", [])
            if isinstance(c, dict)
        )
        for message in messages
    )
    extra_body = {"chat_template_kwargs": {"enable_thinking": enable_thinking}}
    if has_video and fps:
        extra_body["mm_processor_kwargs"] = {"fps": fps, "do_sample_frames": True}
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        extra_body=extra_body,
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    vllm_client = get_vllm_client()
    text = "你好"
    messages = [make_vllm_text_msg(text)]
    response = infer_vllm(vllm_client, messages)
    print(response)
