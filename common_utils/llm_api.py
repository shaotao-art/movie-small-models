from openai import OpenAI
import os
import json

import config


def request_qwen_api(msg_lst: list, model: str = config.DASHSCOPE_MODEL) -> str:
    try:
        client = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url=config.DASHSCOPE_BASE_URL,
        )

        completion = client.chat.completions.create(
            # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
            model=model,
            messages=msg_lst,
            # Qwen3模型通过enable_thinking参数控制思考过程（开源版默认True，商业版默认False）
            # 使用Qwen3开源版模型时，若未启用流式输出，请将下行取消注释，否则会报错
            extra_body={"enable_thinking": False},
        )
        return json.loads(completion.model_dump_json())['choices'][0]['message']['content']
    except Exception as e:
        print(f"Error requesting Qwen: {e}")
        return '{"error": "Request failed"}'


def make_simple_msg(prompt: str, role="user") -> list:
    """制作单轮纯文本对话消息格式"""
    return [{"role": role, "content": prompt}]


if __name__ == "__main__":
    # 需要先设置环境变量 DASHSCOPE_API_KEY
    print(request_qwen_api(make_simple_msg("你好"), model="qwen-plus"))
