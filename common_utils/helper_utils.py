import re
from difflib import SequenceMatcher

# 字幕里残留的 ass 样式标签，例如 {\an5}，只用于排版
ASS_TAG_RE = re.compile(r"\{[^}]*\}")


def text_similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()


def strip_ass_tags(text: str) -> str:
    """去掉字幕文本里的 ass 排版标签，例如 {\\an5}（波士顿以西１０英里）"""
    if not text:
        return text
    return ASS_TAG_RE.sub("", text).strip()


def flatten_list(nested_lst):
    """将2d嵌套列表展开为一维列表"""
    out = []
    for l in nested_lst:
        out.extend(l)
    return out
