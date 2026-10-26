import logging
import os

from typing import Union
from dataclasses import asdict, dataclass, field

import config
from common_utils.file_utils import load_json, save_json
from common_utils.time_utils import timecode_to_second, time_iof
from src.collect_captions import collect_captions


@dataclass
class SrtClip:
    uid: str
    start_time: float
    end_time: float
    duration: float
    text: str  # 有台词片段（line）的台词，无台词片段为空
    clip_type: str
    speaker: str = ""  # 说话人
    caption: str = ""  # 无台词片段（scene）的画面描述，有台词片段为空
    splited_scene: list = field(default_factory=list)

    def __post_init__(self):
        assert self.clip_type in [
            "line",
            "scene",
        ], "clip_type must be 'line' or 'scene'"


class SrtProcesser:
    def __init__(
        self,
        srt_path: str,
        no_line_time_thres: float = config.NO_LINE_TIME_THRES,
        scene_iof_thres: float = config.SCENE_IOF_THRES,
    ) -> None:
        self.total_line_duration = 0.0
        self.lines = None
        self.srt_path = srt_path
        self.no_line_time_thres = no_line_time_thres
        self.scene_iof_thres = scene_iof_thres
        self._no_line_scenes_done = False
        logging.info(f"[SrtProcesser] load srt file from {srt_path}")
        self._load_srt()

    def set_no_line_time_thres(self, thres: float) -> None:
        """设置无台词时间阈值，单位为秒，超过该阈值的无台词时间段将被视为无台词场景"""
        self.no_line_time_thres = thres

    def _load_srt(self) -> None:
        """load srt file and process it to get the lines and their start and end time in seconds"""
        srt_dict_lst = self._read_srt_to_dict(self.srt_path)
        for item in srt_dict_lst:
            item["start_time"] = timecode_to_second(item["start_time"])
            item["end_time"] = timecode_to_second(item["end_time"])
            item["duration"] = item["end_time"] - item["start_time"]
        self.lines = [
            SrtClip(
                uid=item["uid"],
                start_time=item["start_time"],
                end_time=item["end_time"],
                duration=item["duration"],
                text=item["text"],
                clip_type="line",
            )
            for item in srt_dict_lst
        ]
        self.total_line_duration = sum(item.duration for item in self.lines)
        self.total_duration = self.lines[-1].end_time if len(self.lines) > 0 else 0.0
        self.total_no_line_duration = self.total_duration - self.total_line_duration
        return None

    def _read_srt_to_dict(self, srt_path: str) -> list[dict]:
        """Read srt file and convert to dict list
        contains:
        - idx: index of the line
        - start_time: start time of the line
        - end_time: end time of the line
        - text: text of the line"""
        with open(srt_path, "r", encoding="utf-8-sig") as f:
            lines = f.readlines()

        i = 0
        line_lst = []
        line_count = 0
        while i < len(lines):
            if lines[i].strip().isdigit():
                i += 1
                time_line = lines[i].strip()
                i += 1
                caption_lines = []
                # handle multi-line captions
                while i < len(lines) and lines[i].strip() != "":
                    caption_lines.append(lines[i].strip())
                    i += 1

                start_time = time_line.split("-->")[0].replace(",", ".")
                end_time = time_line.split("-->")[1].replace(",", ".")
                caption_text = " ".join(caption_lines).strip()
                line_lst.append(
                    dict(
                        uid=f"line_{line_count:04d}",
                        start_time=start_time,
                        end_time=end_time,
                        text=caption_text,
                    )
                )
                line_count += 1
            i += 1
        return line_lst

    def get_no_line_scenes(self):
        """获得无字幕的场景，定义为相邻字幕间隔大于x秒的部分

        只能在原始的纯 line 列表上执行一次：执行后 self.lines 中会混入 scene 片段，
        再次调用会因为 assert 失败，这里直接提前拦截。
        """
        if self._no_line_scenes_done:
            raise RuntimeError(
                "[SrtProcesser] get_no_line_scenes 只能调用一次，"
                "如需调整阈值请重新构造 SrtProcesser"
            )
        out_lines = []
        num_scene = 0
        for i in range(len(self.lines) - 1):
            before_line = self.lines[i]
            after_line = self.lines[i + 1]
            assert before_line.clip_type == "line" and after_line.clip_type == "line"
            space_time = after_line.start_time - before_line.end_time
            if space_time > self.no_line_time_thres:
                out_lines.append(before_line)
                out_lines.append(
                    SrtClip(
                        uid=f"scene_{num_scene:04d}",
                        start_time=before_line.end_time,
                        end_time=after_line.start_time,
                        duration=space_time,
                        text="",
                        clip_type="scene",
                    )
                )
                num_scene += 1
            else:
                out_lines.append(before_line)
        out_lines.append(self.lines[-1])
        logging.info(
            f"[SrtProcesser] found {num_scene} no line scenes, total lines: {len(out_lines)}"
        )
        self.lines = out_lines
        self.num_lines = len(out_lines)
        self._no_line_scenes_done = True

    def _get_splited_scene(
        self, start_time: float, end_time: float, scene_det_lst: list[dict]
    ) -> list[list]:
        """根据iof 将secne_det切割好的场景分配给大的空白时间
        start_time: float, 空白/无台词场景的开始时间
        end_time: float, 空白/无台词场景的结束时间
        scene_det_lst: list of dict, each dict contains:
            - start_time: start time of the scene
            - end_time: end time of the scene
        """
        out = []
        for scene in scene_det_lst:
            s_start_time, s_end_time = scene["start_time"], scene["end_time"]
            iof = time_iof((s_start_time, s_end_time), (start_time, end_time))
            if iof > self.scene_iof_thres:
                out.append([s_start_time, s_end_time])
        return out

    def get_splited_scenes(self, scene_det_lst: list[dict]):
        """得到每个无台词场景对应的场景切分结果，有台词直接使用台词的时间戳"""
        for line in self.lines:
            if line.clip_type == "scene":
                splited_scene_time = self._get_splited_scene(
                    line.start_time, line.end_time, scene_det_lst
                )
                if len(splited_scene_time) == 0:
                    logging.warning(
                        f"[SrtProcesser] cannot find splited scene for scene {line.start_time}-{line.end_time}, set to [start_time, end_time]"
                    )
                    line.splited_scene = [[line.start_time, line.end_time]]
                else:
                    line.splited_scene = splited_scene_time
            else:
                line.splited_scene = [[line.start_time, line.end_time]]

    def find_line_by_time(self, time: float) -> Union[SrtClip, None]:
        """Find the line by time in seconds"""
        for item in self.lines:
            if item.start_time <= time <= item.end_time:
                return item
        logging.warning(f"[SrtProcesser] cannot find line at time {time}")
        return None

    def get_lines_by_time(self, start_time: float, end_time: float) -> str:
        """Get the lines from start_time to end_time (inclusive)"""
        assert 0 <= start_time <= end_time <= self.total_duration, "Invalid time"
        res = []
        for item in self.lines:
            if item.end_time < start_time:
                continue
            if item.start_time > end_time:
                break
            res.append(item)
        return res

    def get_timestamp_by_idx(self, idx: Union[int, list]) -> list[float]:
        """Get the time_stamp by index"""
        if isinstance(idx, int):
            idx = [idx]
        assert all(0 <= i < len(self.lines) for i in idx), "Invalid index"
        start_idx = idx[0]
        end_idx = idx[-1]
        return [self.lines[start_idx].start_time, self.lines[end_idx].end_time]

    def save_json(self, out_path: str) -> None:
        """把当前所有片段（含 speaker / caption / splited_scene）与统计信息存成一个 json"""
        data = {
            "srt_path": self.srt_path,
            "total_duration": round(self.total_duration, 3),
            "total_line_duration": round(self.total_line_duration, 3),
            "total_no_line_duration": round(self.total_no_line_duration, 3),
            "num_lines": len(self.lines),
            "lines": [asdict(line) for line in self.lines],
        }
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)  # save_json 不建目录
        save_json(data, out_path)
        logging.info(f"[SrtProcesser] 已保存 {len(self.lines)} 个片段到 {out_path}")

    def load_captions(self, caption_path: str) -> int:
        """把画面描述填进对应片段的 caption 字段，返回填上的片段数

        caption_path 是目录时读 make_video_caption 产出的每片段 json，
        是文件时读 collect_captions 汇总出的 {uid: caption}。
        """
        if os.path.isdir(caption_path):
            uid2caption = collect_captions(caption_path)
        else:
            uid2caption = load_json(caption_path)
        matched = 0
        for line in self.lines:
            caption = (uid2caption.get(line.uid) or "").strip()
            if caption:
                line.caption = caption
                matched += 1
        logging.info(
            f"[SrtProcesser] caption 命中 {matched}/{len(self.lines)} 个片段"
            f"（读到 {len(uid2caption)} 条描述）"
        )
        return matched

    def __repr__(self) -> str:
        base_str = (
            f"SrtProcesser(\n"
            f"\tsrt_path={self.srt_path},\n"
            f"\ttotal_line_duration={self.total_line_duration:.2f},\n"
            f"\ttotal_no_line_duration={self.total_no_line_duration:.2f},\n"
            f"\ttotal_duration={self.total_duration:.2f},\n"
            f"\thas_line_ratio={self.total_line_duration / self.total_duration if self.total_duration > 0 else 0:.2f}\n"
            f")\n"
        )
        if len(self.lines) == 0:
            return base_str
        sample_line_str = ">>> sample_lines:"
        sample_line_str += f"\n{self.lines[0]}"
        return base_str + sample_line_str

    def print_line(self, idx: int) -> None:
        """Print the line by index"""
        assert 0 <= idx < len(self.lines), "Invalid index"
        print(self.lines[idx])


if __name__ == "__main__":
    # 跑一遍完整流程并导出：台词 + 无台词场景 + scene detect 切片 + 画面描述
    # python -m src.srt_preprocessor
    from src.split_video import load_scene_det_res

    srt = SrtProcesser(config.SRT_PATH)
    srt.get_no_line_scenes()
    srt.get_splited_scenes(load_scene_det_res(config.SCENES_PKL))
    srt.load_captions(config.CAPTIONS_DIR)
    srt.save_json(config.SRT_JSON)
