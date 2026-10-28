import os
from moviepy import *
from typing import Union
from PIL import Image


class VideoProcessor:
    def __init__(self, video_path: str):
        """read video"""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"错误：找不到视频文件 '{video_path}'")
        self.video_path = video_path
        self.video = VideoFileClip(video_path)
        self.info = None
        self.v_encode = "libx264"
        self.a_encode = "aac"

    def set_video_encode(self, v_encode: str):
        """set video encode, default is libx264"""
        self.v_encode = v_encode

    def set_audio_encode(self, a_encode: str):
        """set audio encode, default is aac"""
        self.a_encode = a_encode

    def get_video_info(self) -> dict:
        """get video info

        返回:
            dict: 一个包含视频信息的字典，格式如下：
            {
                'filename': '文件名',
                'duration': '时长 (秒)',
                'resolution': '分辨率 (宽x高)',
                'fps': '帧率 (FPS)',
                'has_audio': '是否有音频 (True/False)',
                'audio_codec': '音频编码' (如果存在),
                'audio_fps': '音频采样率' (如果存在)
            }
        """
        info = {
            "filename": os.path.basename(self.video_path),
            "duration": self.video.duration,
            "resolution": f"{self.video.w}x{self.video.h}",
            "has_audio": self.video.audio is not None,
            "ori_video_fps": self.video.fps,
        }
        if self.video.audio:
            info["ori_audio_codec"] = self.video.audio.reader.codec
            info["ori_audio_fps"] = self.video.audio.fps
        self.info = info
        return info

    def cut_video(self, start_time: float, end_time: float, output_path: str = None):
        """cut video from start_time to end_time in seconds"""
        if start_time < 0 or end_time > self.video.duration or start_time >= end_time:
            raise ValueError("错误：提供的剪切时间无效或超出视频时长范围。")
        cut_clip = self.video.subclipped(start_time, end_time)
        if output_path is not None:
            cut_clip.write_videofile(
                output_path, codec=self.v_encode, audio_codec=self.a_encode
            )
        return cut_clip

    def cut_audio(self, start_time: float, end_time: float, output_path: str = None):
        """extract audio from video"""
        cut_clip = self.video.subclipped(start_time, end_time)
        audio_clip = cut_clip.audio
        if output_path is not None:
            audio_clip.write_audiofile(output_path)
        return audio_clip

    def extract_audio(self, output_path: str = None):
        """extract audio from video"""
        return self.cut_audio(0, self.video.duration, output_path)

    def write_audiofile(self, audio_clip, output_path: str):
        """write audio clip to file"""
        audio_clip.write_audiofile(output_path, audio_codec=self.a_encode)

    def get_frame_at_t(
        self, time: Union[float, list], output_path=None, out_folder=None
    ):
        """get frame at time t in seconds"""
        if isinstance(time, float):
            if output_path is None:
                return self.video.frame_function(time)
            else:
                frame = self.video.frame_function(time)
                img = Image.fromarray(frame)
                img.save(output_path)
                return img

        elif isinstance(time, list):
            imgs = [self.video.frame_function(t) for t in time]
            if out_folder is not None:
                os.makedirs(out_folder, exist_ok=True)
                for i, img in enumerate(imgs):
                    img = Image.fromarray(img)
                    img.save(os.path.join(out_folder, f"frame_{i}.jpg"))
            return imgs
        else:
            raise ValueError("错误：时间参数必须是浮点数或浮点数列表。")

    def write_videofile(self, video_clip, output_path: str):
        """write video clip to file"""
        video_clip.write_videofile(
            output_path, codec=self.v_encode, audio_codec=self.a_encode
        )

    def merge_video_and_audio(
        self,
        video_clip: Union[str, VideoFileClip],
        audio_clip: Union[str, AudioFileClip],
        output_path: str = None,
    ):
        """merge video and audio into a new video file from local files"""
        if isinstance(video_clip, str):
            if not os.path.exists(video_clip):
                raise FileNotFoundError(f"错误：找不到视频文件 '{video_clip}'")
            video_clip = VideoFileClip(video_clip)

        if isinstance(audio_clip, str):
            if not os.path.exists(audio_clip):
                raise FileNotFoundError(f"错误：找不到音频文件 '{audio_clip}'")
            audio_clip = AudioFileClip(audio_clip)

        final_clip = video_clip.with_audio(audio_clip)
        if output_path is not None:
            final_clip.write_videofile(
                output_path, codec=self.v_encode, audio_codec=self.a_encode
            )
        return final_clip

    def concatenate_videos(
        self,
        clips: Union[list[str], list[VideoClip]],
        output_path: str = None,
        vencode: str = None,
        aencode: str = None,
    ):
        """拼接多个视频文件"""
        if vencode is None:
            vencode = self.v_encode
        if aencode is None:
            aencode = self.a_encode
        if isinstance(clips[0], str):
            clips = [VideoFileClip(path) for path in clips]
        # simple concat for now
        final_clip = concatenate_videoclips(clips)
        if output_path is not None:
            final_clip.write_videofile(output_path, codec=vencode, audio_codec=aencode)
        return final_clip

    def close(self):
        """close video file"""
        if self.video:
            self.video.close()
