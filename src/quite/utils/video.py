from __future__ import annotations

from typing import List, Callable, Tuple
from pathlib import Path
import subprocess
import shutil

import numpy as np
import cv2
import librosa

from .image import resize_image, ImageResizeOptions
from .file import tmp_dir, save_temp_file

__pdoc__ = {"get_video_info": False}


def get_video_info(cap):
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return height, width, fps, frame_count


def video_extract(
    file: bytes | None = None,
    video_path: Path | None = None,
    resize_options: ImageResizeOptions | None = None,
    num_skip_frames: int | Callable[[int], int] = 0,
    num_frames: int | None = None,
    video_length: float | None = None,
    video_length_ratio: float | None = None,
    to_rgb: bool = True,
    extract_frames: bool = True,
    extract_audio: bool = False,
) -> Tuple[List[np.ndarray], int, int, np.ndarray | None, int]:
    """
    Get frames and audio waveform from a video file

    Args:
        file: video file bytes
        video_path: video file path, if file is not provided
        resize_options: resize options for the output frames
        num_skip_frames: number of frames to skip per frame
        num_frames: target number of frames
        video_length: target video length in seconds, if num_frames is not provided
        video_length_ratio: target video length ratio, if video_length is not provided
        to_rgb: convert frames to RGB
        extract_frames: extract frames
        extract_audio: extract audio

    Returns:
        frames: list of numpy array frames
        num_skip_frames: number of frames skipped per frame
        orig_fps: original frames per second
        wf: audio waveform
        sr: audio sample rate
    """
    frames, video_path, wf, sr = [], None, None, 0

    try:
        if file is not None:
            video_path = save_temp_file(file, "video.mp4")
        elif video_path is None:
            raise ValueError("cannot get video frames: no video source")

        if extract_audio and has_audio_stream(video_path):
            wf, sr = librosa.load(str(video_path), sr=None)

        if not extract_frames:
            return frames, 0, 0, wf, sr

        cap = cv2.VideoCapture(str(video_path))
        orig_height, orig_width, orig_fps, frames_count = get_video_info(cap)
        orig_video_length = (frames_count - 1) / orig_fps

        if not isinstance(num_skip_frames, int):
            num_skip_frames = num_skip_frames(orig_fps)

        every_n = num_skip_frames + 1

        if num_frames:
            num_frames = min(num_frames, frames_count)
        elif video_length:
            video_length = min(video_length, orig_video_length)
            num_frames = int(video_length * orig_fps)
        elif video_length_ratio:
            video_length = orig_video_length * video_length_ratio
            num_frames = int(video_length * orig_fps)
        else:
            num_frames = frames_count

        for frame_index in range(num_frames):
            ret, image = cap.read()
            if not ret or image is None:
                break

            if frame_index % every_n != 0:
                continue

            resize_args = (
                resize_options is not None
                and resize_options.get_args()
                or {"size": image.shape[:2][::-1]}
            )
            image = resize_image(image=image, **resize_args)
            if to_rgb:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            frames.append(image)

    except Exception as e:
        if file is not None and video_path is not None:
            video_path.unlink()
        raise e

    # fps = orig_fps / (num_skip_frames + 1)

    return frames, num_skip_frames, orig_fps, wf, sr


def has_audio_stream(video_path: Path) -> bool:
    """
    Check if a video file has an audio stream
    """

    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=codec_type",
        "-of",
        "compact=p=0:nk=1",
        str(video_path),
    ]
    try:
        result = subprocess.check_output(
            command, stderr=subprocess.STDOUT, universal_newlines=True
        )
        return "audio" in result
    except subprocess.CalledProcessError:
        return False


def save_video(
    source_file: bytes | None = None,
    frames: List[np.ndarray] = [],
    save_path: Path | None = None,
    audio_source_path: Path | None = None,
    resize_options: ImageResizeOptions | None = None,
    fps: int | None = 30,
    bitrate: int = 1000,
    codec: str = "hevc",
    return_bytes: bool = False,
) -> Path | bytes:
    """
    Save a list of frames as a video file
    """
    if not len(frames) and source_file is None:
        raise ValueError("no frames or source file")

    bv = f"{bitrate}k"

    if codec not in ["hevc", "av1"]:
        raise ValueError(f"unsupported codec: {codec}")

    if save_path is None:
        is_temp = True
        save_path = tmp_dir() / "video.mp4"
    else:
        is_temp = False

    source_dir = save_path.parent / "frames"
    source_dir.mkdir(exist_ok=True, parents=True)
    if len(frames):
        for i in range(len(frames)):
            f = frames[i]
            h, w = f.shape[:2]
            if resize_options is not None:
                f = resize_image(f, **resize_options.get_args())
            f = cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
            fname = f"{i:05d}.png"
            fpath = source_dir / fname
            cv2.imwrite(str(fpath), f)
    else:
        source_path = source_dir / "video.mp4"
        with open(source_path, "wb") as f:
            f.write(source_file)

    if codec == "hevc":
        cv = "libx265"
        tagv = "hvc1"
    elif codec == "av1":
        # cv = "libaom-av1"
        cv = "libsvtav1"
        tagv = "av01"

    command = ["/usr/local/bin/ffmpeg"]
    if source_file is not None:
        command += [
            "-i",
            str(source_path),
        ]
        if resize_options is not None and resize_options.size:
            command += ["-s", f"{resize_options.size[0]}x{resize_options.size[1]}"]

    else:
        command += [
            "-framerate",
            str(fps),
            "-i",
            str(source_dir) + "/%05d.png",
        ]

    command += [
        "-c:v",
        cv,
        "-pix_fmt",
        "yuv420p",
        "-b:v",
        bv,
        "-tag:v",
        tagv,
    ]
    if source_file is not None and fps:
        command += [
            "-r",
            str(fps),
        ]

    command.append(str(save_path))

    subprocess.run(command)

    if audio_source_path is not None and has_audio_stream(audio_source_path):
        merged_video_path = save_path.parent / "merged_video.mp4"
        ffmpeg_audio_command = [
            "/usr/local/bin/ffmpeg",
            "-i",
            str(save_path),
            "-i",
            str(audio_source_path),
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-strict",
            "experimental",
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-shortest",
            "-tag:v",
            tagv,
            str(merged_video_path),
        ]
        subprocess.run(ffmpeg_audio_command)

        merged_video_path.replace(save_path)
        if merged_video_path.is_file():
            merged_video_path.unlink()

    if return_bytes:
        with open(save_path, "rb") as f:
            output = f.read()
    else:
        output = save_path

    shutil.rmtree(source_dir)

    if return_bytes and is_temp:
        shutil.rmtree(save_path.parent)

    return output
