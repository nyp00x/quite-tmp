from __future__ import annotations

from typing import Annotated, Any, Dict, List

import numpy as np

from msgspec import Meta

from .common import struct_options
from .data import (
    VideoDType,
    VideoCodec,
    VideoTarget,
    VideoFormat,
    VideoInput,
    VideoOutput,
    data_types_map,
)
from .array import NDArray
from ..utils import video_extract, save_video

__pdoc__ = {
    "Video.get_file_name": False,
    "Video.pick_file_name": False,
    "Video.skip_frames": False,
}


class Video(NDArray, **struct_options, tag=True):
    """
    Represents video data
    """

    dtype: Annotated[VideoDType, Meta(description="Data type")] = VideoDType.UINT8
    target: Annotated[VideoTarget, Meta(description="Target representation")] = (
        VideoTarget.FILE
    )
    format: Annotated[VideoFormat, Meta(description="Video container format")] = (
        VideoFormat.MP4
    )
    annotation: Dict[str, Any] | None = None
    batch: Annotated[List[Video], Meta(description="Batch data")] = []
    input: Annotated[VideoInput, Meta(description="Input properties")] | None = None
    output: Annotated[VideoOutput, Meta(description="Output properties")] | None = None
    orig_fps: Annotated[int, Meta(description="Original frames per second")] | None = (
        None
    )

    def __post_init__(self):
        super().__post_init__()

    @classmethod
    def default_format(cls) -> VideoFormat:
        return VideoFormat.MP4

    @classmethod
    def default_codec(cls) -> VideoCodec:
        return VideoCodec.HEVC

    @classmethod
    def pick_file_name(cls, file_names: List[str]) -> str | None:
        for fname in file_names:
            if fname.startswith("video"):
                return fname
        return None

    def get_file_name(self):
        return f"video.{self.format.value}"

    def to(
        self,
        target: VideoTarget | None = None,
        force: bool = True,
        remove: bool = False,
    ) -> Video:
        """
        Convert video data to target representation
        """
        if self.is_batch():
            for item in self.batch:
                if item.is_batch():
                    item.batch = []
                item.to(target, force=force, remove=remove)
            return self

        if target == VideoTarget.EMPTY:
            return self.empty()

        if target == VideoTarget.ARRAY:
            if self.array is not None and not force:
                return self
            elif self.tensor is not None:
                return self.tensor_to_array(remove=remove)
            elif self.file is not None:
                return self.file_to_array(remove=remove)
            elif self.list is not None:
                return self.list_to_array(remove=remove)
            else:
                raise ValueError(f"{self}: cannot convert to array")
        elif target == VideoTarget.FILE:
            if self.file is not None and not force:
                if self.output is not None:
                    return self.file_to_file()
                else:
                    return self
            elif self.array is not None:
                return self.array_to_file(remove=remove)
            elif self.tensor is not None:
                return self.tensor_to_array(remove=remove).array_to_file(remove=remove)
            elif self.list is not None:
                return self.list_to_array(remove=remove).array_to_file(remove=remove)
            else:
                raise ValueError(f"{self}: cannot convert to file")
        elif target == VideoTarget.TENSOR:
            if self.tensor is not None and not force:
                return self
            elif self.array is not None:
                return self.array_to_tensor(remove=remove)
            elif self.file is not None:
                return self.file_to_array(remove=remove).array_to_tensor(remove=remove)
            elif self.list is not None:
                return self.list_to_array(remove=remove).array_to_tensor(remove=remove)
            else:
                raise ValueError(f"{self}: cannot convert to tensor")
        else:
            raise ValueError(f"unknown target {target}")

    def file_to_file(self) -> Video:
        """
        Convert video file to file
        """
        if self.is_batch():
            for item in self.batch:
                item.file_to_file()
            return self

        if self.file is None:
            raise ValueError("missing video file")

        args = {}
        if self.output is not None:
            if self.output.codec is not None:
                args["codec"] = self.output.codec.value
            if self.output.bitrate is not None:
                args["bitrate"] = self.output.bitrate
            if self.output.resize is not None:
                args["resize_options"] = self.output.resize
            if self.output.fps is not None:
                args["fps"] = self.output.fps

        file = save_video(source_file=self.file, return_bytes=True, **args)

        if file is not None:
            self.file = file
            return self
        else:
            raise ValueError("failed to save video file")

    def file_to_array(
        self,
        dtype: VideoDType | None = None,
        force: bool = True,
        remove: bool = False,
        extract_only: bool = False,
        extract_frames: bool = True,
        extract_audio: bool = False,
    ) -> Video:
        """
        Convert video file to array
        """
        if self.is_batch():
            for item in self.batch:
                if item.is_batch():
                    item.batch = []
                item.file_to_array(dtype=dtype, force=force, remove=remove)
            return self

        if self.array is not None and not force:
            if remove:
                self.file = None
            return self.array_as(dtype=dtype)
        if self.file is not None:
            if self.input is None:
                self.input = VideoInput()

            args = {}
            if self.input.num_skip_frames is not None:
                args["num_skip_frames"] = self.skip_frames
            if self.input.num_frames is not None:
                args["num_frames"] = self.input.num_frames
            if self.input.duration is not None:
                args["video_length"] = self.input.duration
            if self.input.duration_ratio is not None:
                args["video_length_ratio"] = self.input.duration_ratio
            if self.input.resize is not None:
                args["resize_options"] = self.input.resize
            if extract_only:
                args["extract_audio"] = extract_audio
                args["extract_frames"] = extract_frames

            frames, num_skip_frames, self.orig_fps, wf, sr = video_extract(
                file=self.file, **args
            )

            if not extract_only:
                self.array = np.stack(frames, axis=0)
                self.shape = self.array.shape
                if dtype is not None:
                    self.array = self.array.astype(dtype.value)
                    self.dtype = dtype

            if remove:
                self.file = None

            if not extract_only:
                return self
            else:
                return frames, num_skip_frames, self.orig_fps, wf, sr
        else:
            raise ValueError("cannot convert Video file to array")

    def skip_frames(self, orig_fps: int) -> int:
        num_skip_frames = 0
        if self.input is not None and self.input.num_skip_frames is not None:
            num_skip_frames = self.input.num_skip_frames
        if orig_fps > 30:
            return (num_skip_frames + 1) * 2 - 1
        return num_skip_frames

    def array_to_file(
        self,
        force: bool = True,
        remove: bool = False,
    ) -> Video:
        """
        Convert video array to file
        """
        if self.is_batch():
            for item in self.batch:
                if item.is_batch():
                    item.batch = []
                item.array_to_file(force=force, remove=remove)
            return self

        if self.file is not None and not force:
            if remove:
                self.array = None
            return self
        if self.array is not None:
            args = {}
            if self.output is not None:
                if self.output.codec is not None:
                    args["codec"] = self.output.codec.value
                if self.output.bitrate is not None:
                    args["bitrate"] = self.output.bitrate
                if self.output.resize is not None:
                    args["resize_options"] = self.output.resize
                if self.output.fps is not None:
                    args["fps"] = self.output.fps

            self.file = save_video(frames=self.array, return_bytes=True, **args)
            self.format = self.default_format()
            if remove:
                self.array = None
            return self
        else:
            raise ValueError("cannot convert Video array to file")

    def serializable(self, convert: bool = True) -> Video:
        """
        Make video serializable. If `convert` is True, convert tensor to array, else remove tensor
        """
        if self.is_batch():
            for item in self.batch:
                if item.is_batch():
                    item.batch = []
                item.serializable(convert=convert)

        if self.tensor is not None:
            if convert:
                self.tensor_to_array()
            else:
                self.tensor = None
        return self

    # def _apply(self, output: VideoOutput) -> Video:
    #     if self.is_batch():
    #         for item in self.batch:
    #             if item.is_batch():
    #                 item.batch = []
    #             item._apply(output)
    #         return self

    #     if output.format is not None:
    #         self.format = output.format

    #     self.output = output

    #     targets = output.target if isinstance(output.target, list) else [output.target]
    #     for target in targets:
    #         self.to(target, force=False, remove=True)

    #     return self

    async def apply(
        self,
        output: VideoOutput,
        repository: Any | None = None,
        log_prefix: str | None = None,
    ) -> Video:
        """
        Apply output properties
        """
        if not output.clone_object and self.is_batch():
            for item in self.batch:
                if item.is_batch():
                    item.batch = []
                await item.apply(output, repository=repository, log_prefix=log_prefix)
            return self

        if self.is_empty():
            return self

        if output.format is not None:
            self.format = output.format

        self.set_output(output)

        targets = output.target if isinstance(output.target, list) else [output.target]

        for target in targets:
            self.target = target

            await self.resolve(
                force=False,
                remove=True,
                repository=repository,
                log_prefix=log_prefix,
            )

        return self

    async def resolve(
        self,
        force: bool = True,
        remove: bool = False,
        repository: Any | None = None,
        log_prefix: str | None = None,
    ) -> Video:
        """
        Convert video to target representation defined in `target` property
        """
        if (self.output is None or not self.output.clone_object) and self.is_batch():
            for item in self.batch:
                if item.is_batch():
                    item.batch = []
                await item.resolve(
                    force=force,
                    remove=remove,
                    repository=repository,
                    log_prefix=log_prefix,
                )
            return self

        if self.target is None:
            raise ValueError(
                f"cannot resolve {self.__class__.__name__} target is not specified"
            )

        if self.url is not None or self.from_id is not None:
            need_download = (
                self.target == VideoTarget.ARRAY and (self.array is None or force)
            ) or (self.target == VideoTarget.FILE and (self.file is None or force))
            if need_download:
                await self.download(
                    repository=repository, force=force, log_prefix=log_prefix
                )

        if self.target == VideoTarget.URL:
            self.to(VideoTarget.FILE, force=force, remove=remove)
            object_folder = (
                self.output is not None and self.output.object_folder or None
            )
            as_clone_object = self.output is not None and self.output.clone_object
            await self.upload(
                repository=repository,
                remove=remove,
                object_folder=object_folder,
                clone_object=as_clone_object,
                log_prefix=log_prefix,
            )
        else:
            self.to(self.target, force=force, remove=remove)

        return self


data_types_map["Video"] = Video
