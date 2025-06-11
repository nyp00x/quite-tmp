from __future__ import annotations

from typing import Annotated, Any, Dict, List

import io

import soundfile
import librosa

from msgspec import Meta

from .common import struct_options
from .data import (
    AudioDType,
    AudioTarget,
    AudioFormat,
    AudioInput,
    AudioOutput,
    data_types_map,
)
from .array import NDArray
from ..utils import name2enum

__pdoc__ = {
    "Audio.get_file_name": False,
    "Audio.pick_file_name": False,
}


class Audio(NDArray, **struct_options, tag=True):
    """
    Represents audio data
    """

    dtype: Annotated[AudioDType, Meta(description="Data type")] = AudioDType.FLOAT32
    target: Annotated[AudioTarget, Meta(description="Target representation")] = (
        AudioTarget.ARRAY
    )
    format: Annotated[AudioFormat, Meta(description="Audio container format")] = (
        AudioFormat.MP3
    )
    annotation: Dict[str, Any] | None = None
    batch: Annotated[List[Audio], Meta(description="Batch data")] = []
    input: Annotated[AudioInput, Meta(description="Input properties")] | None = None
    output: Annotated[AudioOutput, Meta(description="Output properties")] | None = None
    sr: Annotated[int, Meta(description="Sampling rate")] | None = None

    def __post_init__(self):
        super().__post_init__()

    @classmethod
    def default_format(cls) -> AudioFormat:
        return AudioFormat.MP3

    @classmethod
    def pick_file_name(cls, file_names: List[str]) -> str | None:
        for fname in file_names:
            if fname.startswith("audio"):
                return fname
        return None

    def get_file_name(self):
        return f"audio.{self.format.value}"

    def to(
        self,
        target: AudioTarget | None = None,
        force: bool = True,
        remove: bool = False,
    ) -> Audio:
        """
        Convert audio data to target representation
        """
        if self.is_batch():
            for item in self.batch:
                if item.is_batch():
                    item.batch = []
                item.to(target, force=force, remove=remove)
            return self

        if target == AudioTarget.EMPTY:
            return self.empty()

        if target == AudioTarget.ARRAY:
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
        elif target == AudioTarget.FILE:
            if self.file is not None and not force:
                return self
            elif self.array is not None:
                return self.array_to_file(remove=remove)
            elif self.tensor is not None:
                return self.tensor_to_array(remove=remove).array_to_file(remove=remove)
            elif self.list is not None:
                return self.list_to_array(remove=remove).array_to_file(remove=remove)
            else:
                raise ValueError(f"{self}: cannot convert to file")
        elif target == AudioTarget.TENSOR:
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

    def file_to_array(
        self, dtype: AudioDType | None = None, force: bool = True, remove: bool = False
    ) -> Audio:
        """
        Convert audio file to array
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
                self.input = AudioInput()

            audio = io.BytesIO(self.file)
            audio_info = soundfile.info(audio)

            self.array, self.sr = librosa.load(
                io.BytesIO(self.file),
                sr=self.input.sr or self.sr or None,
                duration=self.input.duration,
            )

            self.format = name2enum(
                audio_info.format.upper(), AudioFormat, default=None
            )

            self.shape = self.array.shape
            if dtype is not None:
                self.array = self.array.astype(dtype.value)
                self.dtype = dtype
            if remove:
                self.file = None
            return self
        else:
            raise ValueError("cannot convert Audio file to array")

    def array_to_file(
        self,
        force: bool = True,
        remove: bool = False,
    ) -> Audio:
        """
        Convert audio array to file
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
            new_sr = None

            if self.output is not None:
                new_sr = self.output.sr

                if new_sr is not None and self.sr and new_sr != self.sr:
                    self.array = librosa.resample(
                        self.array, orig_sr=self.sr, target_sr=new_sr
                    )

            new_sr = new_sr or self.sr

            if new_sr is None:
                raise ValueError(
                    "cannot convert Audio array to file: sampling rate is not specified"
                )

            fmt = self.format or self.default_format()
            audio_buffer = io.BytesIO()

            soundfile.write(audio_buffer, self.array, new_sr, format=fmt.name)

            self.file = audio_buffer.getvalue()
            self.format = fmt

            if remove:
                self.array = None
            return self
        else:
            raise ValueError("cannot convert Audio array to file")

    def serializable(self, convert: bool = True) -> Audio:
        """
        Make audio serializable. If `convert` is True, convert tensor to array, else remove tensor
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

    # def _apply(self, output: AudioOutput) -> Audio:
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
        output: AudioOutput,
        repository: Any | None = None,
        log_prefix: str | None = None,
    ) -> Audio:
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

        if output.sr and output.sr != self.sr:
            if not self.sr:
                raise ValueError("cannot apply AudioOutput: no original sampling rate")
            self.resample(output.sr)

        if output.duration:
            self.array = self.array[: int(output.duration * self.sr)]

        return self

    def resample(self, sr: int):
        """
        Resample audio to target sampling rate
        """
        if self.is_batch():
            for item in self.batch:
                if item.is_batch():
                    item.batch = []
                item.resample(sr)
            return self

        array_exists = self.array is not None
        file_exists = self.file is not None

        if not array_exists and not file_exists:
            raise ValueError("cannot resample Audio: no data")

        if not self.sr:
            raise ValueError("cannot resample Audio: no sampling rate")

        if sr == self.sr:
            return self

        if not array_exists:
            self.file_to_array(force=True, remove=True)

        self.array = librosa.resample(self.array, orig_sr=self.sr, target_sr=sr)
        self.sr = sr

        if file_exists:
            self.array_to_file(force=True, remove=not array_exists)

        if not array_exists:
            self.array = None

        return self

    async def resolve(
        self,
        force: bool = True,
        remove: bool = False,
        repository: Any | None = None,
        log_prefix: str | None = None,
    ) -> Audio:
        """
        Convert audio to target representation defined in `target` property
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
                self.target == AudioTarget.ARRAY and (self.array is None or force)
            ) or (self.target == AudioTarget.FILE and (self.file is None or force))
            if need_download:
                await self.download(
                    repository=repository, force=force, log_prefix=log_prefix
                )

        if self.target == AudioTarget.URL:
            self.to(AudioTarget.FILE, force=force, remove=remove)
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


data_types_map["Audio"] = Audio
