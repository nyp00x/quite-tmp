from __future__ import annotations

from typing import Annotated, Any, Dict, List
from concurrent.futures import ThreadPoolExecutor
import io
import time

import numpy as np
from msgspec import Meta

import PIL
import pillow_avif  # noqa

from .common import struct_options
from .data import ImageDType, ImageFormat, ImageInput, ImageOutput, data_types_map
from .array import NDArray, NDArrayTarget
from ..utils import (
    name2enum,
    resize_image,
    encode_image,
    encode_image_sync,
    ImageResizeOptions,
    set_image_mode,
)
from ..globals import new_logger

logger = new_logger("<black>image</black>")

__pdoc__ = {
    "Image.get_file_name": False,
    "Image.pick_file_name": False,
}


class Image(NDArray, **struct_options, tag=True):
    """
    Represents 2D image data
    """

    dtype: Annotated[ImageDType, Meta(description="Data type")] = ImageDType.UINT8
    format: Annotated[ImageFormat, Meta(description="Image format")] = (
        ImageFormat.SOURCE
    )
    target: Annotated[NDArrayTarget, Meta(description="Target representation")] = (
        NDArrayTarget.ARRAY
    )
    annotation: Dict[str, Any] | None = None
    batch: Annotated[List[Image], Meta(description="Batch data")] = []
    input: Annotated[ImageInput, Meta(description="Input properties")] | None = None
    output: Annotated[ImageOutput, Meta(description="Output properties")] | None = None

    def __post_init__(self):
        super().__post_init__()

    @classmethod
    def default_format(cls) -> ImageFormat:
        """
        Default image format
        """
        return ImageFormat.AVIF

    @classmethod
    def pick_file_name(cls, file_names: List[str]) -> str | None:
        for fname in file_names:
            if not fname.startswith("image"):
                continue
            for fmt in ImageFormat:
                if fname.endswith(fmt.value):
                    return fname
        return None

    def get_file_name(self) -> str:
        return str(f"image.{self.format.value}")

    def file_to_array(
        self,
        dtype: ImageDType | None = None,
        force: bool = True,
        remove: bool = False,
    ) -> Image:
        """
        Convert image file to array
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
            image = PIL.Image.open(io.BytesIO(self.file))
            array = np.array(PIL.ImageOps.exif_transpose(image))
            self.array = array
            self.format = name2enum(
                image.format, ImageFormat, default=ImageFormat.SOURCE
            )
            if remove:
                self.file = None
            if self.input is not None:
                if self.input.mode is not None:
                    self.array = set_image_mode(self.array, self.input.mode)
                if self.input.resize is not None:
                    self.resize(self.input.resize)
            if dtype is not None:
                self.array = self.array.astype(dtype.value)
                self.dtype = dtype

            return self
        else:
            raise ValueError("cannot convert Image file to array")

    def array_to_file(
        self,
        format: ImageFormat | None = None,
        force: bool = True,
        remove: bool = False,
        quality: int | None = None,
    ) -> Image:
        """
        Convert image array to file
        """
        format = format or self.format
        if format is None or format == ImageFormat.SOURCE:
            format = self.default_format()

        if self.is_batch():
            for item in self.batch:
                if item.is_batch():
                    item.batch = []
                item.array_to_file(
                    format=format,
                    force=force,
                    remove=remove,
                )
            return self

        if self.file is not None and not force:
            if remove:
                self.array = None
            return self
        if self.array is not None:
            time_start = time.perf_counter()
            image = PIL.Image.fromarray(self.array)
            if quality is None:
                quality = self.output is not None and self.output.quality or None
            self.file = encode_image_sync(image, format.name.upper(), quality)
            logger.debug(
                f"{self} encoded {format.name} ({time.perf_counter() - time_start:.3f}s)"
            )
            self.format = format
            if remove:
                self.array = None
            return self
        else:
            raise ValueError("cannot convert Image array to file")

    async def array_to_file_async(
        self,
        format: ImageFormat,
        force: bool = True,
        remove: bool = False,
        quality: int | None = None,
        thread_pool: ThreadPoolExecutor | None = None,
    ) -> Image:
        """
        Convert image array to file asynchronously
        """
        format = format or self.format
        if format is None or format == ImageFormat.SOURCE:
            format = self.default_format()

        if self.is_batch():
            for item in self.batch:
                await item.array_to_file_async(
                    format=format, force=force, remove=remove, quality=quality
                )
            return self

        if self.file is not None and not force:
            if remove:
                self.array = None
            return self
        if self.array is not None:
            time_start = time.perf_counter()
            image = PIL.Image.fromarray(self.array)
            if quality is None:
                quality = self.output is not None and self.output.quality or None
            if thread_pool is None:
                self.file = encode_image_sync(image, format.name.upper(), quality)
                logger.debug(
                    f"{self} encoded {format.name} ({time.perf_counter() - time_start:.3f}s)"
                )
            else:
                self.file = await encode_image(
                    thread_pool, image, format.name.upper(), quality
                )
                logger.debug(
                    f"{self} async encoded {format.name} ({time.perf_counter() - time_start:.3f}s)"
                )
            self.format = format
            if remove:
                self.array = None
            return self
        else:
            raise ValueError("cannot convert Image array to file")

    async def download(
        self,
        repository: Any | None = None,
        force: bool = True,
        log_prefix: str | None = None,
    ) -> Image:
        """
        Download image.
        If `from_id` is set, delegate to repository, otherwise download from `url` to `file`
        """
        await super().download(
            repository=repository, force=force, log_prefix=log_prefix
        )

        if self.format != ImageFormat.SOURCE:
            self.file_to_array(force=True, remove=True)
            self.array_to_file(force=True, remove=True)
            logger.debug(f"{self} converted to {self.format.name}")

        return self

    # def _apply(self, output: ImageOutput) -> Image:
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
        output: ImageOutput,
        repository: Any | None = None,
        log_prefix: str | None = None,
    ) -> Image:
        """
        Apply output properties
        """
        if not output.clone_object and self.is_batch():
            for item in self.batch:
                if item.is_batch():
                    item.batch = []
                await item.apply(output, repository=repository, log_prefix=log_prefix)
            return self

        if output.format is not None:
            self.format = output.format

        if self.is_empty():
            return self

        self.set_output(output)

        targets = output.target if isinstance(output.target, list) else [output.target]

        if (
            self.array is not None or self.file is not None
        ) and output.resize is not None:
            self.resize(output.resize)

        for target in targets:
            self.target = target
            if (
                repository is not None
                and self.file is None
                and target in [NDArrayTarget.FILE, NDArrayTarget.URL]
            ):
                self.to(NDArrayTarget.ARRAY, force=False, remove=True)
                await self.array_to_file_async(
                    format=self.format,
                    force=False,
                    remove=True,
                    quality=self.output.quality,
                    thread_pool=repository.thread_pool,
                )
            await self.resolve(
                force=False,
                remove=True,
                repository=repository,
                log_prefix=log_prefix,
            )

        return self

    def resize(self, options: ImageResizeOptions) -> Image:
        """
        Resize image
        """
        if self.is_batch():
            for item in self.batch:
                if item.is_batch():
                    item.batch = []
                item.resize(options)
            return self

        if self.array is None and self.file is None:
            raise ValueError("cannot resize Image: array or file is not set")

        need_file, need_array = True, True

        if self.file is None:
            need_file = False

        if self.array is None:
            self.file_to_array(force=False, remove=False)
            need_array = False

        self.array = resize_image(image=self.array, **options.get_args())
        self.shape = self.array.shape

        if need_file:
            self.array_to_file(force=True, remove=not need_array)
        if not need_array:
            self.array = None

        return self

    def pil(self):
        """
        Make new PIL.Image
        """
        if self.is_batch():
            return [image.pil() for image in self.batch]

        if self.array is None and self.file is None:
            raise ValueError("cannot convert Image to PIL.Image: no data")

        if self.array is None:
            self.file_to_array(force=True, remove=True)

        return PIL.Image.fromarray(self.array)


data_types_map["Image"] = Image
