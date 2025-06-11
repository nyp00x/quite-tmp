from typing import Tuple, List, Annotated
from concurrent.futures import ThreadPoolExecutor
import io
import asyncio

import cv2
import numpy as np
import enum
from PIL import Image as PIL_Image

from msgspec import Meta, Struct
from loguru import logger

from ..globals import struct_options

__pdoc__ = {"get_interpolation": False, "cv_algo": False, "cv_border": False}


class ResizeMode(enum.Enum):
    """
    Image resize mode.
    COVER: Image may be cropped to fit the target size.
    CONTAIN: Image may be padded to fit the target size.
    STRETCH: Image will be stretched to fit the target size.
    """

    COVER = "cover"
    CONTAIN = "contain"
    STRETCH = "stretch"


class ImageMode(enum.Enum):
    """
    Image mode.
    RGB: Red, Green, Blue.
    RGBA: Red, Green, Blue, Alpha.
    """

    RGB = "rgb"
    RGBA = "rgba"


def set_image_mode(image: np.ndarray, mode: ImageMode) -> np.ndarray:
    """
    Set image mode
    """
    if len(image.shape) == 2:
        image = np.stack([image, image, image], axis=-1)
    if image.shape[2] == 3 and mode == ImageMode.RGBA:
        return np.dstack(
            [image, np.ones((image.shape[0], image.shape[1]), dtype=np.uint8) * 255]
        )
    elif image.shape[2] == 4 and mode == ImageMode.RGB:
        return image[..., :3]
    return image


def hex_to_rgb(color: str) -> Tuple[int, int, int]:
    """
    Convert hex color to rgb
    """
    color = color.lstrip("#")
    return tuple(int(color[i : i + 2], 16) for i in (0, 2, 4))


def hex_to_bgr(color: str) -> Tuple[int, int, int]:
    """
    Convert hex color to bgr
    """
    color = color.lstrip("#")
    return tuple(int(color[i : i + 2], 16) for i in (4, 2, 0))


def get_interpolation(
    size: Tuple[int, int],
    orig_size: Tuple[int, int],
    algo_up: int | None = cv2.INTER_LINEAR,
    algo_down: int | None = cv2.INTER_AREA,
) -> int:
    """
    Get image interpolation algorithm
    """
    if size[0] > orig_size[0] and size[1] > orig_size[1]:
        return algo_up
    else:
        return algo_down


def resize_image(
    image: np.ndarray | PIL_Image.Image,
    size: Tuple[int, int] | None = None,
    size_mean: float | int | None = None,
    aspect: float | None = None,
    max_area: int = 2073600,
    multiple_of: int = 4,
    mode: ResizeMode = ResizeMode.COVER,
    algo_up: int | None = cv2.INTER_CUBIC,
    algo_down: int | None = cv2.INTER_AREA,
    border_type: int = cv2.BORDER_REPLICATE,
    border_value: Tuple[int, int, int] | str | int = 0,
    box_rel: Tuple[float, float, float, float] | None = None,
) -> np.array:
    """
    Resize image
    """
    if not isinstance(image, np.ndarray):
        image = np.array(image)

    if box_rel is not None:
        box_rel = (
            max(box_rel[0], 0),
            max(box_rel[1], 0),
            min(box_rel[2], 1),
            min(box_rel[3], 1),
        )
        if box_rel[0] >= box_rel[2] or box_rel[1] >= box_rel[3]:
            raise ValueError(f"invalid box_rel: {box_rel}")
        box = [
            round(box_rel[0] * image.shape[1]),
            round(box_rel[1] * image.shape[0]),
            round(box_rel[2] * image.shape[1]),
            round(box_rel[3] * image.shape[0]),
        ]
        box, image = crop_image(box, image=image)

    orig_w, orig_h = image.shape[1], image.shape[0]
    orig_aspect = orig_w / orig_h
    if not aspect:
        aspect = orig_aspect

    if size is not None:
        w, h = size
    else:
        if not size_mean:
            size_mean = (orig_w + orig_h) / 2

        h = 2 * size_mean / (1 + aspect)
        w = h * aspect

        if size_mean:
            num_pixels = w * h
            if num_pixels > max_area:
                scale = np.sqrt(max_area / num_pixels)
                w = w * scale
                h = h * scale

        w = int(round(w / multiple_of) * multiple_of)
        h = int(round(h / multiple_of) * multiple_of)

    if (w, h) == (orig_w, orig_h):
        return image

    target_aspect = w / h
    top, left, bottom, right = 0, 0, 0, 0

    if mode == ResizeMode.COVER:
        if orig_aspect > target_aspect:
            new_h = h
            new_w = new_h * orig_aspect
            left = int((new_w - w) // 2)
        else:
            new_w = w
            new_h = new_w / orig_aspect
            top = int((new_h - h) // 2)

        top = max(top, 0)
        left = max(left, 0)
        bottom = min(top + h, int(new_h))
        right = min(left + w, int(new_w))

        algo = get_interpolation((new_w, new_h), (orig_w, orig_h), algo_up, algo_down)
        image = cv2.resize(image, (int(new_w), int(new_h)), interpolation=algo)

        image = image[top:bottom, left:right]

    elif mode == ResizeMode.CONTAIN:
        if orig_aspect > target_aspect:
            new_w = w
            new_h = int(new_w / orig_aspect)
            top = int((h - new_h) // 2)
            bottom = h - new_h - top
        else:
            new_h = h
            new_w = int(new_h * orig_aspect)
            left = int((w - new_w) // 2)
            right = w - new_w - left

        top = max(top, 0)
        left = max(left, 0)
        bottom = max(bottom, 0)
        right = max(right, 0)

        algo = get_interpolation((new_w, new_h), (orig_w, orig_h), algo_up, algo_down)
        image = cv2.resize(image, (new_w, new_h), interpolation=algo)

        image = cv2.copyMakeBorder(
            image,
            top,
            bottom,
            left,
            right,
            border_type,
            value=border_value
            if not isinstance(border_value, str)
            else hex_to_bgr(border_value),
        )
    elif mode == ResizeMode.STRETCH:
        algo = get_interpolation((w, h), (orig_w, orig_h), algo_up, algo_down)
        image = cv2.resize(image, (w, h), interpolation=algo)
    else:
        raise ValueError(f"invalid resize mode {mode}")

    w, h = image.shape[1], image.shape[0]

    if w % multiple_of or h % multiple_of:
        logger.warning("image is resized second time")
        w = w // multiple_of * multiple_of
        h = h // multiple_of * multiple_of
        image = cv2.resize(image, (w, h), interpolation=algo)

    return image


def encode_image_sync(
    image: np.ndarray | PIL_Image.Image, format: str, quality: int | None = None
) -> bytes:
    try:
        if isinstance(image, np.ndarray):
            image = PIL_Image.fromarray(image)
        with io.BytesIO() as b:
            kwargs = {} if not quality else {"quality": quality}
            image.save(b, format=format, **kwargs)
            return b.getvalue()
    except Exception as e:
        raise RuntimeError(f"Image encoding failed: {e}")


async def encode_image(
    thread_pool: ThreadPoolExecutor,
    image: np.ndarray | PIL_Image.Image,
    format: str,
    quality: int | None = None,
) -> bytes:
    """
    Encode image to bytes
    """
    return await get_event_loop().run_in_executor(
        thread_pool,
        encode_image_sync,
        image,
        format,
        quality,
    )


class ImageResizeAlgorithm(enum.Enum):
    """
    CV image resize algorithm
    """

    NEAREST = "nearest"
    LINEAR = "linear"
    CUBIC = "cubic"
    AREA = "area"
    LANCZOS4 = "lanczos4"


class ImageBorderType(enum.Enum):
    """
    CV image border type
    """

    REPLICATE = "replicate"
    REFLECT = "reflect"
    CONSTANT = "constant"


def cv_algo(algo: ImageResizeAlgorithm):
    if algo == ImageResizeAlgorithm.CUBIC:
        return cv2.INTER_CUBIC
    elif algo == ImageResizeAlgorithm.AREA:
        return cv2.INTER_AREA
    elif algo == ImageResizeAlgorithm.NEAREST:
        return cv2.INTER_NEAREST
    elif algo == ImageResizeAlgorithm.LINEAR:
        return cv2.INTER_LINEAR
    elif algo == ImageResizeAlgorithm.LANCZOS4:
        return cv2.INTER_LANCZOS4


def cv_border(border: ImageBorderType):
    if border == ImageBorderType.REPLICATE:
        return cv2.BORDER_REPLICATE
    elif border == ImageBorderType.REFLECT:
        return cv2.BORDER_REFLECT
    elif border == ImageBorderType.CONSTANT:
        return cv2.BORDER_CONSTANT


class ImageResizeOptions(Struct, **struct_options):
    """
    Image resize options
    """

    size: Annotated[Tuple[int, int], Meta(description="Target size")] | None = None
    size_mean: (
        Annotated[float, Meta(description="Size mean, (w + h) / 2", ge=32, le=8192)]
        | None
    ) = None
    aspect: (
        Annotated[float, Meta(description="Aspect ratio, w / h", ge=0.05, le=20)] | None
    ) = None
    max_area: Annotated[
        int, Meta(description="Max number of pixels", ge=256, le=2**26)
    ] = 2073600
    multiple_of: Annotated[
        int, Meta(description="Each side must be multiple of", ge=1, le=512)
    ] = 16
    mode: Annotated[ResizeMode, Meta(description="Resize mode")] = ResizeMode.COVER
    algo_up: Annotated[ImageResizeAlgorithm, Meta(description="Upscale algorithm")] = (
        ImageResizeAlgorithm.CUBIC
    )
    algo_down: Annotated[
        ImageResizeAlgorithm, Meta(description="Downscale algorithm")
    ] = ImageResizeAlgorithm.AREA
    border_type: Annotated[ImageBorderType, Meta(description="Border type")] = (
        ImageBorderType.CONSTANT
    )
    border_value: Annotated[
        Tuple[int, int, int] | str | int, Meta(description="Border value, rgb or hex")
    ] = 0
    box_rel: Annotated[
        Tuple[float, float, float, float],
        Meta(description="Crop before resize (left, top, right, bottom)"),
    ] = None

    def __post_init__(self):
        if self.size is not None:
            for s in self.size:
                if s < 16 or s > 8192:
                    raise ValueError(f"invalid target image size {self.size}")

    def get_args(self) -> dict:
        """
        Get image resize arguments
        """

        args = {
            name: getattr(self, name)
            for name in [
                "size",
                "size_mean",
                "aspect",
                "max_area",
                "multiple_of",
                "mode",
                "border_value",
                "box_rel",
            ]
        }
        args["algo_up"] = cv_algo(self.algo_up)
        args["algo_down"] = cv_algo(self.algo_down)
        args["border_type"] = cv_border(self.border_type)
        return args


def crop_image(
    box: List[int] | np.ndarray,
    pad: int | List[int] | None = None,
    pad_rel: float | List[float] | None = None,
    image: np.ndarray | None = None,
    size: Tuple[int, int] | None = None,
    square: bool = False,
    crop: bool = True,
    rel_axis: str | None = "y",
    clip: bool = True,
    border: Tuple[int, int, int] | None = None,
    multiple_of: int | Tuple[int, Tuple[int | None, int | None]] | None = None,
):
    box = np.array(box, dtype=int)

    if image is None:
        if size is None:
            raise ValueError("either image or size must be provided")
        h, w = size
    else:
        h, w = image.shape[:2]

    bh = max(box[3] - box[1], 2)
    bw = max(box[2] - box[0], 2)

    if pad is None and pad_rel is not None:
        if isinstance(pad_rel, (int, float)):
            pad_rel = (pad_rel, pad_rel, pad_rel, pad_rel)

        pad = np.array(pad_rel, dtype=float)

        if rel_axis is None:
            pad[0] *= bw
            pad[1] *= bh
            pad[2] *= bw
            pad[3] *= bh
        else:
            k = bh if rel_axis == "y" else bw
            pad *= k

    if pad is not None:
        if isinstance(pad, (int, float)):
            pad = np.array([pad, pad], dtype=float)
        elif not isinstance(pad, np.ndarray):
            pad = np.array(pad, dtype=float)

        if len(pad) < 4:
            pad = np.array([pad[0], pad[1], pad[0], pad[1]], dtype=float)

        halfw, halfh = bw // 2, bh // 2
        # pad = np.maximum(pad, [-halfw + 1, -halfh + 1, -halfw + 1, -halfh + 1])

        if pad[0] < 0:
            pad[0] = max(pad[0], -halfw + 1)
        if pad[1] < 0:
            pad[1] = max(pad[1], -halfh + 1)
        if pad[2] < 0:
            pad[2] = max(pad[2], -halfw + 1)
        if pad[3] < 0:
            pad[3] = max(pad[3], -halfh + 1)

        if square:
            max_pad = np.max(pad)
            if bw > bh:
                pad[0] = max_pad
                pad[1] = (bw - bh) // 2 + max_pad
                pad[2] = max_pad
                pad[3] = pad[1]
            else:
                pad[0] = (bh - bw) // 2 + max_pad
                pad[1] = max_pad
                pad[2] = pad[0]
                pad[3] = max_pad

        new_box = box + np.array([-pad[0], -pad[1], pad[2], pad[3]], dtype=int)
    else:
        new_box = box.copy()

    if clip:
        new_box[0] = np.clip(new_box[0], 0, w)
        new_box[1] = np.clip(new_box[1], 0, h)
        new_box[2] = np.clip(new_box[2], 0, w)
        new_box[3] = np.clip(new_box[3], 0, h)

    if multiple_of is not None:
        bw, bh = new_box[2] - new_box[0], new_box[3] - new_box[1]
        if isinstance(multiple_of, int):
            if multiple_of < 1:
                raise ValueError(f"invalid multiple_of: {multiple_of}")
            mof_x, mof_y = multiple_of, multiple_of
            nbw = bw // mof_x * mof_x
            nbh = bh // mof_y * mof_y
        else:
            mof_x, mof_y, fsize = multiple_of[0], multiple_of[0], multiple_of[1]
            target_w = fsize[0]
            target_h = fsize[1]
            if mof_x < 1 or mof_y < 1:
                raise ValueError(f"invalid multiple_of: {multiple_of}")
            if target_w is not None:
                nbw = min(bw, target_w) // mof_x * mof_x
                scale = target_w / nbw
                nbh = int(round(bh / bw * nbw))
                scaled_h = int(round(nbh * scale))
                scaled_h = (scaled_h // mof_y) * mof_y
                nbh = int(round(scaled_h / scale))
                nbw = max(nbw, 4)
                nbh = max(nbh, 4)
            elif target_h is not None:
                nbh = min(bh, target_h) // mof_y * mof_y
                scale = target_h / nbh
                nbw = int(round(bw / bh * nbh))
                scaled_w = int(round(nbw * scale))
                scaled_w = (scaled_w // mof_x) * mof_x
                nbw = int(round(scaled_w / scale))
                nbw = max(nbw, 4)
                nbh = max(nbh, 4)
            else:
                raise ValueError(f"invalid multiple_of: {multiple_of}")
            # mof_x, mof_y, fsize = multiple_of[0], multiple_of[0], multiple_of[1]
            # if mof_x < 1 or mof_y < 1:
            #     raise ValueError(f"invalid multiple_of: {multiple_of}")
            # if fsize[0] is not None:
            #     fs = fsize[0] / bw
            # else:
            #     fs = fsize[1] / bh
            # fbw = int(np.floor(bw * fs / mof_x) * mof_x)
            # fbh = int(np.floor(bh * fs / mof_y) * mof_y)
            # nbw = max(round(fbw / fs), 4)
            # nbh = max(round(fbh / fs), 4)
        bmid_x = (new_box[0] + new_box[2]) // 2
        bmid_y = (new_box[1] + new_box[3]) // 2
        new_box[0] = bmid_x - nbw // 2
        new_box[1] = bmid_y - nbh // 2
        new_box[2] = new_box[0] + nbw
        new_box[3] = new_box[1] + nbh

    if clip:
        new_image = None
        if crop and image is not None:
            new_image = image[new_box[1] : new_box[3], new_box[0] : new_box[2]]

    elif image is not None:
        left_pad = max(0, -new_box[0])
        top_pad = max(0, -new_box[1])

        new_w = new_box[2] - new_box[0]
        new_h = new_box[3] - new_box[1]

        if border is None:
            border = (0, 0, 0)

        new_image = np.full((new_h, new_w, image.shape[2]), border, dtype=image.dtype)

        copy_x1 = max(0, new_box[0])
        copy_y1 = max(0, new_box[1])
        copy_x2 = min(w, new_box[2])
        copy_y2 = min(h, new_box[3])

        dest_x1 = left_pad
        dest_y1 = top_pad
        dest_x2 = dest_x1 + (copy_x2 - copy_x1)
        dest_y2 = dest_y1 + (copy_y2 - copy_y1)

        new_image[dest_y1:dest_y2, dest_x1:dest_x2] = image[
            copy_y1:copy_y2, copy_x1:copy_x2
        ]

    return new_box, new_image


class BoxTarget(Struct, **struct_options):
    pad: (
        Annotated[List[int] | int, Meta(description="Box padding (x, y) in pixels")]
        | None
    ) = None
    pad_rel: (
        Annotated[
            List[float] | float,
            Meta(
                description="Box padding (x, y) relative to box side ('rel_axis' axis)"
            ),
        ]
        | None
    ) = None
    rel_axis: str | None = "y"
    border: (
        Annotated[Tuple[int, int, int], Meta(description="Out of box color")] | None
    ) = None
    square: Annotated[bool, Meta(description="Make square box")] = False
    clip: Annotated[bool, Meta(description="Clip box to image size")] = True
    crop: Annotated[bool, Meta(description="Make cropped image")] = False
    multiple_of: (
        Annotated[
            int | Tuple[int, Tuple[int | None, int | None]],
            Meta(description="Each side must be multiple of"),
        ]
        | None
    ) = None

    def get_crop_args(self) -> dict:
        """
        Get crop_image arguments
        """
        return {
            name: getattr(self, name)
            for name in [
                "pad",
                "pad_rel",
                "rel_axis",
                "border",
                "square",
                "clip",
                "crop",
                "multiple_of",
            ]
        }


def get_event_loop() -> asyncio.AbstractEventLoop:
    """
    Get asyncio event loop
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError("Loop is closed")
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop
