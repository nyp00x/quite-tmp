# ruff: noqa

from .common import *
from .data import *
from .processor import *
from .adapter import *
from .model import *
from .array import *
from .image import *
from .audio import *
from .video import *
from .json import *
from .request import *
from . import human

__pdoc__ = {"text": False}


def get_data_cls(
    data_type: str,
) -> NDArray | Image | Audio | Video | JSON | None:
    for cls in [NDArray, Image, Audio, Video, JSON]:
        if cls.__name__.lower() == data_type.lower():
            return cls
    return None
