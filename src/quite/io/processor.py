from typing import Annotated, Any

from msgspec import Meta

from quite.globals import struct_options
from .common import Unique


class Processable(Unique, **struct_options):
    """
    Post-processing model can be applied
    """

    processor: (
        Annotated[Any, Meta(description="Post-processing model to be applied")] | None
    ) = None

    def __post_init__(self):
        super().__post_init__()
