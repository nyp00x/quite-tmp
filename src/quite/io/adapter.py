from quite.globals import struct_options
from .common import Unique


class Adapter(Unique, **struct_options):
    object_dir: str | None = None

    def __post_init__(self):
        super().__post_init__()
