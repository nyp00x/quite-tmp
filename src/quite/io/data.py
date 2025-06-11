from __future__ import annotations

from typing import Annotated, List, Any
import enum
import io
import time
import hashlib
import base64
from pathlib import Path

from msgspec import Struct, Meta, convert
import numpy as np

from .common import URL
from .processor import Processable
from quite.globals import struct_options, new_logger
from quite.utils import (
    download_file,
    to_base64,
    from_base64,
    make_id,
    ImageResizeOptions,
    ImageMode,
    copy_struct_fields,
)

logger = new_logger("<black>data</black>")

__pdoc__ = {
    "DataType.pick_file_name": False,
    "DataType.get_file_name": False,
    "TextInput": False,
    "TextOutput": False,
    "TextFormat": False,
    "TextTarget": False,
}

data_types_map = {}


def make_data(data: dict, depth=0) -> DataType:
    """
    Make quite.DataType from a dictionary
    """
    global data_types_map

    t = data.get("type", None)
    if t is None:
        raise ValueError("cannot make data type: 'type' is not set")

    dt = data_types_map.get(t, None)
    if dt is None:
        raise ValueError("cannot make data type: unknown type")

    array = data.get("array", None)
    if isinstance(array, bytes):
        f = io.BytesIO(array)
        f.seek(0)
        data["array"] = np.load(f, allow_pickle=True)

    batch = data.get("batch", None)
    if isinstance(batch, list) and len(batch) and depth < 1:
        for i, item in enumerate(batch):
            batch[i] = make_data(item, depth=depth + 1)

    return convert(data, dt)


UploadObjectFolder = Annotated[
    str,
    Meta(
        description="Folder in the object storage to upload the object to",
        min_length=2,
        max_length=128,
        pattern=r"^([a-zA-Z0-9-]*)$",
    ),
]


class DataOutput(Struct, **struct_options):
    """
    Data output properties
    """

    object_folder: UploadObjectFolder | None = None
    clone_object: bool = False

    def update(self, other: DataOutput):
        copy_struct_fields(self, other, ignore_fields=["object_folder"])
        if other.object_folder is not None and self.object_folder is None:
            self.object_folder = other.object_folder
        return self


class NDType(enum.Enum):
    """
    NDArray data type
    """

    UINT8 = "uint8"
    UINT16 = "uint16"
    UINT32 = "uint32"
    UINT64 = "uint64"

    INT8 = "int8"
    INT16 = "int16"
    INT32 = "int32"
    INT64 = "int64"

    FLOAT16 = "float16"
    FLOAT32 = "float32"

    COMPLEX64 = "complex64"
    COMPLEX128 = "complex128"

    BOOL = "bool"


class NDArrayFormat(enum.Enum):
    """
    NDArray array serialization format
    """

    NPZ = "npz"


class NDArrayTarget(enum.Enum):
    """
    NDArray array target representation
    """

    ARRAY = "array"
    LIST = "list"
    TENSOR = "tensor"
    FILE = "file"
    URL = "url"
    EMPTY = "empty"


class NDArrayInput(Struct, **struct_options):
    """
    NDArray array input properties
    """

    pass


class NDArrayOutput(DataOutput, **struct_options, tag="ndarray"):
    """
    NDArray array output properties
    """

    format: Annotated[NDArrayFormat, Meta(description="Serialized array format")] = (
        NDArrayFormat.NPZ
    )
    target: Annotated[
        NDArrayTarget | List[NDArrayTarget], Meta(description="Target representation")
    ] = NDArrayTarget.URL

    def __post_init__(self):
        if not isinstance(self.target, list):
            self.target = [self.target]

    def update(self, other: NDArrayOutput):
        object_folder = self.object_folder or other.object_folder

        copy_struct_fields(self, other, ignore_fields=["object_folder"])

        if object_folder is not None:
            self.object_folder = object_folder

        return self


class ImageDType(enum.Enum):
    """
    Image data type
    """

    UINT8 = "uint8"
    FLOAT16 = "float16"
    FLOAT32 = "float32"
    BOOL = "bool"


class ImageFormat(enum.Enum):
    """
    Image format
    """

    SOURCE = "source"
    AVIF = "avif"
    WEBP = "webp"
    PNG = "png"
    JPEG = "jpg"
    # JPEGXL = "jxl"


class ImageInput(Struct, **struct_options):
    """
    Image input properties
    """

    resize: ImageResizeOptions | None = None
    mode: ImageMode | None = None


class ImageOutput(DataOutput, **struct_options, tag="image"):
    """
    Image output properties
    """

    format: Annotated[ImageFormat, Meta(description="Image format")] = (
        ImageFormat.SOURCE
    )
    target: Annotated[
        NDArrayTarget | List[NDArrayTarget], Meta(description="Target representation")
    ] = NDArrayTarget.URL
    quality: (
        Annotated[int, Meta(ge=10, le=100, description="Image qulity (10..100)")] | None
    ) = None
    resize: ImageResizeOptions | None = None

    def __post_init__(self):
        if not isinstance(self.target, list):
            self.target = [self.target]


class AudioDType(enum.Enum):
    """
    Audio data type
    """

    FLOAT32 = "float32"


class AudioTarget(enum.Enum):
    """
    Audio target representation
    """

    ARRAY = "array"
    FILE = "file"
    URL = "url"
    EMPTY = "empty"


class AudioFormat(enum.Enum):
    """
    Audio format
    """

    OGG = "ogg"
    MP3 = "mp3"
    WAV = "wav"
    FLAC = "flac"


class AudioInput(Struct, **struct_options):
    """
    Audio input properties
    """

    sr: (
        Annotated[int, Meta(ge=1000, le=384000, description="Sampling rate (1k..384k)")]
        | None
    ) = None
    duration: (
        Annotated[
            float,
            Meta(
                description="Limit audio duration (in seconds) when decoding",
                ge=0,
                le=43200,
            ),
        ]
        | None
    ) = None


class AudioOutput(DataOutput, **struct_options, tag="audio"):
    """
    Audio output properties
    """

    format: Annotated[AudioFormat, Meta(description="Audio format")] = AudioFormat.MP3
    target: Annotated[
        AudioTarget | List[AudioTarget], Meta(description="Target representation")
    ] = AudioTarget.ARRAY
    sr: (
        Annotated[int, Meta(ge=1000, le=384000, description="Sampling rate (1k..384k)")]
        | None
    ) = None
    duration: (
        Annotated[
            float,
            Meta(
                description="Limit audio duration (in seconds) when saving",
                ge=0,
                le=43200,
            ),
        ]
        | None
    ) = None

    def __post_init__(self):
        if not isinstance(self.target, list):
            self.target = [self.target]


class VideoDType(enum.Enum):
    """
    Video data type
    """

    UINT8 = "uint8"


class VideoCodec(enum.Enum):
    """
    Video codec
    """

    HEVC = "hevc"
    AV1 = "av1"


class VideoTarget(enum.Enum):
    """
    Video target representation
    """

    ARRAY = "array"
    FILE = "file"
    URL = "url"
    EMPTY = "empty"
    TENSOR = "tensor"


class VideoFormat(enum.Enum):
    """
    Video format
    """

    MP4 = "mp4"


class VideoInput(Struct, **struct_options):
    """
    Video input properties
    """

    num_skip_frames: (
        Annotated[
            int,
            Meta(
                ge=0,
                le=100000,
                description="Skip n frames for each n + 1 frames when decoding",
            ),
        ]
        | None
    ) = None
    num_frames: (
        Annotated[
            int,
            Meta(description="Limit number of frames when decoding", ge=4, le=108000),
        ]
        | None
    ) = None
    duration: Annotated[
        float,
        Meta(
            description="Limit video duration (in seconds) when decoding",
            ge=0,
            le=3600,
        ),
    ] = 120
    duration_ratio: (
        Annotated[
            float,
            Meta(description="Limit video duration ratio when decoding", ge=0, le=1),
        ]
        | None
    ) = None
    resize: ImageResizeOptions | None = None


class VideoOutput(DataOutput, **struct_options, tag="video"):
    """
    Video output properties
    """

    format: Annotated[VideoFormat, Meta(description="Video format")] = VideoFormat.MP4
    target: Annotated[
        VideoTarget | List[VideoTarget], Meta(description="Target representation")
    ] = VideoTarget.ARRAY
    codec: Annotated[VideoCodec, Meta(description="Video codec")] = VideoCodec.HEVC
    bitrate: Annotated[
        int, Meta(ge=100, le=10000, description="Bitrate (100k..10000k)")
    ] = 1000
    fps: Annotated[int, Meta(ge=1, le=60, description="Frames per second")] | None = (
        None
    )
    resize: ImageResizeOptions | None = None

    def __post_init__(self):
        if not isinstance(self.target, list):
            self.target = [self.target]


class JSONFormat(enum.Enum):
    """
    JSON serialization format
    """

    JSON = "json"


class JSONTarget(enum.Enum):
    """
    JSON target representation
    """

    OBJECT = "object"
    FILE = "file"
    URL = "url"
    EMPTY = "empty"


class JSONInput(Struct, **struct_options):
    """
    JSON input properties
    """

    pass


class JSONOutput(DataOutput, **struct_options, tag="json"):
    """
    JSON output properties
    """

    format: Annotated[JSONFormat, Meta(description="Serialized json format")] = (
        JSONFormat.JSON
    )

    target: Annotated[
        JSONTarget | List[JSONTarget], Meta(description="Target representation")
    ] = JSONTarget.OBJECT

    def __post_init__(self):
        if not isinstance(self.target, list):
            self.target = [self.target]


class TextFormat(enum.Enum):
    TXT = "txt"


class TextTarget(enum.Enum):
    TEXT = "text"
    FILE = "file"
    URL = "url"


class TextInput(Struct, **struct_options):
    pass


class TextOutput(DataOutput, **struct_options, tag="text"):
    format: Annotated[TextFormat, Meta(description="Serialized text format")] = (
        TextFormat.TXT
    )
    target: Annotated[
        TextTarget | List[TextTarget], Meta(description="Target representation")
    ] = TextTarget.TEXT

    def __post_init__(self):
        if not isinstance(self.target, list):
            self.target = [self.target]


class DataType(Processable, **struct_options):
    """
    Base class for data types
    """

    file: Annotated[bytes, Meta(description="Compressed binary")] | None = None
    fileb64: Annotated[str, Meta(description="Base64 encoded binary")] | None = None
    from_name: str | None = None
    from_index: int | None = None
    url: URL | None = None
    object_url: URL | None = None
    _metadata: dict = {}

    def __post_init__(self):
        super().__post_init__()

    def __str__(self):
        s = super().__str__()
        if self.is_batch():
            items_s = ", ".join(str(item) for item in self.batch)
            s += f" batch of {len(self.batch)}: [{items_s}]"
        return s

    def __len__(self):
        return len(self.batch)

    def __iter__(self):
        return iter(self.batch)

    def is_batch(self):
        """
        Check if the data is a batch
        """
        return len(self.batch) > 0

    def is_empty(self):
        raise NotImplementedError

    def empty(self):
        raise NotImplementedError

    @classmethod
    def from_dict(self, d: dict):
        raise NotImplementedError

    def item(self):
        """
        Get the first item in the batch if it is a batch or the instance itself
        """
        if self.is_batch():
            item = self.batch[0]
            if item.is_batch():
                item.batch = []
            return item
        return self

    def to_batch(self, id: str | None = None, from_id: str | None = None):
        """
        Return new instance with the batch set to the current instance if it is not a batch,
        otherwise return the instance itself
        """
        if self.is_batch():
            return self
        if id is not None:
            d = type(self)(id=id, from_id=from_id, batch=[self])
        else:
            d = type(self)(from_id=from_id, batch=[self])
        return d

    def copy_to(
        self,
        obj: Struct,
        fields: List[str] | None = None,
        ignore_fields: list[str] = ["id", "from_id"],
    ):
        """
        Shallow-copy fields to another object
        """
        if fields is None:
            fields = list(self.__struct_fields__)
        if len(ignore_fields):
            fields = [field for field in fields if field not in ignore_fields]
        for field in fields:
            setattr(obj, field, getattr(self, field))
        return self

    def copy(self, id: str | None = None, from_id: str | None = None, copy_batch=True):
        """
        Shallow copy with optional new `id` and `from_id`
        """
        instance = super().copy(id=id, from_id=from_id)
        if copy_batch and instance.is_batch():
            instance.batch = [item.copy() for item in instance.batch]
        return instance

    async def download(
        self,
        repository: Any | None = None,
        force: bool = True,
        log_prefix: str | None = None,
    ) -> Any:
        """
        Download data
        """
        time_start = time.perf_counter()

        if self.is_batch() and not self.from_id:
            for item in self.batch:
                if item.is_batch():
                    item.batch = []
                await item.download(
                    repository=repository, force=force, log_prefix=log_prefix
                )
            return self

        if self.file is not None and not force:
            return self

        if self.from_id:
            object_dir = await repository.object_manager.download(self)
            if object_dir is None:
                raise ValueError(f"cannot download {self} from repository")

            self._metadata["download_time"] = round(time.perf_counter() - time_start, 4)
            self._inc_download_items()
            if log_prefix is not None:
                logger.info(
                    f"{log_prefix} {self} downloaded in {self._metadata['download_time']}s"
                )
            batch = self.get_batch_from_object_dir(object_dir)
            if len(batch):
                self.batch = batch
            else:
                self.file = self.get_file_from_object_dir(object_dir)
            return self

        elif self.url is not None:
            if self.timeout:
                if repository is None:
                    raise ValueError(
                        f"cannot download {self}: repository is not provided"
                    )

                self.from_id = self._get_cache_id()

                if not repository.object_manager.exists(self.from_id):
                    self.file = await download_file(self.url)
                    self._metadata["download_time"] = round(
                        time.perf_counter() - time_start, 4
                    )
                    self._inc_download_items()
                    if log_prefix is not None:
                        logger.info(
                            f"{log_prefix} {self} downloaded in {self._metadata['download_time']}s"
                        )
                    await repository.object_manager.add_data(self, overwrite=False)

                return await self.download(
                    repository=repository, force=False, log_prefix=log_prefix
                )

            self.file = await download_file(self.url)

            self._metadata["download_time"] = round(time.perf_counter() - time_start, 4)
            self._inc_download_items()
            if log_prefix is not None:
                logger.info(
                    f"{log_prefix} {self} downloaded in {self._metadata['download_time']}s"
                )
            return self
        else:
            raise ValueError(f"cannot download {self}")

    def get_batch_from_object_dir(self, object_dir: Path) -> List[Any]:
        batch_dir = object_dir / "_items"
        if not batch_dir.exists():
            return []

        batch, item_dirs = [], {}

        for item_dir in batch_dir.iterdir():
            name = item_dir.name
            if not len(name) == 8:
                continue
            try:
                int(name)
            except ValueError:
                continue
            item_dirs[name] = item_dir

        if not len(item_dirs):
            return []

        item_dirs = sorted(item_dirs.items(), key=lambda x: int(x[0]))

        for name, item_dir in item_dirs:
            file = self.get_file_from_object_dir(item_dir)
            item = type(self)(file=file)
            batch.append(item)

        return batch

    @classmethod
    def pick_file_name(cls, file_names: List[str]) -> str | None:
        raise NotImplementedError

    def get_file_name(self):
        raise NotImplementedError

    async def upload(
        self,
        repository: Any,
        file_name: str | None = None,
        remove: bool = False,
        overwrite: bool = False,
        object_folder: UploadObjectFolder | None = None,
        clone_object: bool = False,
        log_prefix: str | None = None,
    ) -> Any:
        """
        Upload data to object storage
        """
        time_start = time.perf_counter()

        if clone_object and self.is_batch():
            for item in self.batch:
                if item.is_batch():
                    item.batch = []
                await item.upload(
                    repository=repository,
                    file_name=file_name,
                    remove=remove,
                    overwrite=overwrite,
                    object_folder=object_folder,
                    log_prefix=log_prefix,
                )
            return self

        await repository.object_manager.upload_data(
            self,
            file_name=file_name,
            remove=remove,
            overwrite=overwrite,
            object_folder=object_folder,
            as_clone_object=clone_object,
        )

        self._metadata["upload_time"] = round(time.perf_counter() - time_start, 4)
        self._inc_upload_items()
        if log_prefix is not None:
            logger.info(
                f"{log_prefix} {self} uploaded in {self._metadata['upload_time']}s"
            )

        return self

    async def save_to(
        self, repository: Any, overwrite: bool = False, write: bool = False
    ) -> Any:
        """
        Save data to the repository
        """
        # if self.is_batch():
        #     for item in self.batch:
        #         if item.is_batch():
        #             item.batch = []
        #         await item.save_to(repository, overwrite=overwrite, write=write)
        #     return self

        await repository.object_manager.add_data(self, overwrite=overwrite, write=write)

    def set_output(self, output: DataOutput):
        if self.output is None:
            self.output = output
        else:
            self.output.update(output)

    def to_base64(self) -> Any:
        """
        Convert `file` to base64-encoded `fileb64`
        """
        if self.is_batch():
            for item in self.batch:
                if item.is_batch():
                    item.batch = []
                item.to_base64()
            return self

        if self.file is not None:
            if len(self.file) / 1024 / 1024 < 6:
                self.fileb64 = to_base64(self.file)
            else:
                self.fileb64 = None
            self.file = None

        return self

    def from_base64(self) -> Any:
        """
        Convert `fileb64` to binary `file`
        """
        if self.is_batch():
            for item in self.batch:
                if item.is_batch():
                    item.batch = []
                item.from_base64()
            return self

        if self.fileb64 is not None:
            self.file = from_base64(self.fileb64)
            self.fileb64 = None

        return self

    def _reset_metadata(self):
        """
        Reset metadata
        """
        if self.is_batch():
            for item in self.batch:
                if item.is_batch():
                    item.batch = []
                item._reset_metadata()
        self._metadata = {
            "download_time": 0,
            "upload_time": 0,
            "download_items": 0,
            "upload_items": 0,
        }

    def _download_time(self):
        """
        Last download time in seconds
        """
        if self.is_batch():
            t = 0
            for item in self.batch:
                t += item._metadata.get("download_time", 0)
            return t
        return self._metadata.get("download_time", 0)

    def _download_items(self):
        """
        Number of items downloaded
        """
        if self.is_batch():
            n = 0
            for item in self.batch:
                n += item._metadata.get("download_items", 0)
            return n
        return self._metadata.get("download_items", 0)

    def _inc_download_items(self, n: int = 1):
        if "download_items" not in self._metadata:
            self._metadata["download_items"] = 0
        self._metadata["download_items"] += n

    def _upload_time(self):
        """
        Last upload time in seconds
        """
        if self.is_batch():
            t = 0
            for item in self.batch:
                t += item._metadata.get("upload_time", 0)
            return t
        return self._metadata.get("upload_time", 0)

    def _upload_items(self):
        """
        Number of items uploaded
        """
        if self.is_batch():
            n = 0
            for item in self.batch:
                n += item._metadata.get("upload_items", 0)
            return n
        return self._metadata.get("upload_items", 0)

    def _inc_upload_items(self, n: int = 1):
        if "upload_items" not in self._metadata:
            self._metadata["upload_items"] = 0
        self._metadata["upload_items"] += n

    def _hash(self, s: str) -> str:
        hash_object = hashlib.sha256(s.encode())
        b64_hash = base64.b64encode(hash_object.digest()).decode()
        alphanumeric = "".join(c for c in b64_hash if c.isalnum())
        if not alphanumeric[0].isalpha():
            return "a" + alphanumeric[:15]
        else:
            return alphanumeric[:16]

    def _get_cache_id(self) -> str:
        if self.url is None:
            raise ValueError("cannot get cache id: url is not set")
        return make_id(object_path=f"cache/{self._hash(self.url)}")
