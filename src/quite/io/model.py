from __future__ import annotations

from typing import Annotated, Dict, List, Any, Set

from msgspec import Struct, Meta, field

from .common import Unique, SyncTarget, ReloadTarget, ModelRuntime, RuntimeName
from .processor import Processable
from .data import (
    NDArrayOutput,
    ImageOutput,
    AudioOutput,
    VideoOutput,
    JSONOutput,
    DataType,
)
from .array import NDArray
from .image import Image
from .audio import Audio
from .video import Video
from .json import JSON

from ..globals import struct_options


class DataInfo(Struct, **struct_options):
    """
    Time taken for data download and upload
    """

    download_time: Annotated[
        float, Meta(description="Time taken to download, in seconds")
    ] = 0.0
    download_items: Annotated[int, Meta(description="Number of items downloaded")] = 0
    upload_time: Annotated[
        float, Meta(description="Time taken to upload, in seconds")
    ] = 0.0
    upload_items: Annotated[int, Meta(description="Number of items uploaded")] = 0

    def add(self, data_item: DataType):
        self.download_time += data_item._download_time()
        self.download_items += data_item._download_items()
        self.upload_time += data_item._upload_time()
        self.upload_items += data_item._upload_items()


class ModelTimeTaken(Struct, **struct_options):
    """
    Time taken for model download, load and execution
    """

    download: Annotated[
        float, Meta(description="Time taken to download model, in seconds")
    ] = 0.0
    load: Annotated[float, Meta(description="Time taken to load model, in seconds")] = (
        0.0
    )
    execution: Annotated[
        float, Meta(description="Time from the start of execution, in seconds")
    ] = 0.0

    def __post_init__(self):
        if self.download:
            self.download = round(self.download, 3)
        if self.load:
            self.load = round(self.load, 3)
        if self.execution:
            self.execution = round(self.execution, 3)

    def __str__(self):
        return f"[d: {self.download:.3f} l: {self.load:.3f} e: {self.execution:.3f}]"


class ModelExecutionDetails(Struct, **struct_options):
    """
    Execution details
    """

    model_id: Annotated[str, Meta(description="Model ID")]
    node_id: Annotated[str, Meta(description="Node ID")]
    runtime_name: RuntimeName | None = None
    time: Annotated[ModelTimeTaken, Meta(description="Time taken")] = field(
        default_factory=ModelTimeTaken
    )
    result_ids: Annotated[List[str], Meta(description="Result names")] = []

    def __str__(self):
        s = f"{self.model_id} {self.node_id} ({self.runtime_name}) {self.time}"
        if len(self.result_ids):
            s += f" [{' '.join(self.result_ids)}]"
        return s


Output = Annotated[
    Dict[str, NDArray | Image | Audio | Video | JSON],
    Meta(description="General output, name → data"),
]

TargetOutput = Annotated[
    Dict[
        str,
        NDArrayOutput | ImageOutput | AudioOutput | VideoOutput | JSONOutput,
    ],
    Meta(description="Target outputs"),
]

TargetResult = Annotated[
    Dict[str, TargetOutput],
    Meta(description="Batch/iter id -> TargetOutput"),
]


class Iter(Processable, **struct_options):
    """
    Abstract iteration params
    """

    batch_size: Annotated[int, Meta(ge=1, le=2**15, description="Minibatch size")] = 1
    output: (
        Annotated[TargetOutput, Meta(description="Iter postprocessing options")] | None
    ) = None

    def __post_init__(self):
        super().__post_init__()

    def __repr__(self):
        return f"{self.__class__.__name__} {self.id}"


class Batch(Processable, **struct_options):
    """
    Abstract batch params
    """

    seed: int | List[int] | None = None
    size: Annotated[
        int, Meta(ge=1, le=2**15, description="Taarget number of items")
    ] = 1
    iter: Iter | List[Iter] = []
    sequential: Annotated[
        bool,
        Meta(
            description="If sequential, each minibatch takes the previous minibatch as input"
        ),
    ] = False
    output: (
        Annotated[TargetOutput, Meta(description="Batch postprocessing options")] | None
    ) = None

    def __post_init__(self):
        if not isinstance(self.iter, list):
            self.iter = [self.iter]
        for iter in self.iter:
            if iter.batch_size is not None and iter.batch_size > self.size:
                raise ValueError(
                    "Iter batch_size must be less than or equal to batch size"
                )
        super().__post_init__()

    def __repr__(self):
        return (
            f"{self.__class__.__name__} {self.id} "
            f"(size {self.size} seed {self.seed}){' sequential' if self.sequential else ''}"
        )


class Model(Unique, **struct_options):
    """
    Abstract model
    """

    runtime: ModelRuntime = field(default_factory=ModelRuntime)
    anti_tags: Set[str] = set()
    batch: Batch | List[Batch] = []
    sync: (
        Annotated[List[SyncTarget] | SyncTarget, Meta(description="Sync targets")]
        | None
    ) = None
    reload: (
        Annotated[List[ReloadTarget] | ReloadTarget, Meta(description="Reload targets")]
        | None
    ) = None

    def __post_init__(self):
        if self.sync is None:
            self.sync = []
        elif not isinstance(self.sync, list):
            self.sync = [self.sync]
        if self.reload is None:
            self.reload = []
        elif not isinstance(self.reload, list):
            self.reload = [self.reload]
        if not isinstance(self.batch, list):
            self.batch = [self.batch]

        super().__post_init__()

    def get_target_output(self, id: str) -> TargetOutput | None:
        """
        Get the target output for the given id
        """
        for b in self.batch:
            if b.output is not None and id == b.id:
                return b.output
            for i in b.iter:
                if i.output is not None and id == i.id:
                    return i.output
        return None


class ModelOutput(Struct, **struct_options):
    """
    Model execution result
    """

    model_id: Annotated[str, Meta(description="Model ID")]
    output: Annotated[
        Dict[str, Output],
        Meta(description="dict[batch/iter id → Output]"),
    ] = {}
    runtime_name: RuntimeName | None = None
    time: ModelTimeTaken = field(default_factory=ModelTimeTaken)
    data_info: DataInfo = field(default_factory=DataInfo)

    def __repr__(self):
        s = f"{self.__class__.__name__} {self.model_id}"
        try:
            for id, out in self.output.items():
                s += f" {id}:"
                for name, item in out.items():
                    s += f" {name} = {item}"
        except Exception:
            pass
        return s

    def serializable(self):
        for o in self.output.values():
            for v in o.values():
                v.serializable()
        return self

    def to_base64(self):
        for o in self.output.values():
            for v in o.values():
                v.to_base64()
        return self

    def from_base64(self):
        for o in self.output.values():
            for v in o.values():
                v.from_base64()
        return self

    async def apply(
        self,
        target: TargetResult,
        repository: Any = None,
        copy: bool = True,
        log_prefix: str | None = None,
    ) -> ModelOutput:
        """
        Apply the target properties to the output
        """
        model_output = ModelOutput(
            model_id=self.model_id,
            output={},
            runtime_name=self.runtime_name,
            time=self.time,
            data_info=DataInfo(),
        )

        for id, out in self.output.items():
            bi_target = target.get(id, None)
            if bi_target is None:
                continue
            model_output.output[id] = {}
            for name, item in out.items():
                item_target = bi_target.get(name, None)
                if copy:
                    item = item.copy()
                if item_target is None:
                    model_output.output[id][name] = item
                else:
                    if item.is_empty():
                        model_output.output[id][name] = item
                    else:
                        item._reset_metadata()
                        model_output.output[id][name] = await item.apply(
                            item_target, repository=repository, log_prefix=log_prefix
                        )
                        model_output.data_info.add(item)
        return model_output
