from __future__ import annotations

from typing import Annotated, List, Dict, Tuple, Any, Set
import enum
import datetime

import msgspec
from msgspec import Struct, Meta, field

from .common import uid, struct_options, HookURL, TaskRuntime, RuntimeName
from .data import DataType
from .model import (
    ModelOutput,
    TargetResult,
    Model,
    ModelExecutionDetails,
    DataInfo,
)
from .array import NDArray
from .image import Image
from .audio import Audio
from .video import Video
from .json import JSON
from quite.globals import ErrorCode
from quite.utils import iterate_fields, copy_struct_fields, utc_now

__pdoc__ = {"task_input_decoder": False, "TaskInput.get_from_ids": False}


TaskID = Annotated[
    str,
    Meta(
        min_length=12,
        max_length=12,
        pattern="^[a-zA-Z]*$",
        description="Task ID",
    ),
]


def task_id() -> TaskID:
    """
    Generate a task ID
    """
    return uid(length=12)


class TaskState(enum.Enum):
    """
    Task state
    """

    REJECTED = "rejected"
    CREATED = "created"
    QUEUED = "queued"
    STARTED = "started"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskInput(Struct, **struct_options):
    """
    Task input data
    """

    array: NDArray | List[NDArray] = []
    image: Image | List[Image] = []
    audio: Audio | List[Audio] = []
    video: Video | List[Video] = []
    json: JSON | List[JSON] = []

    def __repr__(self) -> str:
        s = ""
        try:
            for u in [self.video, self.audio, self.image, self.array, self.json]:
                for item in u:
                    s += f"{type(item).__name__} {item.id} "
        except Exception:
            pass
        return s.strip()

    def __post_init__(self) -> None:
        if isinstance(self.array, NDArray):
            self.array = [self.array]
        if isinstance(self.image, Image):
            self.image = [self.image]
        if isinstance(self.audio, Audio):
            self.audio = [self.audio]
        if isinstance(self.video, Video):
            self.video = [self.video]
        if isinstance(self.json, JSON):
            self.json = [self.json]

    def get(self, id: str | None) -> Any:
        """
        Get data by ID
        """
        if id is None:
            return None
        for u in [self.video, self.audio, self.image, self.array, self.json]:
            for item in u:
                if item.id == id:
                    return item
        return None

    def add(self, data: Any) -> None:
        """
        Add data to the input
        """
        if isinstance(data, Video):
            self.video.append(data)
        elif isinstance(data, Audio):
            self.audio.append(data)
        elif isinstance(data, Image):
            self.image.append(data)
        elif isinstance(data, NDArray):
            self.array.append(data)
        elif isinstance(data, JSON):
            self.json.append(data)
        else:
            raise ValueError(f"invalid input type {type(data)}")

    def serializable(self):
        """
        Make all items serializable
        """
        for items in [self.array, self.image, self.audio, self.video, self.json]:
            for item in items:
                item.serializable()
        return self

    async def resolve(
        self,
        force: bool = False,
        remove: bool = True,
        repository: Any | None = None,
        log_prefix: str | None = None,
    ) -> Tuple[Dict[str, DataType], DataInfo]:
        """
        Resolve input data to default targets
        """
        data_items = {}
        data_info = DataInfo()
        for array in self.array:
            array._reset_metadata()
            data_items[array.id] = await array.resolve(
                force=force,
                remove=remove,
                repository=repository,
                log_prefix=log_prefix,
            )
            data_info.add(array)
        for image in self.image:
            image._reset_metadata()
            data_items[image.id] = await image.resolve(
                force=force,
                remove=remove,
                repository=repository,
                log_prefix=log_prefix,
            )
            data_info.add(image)
        for audio in self.audio:
            audio._reset_metadata()
            data_items[audio.id] = await audio.resolve(
                force=force,
                remove=remove,
                repository=repository,
                log_prefix=log_prefix,
            )
            data_info.add(audio)
        for video in self.video:
            video._reset_metadata()
            data_items[video.id] = await video.resolve(
                force=force,
                remove=remove,
                repository=repository,
                log_prefix=log_prefix,
            )
            data_info.add(video)
        for json in self.json:
            json._reset_metadata()
            data_items[json.id] = await json.resolve(
                force=force,
                remove=remove,
                repository=repository,
                log_prefix=log_prefix,
            )
            data_info.add(json)
        return data_items, data_info

    def get_from_ids(
        self,
        obj: Struct,
        fill_inplace: bool = False,
        strict: bool = True,
        ignore_fields: List[str] = ["id", "from_id"],
    ) -> List[Tuple[Struct, Struct]]:
        """
        Get input data from IDs
        """
        uniques = iterate_fields(obj, [DataType])
        matches = []
        for u in uniques:
            if u.value is None:
                continue
            if u.value.from_id is None:
                continue
            input_item = self.get(u.value.from_id)
            if input_item is not None:
                if fill_inplace:
                    copy_struct_fields(u.value, input_item, ignore_fields=ignore_fields)
                matches += [(u.value, input_item)]
            elif strict:
                raise ValueError(f"input {u.value.from_id} not found")
        return matches


task_input_decoder = msgspec.msgpack.Decoder(TaskInput)


class TaskRequest(Struct, **struct_options):
    """
    Abstract task request
    """

    id: TaskID | None = None
    parent_id: TaskID | None = None
    runtime: TaskRuntime = field(default_factory=TaskRuntime)
    input: TaskInput | bytes | None = None
    model: Any | Model | None = None
    priority: Annotated[int, Meta(description="Task priority", ge=0, le=2)] = 0
    hook: HookURL | None = None
    skip_states: Set[TaskState] = set()
    result: TargetResult | List[str] | str = {}
    timestamp: (
        Annotated[datetime.datetime, Meta(description="Task creation time")] | None
    ) = None
    timeout: (
        Annotated[
            int,
            Meta(
                description="The task is skipped if arrived after the timeout, in seconds",
                ge=0,
                le=604800,
            ),
        ]
        | None
    ) = None
    verbose: bool = True
    custom_data: Annotated[
        Dict[str, Any], Meta(description="Custom data to include in responses")
    ] = {}

    def __post_init__(self) -> None:
        if self.id is None:
            self.id = task_id()
        if self.model is not None and not isinstance(self.model, list):
            self.model = [self.model]
        if isinstance(self.result, str):
            self.result = [self.result]
        if isinstance(self.result, list):
            self.result = {r: {} for r in self.result}

    def __repr__(self):
        return f"{self.__class__.__name__} {self.id}"

    def decode_input(self):
        if isinstance(self.input, bytes):
            self.input = task_input_decoder.decode(self.input)
        return self

    def timed_out(self) -> bool:
        if self.timeout is None or self.timestamp is None:
            return False
        return (utc_now() - self.timestamp).total_seconds() > self.timeout

    def empty_result(self) -> TaskRequest:
        for bid, res in self.result.items():
            for k, v in res.items():
                v.target = "empty"
        return self


class InferenceTaskRequest(TaskRequest, **struct_options):
    """
    Inference request
    """

    def __post_init__(self) -> None:
        super().__post_init__()


class TaskResponse(Struct, **struct_options):
    """
    Abstract task response
    """

    state: TaskState
    id: TaskID | None = None
    error: ErrorCode | None = None
    message: str | None = None
    index: Annotated[int, Meta(description="Response index")] = 0
    runtime_name: RuntimeName | None = None
    # details: Dict[str, Any] | None = None
    custom_data: Annotated[
        Dict[str, Any], Meta(description="Custom data from the request")
    ] = {}


class TaskExecutionDetails(Struct, **struct_options):
    """
    Task execution details
    """

    resolution_time: float = 0.0
    history: List[ModelExecutionDetails] = []
    graph: Dict[str, Any] | None = None
    metrics: Dict[str, Any] | None = None


class InferenceTaskResponse(TaskResponse, **struct_options):
    """
    Inference response
    """

    model_outputs: List[ModelOutput] = []
    details: TaskExecutionDetails | None = None

    def to_base64(self) -> InferenceTaskResponse:
        for output in self.model_outputs:
            output.to_base64()
        return self

    def from_base64(self) -> InferenceTaskResponse:
        for output in self.model_outputs:
            output.from_base64()
        return self
