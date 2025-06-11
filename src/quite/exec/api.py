import enum
from msgspec import Struct, Raw

import quite
from quite.state import RuntimeState, CloneState


class TaskType(enum.Enum):
    INFERENCE = "inference"
    TRAINING = "training"


class CloneStateRequest(Struct, **quite.struct_options):
    """
    Clone state request
    """

    verbose: bool = True


class CloneStateResponse(Struct, **quite.struct_options):
    """
    Clone state response
    """

    clone_state: CloneState
    error: quite.ErrorCode | None = None
    message: str | None = None


class RuntimeStateRequest(Struct, **quite.struct_options):
    """
    Runtime state request
    """

    runtime_name: quite.RuntimeName
    verbose: bool = True


class RuntimeStateResponse(Struct, **quite.struct_options):
    """
    Runtime state response
    """

    runtime_state: RuntimeState
    error: quite.ErrorCode | None = None
    message: str | None = None


class LoadRequest(Struct, **quite.struct_options):
    """
    Model load request
    """

    runtime_name: quite.RuntimeName
    model: quite.Model
    reload: bool = False


class LoadResponse(Struct, **quite.struct_options):
    """
    Model load response
    """

    error: quite.ErrorCode | None = None
    message: str | None = None


class UnloadRequest(Struct, **quite.struct_options):
    """
    Model unload request
    """

    runtime_name: quite.RuntimeName
    model: quite.Model


class UnloadResponse(Struct, **quite.struct_options):
    """
    Model unload response
    """

    error: quite.ErrorCode | None = None
    message: str | None = None


class RunRequest(Struct, **quite.struct_options):
    """
    Run RPC request
    """

    task_type: TaskType = TaskType.INFERENCE
    task_request: Raw
    internal: bool = False
    native: bool = False


class RunResponse(Struct, **quite.struct_options):
    """
    Run RPC response
    """

    error: quite.ErrorCode | None = None
    message: str | None = None
    state: quite.TaskState | None = None
    task_response: Raw


class DownloadRequest(Struct, **quite.struct_options):
    """
    Download request
    """

    unique: quite.Unique
    sync_model: bool = False
    sync_object: bool = False


class DownloadResponse(Struct, **quite.struct_options):
    """
    Download response
    """

    error: quite.ErrorCode | None = None
    message: str | None = None


class RemoveRequest(Struct, **quite.struct_options):
    """
    Remove request
    """

    unique: quite.Unique


class RemoveResponse(Struct, **quite.struct_options):
    """
    Remove response
    """

    error: quite.ErrorCode | None = None
    message: str | None = None
