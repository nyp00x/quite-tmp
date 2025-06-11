from typing import Annotated, List
import enum

from msgspec import Struct, Meta

import quite

__pdoc__ = {"command_id": False}


CommandID = Annotated[
    str,
    Meta(
        min_length=12,
        max_length=12,
        pattern="^[a-zA-Z_-]*$",
        description="Command ID",
    ),
]


def command_id() -> CommandID:
    return quite.uid(length=12)


class Command(enum.Enum):
    CREATE_CLONE = "create_clone"
    TERMINATE_CLONE = "terminate_clone"


class CommandState(enum.Enum):
    REJECTED = "rejected"
    CREATED = "created"
    SUBMITTED = "submitted"
    FAILED = "failed"


class CommandRequest(Struct, **quite.struct_state_options):
    """
    Cluster command request
    """

    id: CommandID | None = None
    cluster: quite.ClusterName | None = None
    cloneset: quite.CloneSetName | None = None
    clone: quite.CloneName | None = None
    command: Command
    hook: quite.HookURL | None = None
    warmup_task: (
        quite.InferenceTaskRequest
        | str
        | List[quite.InferenceTaskRequest | str | None]
        | None
    ) = None
    delay: (
        Annotated[int, Meta(ge=0, le=3600, description="Delay in seconds")] | None
    ) = None

    def __post_init__(self):
        if self.id is None:
            self.id = command_id()
        if isinstance(self.cluster, str):
            self.cluster = self.cluster.strip()
            if not self.cluster:
                self.cluster = None
        if isinstance(self.cloneset, str):
            self.cloneset = self.cloneset.strip()
            if not self.cloneset:
                self.cloneset = None
        if isinstance(self.clone, str):
            self.clone = self.clone.strip()
            if not self.clone:
                self.clone = None
        if self.command == Command.TERMINATE_CLONE:
            if self.clone is None:
                raise ValueError("clone name is required for TERMINATE_CLONE command")
        if self.command == Command.CREATE_CLONE:
            if self.cloneset is None:
                raise ValueError("cloneset name is required for CREATE_CLONE command")

        if self.warmup_task is not None:
            if not isinstance(self.warmup_task, list):
                self.warmup_task = [self.warmup_task]


class CommandResponse(Struct, **quite.struct_state_options):
    """
    Cluster command response
    """

    id: CommandID | None = None
    state: CommandState
    error: quite.ErrorCode | None = None
    message: str | None = None
