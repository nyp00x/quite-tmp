from __future__ import annotations

from typing import Annotated, Tuple
import enum
import copy
import re

import nanoid
from msgspec import Meta, Struct

from quite.globals import struct_options

__pdoc__ = {"Unique.no_proto": False, "Unique.split_id": False, "uid": False}

alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


def uid(length=10) -> str:
    return str(nanoid.generate(alphabet, length))


NAME = Annotated[
    str,
    Meta(
        min_length=1,
        max_length=64,
        pattern="^[a-zA-Z][a-zA-Z0-9_-]*$",
    ),
]

# unique_id_pattern = (
#     r"^([a-zA-Z0-9-]*)(?:@([a-z][a-z0-9-]+))?(?::([a-zA-Z][a-zA-Z0-9-/]+))?$"
# )
unique_id_pattern = r"^([a-zA-Z0-9-]*)(?:@([a-z][a-z0-9-]+)(?:#([a-zA-Z][a-zA-Z0-9-]+))?)?(?::([a-zA-Z][a-zA-Z0-9-/]+))?$"

UID = Annotated[str, Meta(description="Unique ID", pattern=unique_id_pattern)]

unique_model_object_id_pattern = r"^([a-zA-Z0-9-]+):([a-zA-Z][a-zA-Z0-9-/]+)$"
MODEL_OBJECT_UID = Annotated[
    str, Meta(description="Model object ID", pattern=unique_model_object_id_pattern)
]
unique_data_object_id_pattern = r"^:([a-zA-Z][a-zA-Z0-9-/]+)$"
DATA_OBJECT_UID = Annotated[
    str, Meta(description="Data object ID", pattern=unique_data_object_id_pattern)
]


URL = Annotated[
    str,
    Meta(
        title="URL",
        description="Must start with http:// or https:// or s3://",
        min_length=8,
        max_length=2048,
        pattern=r"^(http://|https://|s3://)",
    ),
]

HookURL = Annotated[
    str,
    Meta(
        title="Hook URL",
        description="Must start with http:// or https://",
        min_length=8,
        max_length=2048,
        pattern=r"^(http://|https://)",
    ),
]

ModelHead = Annotated[str, Meta(description="Model head", min_length=7, max_length=7)]
CloneName = Annotated[NAME, Meta(description="Clone name")]
CloneSetName = Annotated[NAME, Meta(description="CloneSet name")]
ClusterName = Annotated[NAME, Meta(description="Cluster name")]
RuntimeName = Annotated[
    str,
    Meta(
        description="Runtime name",
        min_length=1,
        max_length=64,
        pattern="^[a-zA-Z0-9_-]*$",
    ),
]


class ObjectDomain(enum.Enum):
    """
    Persistent object domain
    """

    DATA = "data"
    MODEL = "model"


class GRPCEndpoint(Struct, **struct_options):
    """
    gRPC endpoint
    """

    host: (
        Annotated[
            str,
            Meta(
                description="Host name or IP address",
                min_length=6,
                max_length=2048,
            ),
        ]
        | None
    ) = None
    port: (
        Annotated[
            int,
            Meta(
                description="Port number",
                ge=1,
            ),
        ]
        | None
    ) = None
    secure: bool = False

    def __str__(self):
        return f"{self.host}:{self.port}"

    @classmethod
    def from_str(cls, s: str, secure: bool = False):
        """
        Create from string
        """
        parts = s.split(":")
        if len(parts) != 2:
            raise ValueError(f"invalid gRPC endpoint: {s}")
        return GRPCEndpoint(host=parts[0], port=int(parts[1]), secure=secure)

    def __eq__(self, other):
        return (
            self.host == other.host
            and self.port == other.port
            and self.secure == other.secure
        )

    def __hash__(self):
        return hash((self.host, self.port, self.secure))


class IDInfo(Struct, **struct_options):
    """
    Repository name, revision and object path
    """

    repo_name: str | None = None
    revision: str | None = None
    variant: str | None = None
    object_path: str | None = None

    def __str__(self):
        return f"{self.repo_name}@{self.revision}#{self.variant}:{self.object_path}"


class Unique(Struct, **struct_options):
    """
    Base class for unique items such as models and data objects
    """

    id: Annotated[str, Meta(description="Instance ID")] | None = None
    from_id: Annotated[str, Meta(description="Prototype ID")] | None = None
    timeout: Annotated[int, Meta(description="Persistence timeout", ge=1)] | None = None

    def __post_init__(self):
        if self.id is not None:
            self.id = self.id.replace(" ", "")
        if self.from_id is not None:
            self.from_id = self.from_id.replace(" ", "")

        if self.from_id is not None:
            (
                repo_name,
                revision,
                variant,
                object_path,
            ) = self.parse_id(self.from_id)

        if self.id is None:
            self.id = uid()
            if self.from_id is not None and repo_name is not None:
                self.id = f"{repo_name}-{self.id}"

        elif self.id == self.from_id:
            raise ValueError(
                f"invalid {self.__class__.__name__} id {self.id}: same as from_id"
            )

    def __str__(self):
        s = f"{self.__class__.__name__} ({self.id or None}"
        s += f" from {self.from_id})" if self.from_id else ")"
        return s

    @classmethod
    def uid(self) -> str:
        """
        Generate unique ID
        """
        return uid()

    @classmethod
    def parse_id(cls, id: str) -> Tuple[str | None, str | None, str | None, str | None]:
        """
        Parse ID into repository name, revision and object name
        """
        repo_name, revision, variant, object_path = cls.split_id(id)

        if object_path is not None:
            if object_path[-1] == "/":
                raise ValueError(
                    f"invalid object_path {cls.__name__} ({object_path}): must not start or end with '/'"
                )

            if repo_name is not None:
                object_path = f"{repo_name}/{object_path}"

            path_parts = object_path.split("/")
            if len(path_parts) != 2 or len(path_parts[0]) < 2:
                raise ValueError(
                    f"invalid object_path {cls.__name__} ({object_path}): must be in format 'some/path'"
                )

        return repo_name, revision, variant, object_path

    def id_info(self, proto: bool = True) -> IDInfo:
        """
        Get ID semantic information
        """
        repo_name, revision, variant, object_path = self.parse_id(
            self.from_id if proto else self.id
        )
        return IDInfo(
            repo_name=repo_name,
            revision=revision,
            variant=variant,
            object_path=object_path,
        )

    @classmethod
    def split_id(cls, id: str) -> Tuple[str | None, str | None, str | None, str | None]:
        match = cls.match_id(id)
        if match:
            groups = match.groups()
            return (
                groups[0] if groups[0] else None,
                groups[1] if groups[1] else None,
                groups[2] if groups[2] else None,
                groups[3] if groups[3] else None,
            )
        else:
            raise ValueError(f"invalid {cls.__name__} id/from_id: {id}")

    @classmethod
    def match_id(cls, id: str) -> re.Match | None:
        """
        Match ID against the unique id pattern
        """
        return re.match(unique_id_pattern, id)

    def no_proto(self):
        if self.is_batch():
            for item in self.batch:
                item.no_proto()
            return self

        if self.id is None:
            raise ValueError(
                f"cannot remove prototype from {self.__class__.__name__} (id is None)"
            )

        self.from_id = None
        return self

    def is_batch(self):
        return False

    def copy(self, id: str | None = None, from_id: str | None = None):
        """
        Shallow copy with optional new `id` and `from_id`
        """
        obj = copy.copy(self)
        if id is not None:
            obj.id = id
        else:
            obj.id = uid()
        if from_id is not None:
            obj.from_id = from_id
        return obj


class ReloadTarget(enum.Enum):
    """
    Items to reload prior to a task execution
    """

    MODEL = "model"


class SyncTarget(enum.Enum):
    """
    Items to sync prior to a task execution
    """

    MODEL = "model"
    OBJECT = "object"


class TaskRuntime(Struct, **struct_options):
    """
    Primary runtime for a task
    """

    cluster: ClusterName | None = None
    name: RuntimeName | None = None
    runner: GRPCEndpoint | None = None
    timeout: (
        Annotated[
            int,
            Meta(
                description="The duration for keeping the runtime alive after the completion, in seconds",
                ge=0,
                le=604800,
            ),
        ]
        | None
    ) = None


class ModelRuntimeAffinity(enum.Enum):
    """
    Defines how the runtime is selected for a model execution.
    ANY: Select automatically.
    PRIMARY: The primary runtime will be used.
    EXTERNAL: Use any external runtime with the model loaded
    """

    ANY = "any"
    PRIMARY = "primary"
    EXTERNAL = "external"


class ModelRuntime(Struct, **struct_options):
    """
    Model runtime
    """

    name: RuntimeName | None = None
    affinity: ModelRuntimeAffinity = ModelRuntimeAffinity.ANY
    runner: GRPCEndpoint | None = None
    timeout: (
        Annotated[
            int,
            Meta(
                description="The duration for which the runtime is kept alive after the completion, in seconds",
            ),
        ]
        | None
    ) = None
