from __future__ import annotations

from typing import Annotated, List, Tuple, Set, Dict

import datetime
import enum

from msgspec import Struct, Meta, field

import quite
from quite.runtime import RuntimeModel, ModelState, DeviceState

__pdoc__ = {"CloneObject.update": False, "CloneModel.update": False}


class CloneTimeTaken(Struct, **quite.struct_state_options):
    """
    Time taken for clone download and sync a persistent object
    """

    download: Annotated[
        float, Meta(description="Time taken to download, in seconds")
    ] = 0.0

    sync: Annotated[float, Meta(description="Time taken to sync, in seconds")] = 0.0

    def __post_init__(self):
        if self.download:
            self.download = round(self.download, 4)
        if self.sync:
            self.sync = round(self.sync, 4)


class Warmup(Struct, **quite.struct_state_options):
    state: quite.TaskState | None = None
    reason: str | None = None


class CloneObject(Struct, **quite.struct_state_options):
    """
    Clone repository object state
    """

    id: Annotated[str, Meta(description="Object ID")]
    time: CloneTimeTaken = field(default_factory=CloneTimeTaken)
    timestamp: datetime.datetime | None = None
    last_needed: datetime.datetime | None = None
    timeout: Annotated[int, Meta(description="Persistence timeout")] | None = None
    size: Annotated[float, Meta(description="Object size in MB")] | None = None

    def update(self, co: CloneObject) -> CloneObject:
        self.id = co.id
        if co.time.download:
            self.time.download = co.time.download
        if co.time.sync:
            self.time.sync = co.time.sync
        if co.timestamp is not None:
            self.timestamp = co.timestamp
        if co.last_needed is not None:
            self.last_needed = co.last_needed
        if co.size is not None:
            self.size = co.size
        if co.timeout is not None:
            self.timeout = co.timeout
        return self

    def __post_init__(self):
        if self.size is not None:
            self.size = round(self.size, 6)


class CloneModel(Struct, **quite.struct_state_options):
    """
    Clone repository model state
    """

    name: Annotated[str, Meta(description="Model name (repo@revision)")]
    config: quite.models.CloneModelConfig | None = None
    variants: Set[str] = set()
    objects: Set[str] = set()
    head: quite.ModelHead | None = None
    time: CloneTimeTaken = field(default_factory=CloneTimeTaken)
    timestamp: datetime.datetime | None = None

    def __repr__(self):
        return f"{self.name} ({self.head}) variants: {self.variants}"

    def update(self, model: CloneModel) -> CloneModel:
        self.name = model.name
        if model.time.download:
            self.time.download = model.time.download
        if model.time.sync:
            self.time.sync = model.time.sync
        if model.timestamp is not None:
            self.timestamp = model.timestamp
        if model.head is not None:
            self.head = model.head
        if model.config is not None:
            self.config = model.config
        if model.variants is not None:
            self.variants = model.variants
        if model.objects is not None:
            self.objects = model.objects
        return self


class ClonePhase(enum.Enum):
    PENDING = "pending"
    NODEASSIGNED = "nodeassigned"
    RUNNING = "running"
    TERMINATING = "terminating"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    UNKNOWN = "unknown"


class ClusterStatus(enum.Enum):
    NORMAL = "normal"
    DISABLED = "disabled"
    UNAVAILABLE = "unavailable"
    CORRUPTED = "corrupted"


class ComputeProvider(Struct, **quite.struct_state_options):
    name: str
    settings: dict = {}


class ComputeProviderInstanceState(Struct, **quite.struct_state_options):
    id: str | None = None


class CloneSet(Struct, **quite.struct_state_options):
    """
    CloneSet state
    """

    name: quite.CloneSetName
    node_group: str | None = None
    max_size: Annotated[
        int, Meta(ge=0, le=1000, description="Maximum cloneset size")
    ] = 0
    topics: List[str] = []
    device_name: str = "cuda"
    devices_mask: Annotated[
        List[int], Meta(description="Enabled devices mask (0 - disabled, 1 - enabled))")
    ] = []
    num_runtime_devices: Annotated[
        int, Meta(ge=1, le=8, description="Number of devices per runtime")
    ] = 1
    executor_index: Annotated[int, Meta(description="Executor index")] = 0
    relax_timeout: Annotated[
        int,
        Meta(
            ge=0,
            le=1000,
            description="Time after which cloneset is considered healthy again",
        ),
    ] = 0
    degraded: bool = False
    provider: ComputeProvider | None = None
    models_hint: List[str] = []

    def __post_init__(self) -> None:
        enabled_device_ids = [i for i in self.devices_mask if i > 0]
        if len(enabled_device_ids) % self.num_runtime_devices != 0:
            raise ValueError(
                f"number of enabled devices ({len(enabled_device_ids)}) "
                f"should be a multiple of num_runtime_devices ({self.num_runtime_devices})"
            )


class CloneRuntime(Struct, **quite.struct_state_options):
    """
    Clone runtime information
    """

    name: quite.RuntimeName
    devices: List[str] = []
    provider: ComputeProvider | None = None
    partition: (
        Annotated[int, Meta(ge=0, le=10000, description="Partition number")] | None
    ) = None
    warmup: Warmup | None = None

    def runtime(self) -> quite.runtime.Runtime:
        return quite.runtime.Runtime(
            name=self.name,
            devices=[quite.runtime.Device.from_str(device) for device in self.devices],
            provider=self.provider.name if self.provider else None,
        )


class Clone(Struct, **quite.struct_state_options):
    """
    Clone information
    """

    name: quite.CloneName
    cloneset_name: quite.CloneSetName
    phase: ClonePhase
    phase_start_time: datetime.datetime
    phase_time: Annotated[float, Meta(description="Current phase time, in seconds")] = (
        0.0
    )
    host_ip: Annotated[str, Meta(description="Host IP address")] | None = None
    cluster_ip: Annotated[str, Meta(description="Cluster IP address")] | None = None
    degraded: Annotated[
        bool, Meta(description="Clone takes too long to enter `running` state")
    ] = False
    runtimes: List[CloneRuntime] = []

    def make_runtimes(self, cloneset: CloneSet) -> None:
        self.runtimes = []
        clone_name = self.name.replace(cloneset.name, "").split("-")[-1]
        enabled_device_ids = [i for i, m in enumerate(cloneset.devices_mask) if m > 0]
        num_runtimes = len(enabled_device_ids) // cloneset.num_runtime_devices
        for i in range(num_runtimes):
            runtime_device_ids = enabled_device_ids[
                i * cloneset.num_runtime_devices : (i + 1)
                * cloneset.num_runtime_devices
            ]
            runtime = CloneRuntime(
                name=f"{clone_name}-{i}",
                devices=[
                    f"{cloneset.device_name}:{runtime_device_ids[j]}"
                    for j in range(cloneset.num_runtime_devices)
                ],
                provider=cloneset.provider,
            )
            self.runtimes += [runtime]


class CloneState(Struct, **quite.struct_state_options):
    """
    Clone state
    """

    namespace: Dict[quite.ObjectDomain, str] | str = {}
    name: quite.CloneName | None = None
    models: List[CloneModel] | None = None
    objects: List[CloneObject] | None = None
    timestamp: datetime.datetime | None = None
    ready: bool = True
    timed_out: bool = False
    task_pool_maxsize: Annotated[
        Dict[quite.RuntimeName, int],
        Meta(
            description="Maximum number of concurrent tasks per runtime",
        ),
    ] = {}
    peers: Dict[quite.RuntimeName, Dict[str, int]] = {}
    provider_instance: ComputeProviderInstanceState | None = None
    reason: str | None = None

    def __str__(self):
        return (
            f"name: {self.name}, "
            f"models: {len(self.models)}, "
            f"objects: {len(self.objects)}"
        )

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = quite.utils.utc_now()


class RuntimeState(Struct, **quite.struct_state_options):
    """
    Clone runtime state
    """

    name: quite.RuntimeName | None = None
    clone_name: quite.CloneName | None = None
    device_states: List[DeviceState] = []
    models: List[RuntimeModel] | None = None
    ready: bool = False
    timeout: int | None = None
    last_needed: datetime.datetime | None = None
    timestamp: datetime.datetime | None = None
    reason: str | None = None
    warmup: Warmup | None = None

    def __str__(self):
        return f"name: {self.name}, models: {len(self.models)}"

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = quite.utils.utc_now()


class ClusterState(Struct, **quite.struct_state_options):
    """
    Cluster state
    """

    timestamp: datetime.datetime | None = None
    name: quite.ClusterName | None = None
    runner: quite.GRPCEndpoint | None = None
    status: ClusterStatus = ClusterStatus.NORMAL
    # load_index: Annotated[
    #     int,
    #     Meta(
    #         ge=0,
    #         le=10,
    #         description="An integer from 0 to 10 representing "
    #         "the current utilization of the cluster's capacity",
    #     ),
    # ]
    clonesets: List[CloneSet] = []
    clones: List[Clone] = []
    num_partitions_taken: int = 0
    task_pool_factor: Annotated[
        float, Meta(ge=0.0, le=10.0, description="Runtime task pool factor")
    ] = 1.0
    paused_runtime_names: List[quite.RuntimeName] = []

    def __str__(self):
        return (
            f"name: {self.name}, "
            f"runner: {self.runner}, "
            f"clonesets: {len(self.clonesets)}, "
            f"clones: {len(self.clones)}, "
            f"runtimes: {sum([len(c.runtimes) for c in self.clones])}, "
            f"num_partitions_taken: {self.num_partitions_taken}"
        )

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = quite.utils.utc_now()


class GlobalState(Struct, **quite.struct_state_options):
    """
    Global cluster state
    """

    cluster_state: ClusterState | None = None
    clone_states: List[CloneState] = []
    runtime_states: List[RuntimeState] = []

    def get_clone(self, name: quite.CloneName) -> Clone | None:
        if self.cluster_state is None:
            return None
        return next((c for c in self.cluster_state.clones if c.name == name), None)

    def get_cloneset(self, name: quite.CloneSetName) -> CloneSet | None:
        if self.cluster_state is None:
            return None
        return next(
            (cs for cs in self.cluster_state.clonesets if cs.name == name), None
        )

    def get_clones_num(self, cloneset_name: quite.CloneSetName) -> int:
        if self.cluster_state is None:
            return 0
        return len(
            [c for c in self.cluster_state.clones if c.cloneset_name == cloneset_name]
        )

    def get_clone_runtime(
        self, name: quite.RuntimeName
    ) -> Tuple[Clone, CloneRuntime] | Tuple[None, None]:
        if self.cluster_state is None:
            return None, None
        for clone in self.cluster_state.clones:
            for runtime in clone.runtimes:
                if runtime.name == name:
                    return clone, runtime
        return None, None

    def get_clone_state(self, name: quite.CloneName) -> CloneState | None:
        return next((cs for cs in self.clone_states if cs.name == name), None)

    def get_runtime_state(self, name: quite.RuntimeName) -> RuntimeState | None:
        return next((rs for rs in self.runtime_states if rs.name == name), None)

    def update_clone_state(self, clone_state: CloneState):
        for i, state in enumerate(self.clone_states):
            if state.name == clone_state.name:
                self.clone_states[i] = clone_state
                return
        self.clone_states += [clone_state]

    def update_runtime_state(self, runtime_state: RuntimeState):
        for i, state in enumerate(self.runtime_states):
            if state.name == runtime_state.name:
                self.runtime_states[i] = runtime_state
                return
        self.runtime_states += [runtime_state]

    def known_only(self):
        self.clone_states = [
            cs for cs in self.clone_states if self.get_clone(cs.name) is not None
        ]
        self.runtime_states = [
            rs
            for rs in self.runtime_states
            if self.get_clone_runtime(rs.name)[1] is not None
        ]

    def get_runtime_models(
        self,
        model_id: str,
        runtime_name: quite.RuntimeName | None = None,
        not_paused: bool = False,
    ) -> List[Tuple[quite.RuntimeName, RuntimeModel]]:
        runtime_models = []
        for rs in self.runtime_states:
            if not rs.ready:
                continue
            if runtime_name is not None and rs.name != runtime_name:
                continue
            if not_paused and self.is_runtime_paused(rs.name):
                continue
            for model in rs.models:
                if model.id == model_id and model.state in [
                    ModelState.READY,
                    ModelState.UNKNOWN,
                ]:
                    # if model.id == model_id and (
                    #     model_variant is None or model.variant == model_variant
                    # ):
                    runtime_models.append((rs.name, model))
        if len(runtime_models):
            runtime_models.sort(key=lambda x: x[1].queue_size)
        return runtime_models

    def num_ready_runtimes(self) -> int:
        num = 0
        for clone in self.cluster_state.clones:
            state = self.get_clone_state(clone.name)
            if state is not None and state.ready:
                num += len(clone.runtimes)
        return num

    def is_runtime_paused(self, runtime_name: quite.RuntimeName) -> bool:
        return runtime_name in self.cluster_state.paused_runtime_names
