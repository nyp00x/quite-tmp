from __future__ import annotations

import os
from typing import List, Tuple, Annotated, Dict
import enum
import shlex
import gc
import asyncio
from loguru import logger as _logger

from msgspec import Struct, Meta

import quite

from quite.globals import struct_options
from quite.io import RuntimeName
from quite.utils import get_torch

__pdoc__ = {"RuntimeModelDelim": False, "Runtime.cuda_set_visible_devices": False}


RuntimeModelDelim = "__"


class DeviceType(enum.Enum):
    CPU = "cpu"
    CUDA = "cuda"


QueueName = Annotated[
    str,
    Meta(
        description="Queue name",
        pattern="^[a-zA-Z0-9_-]+$",
        min_length=1,
        max_length=128,
    ),
]


class CUDADeviceProperties(Struct, **struct_options):
    """
    CUDA device properties
    """

    name: str
    total_memory: Annotated[int, Meta(description="Total memory in GB")]
    compute_capability: Tuple[int, int]

    def __repr__(self):
        return f"{self.name} (capability {self.compute_capability[0]}.{self.compute_capability[1]}, {self.total_memory} GB)"


class Device(Struct, **struct_options):
    """
    Runtime device
    """

    type: DeviceType = DeviceType.CPU
    index_real: int = 0
    index: int | None = None
    cuda_device_properties: CUDADeviceProperties | None = None

    def _index(self):
        return self.index if self.index is not None else self.index_real

    def fetch_properties(self):
        try:
            torch = get_torch()
            if self.type == DeviceType.CUDA:
                index = self._index()
                self.cuda_device_properties = CUDADeviceProperties(
                    name=torch.cuda.get_device_name(index),
                    total_memory=round(
                        torch.cuda.get_device_properties(index).total_memory / 1024**3,
                        2,
                    ),
                    compute_capability=torch.cuda.get_device_capability(index),
                )
        except Exception as e:
            _logger.warning(
                f"failed to retrieve device properties for {self.type} device {self.index_real}: {e}"
            )

    def torch_device(self):
        """
        Get torch.device object
        """
        if self.type == DeviceType.CUDA:
            torch = get_torch()
            return torch.device(f"cuda:{self._index()}")
        else:
            return self.type.name

    def __str__(self):
        if self.type == DeviceType.CUDA:
            index = self._index()
            return (
                f"{self.type.name} {self.index_real} ({self.cuda_device_properties})"
                + (f" visible as {index}" if index != self.index_real else "")
            )
        else:
            return f"{self.type.name} {self.index_real}"

    @classmethod
    def from_str(cls, device: str) -> Device:
        """
        Create instance from string
        """
        parts = device.split(":")
        device_type = quite.utils.val2enum(parts[0], DeviceType)
        if len(parts) > 1:
            index = int(parts[1])
        return cls(type=device_type, index_real=index)

    def empty_cache(self):
        """
        Empty device cache
        """
        gc.collect()
        if quite.utils.torch_available:
            if self.type == DeviceType.CUDA:
                torch = get_torch()
                index = self._index()
                with torch.cuda.device(index):
                    torch.cuda.empty_cache()

    def get_gpu_memory(self) -> Tuple[int | None, int | None]:
        """
        Get total and free gpu memory in bytes
        """
        if self.type == DeviceType.CUDA and quite.utils.torch_available:
            torch = get_torch()
            device = self.torch_device()
            stats = torch.cuda.memory_stats(device)
            mem_active = stats["active_bytes.all.current"]
            mem_reserved = stats["reserved_bytes.all.current"]
            mem_free_cuda, mem_total_cuda = torch.cuda.mem_get_info(device)
            mem_free_torch = mem_reserved - mem_active
            mem_free_total = mem_free_cuda + mem_free_torch
            return mem_total_cuda, mem_free_total
        return None, None

    def cuda_cc_ge(self, cc: Tuple[int, int]) -> bool:
        """
        Check if CUDA device compute capability is greater or equal to specified
        """
        if self.type != DeviceType.CUDA:
            return False
        return (
            self.cuda_device_properties is not None
            and self.cuda_device_properties.compute_capability >= cc
        )


class RuntimeRole(enum.Enum):
    GENERAL = "ganeral"
    INFERENCE = "inference"
    TRAINING = "training"


class Runtime(Struct, **struct_options):
    """
    Runtime information
    """

    name: RuntimeName
    devices: List[Device] = []
    role: RuntimeRole = RuntimeRole.INFERENCE
    provider: str | None = None

    def __repr__(self):
        s = f"{self.name} {self.role.name} ({', '.join(map(str, self.devices))})"
        if quite.utils.torch_available:
            torch = get_torch()
            s += f" torch: {torch.__version__} CUDA: {torch.version.cuda}"
        return s

    def cuda_set_visible_devices(self):
        """
        Set CUDA_VISIBLE_DEVICES variable
        """
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            if len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")) != len(self.devices):
                _logger.warning(
                    "CUDA_VISIBLE_DEVICES is already set and does not match the number of devices"
                )
            for i, device in enumerate(self.devices):
                if device.type == DeviceType.CUDA:
                    device.index = i
        cvd = ""
        for i, device in enumerate(self.devices):
            if device.type == DeviceType.CUDA:
                device.index = i
                cvd += f"{device.index_real}"
                if i < len(self.devices) - 1:
                    cvd += ","
        if cvd:
            os.environ["CUDA_VISIBLE_DEVICES"] = cvd
            _logger.info(f"set CUDA_VISIBLE_DEVICES to {cvd}")

    def fetch_properties(self):
        """
        Fetch device properties for all devices
        """
        for device in self.devices:
            device.fetch_properties()

    def cuda_cc_ge(self, cc: Tuple[int, int]) -> bool:
        """
        Check if all CUDA devices compute capability is greater or equal to specified
        """
        cuda_devices = [d for d in self.devices if d.type == DeviceType.CUDA]
        if not len(cuda_devices):
            return False
        return all(d.cuda_cc_ge(cc) for d in cuda_devices)

    def optimize(self):
        """
        Apply runtime optimizations
        """
        if quite.utils.torch_available:
            torch = get_torch()
            if self.cuda_cc_ge((8, 0)):
                torch.backends.cuda.matmul.allow_tf32 = True

            torch.backends.cudnn.benchmark = True

    def empty_cache(self):
        """
        Empty device caches
        """
        gc.collect()
        for device in self.devices:
            device.empty_cache()

    def log_vram(self, logger=None, debug=False):
        """
        Log VRAM usage
        """
        if logger is None:
            logger = _logger

        for i, device in enumerate(self.devices):
            mem_total, mem_free = device.get_gpu_memory()
            if mem_total is not None:
                f = logger.debug if debug else logger.info
                f(
                    f"device {i} VRAM: {mem_free / 1024**3:.2f}/{mem_total / 1024**3:.2f} GB"
                )


class ModelState(enum.Enum):
    UNKNOWN = "unknown"
    READY = "ready"
    UNAVAILABLE = "unavailable"
    LOADING = "loading"
    UNLOADING = "unloading"


class RuntimeModelTime(Struct, **quite.struct_state_options):
    """
    Time taken to load and unload model
    """

    load: Annotated[float, Meta(description="Time taken to load model, in seconds")] = (
        0.0
    )
    unload: Annotated[
        float, Meta(description="Time taken to unload model, in seconds")
    ] = 0.0
    queue_wait: Annotated[
        float, Meta(description="Last time taken to wait for the execution, in seconds")
    ] = 0.0
    queue_wait_max: Annotated[
        float,
        Meta(description="Maximum time taken to wait for the execution, in seconds"),
    ] = 0.0

    def __post_init__(self):
        if self.load:
            self.load = round(self.load, 4)
        if self.unload:
            self.unload = round(self.unload, 4)
        if self.queue_wait:
            self.queue_wait = round(self.queue_wait, 4)
        if self.queue_wait_max:
            self.queue_wait_max = round(self.queue_wait_max, 4)
        if self.queue_wait > self.queue_wait_max:
            self.queue_wait_max = self.queue_wait

    def update(self, model_time: RuntimeModelTime):
        if model_time.load:
            self.load = model_time.load
        if model_time.unload:
            self.unload = model_time.unload
        self.queue_wait = model_time.queue_wait
        if model_time.queue_wait_max > self.queue_wait_max:
            self.queue_wait_max = model_time.queue_wait_max


class RuntimeModelDeviceMetrics(Struct, **quite.struct_state_options):
    """
    Runtime device metrics
    """

    mem_used: Annotated[int, Meta(description="Memory used, in GB")] = 0
    mem_used_max: Annotated[int, Meta(description="Maximum memory used, in GB")] = 0


class RuntimeModel(Struct, **quite.struct_state_options):
    """
    Runtime model state
    """

    id: Annotated[str, Meta(description="Model ID")]
    queue_size: int | None = None
    time: RuntimeModelTime | None = None
    device_metrics: List[RuntimeModelDeviceMetrics] = []
    state: ModelState | None = None
    reason: str | None = None

    def update(self, model: RuntimeModel):
        self.id = model.id
        if model.time is not None:
            self.time.update(model.time)
        if model.state is not None:
            self.state = model.state
        if model.reason is not None:
            self.reason = model.reason
        if model.queue_size is not None:
            self.queue_size = model.queue_size


def to_float(value):
    try:
        number = float(value)
    except ValueError:
        number = -1.0
    return number


def to_int(value):
    try:
        number = int(value)
    except ValueError:
        number = -1
    return number


def to_gb(value):
    if value < 0:
        return -1
    return round(to_float(value) / 1024, 3)


class DeviceState(Struct, **quite.struct_state_options):
    """
    NVIDIA device state
    """

    NVIDIA_SMI_GET_GPUS = "nvidia-smi --query-gpu=index,uuid,utilization.gpu,memory.total,memory.used,memory.free,driver_version,name,gpu_serial,display_active,display_mode,temperature.gpu --format=csv,noheader,nounits"

    id: Annotated[int, Meta(description="Device ID")] = 0
    uuid: Annotated[str, Meta(description="Device UUID")] = ""
    gpu_util: Annotated[float, Meta(description="GPU utilization, in %")] = 0.0
    mem_total: Annotated[float, Meta(description="Total memory, in GB")] = 0.0
    mem_used: Annotated[float, Meta(description="Used memory, in GB")] = 0.0
    driver: Annotated[str, Meta(description="Driver version")] = ""
    gpu_name: Annotated[str, Meta(description="GPU name")] = ""
    temperature: Annotated[float, Meta(description="Temperature, in C")] = 0.0

    def __str__(self):
        return f"{self.id} {self.gpu_name} {self.gpu_util:5.1f}% {self.mem_used:7.1f}/{self.mem_total:7.1f}GB"

    @classmethod
    async def get_all(cls) -> List[DeviceState | None]:
        try:
            process = await asyncio.create_subprocess_exec(
                *shlex.split(cls.NVIDIA_SMI_GET_GPUS),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await process.communicate()

            if stderr:
                print(f"error in executing command: {stderr.decode().strip()}")
                return []

            lines = stdout.decode("utf-8").split(os.linesep)

            device_states = []

            for line in lines:
                try:
                    line = line.strip()
                    if not line:
                        continue
                    values = line.split(", ")
                    device_state = DeviceState(
                        id=to_int(values[0]),
                        uuid=values[1],
                        gpu_util=to_float(values[2]),
                        mem_total=to_gb(to_float(values[3])),
                        mem_used=to_gb(to_float(values[4])),
                        driver=values[6],
                        gpu_name=values[7],
                        temperature=to_float(values[11]),
                    )
                    device_states.append(device_state)

                except Exception as e:
                    print(f"failed to parse device state: {e}")
                    device_states.append(None)

            return device_states

        except Exception as e:
            print(f"failed to get device states: {e}")
            return []


class DeviceProcess(Struct, **quite.struct_state_options):
    """
    NVIDIA device process
    """

    NVIDIA_SMI_GET_PROCS = "nvidia-smi --query-compute-apps=pid,process_name,gpu_uuid,gpu_name,used_memory --format=csv,noheader,nounits"

    # process_name: Annotated[str, Meta(description="Process name")]
    gpu_uuids: Annotated[List[str], Meta(description="GPU UUIDs")] = []
    mem_used: Annotated[List[float], Meta(description="Used memory, in GB")] = []
    mem_used_max: Annotated[
        List[float], Meta(description="Maximum used memory, in GB")
    ] = []
    command: Annotated[str, Meta(description="Process command")] = ""

    def update(self, process: DeviceProcess):
        if len(process.gpu_uuids) != len(self.gpu_uuids):
            self.gpu_uuids = process.gpu_uuids
            self.mem_used = process.mem_used
            self.mem_used_max = (
                process.mem_used_max
                if len(process.mem_used_max) == len(process.mem_used)
                else self.mem_used
            )
            self.command = process.command
            return

        self.mem_used = process.mem_used
        if len(process.mem_used_max) == len(process.mem_used):
            self.mem_used_max = process.mem_used_max
            return

        if len(self.mem_used) != len(self.mem_used_max):
            self.mem_used_max = self.mem_used
            return

        for i, mem_used in enumerate(process.mem_used):
            if mem_used > self.mem_used_max[i]:
                self.mem_used_max[i] = mem_used

    @classmethod
    async def get_all(
        cls,
    ) -> Tuple[Dict[int, DeviceProcess], List[DeviceState | None]]:
        try:
            process = await asyncio.create_subprocess_exec(
                *shlex.split(cls.NVIDIA_SMI_GET_PROCS),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await process.communicate()

            if stderr:
                print(f"error in executing command: {stderr.decode().strip()}")
                return {}, []

            lines = stdout.decode("utf-8").split(os.linesep)

            device_states = await DeviceState.get_all()
            if not len(device_states):
                raise Exception("no device states")

            device_processes = {}

            for line in lines:
                try:
                    line = line.strip()
                    if not line:
                        continue

                    values = line.split(", ")

                    pid = to_int(values[0])
                    command = await cls.get_process_command(pid)

                    if pid not in device_processes:
                        device_processes[pid] = DeviceProcess(
                            command=command,
                        )
                    device_processes[pid].gpu_uuids += [values[2]]
                    device_processes[pid].mem_used += [to_gb(to_float(values[4]))]

                except Exception as e:
                    print(f"failed to parse device process: {e}")

            return device_processes, device_states

        except Exception as e:
            print(f"failed to get device processes: {e}")
            return {}, []

    @staticmethod
    async def get_process_command(pid: int) -> str | None:
        try:
            process = await asyncio.create_subprocess_exec(
                "ps",
                "-p",
                str(pid),
                "-o",
                "cmd",
                "--no-headers",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await process.communicate()

            if stderr:
                print(
                    f"error fetching command for PID {pid}: {stderr.decode().strip()}"
                )
                return ""

            command = stdout.decode().strip()
            return command or ""

        except Exception as e:
            print(f"failed to get command for PID {pid}: {e}")
            return ""

    @staticmethod
    async def get_subprocesses(pid: int) -> List[int]:
        try:
            process = await asyncio.create_subprocess_exec(
                "ps",
                "--ppid",
                str(pid),
                "-o",
                "pid",
                "--no-headers",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await process.communicate()

            if stderr:
                print(
                    f"error fetching subprocesses for PID {pid}: {stderr.decode().strip()}"
                )
                return []

            subprocess_pids = [
                int(p.strip())
                for p in stdout.decode().splitlines()
                if p.strip().isdigit()
            ]
            return subprocess_pids

        except Exception as e:
            print(f"failed to get subprocesses for PID {pid}: {e}")
            return []
