from __future__ import annotations

from typing import Annotated, Any, List, Tuple
import io

from msgspec import Meta
from pathlib import Path

import numpy as np


from .common import struct_options
from .data import DataType
from .data import (
    NDType,
    NDArrayFormat,
    NDArrayTarget,
    NDArrayInput,
    NDArrayOutput,
    data_types_map,
)
from quite.utils import val2enum, get_torch


__pdoc__ = {
    "NDArray.get_file_from_object_dir": False,
    "NDArray.pick_file_name": False,
}


class NDArray(DataType, **struct_options, tag=True):
    """
    Represents generic array data
    """

    array: Annotated[np.ndarray, Meta(description="Numpy array")] | None = None
    list: Annotated[List[Any], Meta(description="List of values")] | None = None  # noqa
    tensor: Annotated[Any, Meta(description="Torch tensor")] | None = None
    shape: Annotated[Tuple[int, ...], Meta(description="Shape")] | None = None
    dtype: Annotated[NDType, Meta(description="Data type")] = NDType.UINT8
    format: Annotated[NDArrayFormat, Meta(description="Serialized array format")] = (
        NDArrayFormat.NPZ
    )
    target: Annotated[NDArrayTarget, Meta(description="Target representation")] = (
        NDArrayTarget.ARRAY
    )
    batch: Annotated[List[NDArray], Meta(description="Batch data")] = []
    input: Annotated[NDArrayInput, Meta(description="Input properties")] | None = None
    output: Annotated[NDArrayOutput, Meta(description="Output properties")] | None = (
        None
    )

    def __post_init__(self):
        super().__post_init__()

    def __str__(self):
        s = super().__str__()
        if not self.is_batch():
            if self.array is not None:
                s += f" {self.array.shape} {self.array.dtype}"
            if self.tensor is not None:
                s += f" {self.tensor.shape} {self.tensor.dtype}"
            if self.url is not None:
                # urlstr = self.url if len(self.url) < 16 else f"...{self.url[-16:]}"
                s += f" {self.url}"
            if self.file is not None:
                s += f" {len(self.file) / 1024 / 1024:.3f}MB"
        return s

    def is_empty(self) -> bool:
        """
        Check if array is empty
        """
        empty = (
            self.array is None
            and self.tensor is None
            and self.file is None
            and self.list is None
        )
        if self.is_batch():
            return empty and all(item.is_empty() for item in self.batch)
        else:
            return empty

    def empty(self):
        """
        Empty array
        """
        self.array = None
        self.tensor = None
        self.file = None
        self.list = None
        self.batch = []
        return self

    def to(
        self,
        target: NDArrayTarget | None = None,
        force: bool = True,
        remove: bool = False,
    ):
        """
        Convert array to target representation
        """
        if self.is_batch():
            for item in self.batch:
                if item.is_batch():
                    item.batch = []
                item.to(target, force=force, remove=remove)
            return self

        if target == NDArrayTarget.EMPTY:
            return self.empty()

        if target == NDArrayTarget.ARRAY:
            if self.array is not None and not force:
                return self
            elif self.tensor is not None:
                return self.tensor_to_array(remove=remove)
            elif self.file is not None:
                return self.file_to_array(remove=remove)
            elif self.list is not None:
                return self.list_to_array(remove=remove)
            else:
                raise ValueError(f"{self}: cannot convert to array")
        elif target == NDArrayTarget.FILE:
            if self.file is not None and not force:
                return self
            elif self.array is not None:
                return self.array_to_file(remove=remove)
            elif self.tensor is not None:
                return self.tensor_to_array(remove=remove).array_to_file(remove=remove)
            elif self.list is not None:
                return self.list_to_array(remove=remove).array_to_file(remove=remove)
            else:
                raise ValueError(f"{self}: cannot convert to file")
        elif target == NDArrayTarget.TENSOR:
            if self.tensor is not None and not force:
                return self
            elif self.array is not None:
                return self.array_to_tensor(remove=remove)
            elif self.file is not None:
                return self.file_to_array(remove=remove).array_to_tensor(remove=remove)
            elif self.list is not None:
                return self.list_to_array(remove=remove).array_to_tensor(remove=remove)
            else:
                raise ValueError(f"{self}: cannot convert to tensor")
        elif target == NDArrayTarget.LIST:
            if self.list is not None and not force:
                return self
            elif self.array is not None:
                return self.array_to_list(remove=remove)
            elif self.file is not None:
                return self.file_to_array(remove=remove).array_to_list(remove=remove)
            elif self.tensor is not None:
                return self.tensor_to_array(remove=remove).array_to_list(remove=remove)
            else:
                raise ValueError(f"{self}: cannot convert to list")
        else:
            raise ValueError(f"unknown target {target}")

    @classmethod
    def default_format(cls) -> NDArrayFormat:
        """
        Default serialization format
        """
        return NDArrayFormat.NPZ

    @classmethod
    def pick_file_name(cls, file_names: List[str]):
        for fname in file_names:
            if not fname.startswith("array"):
                continue
            for fmt in NDArrayFormat:
                if fname.endswith(fmt.value):
                    return fname
        return None

    def get_file_name(self) -> str:
        return str(f"array.{self.format.value}")

    def get_file_from_object_dir(self, object_dir: Path) -> bytes:
        file_names = [
            file_path.name for file_path in object_dir.iterdir() if file_path.is_file()
        ]
        if not len(file_names):
            raise ValueError(f"empty object directory {object_dir}")
        file_name = next(
            (fname for fname in file_names if fname.endswith(self.format.value)), None
        ) or self.pick_file_name(file_names)
        if file_name is None:
            raise ValueError(f"cannot find file in object directory {object_dir}")
        return (object_dir / file_name).read_bytes()

    def array_to_tensor(
        self, dtype: NDType | None = None, force: bool = True, remove: bool = False
    ) -> NDArray:
        """
        Convert array to tensor
        """
        if self.is_batch():
            for item in self.batch:
                if item.is_batch():
                    item.batch = []
                item.array_to_tensor(dtype=dtype, force=force, remove=remove)
            return self

        if self.tensor is not None and not force:
            if remove:
                self.array = None
            return self
        if self.array is not None:
            self.array_as(dtype=dtype)
            self.tensor = get_torch().from_numpy(self.array)
            if remove:
                self.array = None
            return self
        else:
            raise ValueError(
                f"cannot convert {self.__class__.__name__} `array` to `tensor`"
            )

    def tensor_to_array(
        self, dtype: NDType | None = None, force: bool = True, remove: bool = False
    ) -> NDArray:
        """
        Convert tensor to array
        """
        if self.is_batch():
            for item in self.batch:
                if item.is_batch():
                    item.batch = []
                item.tensor_to_array(dtype=dtype, force=force, remove=remove)
            return self

        if self.array is not None and not force:
            if remove:
                self.tensor = None
            return self.array_as(dtype=dtype)
        if self.tensor is not None:
            self.array = self.tensor.detach().cpu().numpy()
            self.shape = self.array.shape
            if remove:
                self.tensor = None
            if dtype is None:
                dtype = val2enum(str(self.array.dtype), NDType, NDType.UINT8)
            return self.array_as(dtype=dtype)
        else:
            raise ValueError(
                f"cannot convert {self.__class__.__name__} `tensor` to `array`"
            )

    def list_to_array(
        self,
        dtype: NDType | None = None,
        shape: Tuple[int, ...] | None = None,
        force: bool = True,
        remove: bool = False,
    ) -> NDArray:
        """
        Convert list to array
        """
        if self.is_batch():
            for item in self.batch:
                if item.is_batch():
                    item.batch = []
                item.list_to_array(dtype=dtype, shape=shape, force=force, remove=remove)
            return self

        if self.array is not None and not force:
            if remove:
                self.list = None
            return self.array_as(dtype=dtype)
        if self.list is not None:
            self.array = np.array(self.list)
            if dtype is not None:
                self.array_as(dtype=dtype)
            if shape is not None:
                self.array = self.array.reshape(shape)
                self.shape = self.array.shape
            self.dtype = val2enum(str(self.array.dtype), NDType, NDType.UINT8)
            if remove:
                self.list = None
            return self
        else:
            raise ValueError(
                f"cannot convert {self.__class__.__name__} `list` to `array`"
            )

    def array_as(
        self, dtype: Any | None = None, shape: Tuple[int, ...] | None = None
    ) -> NDArray:
        """
        Convert array to specified data type and shape
        """
        if self.is_batch():
            for item in self.batch:
                if item.is_batch():
                    item.batch = []
                item.array_as(dtype=dtype, shape=shape)
            return self

        if self.array is not None:
            if dtype is not None and self.array.dtype != np.dtype(dtype.value):
                self.array = self.array.astype(dtype.value)
                self.dtype = dtype
            if shape is not None and self.array.shape != shape:
                self.array = self.array.reshape(shape)
                self.shape = self.array.shape
            return self
        else:
            raise ValueError(f"{self.__class__.__name__} array_as requires `array`")

    def file_to_array(
        self,
        dtype: NDType | None = None,
        force: bool = True,
        remove: bool = False,
    ) -> NDArray:
        """
        Convert file to array
        """
        if self.is_batch():
            for item in self.batch:
                if item.is_batch():
                    item.batch = []
                item.file_to_array(dtype=dtype, force=force, remove=remove)
            return self

        if self.array is not None and not force:
            if remove:
                self.file = None
            return self.array_as(dtype=dtype)
        if self.file is not None:
            with io.BytesIO(self.file) as f:
                with np.load(f) as data:
                    for key in data.keys():
                        self.array = data[key]
                        break
            if isinstance(self.array, np.ndarray):
                self.shape = self.array.shape
                if dtype is not None:
                    self.array = self.array.astype(dtype.value)
                    self.dtype = dtype
            if remove:
                self.file = None
            return self
        else:
            raise ValueError(
                f"cannot convert {self.__class__.__name__} `file` to `array`"
            )

    def array_to_file(
        self,
        format: NDArrayFormat | None = None,
        force: bool = True,
        remove: bool = False,
    ) -> NDArray:
        """
        Convert array to file
        """
        if self.is_batch():
            for item in self.batch:
                if item.is_batch():
                    item.batch = []
                item.array_to_file(format=format, force=force, remove=remove)
            return self

        if self.file is not None and not force:
            if remove:
                self.array = None
            return self
        if self.array is not None:
            with io.BytesIO() as b:
                if isinstance(self.array, dict):
                    np.savez(b, **self.array)
                else:
                    np.savez(b, data=self.array)
                self.file = b.getvalue()
            if remove:
                self.array = None
            return self
        else:
            raise ValueError(
                f"cannot convert {self.__class__.__name__} `array` to `file`"
            )

    def array_to_list(
        self, force: bool = True, max_size: int = 10000, remove: bool = False
    ) -> NDArray:
        """
        Convert array to list
        """
        if self.is_batch():
            for item in self.batch:
                if item.is_batch():
                    item.batch = []
                item.array_to_list(force=force, max_size=max_size, remove=remove)
            return self

        if self.list is not None and not force:
            if remove:
                self.array = None
            return self
        if self.array is not None:
            if self.array.size > max_size:
                raise ValueError(
                    f"array size ({self.array.size}) is greater than max_size ({max_size}) for array to list conversion"
                )
            self.list = self.array.tolist()
            if remove:
                self.array = None
            return self
        else:
            raise ValueError(
                f"cannot convert {self.__class__.__name__} `array` to `list`"
            )

    def serializable(self, convert: bool = True) -> NDArray:
        """
        Make array serializable. If `convert` is True, convert tensor to array, else remove tensor
        """
        if self.is_batch():
            for item in self.batch:
                if item.is_batch():
                    item.batch = []
                item.serializable(convert=convert)

        if self.tensor is not None:
            if convert:
                self.tensor_to_array()
            else:
                self.tensor = None

        return self

    async def apply(
        self,
        output: NDArrayOutput,
        repository: Any | None = None,
        log_prefix: str | None = None,
    ) -> NDArray:
        """
        Apply output properties
        """
        if not output.clone_object and self.is_batch():
            for item in self.batch:
                if item.is_batch():
                    item.batch = []
                await item.apply(output, repository=repository, log_prefix=log_prefix)
            return self

        if self.is_empty():
            return self

        if output.format is not None:
            self.format = output.format

        self.set_output(output)

        targets = output.target if isinstance(output.target, list) else [output.target]
        for target in targets:
            self.target = target
            await self.resolve(
                force=False,
                remove=True,
                repository=repository,
                log_prefix=log_prefix,
            )
        return self

    async def resolve(
        self,
        force: bool = True,
        remove: bool = False,
        repository: Any | None = None,
        log_prefix: str | None = None,
    ) -> NDArray:
        """
        Convert array to target representation defined in `target` property
        """
        if (self.output is None or not self.output.clone_object) and self.is_batch():
            for item in self.batch:
                if item.is_batch():
                    item.batch = []
                await item.resolve(
                    force=force,
                    remove=remove,
                    repository=repository,
                    log_prefix=log_prefix,
                )
            return self

        if self.target is None:
            raise ValueError(
                f"cannot resolve {self.__class__.__name__} target is not specified"
            )

        if self.url is not None or self.from_id is not None:
            need_download = (
                (self.target == NDArrayTarget.ARRAY and (self.array is None or force))
                or (self.target == NDArrayTarget.FILE and (self.file is None or force))
                or (
                    self.target == NDArrayTarget.TENSOR
                    and (self.tensor is None or force)
                )
            )
            if need_download:
                await self.download(
                    repository=repository, force=force, log_prefix=log_prefix
                )

        if self.target == NDArrayTarget.URL:
            self.to(NDArrayTarget.FILE, force=force, remove=remove)
            object_folder = (
                self.output is not None and self.output.object_folder or None
            )
            as_clone_object = self.output is not None and self.output.clone_object
            await self.upload(
                repository=repository,
                remove=remove,
                object_folder=object_folder,
                clone_object=as_clone_object,
                log_prefix=log_prefix,
            )
        else:
            self.to(self.target, force=force, remove=remove)

        return self


data_types_map["NDArray"] = NDArray
