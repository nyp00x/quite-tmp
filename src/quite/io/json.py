from __future__ import annotations

from typing import Annotated, Any, List, Dict, Generic, TypeVar
import json

import msgspec
from msgspec import Meta
from pathlib import Path


from .common import struct_options
from .data import (
    JSONFormat,
    JSONTarget,
    DataType,
    JSONInput,
    JSONOutput,
    data_types_map,
)

__pdoc__ = {
    "JSON.get_file_name": False,
    "JSON.pick_file_name": False,
    "JSON.get_file_from_object_dir": False,
}

T = TypeVar("T", bound=msgspec.Struct)


class JSON(DataType, Generic[T], **struct_options, tag=True):
    """
    Represents JSON-serializable data
    """

    object: (
        Annotated[Any | T, Meta(description="JSON-serializable dictionary")] | None
    ) = None
    file: Annotated[bytes, Meta(description="JSON file (based on `format`)")] | None = (
        None
    )
    format: Annotated[JSONFormat, Meta(description="Serialized json format")] = (
        JSONFormat.JSON
    )
    target: Annotated[JSONTarget, Meta(description="Target representation")] = (
        JSONTarget.OBJECT
    )
    metadata: Annotated[Dict[str, Any], Meta(description="Metadata")] | None = None
    batch: Annotated[List[JSON], Meta(description="Batch data")] = []
    input: Annotated[JSONInput, Meta(description="Input properties")] | None = None
    output: Annotated[JSONOutput, Meta(description="Output properties")] | None = None

    def __post_init__(self):
        super().__post_init__()

    def is_empty(self) -> bool:
        """
        Check if JSON is empty
        """
        empty = self.object is None and self.file is None
        if self.is_batch():
            return empty and all(item.is_empty() for item in self.batch)
        else:
            return empty

    def empty(self):
        """
        Empty JSON
        """
        self.object = None
        self.file = None
        self.batch = []
        return self

    def to(
        self, target: JSONTarget | None = None, force: bool = True, remove: bool = False
    ) -> JSON:
        """
        Convert JSON to target representation
        """
        if self.is_batch():
            for item in self.batch:
                if item.is_batch():
                    item.batch = []
                item.to(target=target, force=force, remove=remove)
            return self

        if target == JSONTarget.EMPTY:
            return self.empty()

        if target == JSONTarget.OBJECT:
            if self.object is not None and not force:
                return self
            elif self.file is not None:
                return self.file_to_object(remove=remove)
            else:
                raise ValueError(f"{self}: cannot convert to json")
        elif target == JSONTarget.FILE:
            if self.file is not None and not force:
                return self
            elif self.object is not None:
                return self.object_to_file(remove=remove)
            else:
                raise ValueError(f"{self}: cannot convert to file")
        else:
            raise ValueError(f"unknown target {target}")

    @classmethod
    def default_format(cls) -> JSONFormat:
        """
        Default JSON format
        """
        return JSONFormat.JSON

    @classmethod
    def pick_file_name(cls, file_names: List[str]) -> str | None:
        for fname in file_names:
            if not fname.startswith("json"):
                continue
            for fmt in JSONFormat:
                if fname.endswith(fmt.value):
                    return fname
        return None

    def get_file_name(self) -> str:
        return str(f"json.{self.format.value}")

    def get_file_from_object_dir(self, object_dir: Path) -> bytes:
        file_names = [
            file_path.name for file_path in object_dir.iterdir() if file_path.is_file()
        ]
        file_name = next(
            (fname for fname in file_names if fname.endswith(self.format.value)), None
        ) or self.pick_file_name(file_names)
        if file_name is None:
            raise ValueError("cannot find file in object directory")
        with open(object_dir / file_name, "r") as f:
            val = json.load(f)
        return msgspec.json.encode(val)

    def file_to_object(self, force: bool = True, remove: bool = False) -> JSON:
        """
        Convert JSON file to object
        """
        if self.is_batch():
            for item in self.batch:
                if item.is_batch():
                    item.batch = []
                item.file_to_object(force=force, remove=remove)
            return self

        if self.object is not None and not force:
            if remove:
                self.file = None
            return self
        if self.file is not None:
            self.object = msgspec.json.decode(self.file)
            if remove:
                self.file = None
            return self
        else:
            raise ValueError(
                f"cannot convert {self.__class__.__name__} `file` to `object`"
            )

    def object_to_file(
        self, format: JSONFormat | None = None, force: bool = True, remove: bool = False
    ) -> JSON:
        """
        Convert JSON object to file
        """
        if self.is_batch():
            for item in self.batch:
                if item.is_batch():
                    item.batch = []
                item.object_to_file(format=format, force=force, remove=remove)
            return self

        if self.file is not None and not force:
            if remove:
                self.object = None
            return self
        if self.object is not None:
            self.file = msgspec.json.encode(msgspec.to_builtins(self.object))
            if remove:
                self.object = None
            return self
        else:
            raise ValueError(
                f"cannot convert {self.__class__.__name__} `object` to `file`"
            )

    def serializable(self, convert: bool = True) -> JSON:
        """
        Convert JSON to serializable format
        """
        if self.is_batch():
            for item in self.batch:
                if item.is_batch():
                    item.batch = []
                item.serializable(convert=convert)

        if self.object is not None:
            self.object = msgspec.to_builtins(self.object)

        return self

    # def _apply(self, output: JSONOutput) -> JSON:
    #     if self.is_batch():
    #         for item in self.batch:
    #             if item.is_batch():
    #                 item.batch = []
    #             item._apply(output)
    #         return self

    #     if output.format is not None:
    #         self.format = output.format

    #     targets = output.target if isinstance(output.target, list) else [output.target]
    #     for target in targets:
    #         self.to(target, force=False, remove=True)

    #     return self

    async def apply(
        self,
        output: JSONOutput,
        repository: Any | None = None,
        log_prefix: str | None = None,
    ) -> JSON:
        """
        Apply output properties
        """
        if not output.clone_object and self.is_batch():
            for item in self.batch:
                if item.is_batch():
                    item.batch = []
                await item.apply(output, repository=repository, log_prefix=log_prefix)
            return self
        if not isinstance(output, JSONOutput):
            raise ValueError(
                f"cannot apply {type(output).__name__} to {self.__class__}"
            )

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
    ) -> JSON:
        """
        Convert JSON to target representation defined in `target` property
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
            raise ValueError(f"cannot resolve {self.__class__} target is not specified")

        if self.url is not None or self.from_id is not None:
            need_download = (
                self.target == JSONTarget.OBJECT and (self.object is None or force)
            ) or (self.target == JSONTarget.FILE and (self.file is None or force))
            if need_download:
                await self.download(
                    repository=repository, force=force, log_prefix=log_prefix
                )

        if self.target == JSONTarget.URL:
            self.to(JSONTarget.FILE, force=force, remove=remove)
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


data_types_map["JSON"] = JSON
