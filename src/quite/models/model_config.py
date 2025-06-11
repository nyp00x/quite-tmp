from typing import Annotated, Dict, List, Set, Any

from pathlib import Path

from msgspec import Struct, Meta

import quite


ModelVariant = Annotated[
    Dict[quite.NAME, quite.MODEL_OBJECT_UID | List[quite.MODEL_OBJECT_UID]],
    Meta(description="Model variant"),
]

ModelVariantPresets = Annotated[
    Dict[quite.NAME, ModelVariant],
    Meta(description="Model variant presets"),
]

ModelVariantDirectories = Annotated[
    Dict[quite.NAME, Path | List[Path]],
    Meta(description="Variant directories"),
]


class CloneModelRequiredModel(Struct, **quite.struct_options):
    """
    Required model description
    """

    id: quite.UID
    load: Annotated[
        bool, Meta(description="Model must be loaded prior to execution")
    ] = False


class CloneModelRequirements(Struct, **quite.struct_options):
    """
    Model requirements
    """

    models: List[CloneModelRequiredModel] = []
    objects: List[quite.UID] = []
    packages: List[str] = []


class CloneModelConfig(Struct, **quite.struct_options):
    """
    Model configuration
    """

    variants: ModelVariantPresets = {}
    requirements: CloneModelRequirements | None = None
    device: quite.runtime.DeviceType = quite.runtime.DeviceType.CUDA
    tags: Set[str] = set()
    anti_tags: Set[str] = set()
    queue_name: str | None = None
    custom: Dict[str, Any] = {}

    def __str__(self):
        return (
            f"variants: {tuple(self.variants.keys())}, "
            f"tags: {self.tags}, "
            f"anti_tags: {self.anti_tags}, "
            f"queue_name: {self.queue_name}, "
            f"device: {self.device}"
        )

    def __post_init__(self):
        if self.requirements is None:
            self.requirements = CloneModelRequirements()

        for varname in self.variants.keys():
            if quite.runtime.RuntimeModelDelim in varname:
                raise ValueError(
                    f"Variant name '{varname}' "
                    f"contains reserved delimiter '{quite.runtime.RuntimeModelDelim}'"
                )
