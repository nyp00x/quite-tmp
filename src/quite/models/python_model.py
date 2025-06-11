from typing import List, Tuple, AsyncGenerator
from pathlib import Path

import msgspec
from loguru import logger

import quite

from .model_config import CloneModelConfig, ModelVariantDirectories


def iteration(iter_type):
    def decorator(func):
        func.iter_type = iter_type  
        async def wrapper(self, *args, **kwargs):
            return await func(self, *args, **kwargs)
        return wrapper
    return decorator


class PythonModel:
    """
    Base class for Python models
    """

    def __init__(
        self,
        runtime: quite.runtime.Runtime,
        model_config: CloneModelConfig,
        variant: quite.NAME | None = None,
        repo_dir: Path | None = None,
        object_dir: Path | None = None,
        variant_object_dirs: ModelVariantDirectories = {},
    ):
        """
        Initialize the model

        Args:
            runtime: The runtime information
            model_config: The model configuration
            variant: The model variant name
            repo_dir: Repository objects directory
            object_dir: The model's object directory
            variant_object_dirs: The variant object directories
        """
        self.runtime = runtime
        self.model_config = model_config
        self.variant = variant
        self.repo_dir = repo_dir
        self.object_dir = object_dir
        self.variant_object_dirs = variant_object_dirs
        self.logger = logger
        self._runner = None
        self._request = None
        self.runtime.empty_cache()

        self.iter_methods = {}

        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if callable(attr) and hasattr(attr, "iter_type"):
                self.iter_methods[attr.iter_type] = attr

    def load(self):
        """
        Load the model if necessary
        """
        self.runtime.empty_cache()

    def set_model(self, model: quite.Model):
        """
        Set the model to the requested state.  
        Can be overridden

        Args:
            model: The model request
        """
        self.model = model

    def set_batch(self, batch: quite.Batch):
        """
        Prepare the model for the batch.  
        Can be overridden

        Args:
            batch: The input batch parameters
        """
        self.batch = batch

    async def execute(
        self,
    ) -> AsyncGenerator[quite.Output | msgspec.Struct, Tuple[quite.Iter, quite.Output | None]]:
        """
        Iterate over the batch.
        A generator instance is created for every iteration.  
        Must be implemented

        Yields:
            The minibatch output: `quite.Output` or shallow `msgspec.Struct` that can be sconverted to quite.Output

        Receives:
            If batch is sequential:  
            - The current iteration parameters  
            - The minibatch output of the previous iteration  
            otherwise (default):  
            - The current iteration parameters  
            - The previous minibatch output of the same iteration  

        Example:
        ```python
        async def execute(self) -> AsyncGenerator[quite.Output, Tuple[quite.Iter, quite.Output | None]]:
            real_mb_size = 2
            indices = np.arange(self.batch.size)
            for mb in np.array_split(indices, np.ceil(self.batch.size / real_mb_size)):
                yield {
                    quite.NDArray(list=list(mb))
                }
        ```
        """
        raise NotImplementedError

    async def request(
        self, input: quite.TaskInput, model: dict, result: quite.TargetResult | None = None
    ) -> quite.Output:
        """
        Execute other model.  
        To be called from the execute method

        Args:
            input: The input data  
            model: The model  
            result: Target result. Only batch results are allowed  
        """
        if self._runner is None:
            raise quite.Error(
                quite.ErrorCode.MODEL_EXECUTION_ERROR, "model internal runner is not available"
            )

        base_model = msgspec.convert(model, quite.Model)
        if not base_model.from_id:
            raise quite.Error(
                quite.ErrorCode.MODEL_EXECUTION_ERROR, "model `from_id` is required"
            )

        if base_model.from_id == self.model.from_id and base_model.variant == self.model.variant:
            raise quite.Error(
                quite.ErrorCode.MODEL_EXECUTION_ERROR, "cannot run the same model"
            )

        batch = model["batch"]
        if not isinstance(batch, list):
            batch = [batch]

        if not len(batch):
            raise quite.Error(
                quite.ErrorCode.MODEL_EXECUTION_ERROR, "batch is required"
            )

        res_batch_id = batch[-1].get("id", None)
        if res_batch_id is None:
            res_batch_id = batch[-1]["id"] = quite.Unique.uid()

        if result is not None:
            if len(result) > 1:
                raise quite.Error(
                    quite.ErrorCode.MODEL_EXECUTION_ERROR,
                    "only single target batch result is allowed for internal model execution",
                )

            if isinstance(result, list):
                result = {r: {} for r in result}

            for res_batch_id in result.keys():
                if next((b for b in base_model.batch if b.id == res_batch_id), None) is None:
                    raise quite.Error(
                        quite.ErrorCode.MODEL_EXECUTION_ERROR,
                        f"target batch {res_batch_id} not found in model",
                    )

        run_result = await self._runner.run(
            quite.InferenceTaskRequest(
                parent_id=self._request.id if self._request is not None else None,
                input=input.serializable(),
                model=model,
                result=result if result is not None else [res_batch_id],
            )
        )
        if run_result.error:
            raise run_result.error from None

        return run_result.output

    def gather_batch(self, outputs: List[quite.Output]) -> quite.Output:
        """
        Gather the batch output.  
        Can be overridden

        Args:
            outputs: The outputs of the iterations  

        Returns:
            The output of the batch  
        """
        batch_output = {}
        for output in outputs:
            for k, v in output.items():
                item = batch_output.get(k, None)
                if item is None:
                    item = batch_output[k] = v.to_batch()
                else:
                    item.batch.extend(v.to_batch().batch)
        return batch_output

    def finalize(self):
        """
        Clean up the model
        """
        self.runtime.empty_cache()

    def log_vram(self, debug: bool = False):
        """
        Log the VRAM usage
        """
        self.runtime.log_vram(self.logger, debug=debug)

    def _set_logger(self, logger: object):
        self.logger = logger

    def _set_request(self, request: quite.TaskRequest):
        self._request = request

    def __repr__(self):
        return f"{self.__class__.__name__} {self.model_config}"
