# ruff : noqa

from __future__ import annotations

from typing import List

import os
import sys
import copy
import enum
import io as _io
from loguru import logger

import msgspec
import numpy as np
from msgspec import Meta, Struct, field

__pdoc__ = {"logger": False}

log_level = os.environ.get("LOG_LEVEL", "DEBUG")

logger.remove()
logger_templ = copy.deepcopy(logger)
# func_format = (
#     "<c>{name}</c>:<c>{function}</c>:<c>{line}</c> "
#     if log_level == "DEBUG"
#     else "<c>{name}</c> "
# )
func_format = "<c>{name}</c>"

logger.add(
    sys.stderr,
    colorize=True,
    format=(
        "<w>{time:HH:mm:ss.SSS}</w> "
        "<level>{level: <8}</level> "
        f"{func_format} <level>{{message}}</level>"
    ),
    level=log_level,
)


def new_logger(prefix: str | List[str] | None = None):
    """
    Create a new logger with a prefix
    """
    l = copy.deepcopy(logger_templ)
    if prefix is not None:
        if isinstance(prefix, str):
            prefix = [prefix]
        prefix = " ".join(prefix)
        l.add(
            sys.stderr,
            colorize=True,
            format=(
                "<w>{time:HH:mm:ss.SSS}</w> "
                "<level>{level: <8}</level> "
                f"{prefix} "
                "<level>{message}</level>"
            ),
            level=log_level,
        )
    return l


def _enc_hook(obj):
    if isinstance(obj, np.ndarray):
        f = _io.BytesIO()
        np.save(f, obj, allow_pickle=True)
        f.seek(0)
        obj = f.read()
    return obj


def _dec_hook(type, data):
    if type == np.ndarray:
        f = _io.BytesIO(data)
        f.seek(0)
        data = np.load(f, allow_pickle=True)
    return data


def msgpack_encoder(*args, **kwargs):
    """
    Create a new msgpack encoder with additional hooks
    """
    return msgspec.msgpack.Encoder(*args, **kwargs, enc_hook=_enc_hook)


def msgpack_decoder(*args, **kwargs):
    """
    Create a new msgpack decoder with additional hooks
    """
    return msgspec.msgpack.Decoder(*args, **kwargs, dec_hook=_dec_hook)


struct_options = {
    "omit_defaults": True,
    "kw_only": True,
    "dict": False,
}

struct_state_options = dict(struct_options)
struct_state_options["omit_defaults"] = False


def get_tag(obj: msgspec.Struct | type) -> str | None:
    """
    Get the tag of a msgspec.Struct type or instance
    """
    if isinstance(obj, msgspec.Struct):
        return msgspec.inspect.type_info(obj.__class__).tag
    elif isinstance(obj, type):
        return msgspec.inspect.type_info(obj).tag
    return None


def same_tag(a: msgspec.Struct | type, b: msgspec.Struct | type) -> bool:
    """
    Check if two msgspec.Struct types or instances have the same tag
    """
    ta = get_tag(a)
    tb = get_tag(b)
    return ta is not None and tb is not None and ta == tb


class ErrorCode(enum.Enum):
    ERROR = "error"
    INVALID_REQUEST = "invalid request"
    INTERNAL_ERROR = "internal error"
    CLUSTER_ERROR = "cluster error"
    SCALING_ERROR = "scaling error"
    RPC_ERROR = "rpc error"
    SEND_ERROR = "send error"
    RUNTIME_NOT_FOUND = "runtime not found"
    RUNTIME_NOT_READY = "runtime not ready"
    RUNTIME_NOT_LOADED = "runtime not loaded"
    RUNTIME_TIMEOUT = "runtime timeout"
    CLONE_NOT_READY = "clone not ready"
    TASK_RESOLVE_ERROR = "task resolve error"
    TASK_EXECUTION_ERROR = "task execution error"
    TASK_INTERRUPTED = "task interrupted"
    TASK_TIMEOUT = "task timeout"
    TASK_GRAPH_ERROR = "task graph error"
    DOWNLOAD_ERROR = "download error"
    REMOVAL_ERROR = "removal error"
    LOAD_ERROR = "load error"
    UNLOAD_ERROR = "unload error"
    MODEL_ERROR = "model error"
    MODEL_EXECUTION_ERROR = "model execution error"
    OBJECT_UPLOAD_ERROR = "object upload error"
    OBJECT_DOWNLOAD_ERROR = "object download error"
    REPOSITORY_ERROR = "repository error"
    DATA_ERROR = "data error"
    TRITON_ERROR = "triton error"
    PROVIDER_ERROR = "provider error"
    CONFIG_ERROR = "config error"


class Error(Exception):
    """Exception"""

    def __init__(
        self,
        a: Exception | ErrorCode | str,
        b: ErrorCode | str | None = None,
        c: str | None = None,
    ):
        if isinstance(a, Error):
            self.code = b if isinstance(b, ErrorCode) else a.code
            message = f"{b + ': ' if isinstance(b, str) else ''}{a}"
        elif isinstance(a, Exception):
            self.code = b if isinstance(b, ErrorCode) else ErrorCode.ERROR
            message = f"{b + ': ' if isinstance(b, str) else ''}{a}"
        elif isinstance(a, ErrorCode):
            self.code = a
            message = f"{b if isinstance(b, str) else ''}"
        elif isinstance(a, str):
            self.code = ErrorCode.ERROR
            message = a
        else:
            self.code = ErrorCode.ERROR
            message = f"{b if isinstance(b, str) else ''}"

        if isinstance(c, str):
            message = f"{message}: {c}"

        prefix = f"{self.code.value}: "
        message = message.replace(prefix, "")
        message = f"{prefix}{message}"

        super().__init__(message)
