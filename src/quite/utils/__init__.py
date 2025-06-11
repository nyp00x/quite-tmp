from __future__ import annotations
from typing import Any, Tuple, List, Dict, Set

import os
import sys
import re
import enum
import time
import ssl
import datetime
import aiohttp
import asyncio
import msgspec
import importlib
from pathlib import Path
from types import SimpleNamespace
from collections import defaultdict
from omegaconf import OmegaConf, DictConfig

from quite.io.common import Unique, ObjectDomain
from quite.globals import Error, ErrorCode

from loguru import logger as _logger

from .yaml import yaml_loads, yaml_dumps  # noqa
from .file import *
from .image import *
from .video import *
from .request import *

import pkg_resources

__pdoc__ = {
    "request": False,
    "yaml": False,
    "iterate_fields": False,
    "has_duplicate_ids": False,
    "rename_function": False,
}

try:
    pkg_resources.get_distribution("torch")
    torch_available = True
except pkg_resources.DistributionNotFound:
    torch_available = False


GIT_MODEL_PATTERN = r"https://([a-zA-Z0-9.-]+)/([a-zA-Z0-9_-]+)/([a-zA-Z0-9_-]+)\.git"
_torch = None

try:
    DOWNLOAD_TIMEOUT = float(os.getenv("QUITE_DOWNLOAD_TIMEOUT"))
except Exception:
    DOWNLOAD_TIMEOUT = 12

try:
    DOWNLOAD_CONNECT_TIMEOUT = float(os.getenv("QUITE_DOWNLOAD_CONNECT_TIMEOUT"))
except Exception:
    DOWNLOAD_CONNECT_TIMEOUT = 5


def get_torch():
    """
    Deferred torch import
    """
    global _torch
    if _torch is not None:
        return _torch

    if torch_available:
        import torch

        _torch = torch
        return torch
    else:
        raise ImportError("torch is not available")


def load_config(name: str, strict: bool = False) -> DictConfig | None:
    """
    Load a config file
    """
    config_dir = os.getenv("QUITE_CONFIG_DIR", None)
    if not config_dir:
        if strict:
            raise ValueError("QUITE_CONFIG_DIR is not set")
        else:
            # _logger.debug(f"{name} config is not loaded")
            return None

    if not name:
        raise ValueError("config name is not set")

    config_path = Path(config_dir) / f"{name}.yaml"
    if not config_path.is_file():
        if strict:
            raise ValueError(f"config file '{name}' not found")
        return None

    _logger.debug(f"loading config from {config_path}")
    with open(config_path, "r") as f:
        config = yaml_loads(f.read())

    config = OmegaConf.create(config)
    OmegaConf.resolve(config)
    return config


def proto_id(val: str | Unique) -> str:
    """
    Get unique instance id
    """
    if isinstance(val, str):
        id = val
    elif isinstance(val, Unique):
        id = val.from_id
    else:
        raise TypeError(f"invalid type {type(val)}")
    return id


def instance_id(val: str | Unique) -> str:
    """
    Get unique prototype id
    """
    if isinstance(val, str):
        id = val
    elif isinstance(val, Unique):
        id = val.id
    else:
        raise TypeError(f"invalid type {type(val)}")
    return id


def make_id(
    repo_name: str | None = None,
    revision: str | None = None,
    variant: str | None = None,
    object_path: str | None = None,
) -> str:
    """
    Compose a unique id
    """
    if object_path:
        path_parts = object_path.split("/")
        if len(path_parts) != 2 or not path_parts[0] or not path_parts[1]:
            raise ValueError(f"invalid object path '{object_path}'")

    parts = []

    if repo_name:
        parts.append(repo_name)

    if revision:
        parts.append(f"@{revision}")

    if variant:
        if revision is None:
            raise ValueError("variant requires revision")
        parts.append(f"#{variant}")

    if object_path:
        if repo_name:
            repo_part = f"{repo_name}/"
            if not object_path.startswith(repo_part):
                raise ValueError(f"invalid object path '{object_path}'")
            parts.append(f":{object_path[len(repo_part) :]}")
        else:
            parts.append(f":{object_path}")

    return str("".join(parts))


def set_model_variant(id: str, variant: str | None = None) -> str:
    """
    Set a variant to a model id
    """
    repo_name, revision, _, object_path = Unique.parse_id(id)
    if revision is None:
        raise ValueError(f"{id}: invalid model revision")
    return make_id(repo_name, revision, variant, object_path)


def model_name(
    val: str | Unique, proto: bool = True, revision: str = None, strict: bool = False
) -> str:
    """
    Get model name
    """
    id = proto_id(val) if proto else instance_id(val)
    repo_name, _revision, _, _ = Unique.parse_id(id)

    if revision is not None:
        _revision = revision
    if _revision is None or repo_name is None:
        if strict:
            raise ValueError(f"{id}: invalid model revision")
        else:
            return None
    return str(f"{repo_name}@{_revision}")


def object_path(val: str | Unique, proto: bool = True, strict: bool = False) -> str:
    """
    Get object path
    """
    id = proto_id(val) if proto else instance_id(val)
    _, _, _, object_path = Unique.parse_id(id)

    if object_path is None:
        if strict:
            raise ValueError(f"{id}: invalid object path")
        else:
            return None
    return object_path


def object_domain(
    val: str | Unique, proto: bool = True, strict: bool = False
) -> ObjectDomain:
    """
    Get object domain
    """
    id = proto_id(val) if proto else instance_id(val)
    repo_name, _, _, object_path = Unique.parse_id(id)

    if object_path is None and strict:
        raise ValueError(f"{id}: invalid object path")

    if repo_name is not None:
        return ObjectDomain.MODEL
    else:
        return ObjectDomain.DATA


def split_model_url(url: str) -> Tuple[str, str, str]:
    """
    Get domain, org name and model name from model url
    """
    match = re.match(GIT_MODEL_PATTERN, url)
    if match:
        domain = match.group(1)
        org_name = match.group(2)
        repo_name = match.group(3)
    else:
        raise ValueError(f"invalid model url {url}")

    return domain, org_name, repo_name


def package_name(model_name: str) -> str:
    """
    Make package name
    """
    result = re.sub(r"[-@]", "_", model_name)
    if not re.match(r"^[a-z0-9_@-]*$", result):
        raise ValueError(f"invalid model name {model_name}")
    return result


async def download_file(
    url,
    retries=3,
    retry_delay=0.01,
    timeout=None,
    timeout_sock_connect=None,
    logger: Any = None,
):
    """
    Download a file from a specified URL with configurable timeout settings.
    """
    if logger is None:
        logger = _logger

    if timeout is None:
        timeout = DOWNLOAD_TIMEOUT
    if timeout_sock_connect is None:
        timeout_sock_connect = DOWNLOAD_CONNECT_TIMEOUT

    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=timeout, sock_connect=timeout_sock_connect)
    ) as session:
        for attempt in range(1, retries + 1):
            s = f"downloading {url} ({attempt}/{retries}) (timeout {timeout}s)"
            try:
                async with session.get(url) as response:
                    response.raise_for_status()
                    return await response.read()

            except aiohttp.ConnectionTimeoutError as e:
                if attempt < retries:
                    logger.warning(
                        f"connection timeout error: {s}: {e}, retrying in {retry_delay} second(s)..."
                    )
                    await asyncio.sleep(retry_delay)
                else:
                    raise Error(
                        ErrorCode.DOWNLOAD_ERROR,
                        f"{s}: {e}",
                    )

            except Exception as e:
                if attempt < retries:
                    logger.warning(f"{s}: {e}, retrying in {retry_delay} second(s)...")
                    await asyncio.sleep(retry_delay)
                else:
                    raise Error(
                        ErrorCode.DOWNLOAD_ERROR,
                        f"{s}: {type(e).__name__}: {e}",
                    )


async def send_post(
    url: str,
    data: dict | None = None,
    retries: int = 1,
    retry_delay: int = 1,
    timeout: float = 16,
    retry_status: int | List[int] = [],
    timeout_sock_connect: float = 8,
    ssl_verify: bool = True,
    display_url: str = "",
    log_prefix: str = "",
    logger: Any = None,
    silent: bool = False,
    detach: bool = False,
) -> dict | None:
    """
    Send a POST request to a specified URL with optional data
    """
    if logger is None:
        logger = _logger

    if isinstance(retry_status, int):
        retry_status = [retry_status]

    if detach:
        f = asyncio.create_task(
            send_post(
                url,
                data,
                retries,
                retry_delay,
                timeout,
                retry_status,
                timeout_sock_connect,
                ssl_verify,
                display_url,
                log_prefix,
                logger,
                silent,
            )
        )
        f._is_detached = True
        return

    ssl_context = None
    if not ssl_verify:
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE

    try:
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(
                total=timeout, sock_connect=timeout_sock_connect
            )
        ) as session:
            for i in range(retries):
                s = f"{log_prefix} " if log_prefix else ""
                s += f"POST {display_url or url} ({i + 1}/{retries}) (timeout {timeout}s)"
                if not silent:
                    logger.debug(s)

                t = time.perf_counter()
                try:
                    async with session.post(
                        url, json=data, ssl=ssl_context
                    ) as response:
                        dt = time.perf_counter() - t
                        content = await response.text()

                        if not silent:
                            if response.status != 200:
                                logger.warning(
                                    f"{s}: {response.status} ({dt:.3f}s), response length: {len(content)}, content: {content[:512]}"
                                )
                            else:
                                logger.debug(
                                    f"{s}: {response.status} ({dt:.3f}s), response length: {len(content)}"
                                )

                        if response.status in retry_status:
                            raise Error(f"{response.status} status")

                        content_type = (
                            response.headers.get("Content-Type", "")
                            .lower()
                            .split(";")[0]
                            .strip()
                        )

                        if content_type == "application/json":
                            try:
                                return await response.json()
                            except Exception as json_error:
                                if not silent:
                                    logger.warning(
                                        f"{s} error parsing JSON response: {json_error}, content: {content[:512]}"
                                    )
                                return None

                        elif content_type == "text/plain":
                            return content
                        else:
                            if not silent:
                                logger.warning(
                                    f"{s} unexpected content type: {content_type}, content: {content[:512]}"
                                )
                            return None

                except aiohttp.ConnectionTimeoutError as e:
                    msg = f"{s}: connection timeout error: {e}"
                    if i == retries - 1:
                        if not silent:
                            logger.error(msg)
                        raise Error(ErrorCode.SEND_ERROR, msg)
                    else:
                        if not silent:
                            logger.warning(
                                f"{s}: {msg}: retrying connection in {retry_delay} second(s)..."
                            )
                        await asyncio.sleep(retry_delay)

                except Exception as e:
                    msg = f"{s}: {type(e).__name__}: {e}"
                    if i == retries - 1 or "timeout" in msg.lower():
                        if not silent:
                            logger.error(msg)
                        raise Error(ErrorCode.SEND_ERROR, msg)
                    else:
                        if not silent:
                            logger.warning(
                                f"{s}: {msg}: retrying in {retry_delay} second(s)..."
                            )
                        await asyncio.sleep(retry_delay)

    except Exception as e:
        t = asyncio.current_task()
        if t is not None and getattr(t, "_is_detached", False):
            return
        else:
            raise e


def val2enum(value: str, enum_class: enum.Enum, default: Any = None) -> enum.Enum | Any:
    """
    Convert a string value to an enum member
    """
    for member in enum_class:
        if member.value.lower() == value.lower():
            return member
    return default


def name2enum(name: str, enum_class: enum.Enum, default: Any = None) -> enum.Enum | Any:
    """
    Convert a string name to an enum member
    """
    for member in enum_class:
        if member.name.lower() == name.lower():
            return member
    return default


def iterate_fields(
    obj: msgspec.Struct | dict,
    target_types: List[type] = [],
    target_names: List[str] = [],
    target=None,
    depth=0,
    max_depth=10,
    p_name=None,
    pp_name=None,
    ppp_name=None,
    stop_types=[],
    ignore_names=[],
    ignore=None,
    base_name=None,
    base_field=None,
) -> Tuple[Tuple[Any, Any]]:
    fields = []
    if depth > max_depth:
        return fields
    if isinstance(obj, msgspec.Struct):
        items = [(name, getattr(obj, name, None)) for name in obj.__struct_fields__]
    elif isinstance(obj, dict):
        items = obj.items()
    elif isinstance(obj, list):
        items = [(None, item) for item in obj]
    else:
        return fields

    for field_name, field_value in items:
        if (
            field_name in target_names
            or any([isinstance(field_value, t) for t in target_types])
            or target is not None
            and target(field_name, field_value)
        ):
            if field_name is None:
                name = p_name
                parent_name = pp_name or ppp_name
            else:
                name = field_name
                parent_name = p_name or pp_name or ppp_name

            ignored = (
                name in ignore_names or ignore is not None and ignore(name, field_value)
            )
            if not ignored:
                field = SimpleNamespace(
                    name=name,
                    value=field_value,
                    parent_name=parent_name,
                    depth=depth,
                    base_field=base_field,
                )
                fields += [field]

        if any([isinstance(field_value, t) for t in stop_types]):
            continue

        fields.extend(
            iterate_fields(
                field_value,
                target_types,
                target_names,
                target,
                depth + 1,
                max_depth,
                field_name,
                p_name,
                pp_name,
                stop_types,
                ignore_names,
                ignore,
                base_name,
                field_value if base_name == field_name else base_field,
            )
        )
    return fields


def unload_module(module_name):
    """
    Unload a python module
    """
    if module_name in sys.modules:
        del sys.modules[module_name]


def load_module(module_name, force_reload=True) -> Any:
    """
    Load a python module
    """
    if module_name in sys.modules:
        if force_reload:
            _logger.debug(f"reloading module {module_name}")
            return importlib.reload(sys.modules[module_name])
        else:
            return sys.modules[module_name]
    else:
        return __import__(module_name)


def get_directory_size(start_path):
    """
    Get directory size in bytes
    """
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
    return total_size


def copy_struct(obj: msgspec.Struct) -> msgspec.Struct:
    """
    Copy a msgspec.Struct
    """
    return type(obj)(**msgspec.structs.asdict(obj))


def copy_struct_fields(
    obj: msgspec.Struct,
    obj_source: msgspec.Struct,
    fields: list[str] | None = None,
    ignore_fields: list[str] = [],
):
    """
    Copy fields from one msgspec.Struct to another
    """
    if fields is None:
        fields = list(obj_source.__struct_fields__)
    if len(ignore_fields):
        fields = [field for field in fields if field not in ignore_fields]
    for field in fields:
        setattr(obj, field, getattr(obj_source, field))


def concat_lists(dicts: List[Dict[str, list]]) -> Dict[str, list]:
    """
    Concatenate lists in dictionaries
    """
    result = defaultdict(list)
    for d in dicts:
        for key, values in d.items():
            result[key].extend(values)
    return dict(result)


def has_duplicate_ids(
    obj: Dict[str, Any] | List[Any] | msgspec.Struct,
    seen_ids: Set[str] | None = None,
    depth: int = 0,
    max_depth: int = 100,
) -> str | None:
    if seen_ids is None:
        seen_ids = set()

    if depth > max_depth:
        return None

    if isinstance(obj, msgspec.Struct):
        items = [(name, getattr(obj, name, None)) for name in obj.__struct_fields__]
    elif isinstance(obj, dict):
        items = obj.items()
    elif isinstance(obj, list):
        items = [(None, item) for item in obj]
    else:
        return None

    for key, value in items:
        if key == "id" and "from_id" in items:
            if not isinstance(value, str):
                raise ValueError(f"invalid id value: {value}")
            if value in seen_ids:
                return value
            seen_ids.add(value)
        else:
            duplicate_id = has_duplicate_ids(value, seen_ids, depth + 1, max_depth)
            if duplicate_id is not None:
                return duplicate_id
    return None


def rename_function(name: str):
    def decorator(f):
        f.__name__ = name
        return f

    return decorator


def utc_now() -> datetime.datetime:
    """
    Get current UTC time
    """
    return datetime.datetime.now(datetime.timezone.utc)


def is_package_available(package_name: str) -> bool:
    """
    Check if a python package is available
    """
    package_name = package_name.strip().replace("-", "_")
    return importlib.util.find_spec(package_name) is not None


def get_event_loop() -> asyncio.AbstractEventLoop:
    """
    Get asyncio event loop
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError("Loop is closed")
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop
