from typing import Dict

import time
import asyncio

import grpc
from loguru import logger
import msgspec

import quite

from .api import (
    CloneStateResponse,
    CloneStateRequest,
    RuntimeStateResponse,
    RuntimeStateRequest,
    LoadRequest,
    LoadResponse,
    UnloadRequest,
    UnloadResponse,
    RunRequest,
    RunResponse,
    DownloadRequest,
    DownloadResponse,
    RemoveRequest,
    RemoveResponse,
)
from quite.qrpc import QRPCStub, Request, CHANNEL_OPTIONS

encoder = quite.msgpack_encoder()
clone_state_decoder = quite.msgpack_decoder(CloneStateResponse)
runtime_state_decoder = quite.msgpack_decoder(RuntimeStateResponse)
download_decoder = quite.msgpack_decoder(DownloadResponse)
remove_decoder = quite.msgpack_decoder(RemoveResponse)
load_decoder = quite.msgpack_decoder(LoadResponse)
unload_decoder = quite.msgpack_decoder(UnloadResponse)
run_decoder = quite.msgpack_decoder(RunResponse)


class RemoteExecutorConfig(msgspec.Struct, **quite.struct_options):
    """
    Remote RPC executor configuration
    """

    enabled: bool = True
    port: int = 50053
    compression: bool = True
    timeout: Dict[str, int] | None = None
    recreate_every: int = 50

    def __post_init__(self):
        if self.timeout is None:
            self.timeout = {
                "clone_state": 10,
                "runtime_state": 10,
                "download": 1200,
                "remove": 120,
                "load": 120,
                "unload": 120,
                "run": 1200,
            }


class RemoteExecutorChannel:
    def __init__(
        self,
        cfg: RemoteExecutorConfig,
        endpoint: quite.GRPCEndpoint,
        secure: bool = False,
    ):
        compression = (
            grpc.Compression.Gzip if cfg.compression else grpc.Compression.NoCompression
        )
        self.endpoint = endpoint

        options = CHANNEL_OPTIONS

        if endpoint.ssl_target_name_override:
            options = list(options)
            options += [
                ("grpc.ssl_target_name_override", endpoint.ssl_target_name_override)
            ]

        if not secure:
            self.channel = grpc.aio.insecure_channel(
                target=f"{endpoint.host}:{endpoint.port}",
                options=options,
                compression=compression,
            )
        else:
            self.channel = grpc.aio.secure_channel(
                target=f"{endpoint.host}:{endpoint.port}",
                credentials=grpc.ssl_channel_credentials(),
                options=options,
                compression=compression,
            )
        self.stub = QRPCStub(self.channel)

    async def close(self):
        try:
            logger.info(f"closing remote executor channel: {self.endpoint}")
            await self.channel.close()
        except Exception:
            logger.warning("failed to close remote executor channel")
            pass


class RemoteExecutor:
    """
    General executor RPC client
    """

    def __init__(
        self,
        cfg: RemoteExecutorConfig | None = None,
        endpoint: quite.GRPCEndpoint | None = None,
        secure: bool = False,
    ):
        if cfg is None:
            cfg = RemoteExecutorConfig()
        self.cfg = cfg

        if endpoint is None:
            endpoint = quite.GRPCEndpoint(host=self.cfg.host, port=self.cfg.port)

        if endpoint.port is None:
            endpoint.port = self.cfg.port

        if not endpoint.host or not endpoint.port:
            raise quite.Error(
                quite.ErrorCode.CLUSTER_ERROR,
                "remote executor endpoint not specified",
            )

        self.endpoint = endpoint
        self.secure = secure or endpoint.secure
        self.num_channel_retries = 4
        self.sleep_time = 1
        self.run_count = 1
        self.channel_sessions = {}

        self.create_channel()

    async def clone_state(self, req: CloneStateRequest) -> CloneStateResponse:
        response = await self.stub.CloneState(
            Request(data=encoder.encode(req)),
            timeout=self.cfg.timeout.get("clone_state", 10),
        )
        return clone_state_decoder.decode(response.data)

    async def runtime_state(self, req: RuntimeStateRequest) -> RuntimeStateResponse:
        response = await self.stub.RuntimeState(
            Request(data=encoder.encode(req)),
            timeout=self.cfg.timeout.get("runtime_state", 10),
        )
        return runtime_state_decoder.decode(response.data)

    async def download(self, req: DownloadRequest) -> DownloadResponse:
        response = await self.stub.Download(
            Request(data=encoder.encode(req)),
            timeout=self.cfg.timeout.get("download", 800),
        )
        return download_decoder.decode(response.data)

    async def remove(self, req: RemoveRequest) -> RemoveResponse:
        response = await self.stub.Remove(
            Request(data=encoder.encode(req)),
            timeout=self.cfg.timeout.get("remove", 60),
        )
        return remove_decoder.decode(response.data)

    async def load(self, req: LoadRequest) -> LoadResponse:
        response = await self.stub.Load(req, timeout=self.cfg.timeout.get("load", 120))
        return load_decoder.decode(response.data)

    async def unload(self, req: UnloadRequest) -> UnloadResponse:
        response = await self.stub.Unload(
            Request(data=encoder.encode(req)),
            timeout=self.cfg.timeout.get("unload", 120),
        )
        return unload_decoder.decode(response.data)

    async def run_us(self, req: RunRequest, aio_retry: bool = True) -> RunResponse:
        await self.update_channels()
        self.run_count += 1
        started = False
        try:
            data = encoder.encode(req)
            num_retries = self.num_channel_retries if aio_retry else 1
            for i in range(num_retries):
                channel = self.get_channel()
                self.channel_sessions[channel] += 1
                started = True
                try:
                    async for response in channel.stub.RunUS(
                        Request(data=data),
                        timeout=self.cfg.timeout.get("run", 800),
                    ):
                        yield run_decoder.decode(response.data)
                    break
                except grpc.aio.AioRpcError as e:
                    logger.warning(
                        f"AioRpcError unary-stream error: {e.debug_error_string()}"
                    )
                    if (
                        e.code()
                        not in [
                            grpc.StatusCode.UNAVAILABLE,
                            grpc.StatusCode.UNKNOWN,
                            grpc.StatusCode.CANCELLED,
                        ]
                        or i == num_retries - 1
                    ):
                        raise e
                    else:
                        logger.warning(
                            f"recovering from AioRpcError (retry {i}): {e.debug_error_string()}"
                        )
                        self.channel_sessions[channel] -= 1
                        self.create_channel()
                        time.sleep(self.sleep_time)

                except Exception as e:
                    logger.error(f"unary-stream error: {e}")
                    raise e
        finally:
            if started:
                try:
                    self.channel_sessions[channel] -= 1
                except KeyError:
                    pass

    def create_channel(self):
        logger.info(
            f"new remote executor channel: {self.endpoint} (total {len(self.channel_sessions)})"
        )
        channel = RemoteExecutorChannel(
            cfg=self.cfg,
            endpoint=self.endpoint,
            secure=self.secure,
        )
        self.channel_sessions[channel] = 0

    async def update_channels(self):
        if not len(self.channel_sessions):
            self.create_channel()

        if self.cfg.recreate_every and self.run_count % self.cfg.recreate_every == 0:
            self.create_channel()

        for channel in list(self.channel_sessions.keys())[:-1]:
            if self.channel_sessions[channel] <= 0:
                asyncio.create_task(channel.close())
                del self.channel_sessions[channel]

    def get_channel(self):
        return list(self.channel_sessions.keys())[-1]

    async def stop(self):
        for channel in list(self.channel_sessions.keys()):
            await channel.close()
