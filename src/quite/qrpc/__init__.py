# ruff: noqa

from .qrpc_pb2_grpc import QRPCServicer, add_QRPCServicer_to_server, QRPCStub
from .qrpc_pb2 import Response, Request, DESCRIPTOR

CHANNEL_OPTIONS = [
    ("grpc.lb_policy_name", "round_robin"),
    ("grpc.enable_retries", 1),
    ("grpc.max_receive_message_length", 100 * 1024 * 1024),
    ("grpc.max_send_message_length", 100 * 1024 * 1024),
    ("grpc.keepalive_time_ms", 60000),
    ("grpc.keepalive_timeout_ms", 10000),
    ("grpc.keepalive_permit_without_calls", 1),
    ("grpc.max_reconnect_backoff_ms", 4000),
    ("grpc.initial_reconnect_backoff_ms", 1000),
    ("grpc.http2.min_ping_interval_without_data_ms", 1000),
    ("grpc.http2.max_ping_strikes", 0),
]

SERVER_CHANNEL_OPTIONS = [
    ("grpc.keepalive_time_ms", 7200000),
    ("grpc.keepalive_timeout_ms", 10000),
    ("grpc.keepalive_permit_without_calls", 1),
    ("grpc.max_concurrent_streams", 1000),
    ("grpc.max_send_message_length", 100 * 1024 * 1024),
    ("grpc.max_receive_message_length", 100 * 1024 * 1024),
    ("grpc.max_connection_age_ms", 1024 * 1024 * 1024),
    ("grpc.max_connection_age_grace_ms", 1024 * 1024 * 1024),
    ("grpc.max_connection_idle_ms", 1024 * 1024 * 1024),
    ("grpc.server_handshake_timeout_ms", 60000),
    ("grpc.http2.min_ping_interval_without_data_ms", 1000),
    ("grpc.http2.max_ping_strikes", 0),
]
