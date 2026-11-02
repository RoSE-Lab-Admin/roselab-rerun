"""ROS 2 bag streaming and message decoding (via ``rosbags``, no ROS install needed)."""

from __future__ import annotations

import struct
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
from rosbags.highlevel import AnyReader

# compressed_depth_image_transport prepends a ConfigHeader {int32 format; float32 depthParam[2]}.
_DEPTH_HEADER = struct.Struct("<iff")


def stamp_ns(msg, fallback_ns: int) -> int:
    """Header stamp in integer ns if the message has one, otherwise ``fallback_ns`` (bag receive time)."""
    header = getattr(msg, "header", None)
    if header is None:
        return int(fallback_ns)
    return int(header.stamp.sec) * 1_000_000_000 + int(header.stamp.nanosec)


class BagMessage(NamedTuple):
    topic: str
    stamp_ns: int  # header stamp (sensor capture time), or log time for header-less messages
    log_ns: int  # bag log (receive) time; unique-ish per topic and usable for seeking
    msg: object


def count_messages(bag_paths: Iterable[Path], topics: Iterable[str]) -> int:
    """Total number of messages on ``topics`` (from bag metadata, no deserialization)."""
    wanted = set(topics)
    with AnyReader([Path(p) for p in bag_paths]) as reader:
        return sum(c.msgcount for c in reader.connections if c.topic in wanted)


def iter_messages(
    bag_paths: Iterable[Path], topics: Iterable[str], start_ns: int | None = None, stop_ns: int | None = None
) -> Iterator[BagMessage]:
    """Yield :class:`BagMessage` for the requested topics across one or more bags.

    ``start_ns`` / ``stop_ns`` restrict the read to a ``[start, stop)`` window of
    bag log time, which lets parallel workers each read only their own slice.
    """
    wanted = set(topics)
    with AnyReader([Path(p) for p in bag_paths]) as reader:
        connections = [c for c in reader.connections if c.topic in wanted]
        for connection, t_log, rawdata in reader.messages(connections=connections, start=start_ns, stop=stop_ns):
            msg = reader.deserialize(rawdata, connection.msgtype)
            yield BagMessage(connection.topic, stamp_ns(msg, t_log), int(t_log), msg)


def decode_compressed_image(msg) -> np.ndarray:
    """``sensor_msgs/CompressedImage`` (jpeg/png) -> RGB uint8 ``(H, W, 3)``."""
    bgr = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Failed to decode compressed image (format={getattr(msg, 'format', '?')!r})")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def decode_compressed_depth(msg, depth_scale: float = 0.001) -> np.ndarray:
    """``compressedDepth`` CompressedImage -> metric depth float32 ``(H, W)`` in meters, 0 = invalid.

    ``16UC1`` payloads are integer depth units scaled by ``depth_scale``
    (RealSense default: 1 unit = 1 mm). ``32FC1`` payloads are inverse-depth
    quantized and are recovered with the header's quantization parameters.
    """
    data = bytes(msg.data)
    fmt = getattr(msg, "format", "")
    raw = cv2.imdecode(np.frombuffer(data[_DEPTH_HEADER.size :], np.uint8), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise ValueError(f"Failed to decode compressed depth (format={fmt!r})")

    if fmt.startswith("32FC1"):
        _, quant_a, quant_b = _DEPTH_HEADER.unpack_from(data)
        raw = raw.astype(np.float32)
        depth = np.zeros_like(raw)
        nz = raw > 0
        depth[nz] = quant_a / (raw[nz] - quant_b)
        return depth

    return raw.astype(np.float32) * np.float32(depth_scale)
