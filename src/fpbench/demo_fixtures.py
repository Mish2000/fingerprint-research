"""Deterministic software fixtures; no image or subject from a real person.

Reuses the existing SourceAFIS integration fixture geometry.
"""
import binascii
import math
import struct
import zlib
from functools import lru_cache


@lru_cache(maxsize=3)
def synthetic_fingerprint_png(variant: int) -> bytes:
    width = 360
    height = 460
    period = 10.5
    rows: list[bytes] = []
    for y in range(height):
        row = bytearray()
        for x in range(width):
            dx = (x - width / 2) / (width * 0.46)
            dy = (y - height / 2) / (height * 0.43)
            if dx * dx + dy * dy > 1.0:
                row.append(255)
                continue
            warped_y = y + 18.0 * math.sin(x * 0.035 + variant * 0.35) + 5.0 * math.sin(y * 0.02)
            ridge_pos = (warped_y - 52.0) % period
            distance = min(ridge_pos, period - ridge_pos)
            row.append(22 if distance < 1.7 else 245)
        rows.append(bytes([0]) + bytes(row))
    return _png_bytes(width, height, b"".join(rows))


def _png_bytes(width: int, height: int, filtered_rows: bytes) -> bytes:
    def chunk(name: bytes, data: bytes) -> bytes:
        checksum = binascii.crc32(name)
        checksum = binascii.crc32(data, checksum) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + name + data + struct.pack(">I", checksum)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(filtered_rows)) + chunk(b"IEND", b"")
