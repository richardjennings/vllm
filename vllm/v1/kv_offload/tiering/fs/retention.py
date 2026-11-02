# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single-owner idle expiry and write admission for the filesystem KV tier."""

import errno
import fcntl
import logging
import math
import os
import re
import stat
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from vllm.v1.kv_offload.tiering.fs.io import _get_tmp_suffix

logger = logging.getLogger(__name__)
_TEMP_BLOCK = re.compile(r"[0-9a-f]+\.bin_\d+\.tmp\Z")


class DirectoryLock:
    """Shared access without retention, exclusive access with retention."""

    def __init__(self, root_dir: str, *, exclusive: bool):
        os.makedirs(root_dir, exist_ok=True)
        self._fd = os.open(
            os.path.join(root_dir, ".vllm-kv.lock"), os.O_CREAT | os.O_RDWR, 0o644
        )
        mode = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        try:
            fcntl.flock(self._fd, mode | fcntl.LOCK_NB)
        except OSError as exc:
            self.close()
            raise RuntimeError(
                f"KV cache directory {root_dir!r} is already owned by an "
                "incompatible vLLM instance; retention requires exclusive ownership"
            ) from exc

    def close(self) -> None:
        if self._fd >= 0:
            os.close(self._fd)
            self._fd = -1


@dataclass
class _Entry:
    size: int
    last_use: float
    pins: int = 0


class FileRetention:
    """Bound published and reserved block bytes while keeping active I/O pinned.

    Metadata lookups/touches do no filesystem I/O. A separate I/O lock serializes
    admission, eviction and timestamp persistence, without blocking those reads.
    Call close only after the tier's I/O workers have drained.
    """

    def __init__(
        self,
        root_dir: str,
        idle_ttl_seconds: float,
        max_bytes: int,
        *,
        clock: Callable[[], float] = time.time,
    ):
        if not math.isfinite(idle_ttl_seconds) or idle_ttl_seconds <= 0:
            raise ValueError("idle_ttl_seconds must be finite and positive")
        if type(max_bytes) is not int or max_bytes <= 0:
            raise ValueError("max_bytes must be a positive integer")
        self._ttl = idle_ttl_seconds
        self._limit = max_bytes
        self._clock = clock
        self._lock = threading.Lock()
        self._io_lock = threading.Lock()
        self._entries: OrderedDict[str, _Entry] = OrderedDict()
        self._dirty: set[str] = set()
        self._writers: set[str] = set()
        self._bytes = 0
        self._reserved = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._ownership = DirectoryLock(root_dir, exclusive=True)
        try:

            def scan_failed(error: OSError) -> None:
                raise error

            entries = []
            for directory, _, names in os.walk(root_dir, onerror=scan_failed):
                for name in names:
                    path = os.path.join(directory, name)
                    info = os.stat(path, follow_symlinks=False)
                    if not stat.S_ISREG(info.st_mode):
                        continue
                    if _TEMP_BLOCK.fullmatch(name):
                        os.unlink(path)
                    elif name.endswith(".bin"):
                        entries.append((path, _Entry(info.st_size, info.st_mtime)))
            self._entries.update(sorted(entries, key=lambda item: item[1].last_use))
            self._bytes = sum(entry.size for entry in self._entries.values())
            self.maintain()
            if self.accounted_bytes > self._limit:
                raise OSError(errno.ENOSPC, "Cannot bring KV directory under max_bytes")
        except BaseException:
            self._ownership.close()
            raise

    def start(self) -> None:
        """Start periodic maintenance once the owning tier has initialized."""
        if self._thread is None:
            thread = threading.Thread(
                target=self._run, name="vllm_kv_retention", daemon=True
            )
            thread.start()
            self._thread = thread

    @property
    def accounted_bytes(self) -> int:
        """Published bytes plus the maximum size of in-flight new writes."""
        with self._lock:
            return self._bytes + self._reserved

    def contains(self, path: str) -> bool:
        with self._lock:
            entry = self._entries.get(path)
            return entry is not None and self._clock() - entry.last_use < self._ttl

    def touch(self, paths: Iterable[str]) -> None:
        with self._lock:
            now = self._clock()
            for path in paths:
                entry = self._entries.get(path)
                if entry is not None and now - entry.last_use < self._ttl:
                    self._touch(path, entry, now)

    def _touch(self, path: str, entry: _Entry, now: float) -> None:
        entry.last_use = now
        self._entries.move_to_end(path)
        self._dirty.add(path)

    def _evict(self, *, expired_only: bool, exclude: set[str] | None = None) -> bool:
        # Caller holds _io_lock. Removal from the index claims the eviction;
        # actual unlink must not hold the scheduler-facing metadata lock.
        with self._lock:
            now = self._clock()
            for path, entry in self._entries.items():
                if expired_only and now - entry.last_use < self._ttl:
                    return False
                if entry.pins or (exclude and path in exclude):
                    continue
                del self._entries[path]
                self._dirty.discard(path)
                break
            else:
                return False
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass
        except OSError:
            with self._lock:
                # This entry has already been removed from service. Keep its
                # byte charge and retry deletion before admitting more writes.
                entry.last_use = float("-inf")
                self._entries[path] = entry
                self._entries.move_to_end(path, last=False)
            raise
        with self._lock:
            self._bytes -= entry.size
        return True

    def store(
        self, paths: list[str], block_size: int, write: Callable[[], None]
    ) -> None:
        """Reserve before writing; account partial publication even on failure."""
        unique = set(paths)
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        with self._io_lock:
            while self._evict(expired_only=True):
                pass
            with self._lock:
                if unique & self._writers:
                    raise OSError(errno.EAGAIN, "KV blocks already being written")
                present = unique.intersection(self._entries)
                for path in present:
                    entry = self._entries[path]
                    if entry.size != block_size:
                        entry.last_use = float("-inf")
                        self._entries.move_to_end(path, last=False)
                        self._dirty.discard(path)
                        raise OSError(errno.EAGAIN, "KV block has an invalid size")
                    if entry.pins:
                        raise OSError(errno.EAGAIN, "KV block is being loaded")
                if any(
                    self._clock() - self._entries[path].last_use >= self._ttl
                    for path in present
                ):
                    raise OSError(errno.EAGAIN, "Expired KV block still pinned by I/O")
                missing = unique - present
                reservation = len(missing) * block_size
            if reservation > self._limit:
                raise OSError(errno.ENOSPC, "KV store batch exceeds max_bytes")
            while self.accounted_bytes + reservation > self._limit:
                if not self._evict(expired_only=False, exclude=unique):
                    raise OSError(
                        errno.ENOSPC, "KV disk budget is pinned by active I/O"
                    )
            with self._lock:
                for path in present:
                    self._entries[path].pins += 1
                self._writers.update(unique)
                self._reserved += reservation
        try:
            write()
        finally:
            with self._io_lock:
                # The batch can stop after publishing only some of its blocks.
                # Include temp files if the I/O implementation's cleanup failed.
                temporary = {path + _get_tmp_suffix() for path in unique}
                sizes = {}
                for path in missing | temporary:
                    try:
                        sizes[path] = os.stat(path).st_size
                    except FileNotFoundError:
                        pass
                    except OSError:
                        # Keep a conservative charge if the filesystem cannot
                        # tell us whether the reserved bytes were published.
                        sizes[path] = block_size
                        logger.exception("Cannot inspect KV file %s", path)
                with self._lock:
                    now = self._clock()
                    for path, size in sizes.items():
                        self._entries[path] = _Entry(
                            size, float("-inf") if path in temporary else now
                        )
                        if path in temporary:
                            self._entries.move_to_end(path, last=False)
                        self._bytes += size
                    for path in present:
                        self._entries[path].pins -= 1
                    for path in present | (sizes.keys() - temporary):
                        self._touch(path, self._entries[path], now)
                    self._reserved -= reservation
                    self._writers.difference_update(unique)

    def load(self, paths: list[str], read: Callable[[], None]) -> None:
        """Pin a batch against eviction and refresh successfully read blocks."""
        unique = set(paths)
        with self._lock:
            if unique & self._writers:
                raise OSError(errno.EAGAIN, "KV block is being published")
            now = self._clock()
            if any(
                (entry := self._entries.get(path)) is None
                or now - entry.last_use >= self._ttl
                for path in unique
            ):
                raise FileNotFoundError(errno.ENOENT, "KV block expired or evicted")
            for path in unique:
                self._entries[path].pins += 1
        succeeded = 0
        failed = False
        try:
            read()
            succeeded = len(paths)
        except OSError as exc:
            succeeded = getattr(exc, "num_succeeded", 0)
            failed = True
            raise
        finally:
            with self._lock:
                now = self._clock()
                for path in paths[:succeeded]:
                    self._touch(path, self._entries[path], now)
                if failed and succeeded < len(paths):
                    path = paths[succeeded]
                    self._entries[path].last_use = float("-inf")
                    self._entries.move_to_end(path, last=False)
                    self._dirty.discard(path)
                for path in unique:
                    self._entries[path].pins -= 1

    def maintain(self) -> None:
        """Expire idle blocks, enforce the startup limit and persist last reuse."""
        with self._io_lock:
            while self._evict(expired_only=True):
                pass
            while self.accounted_bytes > self._limit:
                if not self._evict(expired_only=False):
                    break
            with self._lock:
                timestamps = [
                    (path, self._entries[path].last_use) for path in self._dirty
                ]
            for path, timestamp in timestamps:
                try:
                    os.utime(path, (timestamp, timestamp))
                except FileNotFoundError:
                    with self._lock:
                        entry = self._entries.get(path)
                        if entry is not None and not entry.pins:
                            del self._entries[path]
                            self._bytes -= entry.size
                            self._dirty.discard(path)
                    continue
                with self._lock:
                    if self._entries[path].last_use == timestamp:
                        self._dirty.discard(path)

    def _run(self) -> None:
        while not self._stop.wait(min(1.0, self._ttl)):
            try:
                self.maintain()
            except OSError:
                logger.exception("KV disk retention maintenance failed")

    def close(self) -> None:
        if self._stop.is_set():
            return
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        try:
            self.maintain()
        finally:
            self._ownership.close()
