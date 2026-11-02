# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for FileSystemTierManager.

These tests use real disk I/O to verify the filesystem tier implementation.
The tier manager writes KV cache blocks to disk and reads them back, verifying
data integrity throughout the process.
"""

import errno
import mmap
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from tests.v1.kv_offload.conftest import make_offloading_config
from vllm.v1.kv_offload.base import (
    Locality,
    LookupResult,
    Medium,
    OffloadingEvent,
    OffloadingKVEventsConfig,
    OffloadKey,
    ReqContext,
    ScheduleEndContext,
    make_offload_key,
)
from vllm.v1.kv_offload.config import (
    OffloadingCacheConfig,
    OffloadingConfig,
    OffloadingModelConfig,
    OffloadingParallelConfig,
)
from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec
from vllm.v1.kv_offload.tiering.base import TransferJob
from vllm.v1.kv_offload.tiering.factory import SecondaryTierFactory
from vllm.v1.kv_offload.tiering.fs.manager import (
    FileSystemTierManager,
)
from vllm.v1.kv_offload.tiering.fs.retention import DirectoryLock, FileRetention
from vllm.v1.kv_offload.tiering.fs.thread_pool import DualQueueThreadPool

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NUM_BLOCKS = 8
_BLOCK_ELEMENTS = 128 * mmap.PAGESIZE  # 2MB per block for pagesize 4096.
_DTYPE: torch.dtype = torch.float32
_CTX = ReqContext(req_id="test")


def _make_offloading_spec(
    enable_kv_cache_events: bool = False,
    *,
    tp_size: int = 1,
    rank: int = 0,
    world_size: int | None = None,
    replicated_layout: bool = False,
    is_parallelism_agnostic: bool = False,
) -> MagicMock:
    """Mock spec with an explicit global KV events flag."""
    if world_size is None:
        world_size = tp_size
    spec = MagicMock()
    spec.config = OffloadingConfig(
        groups=(),
        worker_kv_bytes_per_block=0,
        enable_kv_cache_events=enable_kv_cache_events,
        extra_config={},
        engine_id="test-engine",
        model=OffloadingModelConfig(name="test-model", dtype="float32"),
        cache=OffloadingCacheConfig(tokens_per_hash=16, blocks_per_chunk=1),
        parallel=OffloadingParallelConfig(
            rank=rank,
            world_size=world_size,
            tp_size=tp_size,
            pp_size=1,
            pcp_size=1,
            dcp_size=1,
            data_parallel_index=0,
            data_parallel_size=1,
            data_parallel_rank_local=None,
            is_parallelism_agnostic=is_parallelism_agnostic,
        ),
        replicated_layout=replicated_layout,
    )
    spec.blocks_per_chunk = 1
    spec.kv_events_config = OffloadingKVEventsConfig(
        enable_kv_cache_events=enable_kv_cache_events,
        self_describing_kv_events=False,
    )
    return spec


_MOCK_OFFLOADING_SPEC = _make_offloading_spec(enable_kv_cache_events=False)


def key(n: int) -> OffloadKey:
    return make_offload_key(n.to_bytes(8, "big"), 0)


def make_job(
    job_id: int,
    keys: list[OffloadKey],
    chunk_ids: list[int] | None = None,
    is_promotion: bool = False,
) -> TransferJob:
    if chunk_ids is None:
        chunk_ids = list(range(len(keys)))
    return TransferJob(
        job_id=job_id,
        keys=keys,
        chunk_ids=np.array(chunk_ids, dtype=np.int64),
        is_promotion=is_promotion,
        req_context=_CTX,
    )


def drain(tier: FileSystemTierManager) -> list:
    """Block until all in-flight jobs finish, then collect results."""
    tier.drain_jobs()
    return list(tier.get_finished_jobs())


def lookup_and_wait(
    tier: FileSystemTierManager,
    keys: list[OffloadKey],
    ctx: ReqContext = _CTX,
    timeout: float = 1.0,
) -> list[LookupResult]:
    """Perform a full async lookup cycle and return resolved results."""
    for k in keys:
        tier.lookup(k, ctx)
    tier.on_schedule_end(ScheduleEndContext(new_req_ids=[], preempted_req_ids=()))
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not tier._lookup_manager._pending_results.empty():
            break
        time.sleep(0.01)
    return [tier.lookup(k, ctx) for k in keys]


def _page_aligned_zero_tensor(
    num_blocks: int, block_elements: int, dtype: torch.dtype = _DTYPE
) -> torch.Tensor:
    page_size = mmap.PAGESIZE
    dtype_num_bytes = torch.tensor([], dtype=dtype).element_size()

    num_bytes = num_blocks * block_elements * dtype_num_bytes
    num_bytes_aligned = num_bytes + page_size
    t = torch.zeros(num_bytes_aligned, dtype=torch.uint8)

    ptr = t.data_ptr()
    alignment_offset = ptr % page_size
    # Move tensor to next page regardless.
    shift = page_size - alignment_offset
    t = t[shift : shift + num_bytes]
    return t.view(dtype).view(num_blocks, block_elements)


def _page_aligned_rand_tensor(
    num_blocks: int, block_elements: int, dtype: torch.dtype = _DTYPE
) -> torch.Tensor:
    rand_tensor = _page_aligned_zero_tensor(num_blocks, block_elements)
    rand_tensor[:] = torch.rand(num_blocks, block_elements, dtype=dtype)
    return rand_tensor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fs_tier(tmp_path):
    tensor = _page_aligned_zero_tensor(_NUM_BLOCKS, _BLOCK_ELEMENTS)
    mock_view = memoryview(tensor.numpy())
    tier = FileSystemTierManager(
        offloading_spec=_MOCK_OFFLOADING_SPEC,
        primary_kv_view=mock_view,
        tier_type="fs",
        root_dir=str(tmp_path),
        n_read_threads=4,
        n_write_threads=4,
    )
    yield tier, tensor
    tier.shutdown()


@pytest.fixture
def fs_tier_with_events(tmp_path):
    tensor = _page_aligned_zero_tensor(_NUM_BLOCKS, _BLOCK_ELEMENTS)
    mock_view = memoryview(tensor.numpy())
    tier = FileSystemTierManager(
        offloading_spec=_make_offloading_spec(enable_kv_cache_events=True),
        primary_kv_view=mock_view,
        tier_type="fs",
        root_dir=str(tmp_path),
        n_read_threads=4,
        n_write_threads=4,
        enable_kv_events=True,
        locality="LOCAL",
    )
    yield tier
    tier.shutdown()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_lookup_empty_tier(fs_tier):
    tier, _ = fs_tier
    results = lookup_and_wait(tier, [key(1), key(2)])
    assert results == [LookupResult.MISS, LookupResult.MISS]


def test_store_creates_file_and_lookup_succeeds(fs_tier):
    tier, _ = fs_tier
    job = make_job(1, [key(1)], [0])
    tier.submit_store(job)
    results = drain(tier)
    assert len(results) == 1
    assert results[0].success
    assert lookup_and_wait(tier, [key(1)]) == [LookupResult.HIT]
    dest = tier.file_mapper.get_file_name(key(1))
    assert os.path.exists(dest), f"Expected file at {dest}"


def test_store_accepts_block_published_by_another_writer(fs_tier, monkeypatch):
    """Losing the race to publish a block must not fail the store job.

    Another engine instance sharing the same root_dir can publish the block
    between the existence check and the rename, and some shared filesystems
    report that as EEXIST instead of replacing. Block file names are content
    hashes, so the file already in place holds the same data and the job has
    nothing left to do.
    """
    import vllm.v1.kv_offload.tiering.fs.io as io_mod

    # Python path only: EEXIST cannot be injected into the C extension.
    monkeypatch.setattr(io_mod, "_HAS_FSIO_C", False)
    tier, tensor = fs_tier
    dest = tier.file_mapper.get_file_name(key(1))

    tier.submit_store(make_job(1, [key(1)], [0]))
    assert all(r.success for r in drain(tier))
    with open(dest, "rb") as f:
        published = f.read()

    real_exists = os.path.exists
    real_replace = os.replace

    def racing_replace(src: str, dst: str) -> None:
        if dst == dest:
            raise FileExistsError(errno.EEXIST, "File exists", dst)
        real_replace(src, dst)

    # Hide the published block from the check, then lose the rename to it.
    monkeypatch.setattr(
        io_mod.os.path, "exists", lambda p: False if p == dest else real_exists(p)
    )
    monkeypatch.setattr(io_mod.os, "replace", racing_replace)

    # Distinct bytes, so the comparison below fails if the loser overwrites.
    tensor[1] = 1.0
    tier.submit_store(make_job(2, [key(1)], [1]))
    results = drain(tier)
    assert len(results) == 1
    assert results[0].success
    assert os.listdir(os.path.dirname(dest)) == [os.path.basename(dest)]
    with open(dest, "rb") as f:
        assert f.read() == published


def test_store_then_load_roundtrip(fs_tier):
    tier, _ = fs_tier
    job_s = make_job(1, [key(1), key(2)], [0, 1])
    tier.submit_store(job_s)
    store_results = drain(tier)
    assert all(r.success for r in store_results)

    assert lookup_and_wait(tier, [key(1), key(2)]) == [
        LookupResult.HIT,
        LookupResult.HIT,
    ]

    job_l = make_job(2, [key(1), key(2)], [2, 3], is_promotion=True)
    tier.submit_load(job_l)
    load_results = drain(tier)
    assert all(r.success for r in load_results)
    # A successful load must NOT touch the file: the delete path fires only on
    # a provable short read, so a good block stays on disk (guards against an
    # over-eager delete regressing to upstream's delete-on-any-error).
    for k in (key(1), key(2)):
        assert os.path.exists(tier.file_mapper.get_file_name(k))
    # Blocks stay on disk after load
    assert lookup_and_wait(tier, [key(1), key(2)]) == [
        LookupResult.HIT,
        LookupResult.HIT,
    ]


def test_invalid_path_raises_at_construction():
    """Construction must fail immediately when the config file cannot be written."""
    tensor = _page_aligned_zero_tensor(32, _BLOCK_ELEMENTS)
    mock_view = memoryview(tensor.numpy())

    with pytest.raises(OSError):
        FileSystemTierManager(
            offloading_spec=_MOCK_OFFLOADING_SPEC,
            primary_kv_view=mock_view,
            tier_type="fs",
            root_dir="/dev/null/invalid_path",
        )


@pytest.mark.parametrize("locality", ["local", ""])
def test_invalid_locality_raises_at_construction(tmp_path, locality):
    tensor = _page_aligned_zero_tensor(4, _BLOCK_ELEMENTS)

    with pytest.raises(ValueError, match="Locality"):
        FileSystemTierManager(
            offloading_spec=_MOCK_OFFLOADING_SPEC,
            primary_kv_view=memoryview(tensor.numpy()),
            tier_type="fs",
            root_dir=str(tmp_path),
            locality=locality,
        )


def test_factory_forwards_locality_to_fs_tier(tmp_path):
    tensor = _page_aligned_zero_tensor(4, _BLOCK_ELEMENTS)
    tier = SecondaryTierFactory.create_secondary_tier(
        {
            "type": "fs",
            "root_dir": str(tmp_path),
            "n_read_threads": 1,
            "n_write_threads": 1,
            "locality": "LOCAL",
        },
        memoryview(tensor.numpy()),
        _MOCK_OFFLOADING_SPEC,
    )
    try:
        assert isinstance(tier, FileSystemTierManager)
        assert tier.locality is Locality.LOCAL
    finally:
        tier.shutdown()


def test_failed_load_missing_file(fs_tier):
    """Test that loading a block whose file does not exist results in a failed job."""
    tier, _ = fs_tier
    job = make_job(1, [key(99)], [0], is_promotion=True)
    tier.submit_load(job)
    results = drain(tier)
    assert len(results) == 1
    assert not results[0].success


def test_multiple_jobs_tracked_independently(fs_tier):
    tier, _ = fs_tier
    job1 = make_job(1, [key(1)], [0])
    job2 = make_job(2, [key(2)], [1])
    tier.submit_store(job1)
    tier.submit_store(job2)
    results = drain(tier)
    job_ids = {r.job_id for r in results}
    assert job_ids == {1, 2}
    assert lookup_and_wait(tier, [key(1), key(2)]) == [
        LookupResult.HIT,
        LookupResult.HIT,
    ]


def test_multi_block_job_partial_failure(fs_tier):
    """A load job where one block file is missing yields a single failed JobResult."""
    tier, _ = fs_tier
    # Store two of three keys
    tier.submit_store(make_job(1, [key(10), key(11)], [0, 1]))
    assert all(r.success for r in drain(tier))

    # Load all three — key(99) was never stored
    tier.submit_load(
        make_job(2, [key(10), key(11), key(99)], [0, 1, 2], is_promotion=True)
    )
    results = drain(tier)

    assert len(results) == 1
    assert results[0].job_id == 2
    assert not results[0].success


def test_shutdown_discards_pending_tasks(fs_tier):
    """Shutdown clears both queues and stops all worker threads without draining."""
    tier, _ = fs_tier
    # Submit many tasks to ensure some remain pending
    for i in range(10):
        tier.submit_store(make_job(i, [key(i)], [i % 4]))

    # Shutdown immediately without draining
    tier.shutdown()

    # Verify queues are cleared and threads stopped
    assert len(tier._pool._load_q) == 0
    assert len(tier._pool._store_q) == 0
    assert all(not t.is_alive() for t in tier._pool._threads)


@pytest.mark.parametrize("batch_size", [0, 1, 2, 5])
@pytest.mark.parametrize("use_c_ext", [True, False])
def test_store_load_data_integrity(fs_tier, monkeypatch, use_c_ext, batch_size):
    """Data written by store must be exactly recovered by load, for batches
    of any size -- including the empty batch."""
    import vllm.v1.kv_offload.tiering.fs.io as io_mod

    if use_c_ext and not io_mod._HAS_FSIO_C:
        pytest.skip("fs_io_C extension not built")
    monkeypatch.setattr(io_mod, "_HAS_FSIO_C", use_c_ext)

    tier, tensor = fs_tier
    # Populate tensor with random data
    tensor[:] = _page_aligned_rand_tensor(_NUM_BLOCKS, _BLOCK_ELEMENTS)

    keys = [key(i) for i in range(batch_size)]
    store_chunk_ids = list(range(batch_size))
    load_chunk_ids = list(range(_NUM_BLOCKS - batch_size, _NUM_BLOCKS))
    expected = tensor[:batch_size].clone()

    tier.submit_store(make_job(1, keys, store_chunk_ids))
    store_results = drain(tier)
    assert len(store_results) == 1
    assert store_results[0].success
    assert all(os.path.exists(tier.file_mapper.get_file_name(k)) for k in keys)

    # reset tensor to prove data is read from disk
    tensor[:] = 0.0

    # Load into a range disjoint by index from the store ids, to also
    # exercise loading a chunk into a different id than it was stored from.
    tier.submit_load(make_job(2, keys, load_chunk_ids, is_promotion=True))
    load_results = drain(tier)
    assert len(load_results) == 1
    assert load_results[0].success

    for i, cid in enumerate(load_chunk_ids):
        assert torch.allclose(tensor[cid], expected[i]), (
            f"Chunk {cid} data mismatch after store+load"
        )


def test_store_load_roundtrip_without_o_direct(tmp_path, monkeypatch):
    """Buffered fallback must round-trip data when O_DIRECT is unsupported.

    Simulates filesystems (e.g. overlayfs, some NFS) that reject O_DIRECT by
    forcing the capability probe to report it unavailable.
    """
    monkeypatch.setattr(
        "vllm.v1.kv_offload.tiering.fs.manager.probe_o_direct",
        lambda _dir: False,
    )
    tensor = _page_aligned_rand_tensor(4, _BLOCK_ELEMENTS)
    tier = FileSystemTierManager(
        offloading_spec=_MOCK_OFFLOADING_SPEC,
        primary_kv_view=memoryview(tensor.numpy()),
        tier_type="fs",
        root_dir=str(tmp_path),
        n_read_threads=4,
        n_write_threads=4,
    )
    try:
        assert tier._use_o_direct is False

        keys = [key(0), key(1)]
        expected = tensor[:2].clone()
        tier.submit_store(make_job(1, keys, [0, 1]))
        assert all(r.success for r in drain(tier))

        tensor[:2] = 0.0
        tier.submit_load(make_job(2, keys, [2, 3], is_promotion=True))
        assert all(r.success for r in drain(tier))

        for i, bid in enumerate([2, 3]):
            assert torch.allclose(tensor[bid], expected[i])
    finally:
        tier.shutdown()


def test_wait_idle_blocks_until_tasks_complete():
    """wait_idle must not return while a task is still in flight."""
    pool = DualQueueThreadPool(n_read_threads=1, n_write_threads=1)
    gate = threading.Event()
    pool.enqueue_store(job_id=1, n_tasks=1, tasks=[lambda: gate.wait(timeout=5.0)])

    waiter = threading.Thread(target=pool.wait_idle)
    waiter.start()
    try:
        waiter.join(timeout=0.2)
        assert waiter.is_alive(), "wait_idle returned before task completed"
        gate.set()
        waiter.join(timeout=5.0)
        assert not waiter.is_alive(), "wait_idle did not unblock"
    finally:
        gate.set()
        pool.shutdown(wait=True)
        waiter.join(timeout=5.0)


def test_batch_lookup_c_extension(tmp_path):
    """Validates batch_lookup_C: empty, single, all-existing, all-missing,
    mixed ordering, and input type validation."""
    try:
        from vllm.fs_io_C import batch_lookup as batch_lookup_C
    except ImportError:
        pytest.skip("fs_io_C extension not built")

    # Setup
    all_exist = [str(tmp_path / f"e{i}.bin") for i in range(3)]
    for p in all_exist:
        open(p, "w").close()
    all_missing = [str(tmp_path / f"m{i}.bin") for i in range(3)]

    # Empty list
    assert batch_lookup_C([]) == []

    # Single existing / missing
    assert batch_lookup_C([all_exist[0]]) == [True]
    assert batch_lookup_C([all_missing[0]]) == [False]

    # All existing / all missing
    assert batch_lookup_C(all_exist) == [True, True, True]
    assert batch_lookup_C(all_missing) == [False, False, False]

    # Mixed — verifies index ordering is preserved
    paths = [val for pair in zip(all_exist, all_missing) for val in pair]
    assert batch_lookup_C(paths) == [True, False, True, False, True, False]

    # Input validation: non-list argument
    with pytest.raises(TypeError):
        batch_lookup_C(("/tmp/foo",))
    with pytest.raises(TypeError):
        batch_lookup_C(None)

    # Input validation: non-str elements in list
    with pytest.raises(TypeError):
        batch_lookup_C([None])
    with pytest.raises(TypeError):
        batch_lookup_C([b"/tmp/foo"])
    with pytest.raises(TypeError):
        batch_lookup_C([42])
    with pytest.raises(TypeError):
        batch_lookup_C([all_exist[0], None])  # valid first, invalid mid-list


@pytest.mark.parametrize("use_c_ext", [True, False])
def test_batch_lookup_dispatch(fs_tier, monkeypatch, use_c_ext):
    import vllm.v1.kv_offload.tiering.fs.manager as mgr_mod

    if use_c_ext and not mgr_mod._HAS_BATCH_LOOKUP_C:
        pytest.skip("fs_io_C extension not built")

    monkeypatch.setattr(mgr_mod, "_HAS_BATCH_LOOKUP_C", use_c_ext)

    tier, _ = fs_tier
    tier.submit_store(make_job(1, [key(1)], [0]))
    assert all(r.success for r in drain(tier))

    results = lookup_and_wait(tier, [key(1), key(2)])
    assert results == [LookupResult.HIT, LookupResult.MISS]


@pytest.mark.parametrize("use_c_ext", [True, False])
def test_out_of_bounds_block_id_smoke(fs_tier, monkeypatch, use_c_ext):
    """Smoke test: a block id beyond the primary tensor's block count must
    fail the job, for both the C extension and the Python fallback."""
    import vllm.v1.kv_offload.tiering.fs.io as io_mod

    if use_c_ext and not io_mod._HAS_FSIO_C:
        pytest.skip("fs_io_C extension not built")
    monkeypatch.setattr(io_mod, "_HAS_FSIO_C", use_c_ext)

    tier, tensor = fs_tier
    out_of_bounds_bid = tensor.shape[0]  # one past the last valid block

    tier.submit_store(make_job(1, [key(1)], [out_of_bounds_bid]))
    store_results = drain(tier)
    assert len(store_results) == 1
    assert not store_results[0].success

    tier.submit_load(make_job(2, [key(1)], [out_of_bounds_bid], is_promotion=True))
    load_results = drain(tier)
    assert len(load_results) == 1
    assert not load_results[0].success


@pytest.mark.parametrize("use_c_ext", [True, False])
def test_failed_load_corrects_verdict_and_removes_corrupt_file(
    fs_tier, monkeypatch, use_c_ext
):
    """Failed-load livelock regression, covering the whole contract.

    A successful promotion leaves the cached HIT and the on-disk block intact.
    A promotion that short-reads a truncated (corrupt) block fails, and in
    get_finished_jobs() the tier removes the corrupt file (stores are atomic,
    so a too-short file is genuine corruption) and marks the cached verdict
    False. The SAME request's next lookup is then a MISS served from cache with
    NO re-probe, so the scheduler cannot re-issue the doomed promotion.
    """
    import vllm.v1.kv_offload.tiering.fs.io as io_mod

    if use_c_ext and not io_mod._HAS_FSIO_C:
        pytest.skip("fs_io_C extension not built")
    monkeypatch.setattr(io_mod, "_HAS_FSIO_C", use_c_ext)

    tier, _ = fs_tier
    tier.submit_store(make_job(1, [key(1)], [0]))
    assert all(r.success for r in drain(tier))
    path = tier.file_mapper.get_file_name(key(1))

    ctx = ReqContext(req_id="livelock-req")
    assert lookup_and_wait(tier, [key(1)], ctx=ctx) == [LookupResult.HIT]

    # A successful promotion must NOT touch the verdict or the file.
    tier.submit_load(make_job(2, [key(1)], [0], is_promotion=True))
    results = drain(tier)
    assert len(results) == 1 and results[0].success
    assert tier.lookup(key(1), ctx) == LookupResult.HIT
    assert os.path.exists(path)

    # Truncate below block_size so the next promotion short-reads.
    with open(path, "wb") as f:
        f.write(b"x" * 10)
    tier.submit_load(make_job(3, [key(1)], [0], is_promotion=True))
    results = drain(tier)  # get_finished_jobs() marks the verdict False here
    assert len(results) == 1 and not results[0].success

    # Corrupt file removed; the SAME request now misses from cache, no re-probe.
    assert not os.path.exists(path)
    lm = tier._lookup_manager
    assert tier.lookup(key(1), ctx) == LookupResult.MISS
    assert lm._lookup_batch == []

    # A FRESH request re-probes the tier (no cached verdict) and misses too,
    # since the corrupt file is gone -- the real batch_lookup re-probe path.
    fresh = ReqContext(req_id="fresh-after-short-read")
    assert lookup_and_wait(tier, [key(1)], ctx=fresh) == [LookupResult.MISS]


def test_failed_load_after_abort_does_not_break_newer_probe(fs_tier):
    """A promotion for request A fails after A was aborted and request B
    re-probed the same key. The failure must not flip B's in-flight probe:
    B's next lookup drains its own result (MISS: the file is gone) instead of
    asserting on the scheduler thread."""
    tier, _ = fs_tier
    k = key(1)
    tier.submit_store(make_job(1, [k], [0]))
    assert all(r.success for r in drain(tier))
    ctx_a, ctx_b = ReqContext(req_id="A"), ReqContext(req_id="B")
    assert lookup_and_wait(tier, [k], ctx=ctx_a) == [LookupResult.HIT]

    os.remove(tier.file_mapper.get_file_name(k))
    tier.submit_load(make_job(2, [k], [1], is_promotion=True))
    tier.on_request_finished(ctx_a)

    assert tier.lookup(k, ctx_b) == LookupResult.RETRY
    tier.on_schedule_end(ScheduleEndContext(new_req_ids=[], preempted_req_ids=()))
    deadline = time.monotonic() + 1.0
    while tier._lookup_manager._pending_results.empty():
        assert time.monotonic() < deadline
        time.sleep(0.01)

    assert [r.success for r in drain(tier)] == [False]
    assert tier.lookup(k, ctx_b) == LookupResult.MISS


@pytest.mark.parametrize("use_c_ext", [True, False])
def test_batched_partial_load_failure_keeps_loaded_blocks(
    fs_tier, monkeypatch, use_c_ext
):
    """A batched promotion stops at the first bad block and reports how many
    loaded before it (#50321). Corrupt the LAST block: the earlier blocks load
    fine, so the job reports successful_keys for them and marks only the failed
    tail a miss. The earlier keys stay HIT — including for the same request —
    while the corrupt block stays a MISS (its file was removed)."""
    import vllm.v1.kv_offload.tiering.fs.io as io_mod

    if use_c_ext and not io_mod._HAS_FSIO_C:
        pytest.skip("fs_io_C extension not built")
    monkeypatch.setattr(io_mod, "_HAS_FSIO_C", use_c_ext)

    tier, _ = fs_tier
    keys = [key(1), key(2), key(3)]  # last one is the "bad" block
    tier.submit_store(make_job(1, keys, [0, 1, 2]))
    assert all(r.success for r in drain(tier))
    bad_path = tier.file_mapper.get_file_name(key(3))

    ctx = ReqContext(req_id="batch-req")
    assert lookup_and_wait(tier, keys, ctx=ctx) == [LookupResult.HIT] * 3

    # Corrupt only the last block, then load the whole batch as one job.
    with open(bad_path, "wb") as f:
        f.write(b"x" * 10)
    tier.submit_load(make_job(2, keys, [0, 1, 2], is_promotion=True))
    results = drain(tier)
    # (a) the job fails but reports the two blocks that loaded before the bad one.
    assert len(results) == 1 and not results[0].success
    assert tuple(results[0].successful_keys) == (key(1), key(2))

    # (b) Only the failed tail is a miss; the loaded blocks stay HIT on the same
    # request, and nothing was re-probed.
    lm = tier._lookup_manager
    assert [tier.lookup(k, ctx) for k in keys] == [
        LookupResult.HIT,
        LookupResult.HIT,
        LookupResult.MISS,
    ]
    assert lm._lookup_batch == []

    # (c) A fresh request re-probes: the loaded blocks are still on disk (HIT),
    # only the corrupt block was removed (MISS).
    tier.on_request_finished(ctx)
    fresh = ReqContext(req_id="fresh-batch-req")
    assert lookup_and_wait(tier, keys, ctx=fresh) == [
        LookupResult.HIT,
        LookupResult.HIT,
        LookupResult.MISS,
    ]


@pytest.mark.parametrize("use_c_ext", [True, False])
def test_batched_load_first_block_fails_marks_whole_batch(
    fs_tier, monkeypatch, use_c_ext
):
    """When the FIRST block fails, nothing loaded before it: the job reports no
    successful_keys (None) and the whole batch is marked a miss for the
    request."""
    import vllm.v1.kv_offload.tiering.fs.io as io_mod

    if use_c_ext and not io_mod._HAS_FSIO_C:
        pytest.skip("fs_io_C extension not built")
    monkeypatch.setattr(io_mod, "_HAS_FSIO_C", use_c_ext)

    tier, _ = fs_tier
    keys = [key(1), key(2), key(3)]  # first one is the "bad" block
    tier.submit_store(make_job(1, keys, [0, 1, 2]))
    assert all(r.success for r in drain(tier))

    ctx = ReqContext(req_id="batch-first-fail")
    assert lookup_and_wait(tier, keys, ctx=ctx) == [LookupResult.HIT] * 3

    with open(tier.file_mapper.get_file_name(key(1)), "wb") as f:
        f.write(b"x" * 10)
    tier.submit_load(make_job(2, keys, [0, 1, 2], is_promotion=True))
    results = drain(tier)
    assert len(results) == 1 and not results[0].success
    # Nothing loaded before the failure -> no partial success reported.
    assert results[0].successful_keys is None
    # The whole batch is a miss for this request.
    assert [tier.lookup(k, ctx) for k in keys] == [LookupResult.MISS] * 3


@pytest.mark.parametrize("use_c_ext", [True, False])
def test_transient_load_failure_leaves_file(fs_tier, monkeypatch, use_c_ext):
    """A transient host error (here ELOOP on open) is NOT a short read: the job
    fails but the block file must survive untouched, on both the C and Python
    paths. Deleting on a transient error would turn a passing hiccup into
    permanent data loss."""
    import vllm.v1.kv_offload.tiering.fs.io as io_mod

    if use_c_ext and not io_mod._HAS_FSIO_C:
        pytest.skip("fs_io_C extension not built")
    monkeypatch.setattr(io_mod, "_HAS_FSIO_C", use_c_ext)

    tier, _ = fs_tier
    tier.submit_store(make_job(1, [key(1)], [0]))
    assert all(r.success for r in drain(tier))
    path = tier.file_mapper.get_file_name(key(1))
    with open(path, "rb") as f:
        original = f.read()

    # Make open() fail with ELOOP (fd < 0) without truncating the block. Not
    # chmod 000: CI runs as root, which bypasses permission bits, so open()
    # would succeed and the load would not fail at all.
    saved = path + ".saved"
    loop = path + ".loop"
    os.rename(path, saved)
    os.symlink(loop, path)
    os.symlink(path, loop)

    tier.submit_load(make_job(2, [key(1)], [0], is_promotion=True))
    results = drain(tier)
    assert len(results) == 1 and not results[0].success

    # The path is left alone: a non-short-read error must not unlink.
    assert os.path.lexists(path)

    os.unlink(path)
    os.unlink(loop)
    os.rename(saved, path)
    with open(path, "rb") as f:
        assert f.read() == original


# ---------------------------------------------------------------------------
# KV events
# ---------------------------------------------------------------------------


def test_successful_store_emits_stored_event(fs_tier_with_events):
    """A completed store job emits one stored event with the job's keys."""
    tier = fs_tier_with_events
    keys = [key(1), key(2)]
    tier.submit_store(make_job(1, keys, [0, 1]))
    assert all(r.success for r in drain(tier))

    events = list(tier.take_events())
    assert len(events) == 1
    assert events[0].keys == keys
    assert events[0].medium == Medium.STORAGE
    assert events[0].locality is Locality.LOCAL
    assert not events[0].removed
    # take_events drains the buffer.
    assert list(tier.take_events()) == []


@pytest.mark.parametrize(
    ("locality", "expected"),
    [(None, None), ("REMOTE", Locality.REMOTE)],
)
def test_store_event_uses_configured_locality(tmp_path, locality, expected):
    tensor = _page_aligned_zero_tensor(4, _BLOCK_ELEMENTS)
    locality_config = {} if locality is None else {"locality": locality}
    tier = FileSystemTierManager(
        offloading_spec=_make_offloading_spec(enable_kv_cache_events=True),
        primary_kv_view=memoryview(tensor.numpy()),
        tier_type="fs",
        root_dir=str(tmp_path),
        enable_kv_events=True,
        **locality_config,
    )
    try:
        tier.submit_store(make_job(1, [key(1)], [0]))
        assert all(r.success for r in drain(tier))

        events = list(tier.take_events())
        assert len(events) == 1
        assert events[0].locality is expected
    finally:
        tier.shutdown()


def test_load_job_emits_no_event(fs_tier_with_events):
    tier = fs_tier_with_events
    tier.submit_store(make_job(1, [key(1)], [0]))
    results = drain(tier)
    assert len(results) == 1
    assert results[0].success
    list(tier.take_events())

    tier.submit_load(make_job(2, [key(1)], [1], is_promotion=True))
    results = drain(tier)
    assert len(results) == 1
    assert results[0].success
    assert list(tier.take_events()) == []


def test_mixed_job_results_emit_event_only_for_successful_job(
    fs_tier_with_events, monkeypatch
):
    """With a failed and a successful store job in flight, exactly one event
    is emitted and its keys belong to the successful job."""
    import vllm.v1.kv_offload.tiering.fs.manager as mgr_mod

    tier = fs_tier_with_events
    failing_path = tier.file_mapper.get_file_name(key(1))
    original_batch_store_block = mgr_mod.batch_store_block

    def flaky_batch_store_block(paths, *args, **kwargs):
        if failing_path in paths:
            raise OSError("injected store failure")
        return original_batch_store_block(paths, *args, **kwargs)

    monkeypatch.setattr(mgr_mod, "batch_store_block", flaky_batch_store_block)

    tier.submit_store(make_job(1, [key(1)], [0]))
    tier.submit_store(make_job(2, [key(2)], [1]))
    results = drain(tier)
    assert len(results) == 2
    by_id = {r.job_id: r for r in results}
    assert not by_id[1].success
    assert by_id[2].success

    events = list(tier.take_events())
    assert len(events) == 1
    assert events[0].keys == [key(2)]


def test_partially_failed_store_emits_no_event(fs_tier_with_events, monkeypatch):
    """A store job with any failed block emits no event for the whole job."""
    import vllm.v1.kv_offload.tiering.fs.manager as mgr_mod

    tier = fs_tier_with_events
    failing_path = tier.file_mapper.get_file_name(key(2))
    original_batch_store_block = mgr_mod.batch_store_block

    def flaky_batch_store_block(paths, *args, **kwargs):
        if failing_path in paths:
            raise OSError("injected store failure")
        return original_batch_store_block(paths, *args, **kwargs)

    monkeypatch.setattr(mgr_mod, "batch_store_block", flaky_batch_store_block)

    tier.submit_store(make_job(1, [key(1), key(2)], [0, 1]))
    results = drain(tier)
    assert len(results) == 1
    assert not results[0].success
    assert list(tier.take_events()) == []
    assert tier._store_job_keys == {}


def test_events_disabled_by_default(fs_tier):
    tier, _ = fs_tier
    tier.submit_store(make_job(1, [key(1)], [0]))
    results = drain(tier)
    assert len(results) == 1
    assert results[0].success
    assert tier.events is None
    assert tier._store_job_keys == {}
    assert list(tier.take_events()) == []


def test_events_require_global_kv_events_flag(tmp_path):
    """Tier-level opt-in alone is not enough; the global flag gates events."""
    tensor = _page_aligned_zero_tensor(4, _BLOCK_ELEMENTS)
    tier = FileSystemTierManager(
        offloading_spec=_make_offloading_spec(enable_kv_cache_events=False),
        primary_kv_view=memoryview(tensor.numpy()),
        tier_type="fs",
        root_dir=str(tmp_path),
        enable_kv_events=True,
    )
    try:
        assert tier.events is None
        tier.submit_store(make_job(1, [key(1)], [0]))
        results = drain(tier)
        assert len(results) == 1
        assert results[0].success
        assert list(tier.take_events()) == []
        assert tier._store_job_keys == {}
    finally:
        tier.shutdown()


def test_cascade_store_emits_fs_event_through_tiering_manager(tmp_path):
    """A GPU->CPU->fs cascade surfaces the tier-owned FS stored event via the
    TieringOffloadingManager's aggregated take_events()."""
    from vllm.v1.kv_offload.tiering.manager import (
        CPUPrimaryTierOffloadingManager,
        TieringOffloadingManager,
    )

    tensor = _page_aligned_zero_tensor(4, _BLOCK_ELEMENTS)
    view = memoryview(tensor.numpy())
    mock_region = MagicMock()
    mock_region.create_kv_memoryview.return_value = view
    primary = CPUPrimaryTierOffloadingManager(num_chunks=4, mmap_region=mock_region)
    tier = FileSystemTierManager(
        offloading_spec=_make_offloading_spec(enable_kv_cache_events=True),
        primary_kv_view=primary.get_kv_memoryview(),
        tier_type="fs",
        root_dir=str(tmp_path),
        enable_kv_events=True,
    )
    manager = TieringOffloadingManager(primary_tier=primary, secondary_tiers=[tier])
    try:
        keys = [key(1), key(2)]
        manager.on_new_request(_CTX)
        assert manager.prepare_store(keys, _CTX) is not None
        manager.complete_store(keys, _CTX)  # cascades to the fs tier

        events: list[OffloadingEvent] = []
        ctx = ScheduleEndContext(new_req_ids=[], preempted_req_ids=())
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and not events:
            manager.on_schedule_end(ctx)
            events.extend(manager.take_events())
            time.sleep(0.01)

        fs_events = [e for e in events if e.medium == Medium.STORAGE]
        assert len(fs_events) == 1
        assert set(fs_events[0].keys) == set(keys)
        assert not fs_events[0].removed
    finally:
        tier.shutdown()


def test_fs_tier_cross_tp_round_trip(tmp_path):
    """TP=2 replicated writer and TP=4 reader share namespace and bytes."""
    root = str(tmp_path)
    writer_tensor = _page_aligned_rand_tensor(4, _BLOCK_ELEMENTS)
    expected = writer_tensor[0].clone()
    writer = FileSystemTierManager(
        offloading_spec=_make_offloading_spec(
            tp_size=2, world_size=2, rank=0, replicated_layout=True
        ),
        primary_kv_view=memoryview(writer_tensor.numpy()),
        tier_type="fs",
        root_dir=root,
        n_read_threads=2,
        n_write_threads=2,
    )
    try:
        writer.submit_store(make_job(1, [key(7)], [0]))
        assert all(r.success for r in drain(writer))
        writer_base = writer.file_mapper.base_path
        writer_path = writer.file_mapper.get_file_name(key(7))
    finally:
        writer.shutdown()

    reader_tensor = _page_aligned_zero_tensor(4, _BLOCK_ELEMENTS)
    reader = FileSystemTierManager(
        offloading_spec=_make_offloading_spec(
            tp_size=4, world_size=4, rank=3, replicated_layout=True
        ),
        primary_kv_view=memoryview(reader_tensor.numpy()),
        tier_type="fs",
        root_dir=root,
        n_read_threads=2,
        n_write_threads=2,
    )
    try:
        assert reader.file_mapper.base_path == writer_base
        assert reader.file_mapper.get_file_name(key(7)) == writer_path
        assert lookup_and_wait(reader, [key(7)]) == [LookupResult.HIT]
        reader.submit_load(make_job(2, [key(7)], [1], is_promotion=True))
        assert all(r.success for r in drain(reader))
        assert torch.allclose(reader_tensor[1], expected)
    finally:
        reader.shutdown()


@pytest.fixture
def retention(tmp_path):
    # An explicit clock makes idle expiry deterministic without sleeping.
    now = [time.time()]
    policy = FileRetention(str(tmp_path), 10, 8, clock=lambda: now[0])
    try:
        yield policy, now
    finally:
        policy.close()


def store_bytes(policy: FileRetention, path: Path, value: bytes) -> None:
    def write() -> None:
        path.write_bytes(value)

    policy.store([str(path)], len(value), write)


def test_retention_refreshes_reuse_and_survives_restart(tmp_path):
    now = [time.time()]
    path = tmp_path / "aa.bin"
    policy = FileRetention(str(tmp_path), 10, 8, clock=lambda: now[0])
    try:
        store_bytes(policy, path, b"data")
        now[0] += 9
        policy.touch([str(path)])
        policy.maintain()
        assert path.stat().st_mtime == pytest.approx(now[0])
    finally:
        policy.close()
    policy = FileRetention(str(tmp_path), 10, 8, clock=lambda: now[0])
    try:
        now[0] += 9
        assert policy.contains(str(path))
        # Checking availability must not extend the TTL.
        now[0] += 1
        assert not policy.contains(str(path))
        policy.maintain()
        assert not path.exists()
        assert policy.accounted_bytes == 0
    finally:
        policy.close()


def test_retention_evicts_lru_and_counts_partial_publication(retention, tmp_path):
    policy, now = retention
    a, b, c = [tmp_path / f"{letter}.bin" for letter in "abc"]
    store_bytes(policy, a, b"aaaa")
    now[0] += 1
    store_bytes(policy, b, b"bbbb")
    now[0] += 1
    policy.touch([str(a)])
    store_bytes(policy, c, b"cccc")
    assert a.read_bytes() == b"aaaa"
    assert not b.exists()
    assert c.read_bytes() == b"cccc"
    assert policy.accounted_bytes == 8
    # The first file can be published before a later write fails.
    now[0] += 11
    policy.maintain()

    def partial_write():
        a.write_bytes(b"aaaa")
        raise OSError(errno.EIO, "write failed")

    with pytest.raises(OSError, match="write failed"):
        policy.store([str(a), str(b)], 4, partial_write)
    assert policy.accounted_bytes == 4
    assert policy.contains(str(a))
    assert not policy.contains(str(b))
    store_bytes(policy, b, b"bbbb")
    assert policy.accounted_bytes == 8


def test_retention_reserves_concurrent_writes_before_io(retention, tmp_path):
    policy, _ = retention
    entered, release = threading.Event(), threading.Event()
    a, b = tmp_path / "a.bin", tmp_path / "b.bin"

    def pending_write():
        entered.set()
        assert release.wait(5)
        a.write_bytes(b"12345678")

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(policy.store, [str(a)], 8, pending_write)
        try:
            assert entered.wait(5)
            assert policy.accounted_bytes == 8
            assert not a.exists()
            with pytest.raises(OSError) as error:
                store_bytes(policy, b, b"bbbb")
            assert error.value.errno == errno.ENOSPC
            assert not b.exists()
            with pytest.raises(OSError) as error:
                store_bytes(policy, a, b"12345678")
            assert error.value.errno == errno.EAGAIN
            with pytest.raises(OSError) as error:
                policy.load([str(a)], lambda: a.read_bytes())
            assert error.value.errno == errno.EAGAIN
        finally:
            release.set()
        future.result(timeout=5)
    assert a.read_bytes() == b"12345678"
    assert policy.accounted_bytes == 8


def test_retention_pins_loads_and_refreshes_only_successful_reads(retention, tmp_path):
    policy, now = retention
    a, b = tmp_path / "a.bin", tmp_path / "b.bin"
    store_bytes(policy, a, b"12345678")
    entered, release = threading.Event(), threading.Event()

    def pending_read():
        entered.set()
        assert release.wait(5)
        assert a.read_bytes() == b"12345678"

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(policy.load, [str(a)], pending_read)
        try:
            assert entered.wait(5)
            with pytest.raises(OSError) as error:
                store_bytes(policy, a, b"12345678")
            assert error.value.errno == errno.EAGAIN
            now[0] += 11
            policy.maintain()
            assert a.exists()
            with pytest.raises(OSError) as error:
                store_bytes(policy, b, b"bbbb")
            assert error.value.errno == errno.ENOSPC
        finally:
            release.set()
        future.result(timeout=5)
    assert policy.contains(str(a))
    now[0] += 9

    def failed_read():
        raise OSError(errno.EIO, "read failed")

    with pytest.raises(OSError, match="read failed"):
        policy.load([str(a)], failed_read)
    now[0] += 1
    assert not policy.contains(str(a))
    policy.maintain()
    assert not a.exists()


def test_retention_startup_cleans_crash_files_and_enforces_budget(tmp_path):
    now = time.time()
    for index in range(3):
        path = tmp_path / f"{index}.bin"
        path.write_bytes(b"data")
        os.utime(path, (now - index, now - index))
    abandoned = tmp_path / "abc.bin_123.tmp"
    abandoned.write_bytes(b"partial")
    unrelated = tmp_path / "keep.tmp"
    unrelated.write_bytes(b"unrelated")
    policy = FileRetention(str(tmp_path), 10, 8)
    try:
        assert not abandoned.exists()
        assert unrelated.read_bytes() == b"unrelated"
        assert not (tmp_path / "2.bin").exists()
        assert policy.accounted_bytes == 8
        with pytest.raises(RuntimeError, match="exclusive ownership"):
            FileRetention(str(tmp_path), 10, 8)
        with pytest.raises(RuntimeError, match="exclusive ownership"):
            DirectoryLock(str(tmp_path), exclusive=False)
    finally:
        policy.close()
    # Default, unbounded instances can still share with each other.
    first = DirectoryLock(str(tmp_path), exclusive=False)
    second = DirectoryLock(str(tmp_path), exclusive=False)
    try:
        with pytest.raises(RuntimeError, match="exclusive ownership"):
            FileRetention(str(tmp_path), 10, 8)
    finally:
        second.close()
        first.close()


def test_fs_namespace_isolates_versions_with_one_disk_budget(tmp_path):
    """A replacement cannot load stale KV, but must account for its disk space."""
    spec = CPUOffloadingSpec(make_offloading_config())
    buffer = mmap.mmap(-1, mmap.PAGESIZE)
    view = memoryview(buffer).cast("B", shape=[1, mmap.PAGESIZE])
    try:
        for namespace, expected_hit in [
            ("weights-a/image-a", False),
            ("weights-a/image-a", True),
            ("weights-a/image-b", False),
            ("weights-a/image-a", False),
        ]:
            tier = SecondaryTierFactory.create_secondary_tier(
                {
                    "type": "fs",
                    "root_dir": str(tmp_path),
                    "namespace": namespace,
                    "idle_ttl_seconds": 3600,
                    "n_read_threads": 1,
                    "n_write_threads": 1,
                    "max_bytes": mmap.PAGESIZE,
                },
                view,
                spec,
            )
            try:
                assert lookup_and_wait(tier, [key(1)]) == [
                    LookupResult.HIT if expected_hit else LookupResult.MISS
                ]
                tier.submit_store(make_job(1, [key(1)], [0]))
                assert [result.success for result in drain(tier)] == [True]
                assert (
                    sum(p.stat().st_size for p in tmp_path.rglob("*.bin"))
                    == mmap.PAGESIZE
                )
            finally:
                tier.shutdown()
    finally:
        view.release()
        buffer.close()


def test_fs_retention_budget_preserves_data_and_restarts(tmp_path):
    """Exercise the real tier, worker queues and batched filesystem I/O."""
    spec = CPUOffloadingSpec(make_offloading_config())
    buffer = mmap.mmap(-1, mmap.PAGESIZE * 4)
    view = memoryview(buffer).cast("B", shape=[4, mmap.PAGESIZE])
    flat = view.cast("B")
    flat[: mmap.PAGESIZE] = b"a" * mmap.PAGESIZE
    flat[mmap.PAGESIZE : 2 * mmap.PAGESIZE] = b"b" * mmap.PAGESIZE
    args = dict(
        offloading_spec=spec,
        primary_kv_view=view,
        tier_type="fs",
        root_dir=str(tmp_path),
        n_read_threads=1,
        n_write_threads=1,
        idle_ttl_seconds=3600,
        max_bytes=mmap.PAGESIZE,
    )
    tier = FileSystemTierManager(**args)
    try:
        tier.submit_store(make_job(1, [key(1)], [0]))
        assert [result.success for result in drain(tier)] == [True]
        first = Path(tier.file_mapper.get_file_name(key(1)))
        assert first.read_bytes() == b"a" * mmap.PAGESIZE
        tier.submit_store(make_job(2, [key(2)], [1]))
        assert [result.success for result in drain(tier)] == [True]
        assert not first.exists()
        assert tier.lookup(key(1), _CTX) is LookupResult.MISS
        assert (
            sum(path.stat().st_size for path in tmp_path.rglob("*.bin"))
            <= mmap.PAGESIZE
        )
    finally:
        tier.shutdown()
    tier = FileSystemTierManager(**args)
    try:
        tier.submit_load(make_job(3, [key(2)], [2], is_promotion=True))
        assert [result.success for result in drain(tier)] == [True]
        actual = bytes(flat[2 * mmap.PAGESIZE : 3 * mmap.PAGESIZE])
        assert actual == b"b" * mmap.PAGESIZE
    finally:
        tier.shutdown()
        flat.release()
        view.release()
        buffer.close()


def test_fs_retention_real_generation_reuses_then_expires(
    tmp_path, vllm_runner, monkeypatch
):
    """A disk hit survives clearing faster tiers; expired data is recomputed."""
    from vllm import SamplingParams, TokensPrompt
    from vllm.config import KVTransferConfig

    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    limit = 16 << 20
    config = KVTransferConfig(
        kv_connector="OffloadingConnector",
        kv_role="kv_both",
        kv_connector_extra_config={
            "cpu_bytes_to_use": 256 << 20,
            "block_size": 64,
            "spec_name": "TieringOffloadingSpec",
            "secondary_tiers": [
                {
                    "type": "fs",
                    "root_dir": str(tmp_path),
                    "idle_ttl_seconds": 30,
                    "max_bytes": limit,
                }
            ],
        },
    )
    prompt = TokensPrompt(prompt_token_ids=list(range(100, 356)) + [7])
    sampling = SamplingParams(temperature=0, max_tokens=1)
    with vllm_runner(
        "facebook/opt-125m",
        max_model_len=1024,
        gpu_memory_utilization=0.1,
        enforce_eager=True,
        kv_transfer_config=config,
    ) as runner:
        llm = runner.llm

        def wait_for(condition, timeout=60):
            deadline = time.monotonic() + timeout
            while not condition():
                assert time.monotonic() < deadline, "filesystem tier did not settle"
                llm.llm_engine.step()
                threading.Event().wait(0.02)

        initial = llm.generate(prompt, sampling, use_tqdm=False)[0]
        wait_for(lambda: len(list(tmp_path.rglob("*.bin"))) >= 4)
        paths = list(tmp_path.rglob("*.bin"))
        assert sum(path.stat().st_size for path in paths) <= limit
        latest = max(path.stat().st_mtime for path in paths)

        # This repeat stays in GPU memory. It must still refresh disk lifetime.
        gpu_hit = llm.generate(prompt, sampling, use_tqdm=False)[0]
        assert gpu_hit.num_cached_tokens == 256
        wait_for(lambda: all(path.stat().st_mtime > latest for path in paths))

        # Clearing only GPU KV leaves the primary CPU copies available.
        latest = max(path.stat().st_mtime for path in paths)
        assert llm.reset_prefix_cache()
        cpu_hit = llm.generate(prompt, sampling, use_tqdm=False)[0]
        assert cpu_hit.num_cached_tokens == 256
        assert cpu_hit.outputs[0].token_ids == initial.outputs[0].token_ids
        wait_for(lambda: all(path.stat().st_mtime > latest for path in paths))

        assert llm.reset_prefix_cache(reset_connector=True)
        llm.llm_engine.step()
        disk_hit = llm.generate(prompt, sampling, use_tqdm=False)[0]
        assert disk_hit.num_cached_tokens == 256
        assert disk_hit.outputs[0].token_ids == initial.outputs[0].token_ids

        wait_for(lambda: not list(tmp_path.rglob("*.bin")), timeout=35)
        assert llm.reset_prefix_cache(reset_connector=True)
        llm.llm_engine.step()
        expired = llm.generate(prompt, sampling, use_tqdm=False)[0]
        assert expired.num_cached_tokens == 0
        assert expired.outputs[0].token_ids == initial.outputs[0].token_ids


def test_retention_duplicate_store_refreshes_without_rewriting(retention, tmp_path):
    policy, now = retention
    path = tmp_path / "a.bin"
    store_bytes(policy, path, b"data")
    now[0] += 9
    # The real batch writer also returns immediately for an existing block.
    policy.store([str(path)], 4, lambda: None)
    now[0] += 9
    assert policy.contains(str(path))
    assert policy.accounted_bytes == 4
    assert path.read_bytes() == b"data"
    now[0] += 1
    policy.maintain()
    assert not path.exists()


def test_retention_accounts_abandoned_temporary_write(retention, tmp_path):
    from vllm.v1.kv_offload.tiering.fs.io import _get_tmp_suffix

    policy, _ = retention
    path = tmp_path / "a.bin"
    temporary = Path(str(path) + _get_tmp_suffix())

    def failed_write():
        temporary.write_bytes(b"part")
        raise OSError(errno.EIO, "publication and cleanup failed")

    with pytest.raises(OSError, match="publication and cleanup failed"):
        policy.store([str(path)], 8, failed_write)
    assert policy.accounted_bytes == 4
    assert not policy.contains(str(path))
    policy.maintain()
    assert not temporary.exists()
    assert policy.accounted_bytes == 0


@pytest.mark.parametrize("ttl,limit", [(0, 8), (float("nan"), 8), (10, 0), (10, 1.5)])
def test_retention_rejects_invalid_limits_without_holding_directory(
    tmp_path, ttl, limit
):
    with pytest.raises(ValueError):
        FileRetention(str(tmp_path), ttl, limit)
    owner = DirectoryLock(str(tmp_path), exclusive=True)
    owner.close()


def test_retention_does_not_reuse_corrupt_size_for_write_reservation(tmp_path):
    path = tmp_path / "a.bin"
    path.write_bytes(b"bad")
    policy = FileRetention(str(tmp_path), 10, 8)
    try:
        with pytest.raises(OSError, match="invalid size"):
            store_bytes(policy, path, b"12345678")
        assert policy.accounted_bytes == 3
        assert not policy.contains(str(path))
        policy.maintain()
        store_bytes(policy, path, b"12345678")
        assert policy.accounted_bytes == 8
        assert path.read_bytes() == b"12345678"
    finally:
        policy.close()
