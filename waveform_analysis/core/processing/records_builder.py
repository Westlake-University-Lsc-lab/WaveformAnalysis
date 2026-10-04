"""
Records + wave_pool utilities.

This module separates event metadata from waveform samples so we can keep
fast indexing while reducing the memory cost of large variable-length waves.

Core idea:
- `records` stores per-event metadata such as time, channel, length, and
    offset.
- `wave_pool` stores all waveform samples in one contiguous array.
- `wave_offset` + `event_length` let us jump from a record directly to the
    matching slice inside `wave_pool`.

Why this design helps:
- It preserves efficient event-level queries without allocating one array per
    waveform.
- It works well for streaming pipelines, shard merging, and disk-backed
    workflows.
"""

from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
import tempfile
import time

import numpy as np

from waveform_analysis.core.foundation.utils import exporter
from waveform_analysis.core.hardware.channel import (
    HardwareChannel,
    group_indices_by_hardware_channel,
)
from waveform_analysis.core.processing.dtypes import (
    EVENTS_DTYPE as _EVENTS_DTYPE,
)
from waveform_analysis.core.processing.dtypes import (
    RECORDS_DTYPE as _RECORDS_DTYPE,
)
from waveform_analysis.core.utils.baseline import (
    normalize_baseline_samples as _normalize_baseline_samples,
)
from waveform_analysis.core.utils.baseline import (
    validate_baseline_samples as _validate_baseline_samples,
)

export, __all__ = exporter()

RECORDS_DTYPE = export(_RECORDS_DTYPE, name="RECORDS_DTYPE")
EVENTS_DTYPE = export(_EVENTS_DTYPE, name="EVENTS_DTYPE")
_MAX_V1725_IN_MEMORY_WAVES = 10_000


@export
@dataclass
class RecordsBundle:
    records: np.ndarray
    wave_pool: np.ndarray


EventsBundle = export(RecordsBundle, name="EventsBundle")


@dataclass
class _RecordsPartRef:
    records_path: Path
    wave_pool_path: Path
    n_records: int
    n_samples: int
    time_range: tuple[int, int] | None = None  # (start_time, end_time) for filtering


@export
@dataclass
class RecordsBundleRef:
    """
    磁盘引用式 RecordsBundle，支持流式处理。

    用于处理超大数据集（>50GB），避免将所有数据加载到内存。
    数据保存在磁盘上，通过 memmap 按需加载。

    Attributes:
        part_refs: 分片引用列表
        total_records: 总记录数
        total_samples: 总样本数
        temp_dir: 临时目录路径（用于清理）

    Examples:
        >>> # 分块迭代
        >>> for chunk in bundle_ref.iter_chunks(chunk_size=100_000):
        ...     process(chunk.records, chunk.wave_pool)

        >>> # 时间范围查询
        >>> for chunk in bundle_ref.iter_chunks(time_range=(start, end)):
        ...     process(chunk)

        >>> # 只读取元数据
        >>> records_view = bundle_ref.get_records_view()
        >>> print(f"Total: {len(records_view)}")
    """

    part_refs: list[_RecordsPartRef]
    total_records: int
    total_samples: int
    temp_dir: Path | None = None

    def __post_init__(self):
        """按时间排序分片（如果有时间范围信息）"""
        if self.part_refs and any(p.time_range for p in self.part_refs):
            self.part_refs.sort(key=lambda p: p.time_range[0] if p.time_range else 0)

    @property
    def dtype(self):
        """兼容性：返回 RECORDS_DTYPE"""
        return RECORDS_DTYPE

    def __len__(self):
        """兼容性：返回总记录数"""
        return self.total_records

    def iter_chunks(
        self,
        chunk_size: int = 100_000,
        time_range: tuple[int, int] | None = None,
    ):
        """
        流式迭代分块数据。

        Args:
            chunk_size: 每块的记录数
            time_range: 可选的时间范围过滤 (start_time, end_time)

        Yields:
            RecordsBundle 分块

        Memory:
            单个 chunk 约 200MB (100k events × 1k samples × 2 bytes)
        """

        for part_ref in self.part_refs:
            # 时间范围过滤（粗粒度）
            if time_range and part_ref.time_range:
                start_time, end_time = time_range
                part_start, part_end = part_ref.time_range
                if part_end < start_time or part_start > end_time:
                    continue  # 跳过不相交的分片

            # 加载分片（memmap，不占用内存）
            part_records = np.memmap(
                part_ref.records_path, dtype=RECORDS_DTYPE, mode="r", shape=(part_ref.n_records,)
            )
            part_waves = np.memmap(
                part_ref.wave_pool_path, dtype=np.uint16, mode="r", shape=(part_ref.n_samples,)
            )

            # 分块迭代
            for start_idx in range(0, part_ref.n_records, chunk_size):
                end_idx = min(start_idx + chunk_size, part_ref.n_records)
                chunk_records = part_records[start_idx:end_idx]

                # 时间过滤（细粒度）
                if time_range:
                    start_time, end_time = time_range
                    mask = (chunk_records["time"] >= start_time) & (
                        chunk_records["time"] <= end_time
                    )
                    chunk_records = chunk_records[mask]

                if len(chunk_records) == 0:
                    continue

                # 提取对应的 wave_pool 片段
                wave_start = int(chunk_records[0]["wave_offset"])
                last_rec = chunk_records[-1]
                wave_end = int(last_rec["wave_offset"] + last_rec["event_length"])
                chunk_waves = np.array(part_waves[wave_start:wave_end], copy=True)

                # 调整 wave_offset 为相对偏移
                chunk_records = np.array(chunk_records, copy=True)
                chunk_records["wave_offset"] -= wave_start

                yield RecordsBundle(chunk_records, chunk_waves)

    def load_full(self) -> RecordsBundle:
        """
        完整加载到内存。

        ⚠️ 警告：大数据集会 OOM

        Returns:
            RecordsBundle

        Memory:
            全部数据加载到内存
        """
        if self.total_records == 0:
            return RecordsBundle(np.zeros(0, dtype=RECORDS_DTYPE), np.zeros(0, dtype=np.uint16))

        # 分配输出数组
        records = np.zeros(self.total_records, dtype=RECORDS_DTYPE)
        wave_pool = np.zeros(self.total_samples, dtype=np.uint16)

        rec_cursor = 0
        wave_cursor = 0

        # 逐个分片加载
        for part_ref in self.part_refs:
            part_records = np.memmap(
                part_ref.records_path, dtype=RECORDS_DTYPE, mode="r", shape=(part_ref.n_records,)
            )
            part_waves = np.memmap(
                part_ref.wave_pool_path, dtype=np.uint16, mode="r", shape=(part_ref.n_samples,)
            )

            # 复制到输出数组
            records[rec_cursor : rec_cursor + part_ref.n_records] = part_records[:]
            wave_pool[wave_cursor : wave_cursor + part_ref.n_samples] = part_waves[:]

            # 更新 wave_offset
            records[rec_cursor : rec_cursor + part_ref.n_records]["wave_offset"] += wave_cursor

            rec_cursor += part_ref.n_records
            wave_cursor += part_ref.n_samples

        # 重新分配 record_id
        records["record_id"] = np.arange(self.total_records, dtype=np.int64)

        return RecordsBundle(records, wave_pool)

    def get_records_view(self) -> np.ndarray:
        """
        返回 records 的视图（不加载 wave_pool）。

        适用于只需要 records 元数据的场景：
        - 统计分析（事件数、时间范围、通道分布）
        - 时间戳查询
        - 通道过滤

        Returns:
            np.ndarray (memmap 或合并后的数组)

        Memory:
            单分片：0（memmap）
            多分片：仅 records 大小（无 wave_pool）
        """
        if len(self.part_refs) == 1:
            # 单分片：直接返回 memmap
            part = self.part_refs[0]
            return np.memmap(
                part.records_path, dtype=RECORDS_DTYPE, mode="r", shape=(part.n_records,)
            )
        else:
            # 多分片：需要合并（但只合并 records，不合并 wave_pool）
            records = np.zeros(self.total_records, dtype=RECORDS_DTYPE)
            cursor = 0

            for part_ref in self.part_refs:
                part_records = np.memmap(
                    part_ref.records_path,
                    dtype=RECORDS_DTYPE,
                    mode="r",
                    shape=(part_ref.n_records,),
                )
                records[cursor : cursor + part_ref.n_records] = part_records[:]
                cursor += part_ref.n_records

            return records

    def cleanup(self):
        """清理临时文件"""
        if self.temp_dir and self.temp_dir.exists():
            import shutil

            shutil.rmtree(self.temp_dir)
            self.temp_dir = None


def _resolve_baseline_window(
    baseline_samples: int | tuple[int, int] | list[int] | None,
    samples_start: int,
    baseline_start: int,
    baseline_end: int,
) -> tuple[int, int]:
    baseline_samples = _normalize_baseline_samples(baseline_samples)
    if baseline_samples is None:
        return baseline_start, baseline_end
    if isinstance(baseline_samples, tuple):
        return samples_start + baseline_samples[0], samples_start + baseline_samples[1]
    return baseline_start, baseline_start + int(baseline_samples)


def _clip_wave_to_uint16(wave: np.ndarray) -> np.ndarray:
    wave = np.asarray(wave)
    if wave.dtype == np.uint16:
        return wave
    return wave.astype(np.uint16, copy=False)


def _records_sort_order(records: np.ndarray) -> np.ndarray:
    """Return a stable global sort order for final records output."""
    seq = np.arange(len(records), dtype=np.int64)
    return np.lexsort(
        (seq, records["channel"], records["board"], records["pid"], records["timestamp"])
    )


@export
def split_by_hardware_channel(st_waveforms: np.ndarray) -> list[tuple[HardwareChannel, np.ndarray]]:
    """Split a structured array into per-hardware-channel views."""
    if st_waveforms is None or len(st_waveforms) == 0:
        return []
    if not isinstance(st_waveforms, np.ndarray) or st_waveforms.dtype.names is None:
        raise ValueError("st_waveforms must be a structured numpy array")
    if "board" not in st_waveforms.dtype.names or "channel" not in st_waveforms.dtype.names:
        raise ValueError("st_waveforms missing required 'board'/'channel' fields")

    groups = group_indices_by_hardware_channel(st_waveforms["board"], st_waveforms["channel"])
    return [(hw_channel, st_waveforms[indices]) for hw_channel, indices in groups.items()]


def _hardware_channel_index_groups(
    st_waveforms: np.ndarray,
) -> list[tuple[HardwareChannel, np.ndarray]]:
    """Return per-channel row indices without copying the full structured array."""
    if st_waveforms is None or len(st_waveforms) == 0:
        return []
    if not isinstance(st_waveforms, np.ndarray) or st_waveforms.dtype.names is None:
        raise ValueError("st_waveforms must be a structured numpy array")
    if "board" not in st_waveforms.dtype.names or "channel" not in st_waveforms.dtype.names:
        raise ValueError("st_waveforms missing required 'board'/'channel' fields")

    groups = group_indices_by_hardware_channel(st_waveforms["board"], st_waveforms["channel"])
    return list(groups.items())


@export
def split_by_channel(st_waveforms: np.ndarray) -> list[tuple[int, np.ndarray]]:
    """Backward-compatible helper for single-board inputs only."""
    groups = split_by_hardware_channel(st_waveforms)
    if any(hw_channel.board != 0 for hw_channel, _ in groups):
        raise ValueError(
            "split_by_channel no longer supports multi-board data; use split_by_hardware_channel instead."
        )
    return [(hw_channel.channel, group) for hw_channel, group in groups]


def _build_records_from_wave_list(
    waves: Sequence[tuple[int, int, int, int, int, np.ndarray]],
    default_dt_ns: int,
) -> RecordsBundle:
    if not waves:
        return RecordsBundle(np.zeros(0, dtype=RECORDS_DTYPE), np.zeros(0, dtype=np.uint16))

    total_records = len(waves)
    records = np.zeros(total_records, dtype=RECORDS_DTYPE)
    source_idx = np.arange(total_records, dtype=np.int64)

    for i, (board, channel, timestamp_ps, baseline, flags, waveform) in enumerate(waves):
        records["timestamp"][i] = timestamp_ps
        records["pid"][i] = 0
        records["board"][i] = board
        records["channel"][i] = channel
        records["baseline"][i] = baseline
        records["baseline_upstream"][i] = np.nan
        records["polarity"][i] = "unknown"
        records["dt"][i] = np.int32(default_dt_ns)
        records["trigger_type"][i] = 0
        records["flags"][i] = np.uint32(flags)
        length = int(len(waveform))
        if length > np.iinfo(np.int32).max:
            raise ValueError("event_length exceeds int32 range")
        records["event_length"][i] = np.int32(length)
        records["time"][i] = int(timestamp_ps // 1000)

    order = _records_sort_order(records)
    records = records[order]
    source_idx = source_idx[order]

    total_samples = int(records["event_length"].astype(np.int64, copy=False).sum())
    wave_pool = np.zeros(total_samples, dtype=np.uint16)

    wave_cursor = 0
    for idx in range(total_records):
        wave = waves[int(source_idx[idx])][5]
        length = int(records["event_length"][idx])
        if length > 0:
            wave_pool[wave_cursor : wave_cursor + length] = _clip_wave_to_uint16(wave[:length])
        records["wave_offset"][idx] = wave_cursor
        wave_cursor += length

    records["record_id"] = np.arange(total_records, dtype=np.int64)
    return RecordsBundle(records=records, wave_pool=wave_pool)


def _build_records_part_from_raw_array(
    raw_arr: np.ndarray,
    *,
    channel_idx: int,
    default_dt_ns: int,
    cols: object,
    normalize_timestamp_to_ps,
    baseline_samples: int | tuple[int, int] | list[int] | None,
) -> RecordsBundle:
    if raw_arr.size == 0:
        return RecordsBundle(np.zeros(0, dtype=RECORDS_DTYPE), np.zeros(0, dtype=np.uint16))

    if raw_arr.ndim != 2:
        raise ValueError("raw waveform array must be 2D")

    try:
        timestamps = raw_arr[:, cols.timestamp].astype(np.int64)
    except Exception:
        timestamps = np.array([int(row[cols.timestamp]) for row in raw_arr], dtype=np.int64)
    timestamps = normalize_timestamp_to_ps(timestamps, dt_ns=int(default_dt_ns))

    try:
        board_vals = raw_arr[:, cols.board].astype(np.int16)
    except Exception:
        board_vals = np.zeros(len(raw_arr), dtype=np.int16)

    try:
        channel_vals = raw_arr[:, cols.channel].astype(np.int16)
    except Exception:
        channel_vals = np.full(len(raw_arr), int(channel_idx), dtype=np.int16)

    baseline_start, baseline_end = _resolve_baseline_window(
        baseline_samples,
        cols.samples_start,
        cols.baseline_start,
        cols.baseline_end,
    )
    if baseline_end > raw_arr.shape[1]:
        baseline_end = raw_arr.shape[1]
    if baseline_end <= baseline_start:
        baseline_vals = np.full(len(raw_arr), np.nan, dtype=np.float64)
    else:
        try:
            baseline_vals = np.mean(raw_arr[:, baseline_start:baseline_end].astype(float), axis=1)
        except Exception:
            baseline_vals = np.full(len(raw_arr), np.nan, dtype=np.float64)

    samples_end = cols.samples_end if cols.samples_end is not None else raw_arr.shape[1]
    if samples_end > raw_arr.shape[1]:
        samples_end = raw_arr.shape[1]
    if raw_arr.shape[1] <= cols.samples_start or samples_end <= cols.samples_start:
        wave_data = np.zeros((len(raw_arr), 0), dtype=np.int16)
    else:
        wave_data = raw_arr[:, cols.samples_start : samples_end]

    n_records = len(raw_arr)
    records = np.zeros(n_records, dtype=RECORDS_DTYPE)
    records["timestamp"] = timestamps.astype(np.int64, copy=False)
    records["pid"] = 0
    records["board"] = board_vals.astype(np.int16, copy=False)
    records["channel"] = channel_vals.astype(np.int16, copy=False)
    records["baseline"] = baseline_vals.astype(np.float64, copy=False)
    records["baseline_upstream"] = np.nan
    records["polarity"] = "unknown"
    records["dt"] = np.int32(default_dt_ns)
    records["trigger_type"] = 0
    records["flags"] = np.uint32(0)
    records["time"] = records["timestamp"].astype(np.int64, copy=False) // 1000

    wave_data = np.asarray(wave_data)
    if wave_data.ndim != 2:
        raise ValueError("waveform samples must be a 2D array")

    wave_length = int(wave_data.shape[1]) if wave_data.ndim == 2 else 0
    if wave_length > np.iinfo(np.int32).max:
        raise ValueError("event_length exceeds int32 range")
    records["event_length"] = np.int32(wave_length)

    order = _records_sort_order(records)
    records = records[order]

    if wave_length <= 0:
        records["wave_offset"] = 0
        records["record_id"] = np.arange(n_records, dtype=np.int64)
        return RecordsBundle(records=records, wave_pool=np.zeros(0, dtype=np.uint16))

    ordered_waves = wave_data[order]
    wave_pool = np.asarray(ordered_waves, dtype=np.uint16).reshape(-1)
    records["wave_offset"] = np.arange(n_records, dtype=np.int64) * wave_length
    records["record_id"] = np.arange(n_records, dtype=np.int64)
    return RecordsBundle(records=records, wave_pool=wave_pool)


def _write_records_part(
    bundle: RecordsBundle, part_dir: Path, part_idx: int
) -> _RecordsPartRef | None:
    if len(bundle.records) == 0:
        return None

    records_path = part_dir / f"records_part_{part_idx}.dat"
    wave_pool_path = part_dir / f"wave_pool_part_{part_idx}.dat"

    records_mm = np.memmap(
        records_path,
        dtype=RECORDS_DTYPE,
        mode="w+",
        shape=(len(bundle.records),),
    )
    records_mm[:] = bundle.records
    records_mm.flush()

    wave_pool_mm = np.memmap(
        wave_pool_path,
        dtype=np.uint16,
        mode="w+",
        shape=(len(bundle.wave_pool),),
    )
    if len(bundle.wave_pool) > 0:
        wave_pool_mm[:] = bundle.wave_pool
    wave_pool_mm.flush()
    records_mm._mmap.close()
    wave_pool_mm._mmap.close()

    if len(bundle.records) > 0:
        time_range = (int(bundle.records["time"].min()), int(bundle.records["time"].max()))
    else:
        time_range = None

    return _RecordsPartRef(
        records_path=records_path,
        wave_pool_path=wave_pool_path,
        n_records=len(bundle.records),
        n_samples=len(bundle.wave_pool),
        time_range=time_range,
    )


def _merge_key(
    records: np.ndarray,
    source_idx: int,
    row_idx: int,
) -> tuple[int, int, int, int, int, int]:
    rec = records[row_idx]
    return (
        int(rec["timestamp"]),
        int(rec["pid"]),
        int(rec["board"]),
        int(rec["channel"]),
        int(source_idx),
        int(row_idx),
    )


def _find_run_stop_by_next_heap_key(
    records: np.ndarray,
    source_idx: int,
    row_idx: int,
    boundary_key: tuple[int, int, int, int, int, int] | None,
) -> int:
    """Find the largest consecutive run from one sorted part that can be emitted."""
    n_records = len(records)
    if row_idx >= n_records:
        return row_idx
    if boundary_key is None:
        return n_records

    boundary_ts = boundary_key[0]
    stop = int(np.searchsorted(records["timestamp"], boundary_ts, side="left"))
    if stop <= row_idx:
        stop = row_idx + 1

    while stop < n_records:
        if _merge_key(records, source_idx, stop) <= boundary_key:
            stop += 1
        else:
            break
    return stop


def _copy_records_run_with_waves(
    *,
    records_src: np.ndarray,
    wave_pool_src: np.ndarray,
    row_start: int,
    row_stop: int,
    records_out: np.ndarray,
    wave_pool_out: np.ndarray,
    out_idx: int,
    wave_cursor: int,
) -> tuple[int, int]:
    """Copy a sorted records run and rebase its contiguous wave_pool segment."""
    if row_stop <= row_start:
        return out_idx, wave_cursor

    run_records = records_src[row_start:row_stop]
    run_n = len(run_records)
    records_out[out_idx : out_idx + run_n] = run_records

    first_offset = int(run_records[0]["wave_offset"])
    last = run_records[-1]
    last_length = max(int(last["event_length"]), 0)
    last_end = int(last["wave_offset"]) + last_length
    run_samples = max(last_end - first_offset, 0)

    if run_samples > 0:
        wave_pool_out[wave_cursor : wave_cursor + run_samples] = wave_pool_src[
            first_offset:last_end
        ]

    records_out[out_idx : out_idx + run_n]["wave_offset"] = (
        run_records["wave_offset"].astype(np.int64, copy=False) - first_offset + wave_cursor
    )
    return out_idx + run_n, wave_cursor + run_samples


def _copy_records_run_with_wave_base(
    *,
    records_src: np.ndarray,
    row_start: int,
    row_stop: int,
    records_out: np.ndarray,
    out_idx: int,
    wave_base: int,
) -> int:
    """Copy a sorted records run and rebase offsets into the final wave_pool."""
    if row_stop <= row_start:
        return out_idx

    run_records = records_src[row_start:row_stop]
    run_n = len(run_records)
    records_out[out_idx : out_idx + run_n] = run_records
    records_out[out_idx : out_idx + run_n]["wave_offset"] = run_records["wave_offset"].astype(
        np.int64, copy=False
    ) + int(wave_base)
    return out_idx + run_n


def _merge_records_part_refs_batched(
    parts: Sequence[_RecordsPartRef],
    batch_size: int = 50,
    output_dir: Path | None = None,
) -> list[_RecordsPartRef]:
    """
    分批合并分片，减少内存占用。

    策略：
    1. 将 N 个小分片分成多批（每批 batch_size 个）
    2. 每批合并到一个临时分片
    3. 返回合并后的中等分片列表

    Args:
        parts: 输入分片列表
        batch_size: 每批合并的分片数量
        output_dir: 输出目录（None 则使用第一个分片的目录）

    Returns:
        合并后的分片列表
    """
    if not parts:
        return []

    if len(parts) <= batch_size:
        # 分片数量少，直接返回
        return list(parts)

    # 确定输出目录
    if output_dir is None:
        # 使用第一个分片的父目录（而不是 parent.parent，避免逃逸到 /tmp/merged）
        output_dir = parts[0].records_path.parent / "merged"
    output_dir.mkdir(parents=True, exist_ok=True)

    merged_parts = []

    # 分批处理
    for batch_idx in range(0, len(parts), batch_size):
        batch = parts[batch_idx : batch_idx + batch_size]

        # 合并当前批次
        merged_bundle = _merge_records_part_refs_to_memory(batch)

        # 写入新的分片
        merged_part_ref = _write_records_part(merged_bundle, output_dir, len(merged_parts))

        if merged_part_ref:
            merged_parts.append(merged_part_ref)

        # 释放内存
        del merged_bundle

    return merged_parts


def _merge_records_part_refs_batched_to_disk(
    parts: Sequence[_RecordsPartRef],
    batch_size: int,
    output_dir: Path,
    show_progress: bool = False,
    n_workers: int | None = None,
    executor_type: str = "thread",
) -> list[_RecordsPartRef]:
    if not parts:
        return []
    if len(parts) <= batch_size:
        return list(parts)

    output_dir.mkdir(parents=True, exist_ok=True)
    starts = list(range(0, len(parts), batch_size))

    # 确定并行度
    if n_workers is None:
        import os

        # 保守的默认值：最多 4 个 worker，避免文件描述符耗尽
        # 每个 worker 在 batch_size=50 时可能打开 100+ 文件描述符
        n_workers = min(4, max(1, os.cpu_count() or 1), len(starts))
    else:
        n_workers = max(1, int(n_workers))

    executor_type = str(executor_type or "thread").lower()
    if executor_type not in {"thread", "process"}:
        raise ValueError(
            f"records merge executor_type must be 'thread' or 'process', got {executor_type!r}"
        )

    use_parallel = n_workers > 1 and len(starts) > 1

    # 调试信息
    if show_progress:
        mode = "并行" if use_parallel else "串行"
        print(
            f"[Records 合并] 模式={mode}, executor={executor_type}, "
            f"workers={n_workers}, 批次数={len(starts)}, 分片数={len(parts)}"
        )

    pbar = None
    if show_progress:
        try:
            from tqdm import tqdm

            pbar = tqdm(total=len(starts), desc="Merging parts (disk)", leave=False)
        except ImportError:
            pbar = None

    try:
        if not use_parallel:
            # 串行模式（单核）
            merged_parts: list[_RecordsPartRef] = []
            for batch_idx, start in enumerate(starts):
                batch = parts[start : start + batch_size]
                merged = _merge_records_part_refs_to_disk(
                    batch,
                    output_dir=output_dir,
                    part_idx=batch_idx,
                )
                if merged is not None:
                    merged_parts.append(merged)
                if pbar is not None:
                    pbar.update(1)
        else:
            # 并行模式（多核）
            from concurrent.futures import as_completed

            from waveform_analysis.core.execution.manager import get_executor

            with get_executor(
                "records_batch_merge",
                executor_type=executor_type,
                max_workers=n_workers,
                reuse=True,
            ) as executor:
                futures = {
                    executor.submit(
                        _merge_records_part_refs_to_disk,
                        parts[start : start + batch_size],
                        output_dir,
                        batch_idx,
                    ): batch_idx
                    for batch_idx, start in enumerate(starts)
                }

                results = []
                for future in as_completed(futures):
                    batch_idx = futures[future]
                    merged = future.result()
                    if merged is not None:
                        results.append((batch_idx, merged))
                    if pbar is not None:
                        pbar.update(1)

                # 按批次索引排序，保持确定性顺序
                results.sort(key=lambda x: x[0])
                merged_parts = [merged for _, merged in results]
    finally:
        if pbar is not None:
            pbar.close()

    return merged_parts


def _merge_records_part_refs_to_memory(parts: Sequence[_RecordsPartRef]) -> RecordsBundle:
    """
    将分片合并到内存（原 _merge_records_part_refs 的实现）。

    Args:
        parts: 分片列表

    Returns:
        合并后的 RecordsBundle
    """
    if not parts:
        return RecordsBundle(np.zeros(0, dtype=RECORDS_DTYPE), np.zeros(0, dtype=np.uint16))

    if len(parts) == 1:
        part = parts[0]
        records = np.array(
            np.memmap(part.records_path, dtype=RECORDS_DTYPE, mode="r", shape=(part.n_records,)),
            copy=True,
        )
        wave_pool = np.array(
            np.memmap(part.wave_pool_path, dtype=np.uint16, mode="r", shape=(part.n_samples,)),
            copy=True,
        )
        if len(records) > 0:
            records["record_id"] = np.arange(len(records), dtype=np.int64)
        return RecordsBundle(records=records, wave_pool=wave_pool)

    total_records = sum(part.n_records for part in parts)
    if total_records == 0:
        return RecordsBundle(np.zeros(0, dtype=RECORDS_DTYPE), np.zeros(0, dtype=np.uint16))

    total_samples = sum(part.n_samples for part in parts)
    records_out = np.empty(total_records, dtype=RECORDS_DTYPE)
    wave_pool_out = np.empty(total_samples, dtype=np.uint16)

    import heapq

    records_parts = [
        np.memmap(part.records_path, dtype=RECORDS_DTYPE, mode="r", shape=(part.n_records,))
        for part in parts
    ]
    wave_pool_parts = [
        np.memmap(part.wave_pool_path, dtype=np.uint16, mode="r", shape=(part.n_samples,))
        for part in parts
    ]

    heap: list[tuple[int, int, int, int, int, int]] = []
    for source_idx, records in enumerate(records_parts):
        if len(records) == 0:
            continue
        heapq.heappush(heap, _merge_key(records, source_idx, 0))

    out_idx = 0
    wave_cursor = 0
    while heap:
        key = heapq.heappop(heap)
        source_idx = key[4]
        row_idx = key[5]
        records_src = records_parts[source_idx]
        wave_pool_src = wave_pool_parts[source_idx]
        boundary_key = heap[0] if heap else None

        row_stop = _find_run_stop_by_next_heap_key(
            records=records_src,
            source_idx=source_idx,
            row_idx=row_idx,
            boundary_key=boundary_key,
        )
        out_idx, wave_cursor = _copy_records_run_with_waves(
            records_src=records_src,
            wave_pool_src=wave_pool_src,
            row_start=row_idx,
            row_stop=row_stop,
            records_out=records_out,
            wave_pool_out=wave_pool_out,
            out_idx=out_idx,
            wave_cursor=wave_cursor,
        )

        if row_stop < len(records_src):
            heapq.heappush(heap, _merge_key(records_src, source_idx, row_stop))

    if out_idx != total_records:
        records_out = records_out[:out_idx]
    if wave_cursor != total_samples:
        wave_pool_out = wave_pool_out[:wave_cursor]

    records_out["record_id"] = np.arange(len(records_out), dtype=np.int64)
    return RecordsBundle(records=records_out, wave_pool=wave_pool_out)


def _merge_records_part_refs_to_disk(
    parts: Sequence[_RecordsPartRef], output_dir: Path, part_idx: int = 0
) -> _RecordsPartRef | None:
    """Merge sorted part refs into a single disk-backed part."""
    if not parts:
        return None

    total_records = sum(part.n_records for part in parts)
    if total_records == 0:
        return None

    total_samples = sum(part.n_samples for part in parts)
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / f"records_merged_{part_idx}.dat"
    wave_pool_path = output_dir / f"wave_pool_merged_{part_idx}.dat"

    records_out = np.memmap(
        records_path,
        dtype=RECORDS_DTYPE,
        mode="w+",
        shape=(total_records,),
    )
    wave_pool_out = np.memmap(
        wave_pool_path,
        dtype=np.uint16,
        mode="w+",
        shape=(total_samples,),
    )

    import heapq

    records_parts = [
        np.memmap(part.records_path, dtype=RECORDS_DTYPE, mode="r", shape=(part.n_records,))
        for part in parts
    ]
    wave_pool_parts = [
        np.memmap(part.wave_pool_path, dtype=np.uint16, mode="r", shape=(part.n_samples,))
        for part in parts
    ]

    heap: list[tuple[int, int, int, int, int, int]] = []
    for source_idx, records in enumerate(records_parts):
        if len(records) == 0:
            continue
        heapq.heappush(heap, _merge_key(records, source_idx, 0))

    out_idx = 0
    wave_cursor = 0
    while heap:
        key = heapq.heappop(heap)
        source_idx = key[4]
        row_idx = key[5]
        records_src = records_parts[source_idx]
        wave_pool_src = wave_pool_parts[source_idx]
        boundary_key = heap[0] if heap else None

        row_stop = _find_run_stop_by_next_heap_key(
            records=records_src,
            source_idx=source_idx,
            row_idx=row_idx,
            boundary_key=boundary_key,
        )
        out_idx, wave_cursor = _copy_records_run_with_waves(
            records_src=records_src,
            wave_pool_src=wave_pool_src,
            row_start=row_idx,
            row_stop=row_stop,
            records_out=records_out,
            wave_pool_out=wave_pool_out,
            out_idx=out_idx,
            wave_cursor=wave_cursor,
        )

        if row_stop < len(records_src):
            heapq.heappush(heap, _merge_key(records_src, source_idx, row_stop))

    if out_idx != total_records:
        raise RuntimeError(f"merged records count mismatch: {out_idx} != {total_records}")
    if wave_cursor != total_samples:
        raise RuntimeError(f"merged wave_pool size mismatch: {wave_cursor} != {total_samples}")

    records_out["record_id"] = np.arange(total_records, dtype=np.int64)
    time_range = (int(records_out["time"].min()), int(records_out["time"].max()))
    records_out.flush()
    wave_pool_out.flush()
    del records_out
    del wave_pool_out

    return _RecordsPartRef(
        records_path=records_path,
        wave_pool_path=wave_pool_path,
        n_records=total_records,
        n_samples=total_samples,
        time_range=time_range,
    )


def _write_records_only_part(
    records_path: Path,
    n_records: int,
) -> _RecordsPartRef | None:
    if n_records == 0:
        return None
    records = np.memmap(records_path, dtype=RECORDS_DTYPE, mode="r", shape=(n_records,))
    time_range = (int(records["time"].min()), int(records["time"].max()))
    records._mmap.close()
    del records
    return _RecordsPartRef(
        records_path=records_path,
        wave_pool_path=records_path,
        n_records=n_records,
        n_samples=0,
        time_range=time_range,
    )


def _merge_records_part_refs_records_only_to_disk(
    parts: Sequence[_RecordsPartRef],
    wave_bases: Sequence[int],
    output_dir: Path,
    part_idx: int = 0,
    assign_record_ids: bool = True,
) -> _RecordsPartRef | None:
    """Merge records globally while rebasing offsets into a final concatenated wave_pool."""
    if not parts:
        return None
    if len(parts) != len(wave_bases):
        raise ValueError("parts and wave_bases length mismatch")

    total_records = sum(part.n_records for part in parts)
    if total_records == 0:
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / f"records_merged_{part_idx}.dat"
    records_out = np.memmap(
        records_path,
        dtype=RECORDS_DTYPE,
        mode="w+",
        shape=(total_records,),
    )

    import heapq

    records_parts = [
        np.memmap(part.records_path, dtype=RECORDS_DTYPE, mode="r", shape=(part.n_records,))
        for part in parts
    ]

    heap: list[tuple[int, int, int, int, int, int]] = []
    for source_idx, records in enumerate(records_parts):
        if len(records) == 0:
            continue
        heapq.heappush(heap, _merge_key(records, source_idx, 0))

    out_idx = 0
    while heap:
        key = heapq.heappop(heap)
        source_idx = key[4]
        row_idx = key[5]
        records_src = records_parts[source_idx]
        boundary_key = heap[0] if heap else None

        row_stop = _find_run_stop_by_next_heap_key(
            records=records_src,
            source_idx=source_idx,
            row_idx=row_idx,
            boundary_key=boundary_key,
        )
        out_idx = _copy_records_run_with_wave_base(
            records_src=records_src,
            row_start=row_idx,
            row_stop=row_stop,
            records_out=records_out,
            out_idx=out_idx,
            wave_base=int(wave_bases[source_idx]),
        )

        if row_stop < len(records_src):
            heapq.heappush(heap, _merge_key(records_src, source_idx, row_stop))

    if out_idx != total_records:
        raise RuntimeError(f"merged records count mismatch: {out_idx} != {total_records}")

    if assign_record_ids:
        records_out["record_id"] = np.arange(total_records, dtype=np.int64)
    records_out.flush()
    for records_part in records_parts:
        records_part._mmap.close()
    records_out._mmap.close()
    del records_out

    return _write_records_only_part(records_path, total_records)


def _merge_records_only_part_refs_to_disk(
    parts: Sequence[_RecordsPartRef],
    output_dir: Path,
    part_idx: int = 0,
    assign_record_ids: bool = True,
) -> _RecordsPartRef | None:
    """Merge records-only intermediate refs; wave offsets are already final."""
    if not parts:
        return None

    total_records = sum(part.n_records for part in parts)
    if total_records == 0:
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / f"records_merged_{part_idx}.dat"
    unsorted_path = output_dir / f"records_unsorted_{part_idx}.dat"

    records_unsorted = np.memmap(
        unsorted_path,
        dtype=RECORDS_DTYPE,
        mode="w+",
        shape=(total_records,),
    )

    cursor = 0
    try:
        for part in parts:
            records_src = np.memmap(
                part.records_path,
                dtype=RECORDS_DTYPE,
                mode="r",
                shape=(part.n_records,),
            )
            records_unsorted[cursor : cursor + part.n_records] = records_src
            cursor += part.n_records
            records_src._mmap.close()

        if cursor != total_records:
            raise RuntimeError(f"merged records count mismatch: {cursor} != {total_records}")

        order = _records_sort_order(records_unsorted)
        records_out = np.memmap(
            records_path,
            dtype=RECORDS_DTYPE,
            mode="w+",
            shape=(total_records,),
        )

        copy_chunk_size = 250_000
        for start in range(0, total_records, copy_chunk_size):
            stop = min(start + copy_chunk_size, total_records)
            records_out[start:stop] = records_unsorted[order[start:stop]]

        if assign_record_ids:
            records_out["record_id"] = np.arange(total_records, dtype=np.int64)
        records_out.flush()
        records_out._mmap.close()
        del records_out
        del order
    finally:
        records_unsorted._mmap.close()
        del records_unsorted
        try:
            unsorted_path.unlink()
        except FileNotFoundError:
            pass

    return _write_records_only_part(records_path, total_records)


def _concat_wave_pool_part_refs_to_disk(
    parts: Sequence[_RecordsPartRef], output_dir: Path, part_idx: int = 0
) -> Path | None:
    total_samples = sum(part.n_samples for part in parts)
    if total_samples == 0:
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    wave_pool_path = output_dir / f"wave_pool_merged_{part_idx}.dat"
    wave_pool_out = np.memmap(
        wave_pool_path,
        dtype=np.uint16,
        mode="w+",
        shape=(total_samples,),
    )

    cursor = 0
    for part_idx_local, part in enumerate(parts):
        if part.n_samples <= 0:
            continue

        # 验证文件实际大小
        actual_size = part.wave_pool_path.stat().st_size
        expected_size = part.n_samples * 2  # uint16 = 2 bytes
        if actual_size < expected_size:
            raise RuntimeError(
                f"wave_pool part {part_idx_local} truncated: "
                f"expected {expected_size} bytes ({part.n_samples} samples), "
                f"got {actual_size} bytes. Path: {part.wave_pool_path}"
            )

        wave_pool_src = np.memmap(
            part.wave_pool_path,
            dtype=np.uint16,
            mode="r",
            shape=(part.n_samples,),
        )
        wave_pool_out[cursor : cursor + part.n_samples] = wave_pool_src
        cursor += part.n_samples
        wave_pool_src._mmap.close()

    if cursor != total_samples:
        raise RuntimeError(f"merged wave_pool size mismatch: {cursor} != {total_samples}")

    wave_pool_out.flush()
    wave_pool_out._mmap.close()
    del wave_pool_out
    return wave_pool_path


def _merge_records_part_refs_indexed_to_disk(
    parts: Sequence[_RecordsPartRef],
    output_dir: Path,
    batch_size: int,
    part_idx: int = 0,
    show_progress: bool = False,
    n_workers: int | None = None,
    executor_type: str = "thread",
    profiler=None,
) -> _RecordsPartRef | None:
    """
    Build one formal disk part without sorting wave samples by record order.

    The output wave_pool is a single memmap created by concatenating input part
    wave_pools once. Records are globally sorted and their wave_offset fields
    point into that concatenated physical layout.
    """
    if not parts:
        return None

    total_records = sum(part.n_records for part in parts)
    if total_records == 0:
        return None

    batch_size = max(1, int(batch_size))
    total_samples = sum(part.n_samples for part in parts)
    sample_bases: list[int] = []
    sample_cursor = 0
    for part in parts:
        sample_bases.append(sample_cursor)
        sample_cursor += part.n_samples

    output_dir.mkdir(parents=True, exist_ok=True)
    timer = profiler.timeit if profiler else None

    records_only: _RecordsPartRef | None
    if len(parts) > batch_size:
        batch_output_dir = output_dir / "records_only_batches"
        starts = list(range(0, len(parts), batch_size))

        if n_workers is None:
            import os

            n_workers = min(4, max(1, os.cpu_count() or 1), len(starts))
        else:
            n_workers = max(1, int(n_workers))

        executor_type = str(executor_type or "thread").lower()
        if executor_type not in {"thread", "process"}:
            raise ValueError(
                f"records merge executor_type must be 'thread' or 'process', got {executor_type!r}"
            )

        use_parallel = n_workers > 1 and len(starts) > 1

        if show_progress:
            mode = "并行" if use_parallel else "串行"
            print(
                f"[Records 合并] records-only batch merge, 模式={mode}, "
                f"executor={executor_type}, workers={n_workers}, "
                f"批次数={len(starts)}, 分片数={len(parts)}"
            )

        pbar = None
        if show_progress:
            try:
                from tqdm import tqdm

                pbar = tqdm(total=len(starts), desc="Merging records (disk)", leave=False)
            except ImportError:
                pbar = None

        batched_records: list[_RecordsPartRef] = []
        try:
            with timer("records.merge.records_only_batches.disk") if timer else nullcontext():
                if not use_parallel:
                    for batch_idx, start in enumerate(starts):
                        batch_parts = parts[start : start + batch_size]
                        batch_bases = sample_bases[start : start + batch_size]
                        batch_records = _merge_records_part_refs_records_only_to_disk(
                            batch_parts,
                            batch_bases,
                            output_dir=batch_output_dir,
                            part_idx=batch_idx,
                            assign_record_ids=False,
                        )
                        if batch_records is not None:
                            batched_records.append(batch_records)
                        if pbar is not None:
                            pbar.update(1)
                else:
                    from concurrent.futures import as_completed

                    from waveform_analysis.core.execution.manager import get_executor

                    with get_executor(
                        "records_only_batch_merge",
                        executor_type=executor_type,
                        max_workers=n_workers,
                        reuse=True,
                    ) as executor:
                        futures = {
                            executor.submit(
                                _merge_records_part_refs_records_only_to_disk,
                                parts[start : start + batch_size],
                                sample_bases[start : start + batch_size],
                                batch_output_dir,
                                batch_idx,
                                False,
                            ): batch_idx
                            for batch_idx, start in enumerate(starts)
                        }

                        results: list[tuple[int, _RecordsPartRef]] = []
                        for future in as_completed(futures):
                            batch_idx = futures[future]
                            batch_records = future.result()
                            if batch_records is not None:
                                results.append((batch_idx, batch_records))
                            if pbar is not None:
                                pbar.update(1)

                        results.sort(key=lambda item: item[0])
                        batched_records = [batch_records for _, batch_records in results]
        finally:
            if pbar is not None:
                pbar.close()

        if show_progress:
            print(
                f"[Records 合并] records-only final sort merge, " f"分片数={len(batched_records)}"
            )
        with timer("records.merge.records_only_final.disk") if timer else nullcontext():
            records_only = _merge_records_only_part_refs_to_disk(
                batched_records,
                output_dir=output_dir,
                part_idx=part_idx,
                assign_record_ids=True,
            )
    else:
        with timer("records.merge.records_only.disk") if timer else nullcontext():
            records_only = _merge_records_part_refs_records_only_to_disk(
                parts,
                sample_bases,
                output_dir=output_dir,
                part_idx=part_idx,
                assign_record_ids=True,
            )

    if records_only is None:
        return None

    if show_progress:
        print(f"[Records 合并] wave_pool concat, 分片数={len(parts)}, samples={total_samples}")
    with timer("records.merge.wave_pool_concat.disk") if timer else nullcontext():
        wave_pool_path = _concat_wave_pool_part_refs_to_disk(parts, output_dir, part_idx=part_idx)

    if wave_pool_path is None:
        wave_pool_path = output_dir / f"wave_pool_merged_{part_idx}.dat"
        wave_pool_path.touch()

    return _RecordsPartRef(
        records_path=records_only.records_path,
        wave_pool_path=wave_pool_path,
        n_records=total_records,
        n_samples=total_samples,
        time_range=records_only.time_range,
    )


def _can_concat_records_part_refs(parts: Sequence[_RecordsPartRef]) -> bool:
    if not parts:
        return False
    previous_stop: int | None = None
    for part in parts:
        if part.time_range is None:
            return False
        start, stop = part.time_range
        if previous_stop is not None and start < previous_stop:
            return False
        previous_stop = stop
    return True


def _concat_records_part_refs_to_disk(
    parts: Sequence[_RecordsPartRef], output_dir: Path, part_idx: int = 0
) -> _RecordsPartRef | None:
    if not parts:
        return None

    total_records = sum(part.n_records for part in parts)
    if total_records == 0:
        return None

    total_samples = sum(part.n_samples for part in parts)
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / f"records_merged_{part_idx}.dat"
    wave_pool_path = output_dir / f"wave_pool_merged_{part_idx}.dat"

    records_out = np.memmap(records_path, dtype=RECORDS_DTYPE, mode="w+", shape=(total_records,))
    wave_pool_out = np.memmap(wave_pool_path, dtype=np.uint16, mode="w+", shape=(total_samples,))

    record_cursor = 0
    wave_cursor = 0
    for part in parts:
        records_src = np.memmap(
            part.records_path,
            dtype=RECORDS_DTYPE,
            mode="r",
            shape=(part.n_records,),
        )
        wave_pool_src = np.memmap(
            part.wave_pool_path,
            dtype=np.uint16,
            mode="r",
            shape=(part.n_samples,),
        )
        records_out[record_cursor : record_cursor + part.n_records] = records_src
        if part.n_samples > 0:
            wave_pool_out[wave_cursor : wave_cursor + part.n_samples] = wave_pool_src
        records_out[record_cursor : record_cursor + part.n_records]["wave_offset"] += wave_cursor
        record_cursor += part.n_records
        wave_cursor += part.n_samples

    records_out["record_id"] = np.arange(total_records, dtype=np.int64)
    time_range = (int(records_out["time"].min()), int(records_out["time"].max()))
    records_out.flush()
    wave_pool_out.flush()
    del records_out
    del wave_pool_out

    return _RecordsPartRef(
        records_path=records_path,
        wave_pool_path=wave_pool_path,
        n_records=total_records,
        n_samples=total_samples,
        time_range=time_range,
    )


def _merge_records_part_refs(
    parts: Sequence[_RecordsPartRef],
    memory_budget_gb: float = 50.0,
    batch_size: int = 50,
    keep_on_disk: bool | None = None,
    output_dir: Path | None = None,
    transfer_temp_dir_ownership: bool = False,
    show_progress: bool = False,
    n_workers: int | None = None,
    executor_type: str = "thread",
    profiler=None,
) -> RecordsBundle | RecordsBundleRef:
    """
    智能合并分片：根据数据量自动选择策略。

    Args:
        parts: 分片列表
        memory_budget_gb: 内存预算（GB）
        batch_size: 分批合并时每批的分片数量
        keep_on_disk: 强制选择策略（None=自动，True=磁盘引用，False=加载内存）
        output_dir: 分批合并临时目录
        transfer_temp_dir_ownership: 如果为 True，将 output_dir 的所有权转移给 RecordsBundleRef
        show_progress: 是否显示磁盘分批合并进度
        n_workers: 合并阶段并行 worker 数量（None=自动检测 CPU 核心数）
        executor_type: 合并阶段执行器类型（"thread" 或 "process"）
        profiler: Optional profiler for merge sub-stage timings

    Returns:
        RecordsBundle 或 RecordsBundleRef
    """
    if not parts:
        return RecordsBundle(np.zeros(0, dtype=RECORDS_DTYPE), np.zeros(0, dtype=np.uint16))

    # 诊断信息
    if show_progress:
        print(
            f"[Records 合并入口] 分片数={len(parts)}, batch_size={batch_size}, "
            f"n_workers={n_workers}, executor={executor_type}"
        )

    # 估算总大小
    total_records = sum(part.n_records for part in parts)
    total_samples = sum(part.n_samples for part in parts)

    records_size_gb = (total_records * RECORDS_DTYPE.itemsize) / (1024**3)
    wave_pool_size_gb = (total_samples * 2) / (1024**3)  # uint16 = 2 bytes
    total_size_gb = records_size_gb + wave_pool_size_gb

    if keep_on_disk is None:
        if total_size_gb >= memory_budget_gb:
            raise MemoryError(
                "Estimated records output size "
                f"({total_size_gb:.2f} GB) exceeds memory budget "
                f"memory_budget_gb="
                f"{memory_budget_gb:.2f}. Use keep_on_disk=True to return a "
                "disk-backed RecordsBundleRef, or increase memory_budget_gb "
                "if a full in-memory RecordsBundle is required."
            )
        use_disk_ref = False
    else:
        # 强制选择
        use_disk_ref = keep_on_disk

    if not use_disk_ref:
        # 小数据：加载到内存
        if len(parts) <= batch_size:
            if show_progress:
                print(f"[Records 合并] 分片数 <= {batch_size}，直接合并到内存（无需并行）")
            return _merge_records_part_refs_to_memory(parts)
        else:
            # 分批合并后加载到内存
            if show_progress:
                print("[Records 合并] 内存模式分批合并（不支持并行）")
            merged_parts = _merge_records_part_refs_batched(
                parts, batch_size=batch_size, output_dir=output_dir
            )
            return _merge_records_part_refs_to_memory(merged_parts)
    else:
        # 大数据：返回磁盘引用
        import shutil

        cleanup_dir: Path | None = None
        # 确定输出目录
        if output_dir is not None:
            # 在 output_dir 下创建 merged 子目录
            ref_dir = output_dir / "merged"
            ref_dir.mkdir(parents=True, exist_ok=True)
            cleanup_on_error = False  # 调用者管理生命周期
            # 如果调用者要求转移所有权，则由 RecordsBundleRef 管理整个 output_dir
            managed_temp_dir = output_dir if transfer_temp_dir_ownership else None
        else:
            ref_dir = Path(tempfile.mkdtemp(prefix="records_bundle_ref_"))
            cleanup_on_error = True  # 我们创建的，出错时清理
            cleanup_dir = ref_dir
            managed_temp_dir = cleanup_dir

        try:
            final_part = _merge_records_part_refs_indexed_to_disk(
                parts,
                output_dir=ref_dir,
                batch_size=batch_size,
                show_progress=show_progress,
                n_workers=n_workers,
                executor_type=executor_type,
                profiler=profiler,
            )
        except Exception:
            if cleanup_on_error:
                shutil.rmtree(cleanup_dir or ref_dir, ignore_errors=True)
            raise

        if final_part is None:
            if cleanup_on_error:
                shutil.rmtree(cleanup_dir or ref_dir, ignore_errors=True)
            return RecordsBundle(np.zeros(0, dtype=RECORDS_DTYPE), np.zeros(0, dtype=np.uint16))

        return RecordsBundleRef(
            part_refs=[final_part],
            total_records=total_records,
            total_samples=total_samples,
            temp_dir=managed_temp_dir,
        )


def _build_records_part_refs_for_channel(
    *,
    channel_idx: int,
    channel_files: Sequence[str],
    part_root: str | Path,
    adapter_name: str,
    default_dt_ns: int,
    part_size: int | None,
    baseline_samples: int | tuple[int, int] | list[int] | None,
    epoch_ns: int | None,
    parse_engine: str | None,
    n_jobs: int | None,
    chunksize: int | None,
    use_process_pool: bool,
) -> tuple[int, list[_RecordsPartRef], dict[str, tuple[float, int]]]:
    from waveform_analysis.acquisition.formats import get_adapter

    adapter = get_adapter(adapter_name)
    reader = adapter.format_reader
    cols = adapter.format_spec.columns

    effective_n_jobs = 1 if n_jobs is None else max(int(n_jobs), 1)
    file_batch_size = max(1, effective_n_jobs)
    channel_part_dir = Path(part_root) / f"channel_{channel_idx}"
    channel_part_dir.mkdir(parents=True, exist_ok=True)

    try:
        raw_iter = reader.read_files_generator(
            list(channel_files),
            chunk_size=file_batch_size,
            chunksize=chunksize,
            n_jobs=effective_n_jobs,
            use_process_pool=use_process_pool,
            parse_engine=parse_engine,
            show_progress=False,
        )
    except TypeError:
        raw_iter = reader.read_files_generator(list(channel_files), chunk_size=file_batch_size)

    part_refs: list[_RecordsPartRef] = []
    profile = {
        "records.read": [0.0, 0],
        "records.part_build": [0.0, 0],
    }
    part_idx = 0

    while True:
        read_started = time.perf_counter()
        try:
            raw_arr = next(raw_iter)
        except StopIteration:
            break
        profile["records.read"][0] += time.perf_counter() - read_started
        profile["records.read"][1] += 1

        if raw_arr.size == 0:
            continue
        if part_size is None or part_size <= 0:
            slices = [raw_arr]
        else:
            slices = [
                raw_arr[start : start + part_size] for start in range(0, len(raw_arr), part_size)
            ]

        for raw_slice in slices:
            build_started = time.perf_counter()
            part = _build_records_part_from_raw_array(
                raw_slice,
                channel_idx=channel_idx,
                default_dt_ns=default_dt_ns,
                cols=cols,
                normalize_timestamp_to_ps=adapter.format_spec.normalize_timestamp_to_ps,
                baseline_samples=baseline_samples,
            )
            profile["records.part_build"][0] += time.perf_counter() - build_started
            profile["records.part_build"][1] += 1
            if len(part.records) == 0:
                continue
            if epoch_ns is not None and "time" in part.records.dtype.names:
                part.records["time"] = np.int64(epoch_ns) + (
                    part.records["timestamp"].astype(np.int64, copy=False) // 1000
                )
            part_ref = _write_records_part(part, channel_part_dir, part_idx)
            part_idx += 1
            if part_ref is not None:
                part_refs.append(part_ref)

    return (
        channel_idx,
        part_refs,
        {key: (float(values[0]), int(values[1])) for key, values in profile.items()},
    )


@export
def build_records_from_raw_files_streaming(
    raw_files: list[list[str]],
    adapter_name: str,
    default_dt_ns: int = 1,
    part_size: int | None = 250_000,
    baseline_samples: int | tuple[int, int] | list[int] | None = None,
    epoch_ns: int | None = None,
    show_progress: bool = False,
    parse_engine: str | None = "auto",
    n_jobs: int | None = None,
    chunksize: int | None = None,
    use_process_pool: bool = False,
    channel_workers: int | None = None,
    channel_executor: str = "thread",
    profiler=None,
    keep_on_disk: bool | None = None,
    memory_budget_gb: float = 50.0,
) -> RecordsBundle | RecordsBundleRef:
    _validate_baseline_samples(baseline_samples)

    timer = profiler.timeit if profiler else None

    channel_entries = list(enumerate(raw_files))
    if not channel_entries:
        return RecordsBundle(np.zeros(0, dtype=RECORDS_DTYPE), np.zeros(0, dtype=np.uint16))

    nonempty_channels = [
        (channel_idx, channel_files)
        for channel_idx, channel_files in channel_entries
        if channel_files
    ]
    if not nonempty_channels:
        return RecordsBundle(np.zeros(0, dtype=RECORDS_DTYPE), np.zeros(0, dtype=np.uint16))

    pbar = None
    if show_progress:
        try:
            from tqdm import tqdm

            pbar = tqdm(total=len(nonempty_channels), desc="Building records", leave=False)
        except ImportError:
            pbar = None

    with tempfile.TemporaryDirectory(prefix="records_parts_") as tmp_dir:
        part_dir = Path(tmp_dir)
        part_refs: list[_RecordsPartRef] = []
        channel_results: dict[int, list[_RecordsPartRef]] = {}

        try:
            effective_channel_workers = (
                1 if channel_workers is None else max(int(channel_workers), 1)
            )
            if effective_channel_workers <= 1 or len(nonempty_channels) <= 1:
                for channel_idx, channel_files in nonempty_channels:
                    result_idx, result_parts, profile = _build_records_part_refs_for_channel(
                        channel_idx=channel_idx,
                        channel_files=channel_files,
                        part_root=part_dir,
                        adapter_name=adapter_name,
                        default_dt_ns=default_dt_ns,
                        part_size=part_size,
                        baseline_samples=baseline_samples,
                        epoch_ns=epoch_ns,
                        parse_engine=parse_engine,
                        n_jobs=n_jobs,
                        chunksize=chunksize,
                        use_process_pool=use_process_pool,
                    )
                    channel_results[result_idx] = result_parts
                    if profiler:
                        for key, (duration, count) in profile.items():
                            profiler.durations[key] += duration
                            profiler.counts[key] += count
                    if pbar is not None:
                        pbar.update(1)
            else:
                from concurrent.futures import as_completed

                from waveform_analysis.core.execution.manager import get_executor

                max_workers = min(effective_channel_workers, len(nonempty_channels))
                with get_executor(
                    "records_channel_build",
                    executor_type=channel_executor,
                    max_workers=max_workers,
                    reuse=True,
                ) as executor:
                    futures = {
                        executor.submit(
                            _build_records_part_refs_for_channel,
                            channel_idx=channel_idx,
                            channel_files=channel_files,
                            part_root=part_dir,
                            adapter_name=adapter_name,
                            default_dt_ns=default_dt_ns,
                            part_size=part_size,
                            baseline_samples=baseline_samples,
                            epoch_ns=epoch_ns,
                            parse_engine=parse_engine,
                            n_jobs=n_jobs,
                            chunksize=chunksize,
                            use_process_pool=use_process_pool,
                        ): channel_idx
                        for channel_idx, channel_files in nonempty_channels
                    }
                    for future in as_completed(futures):
                        result_idx, result_parts, profile = future.result()
                        channel_results[result_idx] = result_parts
                        if profiler:
                            for key, (duration, count) in profile.items():
                                profiler.durations[key] += duration
                                profiler.counts[key] += count
                        if pbar is not None:
                            pbar.update(1)
        finally:
            if pbar is not None:
                pbar.close()

        for channel_idx, _ in channel_entries:
            part_refs.extend(channel_results.get(channel_idx, []))

        with timer("records.merge") if timer else nullcontext():
            return _merge_records_part_refs(
                part_refs,
                keep_on_disk=keep_on_disk,
                memory_budget_gb=memory_budget_gb,
                # Disk output owns a separate directory beyond the input parts' lifetime.
                output_dir=None if keep_on_disk else part_dir,
                n_workers=channel_workers,
                executor_type=channel_executor,
            )


def _build_records_from_channels(
    channels: Sequence[tuple[np.ndarray, HardwareChannel]], default_dt_ns: int
) -> RecordsBundle:
    if not channels:
        return RecordsBundle(np.zeros(0, dtype=RECORDS_DTYPE), np.zeros(0, dtype=np.uint16))

    total_records = sum(len(ch) for ch, _ in channels)
    if total_records == 0:
        return RecordsBundle(np.zeros(0, dtype=RECORDS_DTYPE), np.zeros(0, dtype=np.uint16))

    records = np.zeros(total_records, dtype=RECORDS_DTYPE)
    source_channel = np.zeros(total_records, dtype=np.int32)
    source_row = np.zeros(total_records, dtype=np.int64)
    source_record_id = np.full(total_records, -1, dtype=np.int64)

    cursor = 0
    for local_idx, (ch, hw_channel) in enumerate(channels):
        count = len(ch)
        if count == 0:
            continue

        if "timestamp" in ch.dtype.names:
            records["timestamp"][cursor : cursor + count] = ch["timestamp"]
        else:
            records["timestamp"][cursor : cursor + count] = 0

        records["pid"][cursor : cursor + count] = 0

        if "channel" in ch.dtype.names:
            records["channel"][cursor : cursor + count] = ch["channel"]
        else:
            records["channel"][cursor : cursor + count] = hw_channel.channel
        if "board" in ch.dtype.names:
            records["board"][cursor : cursor + count] = ch["board"].astype(np.int16, copy=False)
        else:
            records["board"][cursor : cursor + count] = hw_channel.board

        if "baseline" in ch.dtype.names:
            records["baseline"][cursor : cursor + count] = ch["baseline"]
        else:
            records["baseline"][cursor : cursor + count] = 0.0

        if "baseline_upstream" in ch.dtype.names:
            records["baseline_upstream"][cursor : cursor + count] = ch["baseline_upstream"]
        else:
            records["baseline_upstream"][cursor : cursor + count] = np.nan

        if "polarity" in ch.dtype.names:
            records["polarity"][cursor : cursor + count] = ch["polarity"]
        else:
            records["polarity"][cursor : cursor + count] = "unknown"

        if "event_length" in ch.dtype.names:
            lengths = ch["event_length"].astype(np.int64, copy=False)
            if lengths.size and lengths.max() > np.iinfo(np.int32).max:
                raise ValueError("event_length exceeds int32 range")
            records["event_length"][cursor : cursor + count] = lengths.astype(np.int32, copy=False)
        elif "wave" in ch.dtype.names:
            wave_len = ch["wave"].shape[1]
            records["event_length"][cursor : cursor + count] = np.int32(wave_len)
        else:
            records["event_length"][cursor : cursor + count] = 0

        if "dt" in ch.dtype.names:
            records["dt"][cursor : cursor + count] = ch["dt"].astype(np.int32, copy=False)
        else:
            records["dt"][cursor : cursor + count] = np.int32(default_dt_ns)

        if "trigger_type" in ch.dtype.names:
            records["trigger_type"][cursor : cursor + count] = ch["trigger_type"].astype(
                np.int16, copy=False
            )
        else:
            records["trigger_type"][cursor : cursor + count] = 0

        if "flags" in ch.dtype.names:
            records["flags"][cursor : cursor + count] = ch["flags"].astype(np.uint32, copy=False)
        else:
            records["flags"][cursor : cursor + count] = 0

        if "time" in ch.dtype.names:
            records["time"][cursor : cursor + count] = ch["time"]
        else:
            records["time"][cursor : cursor + count] = (
                records["timestamp"][cursor : cursor + count] // 1000
            )
        if "record_id" in ch.dtype.names:
            source_record_id[cursor : cursor + count] = ch["record_id"].astype(np.int64, copy=False)

        source_channel[cursor : cursor + count] = local_idx
        source_row[cursor : cursor + count] = np.arange(count, dtype=np.int64)
        cursor += count

    order = _records_sort_order(records)
    records = records[order]
    source_channel = source_channel[order]
    source_row = source_row[order]
    source_record_id = source_record_id[order]
    if np.all(source_record_id >= 0):
        records["record_id"] = source_record_id
    else:
        records["record_id"] = np.arange(total_records, dtype=np.int64)

    adjusted_lengths = np.zeros(total_records, dtype=np.int64)
    total_samples = 0
    for idx in range(total_records):
        length = int(records["event_length"][idx])
        if length < 0:
            length = 0
        ch = channels[int(source_channel[idx])][0]
        if "wave" not in ch.dtype.names:
            raise ValueError("st_waveforms missing 'wave' field required for wave_pool")
        wave = ch["wave"][int(source_row[idx])]
        max_len = wave.shape[-1]
        if length > max_len:
            length = max_len
        adjusted_lengths[idx] = length
        total_samples += length

    records["event_length"] = adjusted_lengths.astype(np.int32, copy=False)

    wave_pool = np.zeros(total_samples, dtype=np.uint16)
    wave_cursor = 0
    for idx in range(total_records):
        length = int(adjusted_lengths[idx])
        ch = channels[int(source_channel[idx])][0]
        wave = ch["wave"][int(source_row[idx])]
        if length > 0:
            wave_pool[wave_cursor : wave_cursor + length] = _clip_wave_to_uint16(wave[:length])
        records["wave_offset"][idx] = wave_cursor
        wave_cursor += length

    return RecordsBundle(records=records, wave_pool=wave_pool)


@export
def build_records_from_st_waveforms(
    st_waveforms: np.ndarray,
    default_dt_ns: int = 1,
) -> RecordsBundle:
    """
    Build records + wave_pool from st_waveforms.

    Baseline implementation: single pass, sorted by (timestamp, pid, board, channel).
    """
    channels = [
        (st_waveforms[indices], hw_channel)
        for hw_channel, indices in _hardware_channel_index_groups(st_waveforms)
    ]
    return _build_records_from_channels(channels, default_dt_ns=default_dt_ns)


def _process_v1725_file_to_disk(
    file_path: str,
    reader,
    dt_ns: int,
    part_dir: Path,
    part_idx: int,
    part_size: int | None = 100_000,
    use_parallel_metadata: bool = False,
) -> list[_RecordsPartRef]:
    """
    处理单个 V1725 文件并写入磁盘。

    Args:
        file_path: 文件路径
        reader: V1725Reader 实例
        dt_ns: 采样间隔（纳秒）
        part_dir: 分片输出目录
        part_idx: 分片索引
        part_size: 每个中间分片的 wave 数；<=0 表示单文件一个分片

    Returns:
        _RecordsPartRef 列表
    """
    file_part_dir = part_dir / f"file_{part_idx}"
    file_part_dir.mkdir(parents=True, exist_ok=True)
    effective_part_size = (
        None
        if part_size is None or part_size <= 0
        else min(int(part_size), _MAX_V1725_IN_MEMORY_WAVES)
    )

    part_refs: list[_RecordsPartRef] = []
    wave_batch = []
    local_part_idx = 0

    def flush_batch() -> None:
        nonlocal local_part_idx
        if not wave_batch:
            return
        bundle = _build_v1725_records_part_from_waves(
            wave_batch,
            default_dt_ns=dt_ns,
            use_parallel_metadata=use_parallel_metadata,
        )
        part_ref = _write_records_part(bundle, file_part_dir, local_part_idx)
        local_part_idx += 1
        if part_ref is not None:
            part_refs.append(part_ref)
        wave_batch.clear()

    for wave in reader.iter_waves([file_path]):
        wave_batch.append(wave)
        if effective_part_size is not None and len(wave_batch) >= effective_part_size:
            flush_batch()

    flush_batch()
    return part_refs


def _build_v1725_records_part_from_waves(
    waves: Sequence[object],
    default_dt_ns: int,
    use_parallel_metadata: bool = False,
) -> RecordsBundle:
    if not waves:
        return RecordsBundle(np.zeros(0, dtype=RECORDS_DTYPE), np.zeros(0, dtype=np.uint16))

    from waveform_analysis.acquisition.formats.v1725_numba import (
        fill_v1725_records_metadata_parallel,
        fill_v1725_records_metadata_serial,
    )

    n_records = len(waves)
    boards = np.empty(n_records, dtype=np.int16)
    channels = np.empty(n_records, dtype=np.int16)
    timestamp_ticks = np.empty(n_records, dtype=np.int64)
    baselines = np.empty(n_records, dtype=np.uint16)
    truncs = np.empty(n_records, dtype=np.bool_)
    event_lengths = np.empty(n_records, dtype=np.int32)
    wave_refs = []

    for idx, wave in enumerate(waves):
        waveform = np.asarray(wave.waveform)
        length = int(len(waveform))
        if length > np.iinfo(np.int32).max:
            raise ValueError("event_length exceeds int32 range")
        boards[idx] = np.int16(wave.board)
        channels[idx] = np.int16(wave.channel)
        timestamp_ticks[idx] = np.int64(wave.timestamp)
        baselines[idx] = np.uint16(wave.baseline)
        truncs[idx] = bool(wave.trunc)
        event_lengths[idx] = np.int32(length)
        wave_refs.append(waveform)

    records = np.zeros(n_records, dtype=RECORDS_DTYPE)
    fill_metadata = (
        fill_v1725_records_metadata_parallel
        if use_parallel_metadata and n_records >= 4096
        else fill_v1725_records_metadata_serial
    )
    fill_metadata(
        timestamp_ticks,
        boards,
        channels,
        baselines,
        truncs,
        event_lengths,
        int(default_dt_ns),
        records["timestamp"],
        records["pid"],
        records["board"],
        records["channel"],
        records["baseline"],
        records["baseline_upstream"],
        records["dt"],
        records["trigger_type"],
        records["flags"],
        records["event_length"],
        records["time"],
    )
    records["polarity"] = "unknown"

    source_idx = np.arange(n_records, dtype=np.int64)
    order = _records_sort_order(records)
    records = records[order]
    source_idx = source_idx[order]

    total_samples = int(records["event_length"].astype(np.int64, copy=False).sum())
    wave_pool = np.zeros(total_samples, dtype=np.uint16)
    wave_cursor = 0
    for idx in range(n_records):
        length = int(records["event_length"][idx])
        if length > 0:
            wave = wave_refs[int(source_idx[idx])]
            wave_pool[wave_cursor : wave_cursor + length] = _clip_wave_to_uint16(wave[:length])
        records["wave_offset"][idx] = wave_cursor
        wave_cursor += length

    records["record_id"] = np.arange(n_records, dtype=np.int64)
    return RecordsBundle(records=records, wave_pool=wave_pool)


def _resolve_v1725_file_workers(file_count: int, n_jobs: int | None) -> int:
    if file_count <= 0:
        return 1
    if n_jobs is None:
        return min(file_count, 4)
    return max(int(n_jobs), 1)


def _reclaim_v1725_merge_inputs(
    part_dir: Path, parts: Sequence[_RecordsPartRef], result: RecordsBundleRef
) -> None:
    """Validate final output before removing intermediates owned by this V1725 build.

    This is deliberately outside the shared merger: its callers may still own
    and use input parts. The returned bundle retains ownership of final files.
    """
    if result.temp_dir != part_dir or len(result.part_refs) != 1:
        raise RuntimeError("V1725 merge did not return one owned final part")
    final = result.part_refs[0]
    total_records = sum(part.n_records for part in parts)
    total_samples = sum(part.n_samples for part in parts)
    if (result.total_records, result.total_samples) != (total_records, total_samples) or (
        final.n_records,
        final.n_samples,
    ) != (total_records, total_samples):
        raise RuntimeError("V1725 merged counts do not match input parts")

    final_paths = {final.records_path, final.wave_pool_path}
    if len(final_paths) != 2 or any(path.parent != part_dir / "merged" for path in final_paths):
        raise RuntimeError("V1725 final files are outside the owned merged directory")
    batch_dir = part_dir / "merged" / "records_only_batches"
    inputs = {path for part in parts for path in (part.records_path, part.wave_pool_path)}
    if batch_dir.exists():
        inputs.update(batch_dir.glob("records_merged_*.dat"))
    if inputs & final_paths:
        raise RuntimeError("V1725 final files overlap intermediate inputs")

    # Check every candidate before the first unlink; never follow a symlink out
    # of the private build directory or remove a caller's unrelated files.
    root = part_dir.resolve()
    for path in inputs | final_paths:
        if not path.is_file() or path.resolve() != path.absolute() or not path.is_relative_to(root):
            raise RuntimeError(f"V1725 merge file is not an owned regular file: {path}")
    for path, expected in (
        (final.records_path, total_records * RECORDS_DTYPE.itemsize),
        (final.wave_pool_path, total_samples * np.dtype(np.uint16).itemsize),
    ):
        if path.stat().st_size != expected:
            raise RuntimeError(f"V1725 merged file size mismatch: {path}")

    records = np.memmap(final.records_path, dtype=RECORDS_DTYPE, mode="r", shape=(total_records,))
    try:
        samples_seen = 0
        for start in range(0, total_records, 250_000):
            chunk = records[start : start + 250_000]
            offsets = chunk["wave_offset"]
            lengths = chunk["event_length"].astype(np.int64)
            if (
                np.any(offsets < 0)
                or np.any(offsets > total_samples)
                or np.any(lengths < 0)
                or np.any(lengths > total_samples - offsets)
            ):
                raise RuntimeError("V1725 merged wave offsets or lengths are invalid")
            samples_seen += int(lengths.sum())
        if samples_seen != total_samples:
            raise RuntimeError("V1725 merged event lengths do not match input sample count")
    finally:
        records._mmap.close()

    for path in sorted(inputs):
        path.unlink()
    for directory in {path.parent for path in inputs}:
        if not any(directory.iterdir()):
            directory.rmdir()


@export
def build_records_from_v1725_files(
    file_paths: list[str],
    dt_ns: int,
    n_jobs: int | None = None,
    executor_type: str = "thread",
    memory_budget_gb: float = 50.0,
    batch_size: int = 50,
    keep_on_disk: bool | None = None,
    v1725_part_size: int | None = 100_000,
    show_progress: bool = False,
    profiler=None,
) -> RecordsBundle | RecordsBundleRef:
    """
    从 V1725 文件构建 records + wave_pool。

    使用临时文件和 memmap 减少内存占用，支持文件级并行处理和分批合并。

    Args:
        file_paths: V1725 文件路径列表
        dt_ns: 采样间隔（纳秒）
        n_jobs: 并行 worker 数量（None=auto，1=串行）
        executor_type: 执行器类型（"thread" 或 "process"）
        memory_budget_gb: 内存预算（GB），用于智能选择合并策略
        batch_size: 分批合并时每批的分片数量
        keep_on_disk: 强制选择策略（None=自动，True=磁盘引用，False=加载内存）
        v1725_part_size: 单个 V1725 文件内每个中间分片的 wave 数；<=0 表示单文件一个分片
        show_progress: 是否显示文件级 records 构建进度
        profiler: Optional profiler for timing V1725 records build stages

    Returns:
        RecordsBundle 或 RecordsBundleRef
    """
    if not file_paths:
        return RecordsBundle(np.zeros(0, dtype=RECORDS_DTYPE), np.zeros(0, dtype=np.uint16))

    from waveform_analysis.acquisition.formats import get_adapter

    adapter = get_adapter("v1725")
    reader = adapter.format_reader
    if not hasattr(reader, "iter_waves"):
        raise RuntimeError("V1725 adapter does not provide iter_waves")

    import shutil

    part_dir = Path(tempfile.mkdtemp(prefix="v1725_parts_"))
    pbar = None
    if show_progress:
        try:
            from tqdm import tqdm

            pbar = tqdm(total=len(file_paths), desc="Building V1725 records", leave=False)
        except ImportError:
            pbar = None
    try:
        part_refs: list[_RecordsPartRef] = []
        timer = profiler.timeit if profiler else None

        # 确定并行度
        effective_workers = _resolve_v1725_file_workers(len(file_paths), n_jobs)

        with timer("records.v1725.build") if timer else nullcontext():
            if effective_workers <= 1 or len(file_paths) <= 1:
                # 串行处理
                for part_idx, file_path in enumerate(file_paths):
                    file_part_refs = _process_v1725_file_to_disk(
                        file_path,
                        reader,
                        dt_ns,
                        part_dir,
                        part_idx,
                        part_size=v1725_part_size,
                    )
                    part_refs.extend(file_part_refs)
                    if pbar is not None:
                        pbar.update(1)
            else:
                # 并行处理
                from concurrent.futures import as_completed

                from waveform_analysis.core.execution.manager import get_executor

                max_workers = min(effective_workers, len(file_paths))
                with get_executor(
                    "v1725_file_build",
                    executor_type=executor_type,
                    max_workers=max_workers,
                    reuse=True,
                ) as executor:
                    futures = {
                        executor.submit(
                            _process_v1725_file_to_disk,
                            file_path,
                            reader,
                            dt_ns,
                            part_dir,
                            idx,
                            v1725_part_size,
                        ): idx
                        for idx, file_path in enumerate(file_paths)
                    }

                    # 收集结果并按原始顺序排序（避免竞态条件）
                    results = []
                    for future in as_completed(futures):
                        idx = futures[future]
                        file_part_refs = future.result()
                        if file_part_refs:
                            results.append((idx, file_part_refs))
                        if pbar is not None:
                            pbar.update(1)

                    # 按索引排序以保持确定性顺序
                    results.sort(key=lambda x: x[0])
                    part_refs = [ref for _, refs in results for ref in refs]

        # 智能合并分片
        if not part_refs:
            shutil.rmtree(part_dir, ignore_errors=True)
            return RecordsBundle(np.zeros(0, dtype=RECORDS_DTYPE), np.zeros(0, dtype=np.uint16))

        with timer("records.merge") if timer else nullcontext():
            result = _merge_records_part_refs(
                part_refs,
                memory_budget_gb=memory_budget_gb,
                batch_size=batch_size,
                keep_on_disk=keep_on_disk,
                output_dir=part_dir,
                transfer_temp_dir_ownership=True,
                show_progress=show_progress,
                n_workers=n_jobs,
                executor_type=executor_type,
                profiler=profiler,
            )

        # 如果返回 RecordsBundle（内存模式），清理临时目录
        if isinstance(result, RecordsBundle):
            shutil.rmtree(part_dir, ignore_errors=True)
        else:
            # Workers and merge helpers have closed their intermediate mappings.
            # Keep only validated final files until the returned ref is cleaned up.
            _reclaim_v1725_merge_inputs(part_dir, part_refs, result)

        return result
    except Exception:
        shutil.rmtree(part_dir, ignore_errors=True)
        raise
    finally:
        if pbar is not None:
            pbar.close()


@export
def build_records_from_raw_files(
    raw_files: list[list[str]],
    adapter_name: str,
    default_dt_ns: int = 1,
    part_size: int | None = 250_000,
    baseline_samples: int | tuple[int, int] | list[int] | None = None,
    epoch_ns: int | None = None,
    show_progress: bool = False,
    parse_engine: str | None = "auto",
    n_jobs: int | None = None,
    chunksize: int | None = None,
    use_process_pool: bool = False,
    channel_workers: int | None = None,
    channel_executor: str = "thread",
    profiler=None,
    keep_on_disk: bool | None = None,
    memory_budget_gb: float = 50.0,
) -> RecordsBundle | RecordsBundleRef:
    """Build records + wave_pool from raw files using the streaming part builder."""
    return build_records_from_raw_files_streaming(
        raw_files=raw_files,
        adapter_name=adapter_name,
        default_dt_ns=default_dt_ns,
        part_size=part_size,
        baseline_samples=baseline_samples,
        epoch_ns=epoch_ns,
        show_progress=show_progress,
        parse_engine=parse_engine,
        n_jobs=n_jobs,
        chunksize=chunksize,
        use_process_pool=use_process_pool,
        channel_workers=channel_workers,
        channel_executor=channel_executor,
        profiler=profiler,
        keep_on_disk=keep_on_disk,
        memory_budget_gb=memory_budget_gb,
    )


@export
def merge_records_parts(parts: Sequence[RecordsBundle]) -> RecordsBundle:
    """
    Merge sorted records parts and build a global wave_pool.

    Each part must have records sorted by (timestamp, pid, board, channel).
    """
    if not parts:
        return RecordsBundle(np.zeros(0, dtype=RECORDS_DTYPE), np.zeros(0, dtype=np.uint16))

    total_records = sum(len(part.records) for part in parts)
    if total_records == 0:
        return RecordsBundle(np.zeros(0, dtype=RECORDS_DTYPE), np.zeros(0, dtype=np.uint16))

    total_samples = 0
    for part in parts:
        if len(part.records) > 0:
            total_samples += int(part.records["event_length"].astype(np.int64, copy=False).sum())

    records_out = np.zeros(total_records, dtype=RECORDS_DTYPE)
    wave_pool_out = np.zeros(total_samples, dtype=np.uint16)

    import heapq

    heap = []
    for part_idx, part in enumerate(parts):
        if len(part.records) == 0:
            continue
        rec = part.records[0]
        key = (
            int(rec["timestamp"]),
            int(rec["pid"]),
            int(rec["board"]),
            int(rec["channel"]),
            part_idx,
            0,
        )
        heapq.heappush(heap, key)

    out_idx = 0
    wave_cursor = 0
    while heap:
        _, _, _, _, part_idx, row_idx = heapq.heappop(heap)
        part = parts[part_idx]
        rec = part.records[row_idx]
        length = int(rec["event_length"])
        if length < 0:
            length = 0

        if length > 0:
            offset = int(rec["wave_offset"])
            wave_pool_out[wave_cursor : wave_cursor + length] = part.wave_pool[
                offset : offset + length
            ]

        records_out[out_idx] = rec
        records_out[out_idx]["wave_offset"] = wave_cursor
        out_idx += 1
        wave_cursor += length

        next_row = row_idx + 1
        if next_row < len(part.records):
            next_rec = part.records[next_row]
            key = (
                int(next_rec["timestamp"]),
                int(next_rec["pid"]),
                int(next_rec["board"]),
                int(next_rec["channel"]),
                part_idx,
                next_row,
            )
            heapq.heappush(heap, key)

    record_ids = records_out["record_id"].astype(np.int64, copy=False)
    if len(np.unique(record_ids)) != len(record_ids):
        records_out["record_id"] = np.arange(total_records, dtype=np.int64)
    return RecordsBundle(records=records_out, wave_pool=wave_pool_out)


@export
def build_records_from_st_waveforms_sharded(
    st_waveforms: np.ndarray,
    part_size: int = 200_000,
    default_dt_ns: int = 1,
) -> RecordsBundle:
    """
    Build records + wave_pool using sharded parts and k-way merge.

    part_size controls the max number of events per part. If part_size <= 0 or
    total records <= part_size, this falls back to the baseline builder.
    """
    if part_size is None or part_size <= 0:
        return build_records_from_st_waveforms(st_waveforms, default_dt_ns=default_dt_ns)

    total_records = len(st_waveforms)
    if total_records <= part_size:
        return build_records_from_st_waveforms(st_waveforms, default_dt_ns=default_dt_ns)

    parts: list[RecordsBundle] = []
    for hw_channel, indices in _hardware_channel_index_groups(st_waveforms):
        count = len(indices)
        if count == 0:
            continue
        start = 0
        while start < count:
            stop = min(count, start + part_size)
            shard_indices = indices[start:stop]
            part = _build_records_from_channels(
                [(st_waveforms[shard_indices], hw_channel)],
                default_dt_ns=default_dt_ns,
            )
            if len(part.records) > 0:
                parts.append(part)
            start = stop

    if not parts:
        return RecordsBundle(np.zeros(0, dtype=RECORDS_DTYPE), np.zeros(0, dtype=np.uint16))
    if len(parts) == 1:
        return parts[0]
    return merge_records_parts(parts)
