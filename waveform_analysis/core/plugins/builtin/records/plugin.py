"""RecordsPlugin 类实现 - 从共享 RecordsBundle 暴露 records 元数据。"""

from typing import Any

import numpy as np

from waveform_analysis.core.plugins.builtin.records._compute import (
    _records_from_bundle,
    _RecordsBundlePluginBase,
    get_records_bundle,
)
from waveform_analysis.core.plugins.core.base import Option
from waveform_analysis.core.processing.dtypes import RECORDS_DTYPE


class RecordsPlugin(_RecordsBundlePluginBase):
    """Build records (event index table) from the shared internal bundle."""

    provides = "records"
    depends_on = []
    description = "Build records (event index table) from the shared internal records bundle."
    output_dtype = RECORDS_DTYPE
    agent_doc = {
        "overview": (
            "RecordsPlugin 是分析链最底层的基础插件，把共享的 RecordsBundle / RecordsBundleRef "
            "（由 raw_files 或 st_waveforms 构建）产出的记录元数据暴露为正式的 `records` 结构化"
            "数组。每条记录对应一次事件，包含时间戳、板卡/通道、基线、极性、触发类型、dt，以及"
            "指向 wave_pool 的 `wave_offset` 与 `event_length` 等关键索引字段。\n\n"
            "records 是绝大多数 records-backed 产物的源头：波形池的切片访问、通道角色掩码、"
            "不对称性筛选、滤波波形池与 peaklet 波形还原等都从 records 的行结构与字段约定取得"
            "语义。该插件不重复读取原始波形，而是复用同一份内存或磁盘 bundle（单分片直接 memmap "
            "records_path，多分片仅合并元数据视图），保证整条链共享一致的记录视图。\n\n"
            "插件通过 `_RecordsBundlePluginBase` 共享配置源，并支持 `input_source` 在 raw_files "
            "与 st_waveforms 之间切换（V1725 仅支持 raw_files）；`resolve_depends_on` 按所选输入源"
            "动态声明上游依赖。"
        ),
        "workflow_steps": [
            "解析共享 bundle：调用 `get_records_bundle(context, run_id)` 获取（必要时构建）本 run 的 RecordsBundle / RecordsBundleRef。",
            "选择元数据视图：单分片 `RecordsBundleRef` 时直接 memmap `records_path`，多分片时通过 `get_records_view()` 仅合并 records 元数据（不载入 wave_pool），内存 bundle 直接返回 `bundle.records`。",
            "返回结果：输出按行对齐的 `RECORDS_DTYPE` 元数据数组，行序即后续 `record_id` 对齐约定。",
        ],
        "behavior_notes": [
            "Non-V1725 raw_files builds honor keep_on_disk and memory_budget_gb; disk output stays readable until bundle cleanup, while temporary input shards are reclaimed after merging.",
            "V1725 disk builds validate merged counts, file sizes and wave bounds before reclaiming owned input shards and records-only batches; final files remain readable until bundle cleanup.",
            "The plugin never re-parses raw waveforms itself; it only materializes the record metadata view from the shared bundle.",
            "Single-part `RecordsBundleRef` returns a memmap over `records_path` (zero-copy); multi-part falls back to a merged metadata-only view.",
            "`wave_offset` + `event_length` references into the `wave_pool` array (uint16), so `records` and `wave_pool` must stay index-consistent.",
            "Polarity and baseline enrichment are applied while the shared bundle is built, not inside this plugin's compute.",
        ],
        "field_notes": {
            "timestamp": "ADC 时间戳（ps）。",
            "pid": "分区 id（partition id），作为排序平局决胜字段。",
            "board": "板卡索引（int16）。",
            "channel": "物理通道号（int16）。",
            "baseline": "基线（float64），由 bundle 构建阶段计算。",
            "baseline_upstream": "上游插件写入的基线（float64，可选）。",
            "polarity": "硬件真实极性（'positive' | 'negative' | 'unknown' 或按设备约定）。",
            "record_id": "排序后的顺序记录 id（int64）。",
            "dt": "采样间隔（ns，与 time 对齐，int32）。",
            "trigger_type": "触发类型码（int16）。",
            "flags": "位标志（uint32）。",
            "wave_offset": "事件波形在 wave_pool 中的起始索引（int64）。",
            "event_length": "事件波形长度（采样点数，int32）。",
            "time": "系统时间（ns，语义可选）。",
        },
        "config_notes": {
            "daq_adapter": "DAQ 适配器名称（vx2730/v1725 等），决定 bundle 的解析路径与默认 dt。",
            "input_source": "records bundle 输入源：'raw_files' 或 'st_waveforms'（V1725 仅支持 'raw_files'）。",
            "dt": "采样间隔（ns），写回 records.dt；缺省取适配器采样率或 1ns。",
            "keep_on_disk": "是否保持 bundle 磁盘驻留；None 时 V1725 默认 True、其余适配器默认 False。",
            "memory_budget_gb": "非 V1725 raw_files 在 keep_on_disk=None 时检查内存预算（GB）；显式 True/False 沿用磁盘/内存模式。",
            "baseline_samples": "基线范围：int（距适配器起始的采样数）或 (start, end) 元组，相对 samples_start。",
            "channel_workers / channel_executor / n_jobs / use_process_pool": "通道级与文件级加载/合并不的并行控制参数（不参与血缘 track）。",
            "v1725_part_size": "V1725 每文件 records 分片的最大波形数；<=0 表示每文件一个分片。",
        },
        "failure_modes": [
            "所选 `input_source` 非 'raw_files'/'st_waveforms' 时抛出 `ValueError`。",
            "V1725 使用 `input_source='st_waveforms'` 时抛出 `ValueError`（不支持该组合）。",
            "上游 `raw_files` 数据缺失（非 list）时由共享 bundle 构建逻辑抛出 `ValueError`。",
            "多分片 bundle 的元数据视图只合并 records、不合并 wave_pool，若下游误按 records 行取波形会越界——属消费方契约错误，本插件不单独拦截。",
        ],
        "downstream_consumers": [
            "records_asymmetry_mask",
            "records_detector_mask",
            "records_veto_mask",
            "wave_pool_filtered",
            "peaklet_waveforms",
        ],
        "downstream_notes": [
            "行序与 `record_id` 语义的变更会影响所有 mask 类产物（其输出长度必须与 records 一致）以及 align 到 records 的派生数组。",
            "`wave_offset`/`event_length` 与 `wave_pool` 的索引一致性由下游切片访问共享，修改 records 布局需同步校验 `wave_pool_filtered` 与 `peaklet_waveforms`。",
        ],
        "agent_change_notes": [
            "RECORDS_DTYPE 字段或行序变化会级联影响 records 的 mask/滤波/peaklet 消费链，请同步运行对应定向测试并重新生成文档。",
        ],
    }
    options = {
        **_RecordsBundlePluginBase.options,
        "channel_workers": Option(
            default=16,
            help="Workers for channel-level waveform loading.",
            track=False,
        ),
        "channel_executor": Option(
            default="process",
            type=str,
            help="Executor type for channel-level loading and records merge: 'thread' or 'process'.",
            track=False,
        ),
        "n_jobs": Option(
            default=16,
            type=int,
            help="Workers per channel for file-level parsing; V1725 None=auto caps file readers at 4.",
            track=False,
        ),
        "use_process_pool": Option(
            default=True,
            type=bool,
            help="Use a process pool for file-level parsing (False=thread pool).",
            track=False,
        ),
        "v1725_part_size": Option(
            default=20_000,
            type=int,
            help="Max V1725 waves per per-file records shard; <=0 uses one shard per file.",
        ),
        "keep_on_disk": Option(
            default=True,
            type=None,
            validate=lambda v: v is None or isinstance(v, bool),
            help="Keep merged records bundle disk-backed. None defaults to True for V1725 and False otherwise.",
        ),
    }

    def compute(self, context: Any, run_id: str, **kwargs) -> np.ndarray:
        bundle = get_records_bundle(context, run_id)
        return _records_from_bundle(bundle)


__all__ = ["RecordsPlugin"]
