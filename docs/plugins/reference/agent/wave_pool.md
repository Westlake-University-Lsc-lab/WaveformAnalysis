---
schema_version: 2
document_type: "plugin_reference"
profile: "agent"
provides: "wave_pool"
plugin_class: "WavePoolPlugin"
module: "waveform_analysis.core.plugins.builtin.wave_pool.plugin"
version: "0.14.4"
summary: "Build wave_pool from the shared internal records bundle."
depends_on: []
declared_depends_on: []
resolved_depends_on: ["raw_files"]
dependency_profile: "documentation-default-v1"
dependency_profile_values: {"daq_adapter": "vx2730", "use_filtered": false, "wave_source": "records"}
dependency_config_keys: ["daq_adapter", "input_source"]
output_kind: "array"
execution_kind: "static"
narrative_source: "source"
narrative_source_reason: null
source_fingerprint: "ec6c0538db1567e967a6269e93dbcfc4aa2adb5efd92cbbe73327438c8831b59"
generated: true
---
# wave_pool

## Overview

Build wave_pool from the shared internal records bundle.
WavePoolPlugin 把共享 RecordsBundle 中的原始 ADC 波形样本暴露为正式的 `wave_pool` 插件输出。wave_pool 是一维 uint16 数组，事件波形通过 `records['wave_offset']` 与 `records['event_length']` 切片获得，因此它必须与 `records` 保持行对齐的索引约定。

与 `records` 一样，wave_pool 复用同一份内存或磁盘 bundle：单分片 `RecordsBundleRef` 时直接 memmap `wave_pool_path`，避免二次拷贝；`lineage_virtual=True` 标记其血缘推导为虚拟，配置与血缘共享 `records` 插件的来源，避免两处配置漂移。

它是 records-backed 波形访问与滤波产物（如 `wave_pool_filtered`）的直接数据源，也是 peaklet 波形还原（peaklet_waveforms，在 use_filtered=False 时）的原始波形池。

| Item | Value |
| --- | --- |
| Provides | `wave_pool` |
| Plugin Class | `WavePoolPlugin` |
| Module | `waveform_analysis.core.plugins.builtin.wave_pool.plugin` |
| Version | `0.14.4` |
| Category | 波形处理 |
| Output Container | `array` |
| Execution Mode | `static` |
| Save Policy | `always` |
| Uses Run Config | yes |
| Timeout | `none` |
| Side Effect | no |
| Narrative Source | `source` |
| Source Fingerprint | `ec6c0538db1567e967a6269e93dbcfc4aa2adb5efd92cbbe73327438c8831b59` |

### Dependencies

默认文档画像：`documentation-default-v1`（{"daq_adapter": "vx2730", "use_filtered": false, "wave_source": "records"}）。
该插件通过 `resolve_depends_on(context, run_id)` 动态解析依赖；可能影响解析的配置键：`daq_adapter`, `input_source`。

| Dependency | Version Constraint | Resolution | Required Fields | Description |
| --- | --- | --- | --- | --- |
| `raw_files` | - | dynamic-default | - | Scan the data directory and group raw CSV files by channel number. |
### How It Works

1. 解析共享 bundle：调用 `get_records_bundle(context, run_id)` 获取本 run 的 RecordsBundle / RecordsBundleRef。
2. 选择波形池视图：`RecordsBundleRef` 单分片时直接 memmap `wave_pool_path`（uint16, shape=(n_samples,)），内存 bundle 直接返回 `bundle.wave_pool`。
3. 返回结果：输出一维 uint16 数组，供 `records` 的 `wave_offset`/`event_length` 切片访问。

## Configuration

| Name | Type | Default | Unit | Tracked | Deprecated | Description |
| --- | --- | --- | --- | --- | --- | --- |
| `daq_adapter` | `str` | `vx2730` | - | yes | no | DAQ 适配器名称，决定 bundle 的解析路径（vx2730/v1725 等）；与 records 共享。 |
| `channel_workers` | `any` | `None` | - | no | no | Workers for channel-level waveform loading (None=auto). |
| `channel_executor` | `str` | `thread` | - | no | no | Executor type for channel-level loading and records merge: 'thread' or 'process'. |
| `n_jobs` | `int` | `None` | - | no | no | Workers per channel for file-level parsing; V1725 None=auto caps file readers at 4. |
| `use_process_pool` | `bool` | `False` | - | no | no | Use a process pool for file-level parsing (False=thread pool). |
| `chunksize` | `int` | `None` | - | no | no | CSV read chunk size; None reads full file (PyArrow if available). |
| `parse_engine` | `str` | `auto` | - | no | no | CSV engine: auto \| polars \| pyarrow \| pandas |
| `records_part_size` | `int` | `250000` | - | yes | no | Max events per records shard; <=0 disables sharding. |
| `v1725_part_size` | `int` | `100000` | - | yes | no | Max V1725 waves per per-file records shard; <=0 uses one shard per file. |
| `keep_on_disk` | `any` | `None` | - | yes | no | 是否保持 bundle 磁盘驻留；None 时 V1725 默认 True、其余适配器默认 False。 |
| `memory_budget_gb` | `float` | `50.0` | - | yes | no | Memory budget in GB for in-memory records bundle materialization. |
| `dt` | `int` | `None` | - | yes | no | 采样间隔（ns），写回 records.dt；缺省取适配器采样率或 1ns。 |
| `baseline_samples` | `any` | `None` | - | yes | no | 基线范围（int 或 (start, end)），在 bundle 构建时同步用于 records。 |
| `input_source` | `str` | `raw_files` | - | yes | no | records bundle 输入源：raw_files 或 st_waveforms（V1725 仅支持 raw_files）。 |
## Output

array output with fields: value.

| Field | DType | Unit | Meaning |
| --- | --- | --- | --- |
| `value` | `uint16` | ADC counts | Flattened uint16 ADC sample value |
## Usage

### Minimal Example

```python
from waveform_analysis import Context
from waveform_analysis.plugins import profiles

ctx = Context(config={"data_root": "DAQ", "daq_adapter": "vx2730"})
ctx.register(*profiles.cpu_default())
result = ctx.get_data("run_001", "wave_pool")
```

示例使用 `run_id="run_001"` 和文档默认运行画像；真实数据路径与配置应以当前实验设置为准。

## Operational Notes

### Behavior

- The returned array is a flat `uint16` pool; per-event waveforms are slices `pool[offset : offset + length]`.
- `wave_pool` is paired 1:1 with `records` through `wave_offset`/`event_length`; keeping both plugins consistent is part of the shared bundle contract.
- Config resolution follows the `records` plugin (`_resolve_bundle_config_plugin`) so the two outputs never drift in dtype/dt/bundle semantics.
### Failure Modes

- `RecordsBundleRef` 为多分片且未合并为单分片视图时，`_wave_pool_from_bundle` 抛出 `ValueError`（wave_pool 要求单分片 memmap 视图）。
- 上游 bundle 缺失或 `input_source` 非法时，由共享 bundle 构建逻辑抛出 `ValueError`。
### Downstream Impact

直接消费者：`basic_features`、`hit`、`hit_merged_features`、`hit_threshold`、`peaklet_channels`、`peaklet_waveforms`、`records_asymmetry_mask`、`wave_pool_filtered`、`waveform_width_integral`- `wave_pool_filtered` 以 wave_pool 为输入做滤波，输出同为 records 对齐的 float32 池。
- `peaklet_waveforms` 在 `use_filtered=False` 时直接消费 wave_pool；池的索引约定必须与 records 保持一致。

## Maintenance

### Change Playbook

1. 保持 `provides`、依赖和输出字段语义稳定，或同步所有下游消费者。
2. 行为、配置或输出契约改变时升级插件 `version`。
3. 修改插件源码后重新生成 Auto、Agent 和 HTML 参考。
### Validation

```bash
waveform-docs generate plugins-agent --plugin wave_pool
waveform-docs check coverage --strict --fail-on-warning
```
