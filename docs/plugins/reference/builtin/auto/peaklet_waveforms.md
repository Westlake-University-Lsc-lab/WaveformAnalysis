---
schema_version: 2
document_type: "plugin_reference"
profile: "auto"
provides: "peaklet_waveforms"
plugin_class: "PeakletWaveformPlugin"
module: "waveform_analysis.core.plugins.builtin.peaklet_waveforms.plugin"
version: "2.1.2"
summary: "Build peaklet waveform index rows from records-backed hit_merged samples. Supports cross-record hits via component expansion."
depends_on: []
declared_depends_on: []
resolved_depends_on: ["peaklets", "peaklet_components", "hit_merged", "hit_merged_components", "hit_threshold", "records", "wave_pool"]
dependency_profile: "documentation-default-v1"
dependency_profile_values: {"daq_adapter": "vx2730", "use_filtered": false, "wave_source": "records"}
dependency_config_keys: ["clip_negative_signal", "use_filtered"]
output_kind: "structured_array"
execution_kind: "static"
narrative_source: "source"
narrative_source_reason: null
source_fingerprint: "0c629f8740dac250196690123e37d6f58834d750e168bcfa84f50a23fb34e158"
generated: true
---
# peaklet_waveforms

## Overview

Build peaklet waveform index rows from records-backed hit_merged samples. Supports cross-record hits via component expansion.
`peaklet_waveforms` 将 peaklet 的 `hit_merged` 组件还原为按绝对时间对齐的 ragged 求和波形，并与 `peaklet_waveform_pool` 在同一次构建中写入。输入先按 peaklet、硬件键 `(board, channel)` 与绝对起点整理；普通单记录、无同通道重叠的 peaklet 使用轻量 Numba 累加，cross-record 或重叠 peaklet 使用严格的 canonical Numba 合并。

| Item | Value |
| --- | --- |
| Provides | `peaklet_waveforms` |
| Plugin Class | `PeakletWaveformPlugin` |
| Module | `waveform_analysis.core.plugins.builtin.peaklet_waveforms.plugin` |
| Version | `2.1.2` |
| Category | 峰构建 |
| Output Container | `structured_array` |
| Execution Mode | `static` |
| Save Policy | `always` |
| Uses Run Config | no |
| Timeout | `none` |
| Side Effect | no |
| Narrative Source | `source` |
| Source Fingerprint | `0c629f8740dac250196690123e37d6f58834d750e168bcfa84f50a23fb34e158` |

### Dependencies

默认文档画像：`documentation-default-v1`（{"daq_adapter": "vx2730", "use_filtered": false, "wave_source": "records"}）。
该插件通过 `resolve_depends_on(context, run_id)` 动态解析依赖；可能影响解析的配置键：`clip_negative_signal`, `use_filtered`。

| Dependency | Version Constraint | Resolution | Required Fields | Description |
| --- | --- | --- | --- | --- |
| `peaklets` | - | dynamic-default | - | Build lightweight cross-channel peaklets from hit_merged intervals. |
| `peaklet_components` | - | dynamic-default | - | Return per-peaklet component hit_merged indices. |
| `hit_merged` | - | dynamic-default | - | Merge nearby threshold hits per channel with time-gap and max-width constraints. |
| `hit_merged_components` | - | dynamic-default | - | Return per-cluster component hit indices for hit_merged rows. |
| `hit_threshold` | - | dynamic-default | - | Threshold-only hit detector with THRESHOLD_HIT_DTYPE output. |
| `records` | - | dynamic-default | - | Build records (event index table) from the shared internal records bundle. |
| `wave_pool` | - | dynamic-default | - | Build wave_pool from the shared internal records bundle. |
### How It Works

1. 读取 peaklets、peaklet_components、hit_merged、records 与所选 wave pool；cross-record merged 行通过 hit_merged_components 展开为 threshold-hit 片段。
2. 将片段按 `(peaklet_id, board, channel, absolute_start)` 排序并构建每个 peaklet 的 CSR 范围。
3. Numba 分类阶段验证 record、dt、pool 边界、有限采样和共同绝对时间网格，同时计算输出行及 fast/canonical 路由。
4. fast 路径直接累加无重叠的片段；canonical 路径按硬件通道使用 occupancy buffer 去重后，按确定顺序跨通道求和。
5. 以 float64 累加器完成通道求和、最终物化为 float32 pool；index 行和 pool 一起缓存，供 peaklet_features 与 peaklet_waveform_pool 消费。

## Configuration

| Name | Type | Default | Unit | Tracked | Deprecated | Description |
| --- | --- | --- | --- | --- | --- | --- |
| `use_filtered` | `bool` | `False` | - | yes | no | 选择 wave_pool_filtered 而非原始 wave_pool；此选择参与 cache lineage。 |
| `clip_negative_signal` | `bool` | `False` | - | yes | no | 控制 canonical 与 fast 路径共同使用的采样裁剪口径，默认 False。 |
| `debug_numba` | `bool` | `False` | - | yes | no | 仅用于排查 Numba 内部异常；契约性输入错误始终直接抛出。 |
| `log_waveform_diagnostics` | `bool` | `False` | - | yes | no | 记录 fast/canonical/fallback peaklet 数、输入与唯一采样数、展开/排序/物化分阶段耗时以及 JIT signature 状态。 |
| `n_workers` | `int` | `1` | - | yes | no | 保留公开兼容；只用于 Python canonical fallback，不改变 Numba routed 路径的并行度。 |
| `parallel_threshold` | `int` | `5000` | - | yes | no | 仅控制 Python fallback 何时尝试 process pool。 |
## Output

structured_array output with fields: peak_id, time_start, time_end, dt, wave_offset, wave_length.

| Field | DType | Unit | Meaning |
| --- | --- | --- | --- |
| `peak_id` | `int64` | None | Peaklet identifier, matching the row index in the peaklets table |
| `time_start` | `int64` | ps | Absolute start time of the waveform slice (ps) |
| `time_end` | `int64` | ps | Absolute end time of the waveform slice (ps) |
| `dt` | `int32` | ns | Sample interval in nanoseconds |
| `wave_offset` | `int64` | None | Starting index in peaklet_waveform_pool |
| `wave_length` | `int32` | samples | Number of samples in the waveform slice |
## Usage

### Minimal Example

```python
from waveform_analysis import Context
from waveform_analysis.plugins import profiles

ctx = Context(config={"data_root": "DAQ", "daq_adapter": "vx2730"})
ctx.register(*profiles.cpu_default())
result = ctx.get_data("run_001", "peaklet_waveforms")
```

示例使用 `run_id="run_001"` 和文档默认运行画像；真实数据路径与配置应以当前实验设置为准。

### Downstream Consumers

- `peaklet_features`
- `peaklet_waveform_pool`
