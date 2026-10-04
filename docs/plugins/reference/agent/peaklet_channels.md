---
schema_version: 2
document_type: "plugin_reference"
profile: "agent"
provides: "peaklet_channels"
plugin_class: "PeakletChannelsPlugin"
module: "waveform_analysis.core.plugins.builtin.peaklet_channels.plugin"
version: "2.0.6"
summary: "Reconstruct deduplicated per-peaklet channel waveform contributions."
depends_on: ["peaklets", "peaklet_components", "hit_merged", "hit_merged_components", "hit_threshold", "hit_merged_features", "peaklet_features", "records", "wave_pool"]
declared_depends_on: ["peaklets", "peaklet_components", "hit_merged", "hit_merged_components", "hit_threshold", "hit_merged_features", "peaklet_features", "records", "wave_pool"]
resolved_depends_on: ["peaklets", "peaklet_components", "hit_merged", "hit_merged_components", "hit_threshold", "hit_merged_features", "peaklet_features", "records", "wave_pool"]
dependency_profile: "documentation-default-v1"
dependency_profile_values: {"daq_adapter": "vx2730", "use_filtered": false, "wave_source": "records"}
dependency_config_keys: ["clip_negative_signal", "use_filtered", "wave_source"]
output_kind: "structured_array"
execution_kind: "static"
narrative_source: "source"
narrative_source_reason: null
source_fingerprint: "ee6ab8fd8b5395b3fcedde316b8ab28a3db9f730cd0234eed1955b6c805c3d38"
generated: true
---
# peaklet_channels

## Overview

Reconstruct deduplicated per-peaklet channel waveform contributions.
Reconstruct peaklets into deduplicated per-board/channel contribution rows.

| Item | Value |
| --- | --- |
| Provides | `peaklet_channels` |
| Plugin Class | `PeakletChannelsPlugin` |
| Module | `waveform_analysis.core.plugins.builtin.peaklet_channels.plugin` |
| Version | `2.0.6` |
| Category | 峰构建 |
| Output Container | `structured_array` |
| Execution Mode | `static` |
| Save Policy | `always` |
| Uses Run Config | no |
| Timeout | `none` |
| Side Effect | no |
| Narrative Source | `source` |
| Source Fingerprint | `ee6ab8fd8b5395b3fcedde316b8ab28a3db9f730cd0234eed1955b6c805c3d38` |

### Dependencies

默认文档画像：`documentation-default-v1`（{"daq_adapter": "vx2730", "use_filtered": false, "wave_source": "records"}）。
该插件通过 `resolve_depends_on(context, run_id)` 动态解析依赖；可能影响解析的配置键：`clip_negative_signal`, `use_filtered`, `wave_source`。

| Dependency | Version Constraint | Resolution | Required Fields | Description |
| --- | --- | --- | --- | --- |
| `peaklets` | - | dynamic-default | - | Build lightweight cross-channel peaklets from hit_merged intervals. |
| `peaklet_components` | - | dynamic-default | - | Return per-peaklet component hit_merged indices. |
| `hit_merged` | - | dynamic-default | - | Merge nearby threshold hits per channel with time-gap and max-width constraints. |
| `hit_merged_components` | - | dynamic-default | - | Return per-cluster component hit indices for hit_merged rows. |
| `hit_threshold` | - | dynamic-default | - | Threshold-only hit detector with THRESHOLD_HIT_DTYPE output. |
| `hit_merged_features` | - | dynamic-default | - | Compute per-hit_merged local waveform features from records-backed samples. |
| `peaklet_features` | - | dynamic-default | - | Compute peaklet waveform features from ragged signal pools. |
| `records` | - | dynamic-default | - | Build records (event index table) from the shared internal records bundle. |
| `wave_pool` | - | dynamic-default | - | Build wave_pool from the shared internal records bundle. |
### How It Works

1. Reconstruct peaklets into deduplicated per-board/channel contribution rows.

## Configuration

| Name | Type | Default | Unit | Tracked | Deprecated | Description |
| --- | --- | --- | --- | --- | --- | --- |
| `wave_source` | `str` | `records` | - | yes | no | 波形来源；peaklet_channels 当前正式支持 records。 |
| `use_filtered` | `bool` | `False` | - | yes | no | 是否从 wave_pool_filtered 重建通道波形。 |
| `clip_negative_signal` | `bool` | `False` | - | yes | no | 是否在通道波形合并与积分前把负采样裁剪为 0。 |
## Output

structured_array output with fields: peaklet_id, board, channel, area, height, n_hits, area_fraction.

| Field | DType | Unit | Meaning |
| --- | --- | --- | --- |
| `peaklet_id` | `int64` | None | Peaklet identifier |
| `board` | `int16` | None | Hardware board index |
| `channel` | `int16` | None | Physical channel number |
| `area` | `float32` | ADC counts | Total area contribution from this channel |
| `height` | `float32` | ADC counts | Maximum height contribution from this channel |
| `n_hits` | `int32` | None | Number of component hits from this channel |
| `area_fraction` | `float32` | None | Fraction of the peaklet total area contributed by this channel |
## Usage

### Minimal Example

```python
from waveform_analysis import Context
from waveform_analysis.plugins import profiles

ctx = Context(config={"data_root": "DAQ", "daq_adapter": "vx2730"})
ctx.register(*profiles.cpu_default())
result = ctx.get_data("run_001", "peaklet_channels")
```

示例使用 `run_id="run_001"` 和文档默认运行画像；真实数据路径与配置应以当前实验设置为准。

## Operational Notes

### Behavior

- peaklet_channels bundle - provides 'peaklet_channels'。
### Failure Modes

- `peaklet_channels` 的实际输入由 `resolve_depends_on(context, run_id)` 决定；默认画像之外的配置需要重新确认依赖是否可用。
- 动态依赖无法解析、所需配置不合法或上游产物缺失时，插件不会生成有效输出。
### Downstream Impact

直接消费者：`peaks`、`position_reconstruction`
## Maintenance

### Change Playbook

1. 保持 `provides`、依赖和输出字段语义稳定，或同步所有下游消费者。
2. 行为、配置或输出契约改变时升级插件 `version`。
3. 修改插件源码后重新生成 Auto、Agent 和 HTML 参考。
### Validation

```bash
waveform-docs generate plugins-agent --plugin peaklet_channels
waveform-docs check coverage --strict --fail-on-warning
```
