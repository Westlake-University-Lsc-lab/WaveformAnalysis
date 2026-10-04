---
schema_version: 2
document_type: "plugin_reference"
profile: "auto"
provides: "events"
plugin_class: "EventPlugin"
module: "waveform_analysis.core.plugins.builtin.events.plugin"
version: "0.0.3"
summary: "Complete event reconstruction from S1-S2 pairs and position"
depends_on: ["s1_s2_pairs", "position_reconstruction"]
declared_depends_on: ["s1_s2_pairs", "position_reconstruction"]
resolved_depends_on: ["s1_s2_pairs", "position_reconstruction"]
dependency_profile: "declared"
dependency_profile_values: {}
dependency_config_keys: []
output_kind: "structured_array"
execution_kind: "static"
narrative_source: "source"
narrative_source_reason: null
source_fingerprint: "f40afade8aefd9aff2e666625178afd057773f52f5b0678ca8370cedccac5c11"
generated: true
---
# events

## Overview

Complete event reconstruction from S1-S2 pairs and position
完整事件重建插件

整合 S1-S2 配对和位置重建，输出完整的物理事件记录。

输入: - s1_s2_pairs: S1-S2 配对结果 - position_reconstruction: 位置重建结果

输出: - events: 完整事件记录

v0.0.0 状态: - 仅建立数据结构和依赖关系 - 基本特征从输入数据复制 - 拓扑特征使用占位值（预留接口） - 质量评估使用简单阈值检查

未来版本计划: - v0.1.0: 基础特征提取和质量检查 - v0.2.0: 拓扑特征计算（需要通道波形信息） - v0.3.0: 高级质量评估 - v1.0.0: 完整的事件重建和分类

| Item | Value |
| --- | --- |
| Provides | `events` |
| Plugin Class | `EventPlugin` |
| Module | `waveform_analysis.core.plugins.builtin.events.plugin` |
| Version | `0.0.3` |
| Category | 事件分析 |
| Output Container | `structured_array` |
| Execution Mode | `static` |
| Save Policy | `always` |
| Uses Run Config | no |
| Timeout | `none` |
| Side Effect | no |
| Narrative Source | `source` |
| Source Fingerprint | `f40afade8aefd9aff2e666625178afd057773f52f5b0678ca8370cedccac5c11` |

### Dependencies

默认文档画像：`declared`。

| Dependency | Version Constraint | Resolution | Required Fields | Description |
| --- | --- | --- | --- | --- |
| `s1_s2_pairs` | - | declared | - | Select best S1-S2 pairs from candidates |
| `position_reconstruction` | - | declared | - | Reconstruct 3D position from S1-S2 pairs using CoG, Jun or Junshi |
### How It Works

1. 执行完整事件重建
2. v0.0.0 实现: 1. 关联 pairs 和 positions（通过 pair_id） 2. 复制基本特征 3. 设置拓扑特征占位值 4. 应用简单质量标志

## Configuration

| Name | Type | Default | Unit | Tracked | Deprecated | Description |
| --- | --- | --- | --- | --- | --- | --- |
| `min_s1` | `float` | `0.0` | - | yes | no | 最小 S1 阈值（用于质量筛选）；范围：0.0 至 +∞ |
| `min_s2` | `float` | `0.0` | - | yes | no | 最小 S2 阈值（用于质量筛选）；范围：0.0 至 +∞ |
| `fiducial_radius` | `(<class 'float'>, <class 'NoneType'>)` | `None` | - | yes | no | 基准体积半径 (mm)。None 表示不应用 |
| `fiducial_z_range` | `(<class 'tuple'>, <class 'NoneType'>)` | `None` | - | yes | no | 基准体积 Z 范围 (z_min, z_max) mm。None 表示不应用 |
## Output

structured_array output with fields: event_id, event_number, run_id, pair_id, s1_peak_id, s2_peak_id, x, y, ....

| Field | DType | Unit | Meaning |
| --- | --- | --- | --- |
| `event_id` | `int64` | None | Globally unique event identifier |
| `event_number` | `int64` | None | Sequential event number within the run, 0-based |
| `run_id` | `<U32` | None | Run identifier string |
| `pair_id` | `int64` | None | S1-S2 pair identifier |
| `s1_peak_id` | `int64` | None | S1 peak identifier |
| `s2_peak_id` | `int64` | None | S2 peak identifier |
| `x` | `float32` | mm | X coordinate (mm) |
| `y` | `float32` | mm | Y coordinate (mm) |
| `z` | `float32` | mm | Z coordinate (drift distance, mm) |
| `r` | `float32` | mm | Radial coordinate sqrt(x^2 + y^2) (mm) |
| `drift_time_ns` | `float32` | ns | Drift time (ns) |
| `s1_time` | `float64` | ps | S1 time relative to run start (ps) |
| `s2_time` | `float64` | ps | S2 time (ps) |
| `s1_area` | `float32` | ADC counts | S1 raw area |
| `s2_area` | `float32` | ADC counts | S2 raw area |
| `log10_s2_s1` | `float32` | None | log10(S2/S1) |
| `s1_n_channels` | `int16` | None | S1 channel count |
| `s2_n_channels` | `int16` | None | S2 channel count |
| `s1_area_fraction_top` | `float32` | None | S1 area fraction in top PMT array |
| `s2_area_fraction_top` | `float32` | None | S2 area fraction in top PMT array |
| `s1_rise_time` | `float32` | ns | S1 rise time (ns) |
| `s2_rise_time` | `float32` | ns | S2 rise time (ns) |
| `n_s1_candidates_for_s2` | `int32` | None | Number of S1 candidates for this S2 |
| `n_s2_candidates_for_s1` | `int32` | None | Number of S2 candidates for this S1 |
| `flags` | `uint32` | None | Bit-field status flags |
## Usage

### Minimal Example

```python
from waveform_analysis import Context
from waveform_analysis.plugins import profiles

ctx = Context(config={"data_root": "DAQ", "daq_adapter": "vx2730"})
ctx.register(*profiles.cpu_default())
result = ctx.get_data("run_001", "events")
```

示例使用 `run_id="run_001"` 和文档默认运行画像；真实数据路径与配置应以当前实验设置为准。

### Downstream Consumers

- 没有声明直接的内置消费者。
