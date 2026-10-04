---
schema_version: 2
document_type: "plugin_reference"
profile: "auto"
provides: "position_reconstruction"
plugin_class: "PositionReconstructionPlugin"
module: "waveform_analysis.core.plugins.builtin.position_reconstruction.plugin"
version: "0.5.0"
summary: "Reconstruct 3D position from S1-S2 pairs using vectorized CoG method"
depends_on: ["s1_s2_pairs", "peaklet_channels"]
declared_depends_on: ["s1_s2_pairs", "peaklet_channels"]
resolved_depends_on: ["s1_s2_pairs", "peaklet_channels"]
dependency_profile: "declared"
dependency_profile_values: {}
dependency_config_keys: []
output_kind: "structured_array"
execution_kind: "static"
narrative_source: "source"
narrative_source_reason: null
source_fingerprint: "682d2476ed6150ab6638657ae1f4ce9c91a20a24def28d8c71751419ff9e2760"
generated: true
---
# position_reconstruction

## Overview

Reconstruct 3D position from S1-S2 pairs using vectorized CoG method
位置重建插件（向量化优化版本）

从选定的 S1-S2 配对重建事件的三维空间位置。

输入: - s1_s2_pairs: S1-S2 配对结果（仅处理 selected=True 的配对）

输出: - position_reconstruction: 位置重建结果

v0.2.0 功能: - Z 坐标: 基于 drift_time * drift_velocity（向量化） - XY 坐标: 电荷重心法 (Center of Gravity, 向量化) * 批量加载所有通道数据 * 预计算 PMT 映射表 * 使用 NumPy 广播加速计算 - 质量评估: 边缘事件检测、低信号标记（向量化）

性能优化: - 避免 Python for 循环 - 批量处理所有事件 - 预计算和缓存映射关系 - 典型性能提升: 10-100x（取决于事件数）

未来版本计划: - v0.3.0: 高级 XY 重建算法 (ML, 模板匹配) - v1.0.0: 位置相关修正 (电场、光收集效率)

| Item | Value |
| --- | --- |
| Provides | `position_reconstruction` |
| Plugin Class | `PositionReconstructionPlugin` |
| Module | `waveform_analysis.core.plugins.builtin.position_reconstruction.plugin` |
| Version | `0.5.0` |
| Category | 其他 |
| Output Container | `structured_array` |
| Execution Mode | `static` |
| Save Policy | `always` |
| Uses Run Config | no |
| Timeout | `none` |
| Side Effect | no |
| Narrative Source | `source` |
| Source Fingerprint | `682d2476ed6150ab6638657ae1f4ce9c91a20a24def28d8c71751419ff9e2760` |

### Dependencies

默认文档画像：`declared`。

| Dependency | Version Constraint | Resolution | Required Fields | Description |
| --- | --- | --- | --- | --- |
| `s1_s2_pairs` | - | declared | - | Select best S1-S2 pairs from candidates |
| `peaklet_channels` | - | declared | - | Reconstruct deduplicated per-peaklet channel waveform contributions. |
### How It Works

1. 执行位置重建（向量化优化版本）
2. v0.2.0 实现: 1. 筛选 selected=True 的配对 2. 加载 PMT 几何布局 3. 计算 Z 坐标（向量化） 4. 计算 XY 坐标（批量向量化） 5. 设置质量标志位（向量化）
3. 性能优化： - 所有数组操作使用 NumPy 向量化 - 批量处理，避免 Python 循环 - 预计算映射表 - 典型加速比：10-100x

## Configuration

| Name | Type | Default | Unit | Tracked | Deprecated | Description |
| --- | --- | --- | --- | --- | --- | --- |
| `detector_geometry` | `dict` | `None` | - | yes | no | PMT 几何、硬件通道映射及相对增益；未设置时使用七 PMT 布局 |
| `drift_velocity` | `float` | `0.0013` | - | yes | no | 漂移速度 (mm/ns)，用于 Z 坐标计算。典型值：液氙 ~0.001 mm/ns, 液氩 ~0.0013 mm/ns；范围：0.0 至 +∞ |
| `min_s2_area_for_xy` | `float` | `100.0` | - | yes | no | XY 重建所需的最小 S2 面积 (ADC counts，与 s2_area 一致)；范围：0.0 至 +∞ |
| `edge_threshold_mm` | `float` | `5.0` | - | yes | no | 边缘事件判定阈值：距离 TPC 边界的最小距离 (mm)；范围：0.0 至 +∞ |
| `detector_radius_mm` | `float` | `62.5` | - | yes | no | 探测器有效半径 (mm)，用于边缘事件检测；范围：0.0 至 +∞ |
## Output

structured_array output with fields: event_id, pair_id, s1_peak_id, s2_peak_id, x, y, z, r, ....

| Field | DType | Unit | Meaning |
| --- | --- | --- | --- |
| `event_id` | `int64` | None | Unique event identifier |
| `pair_id` | `int64` | None | S1-S2 pair identifier |
| `s1_peak_id` | `int64` | None | S1 peak identifier |
| `s2_peak_id` | `int64` | None | S2 peak identifier |
| `x` | `float32` | mm | X coordinate (mm) |
| `y` | `float32` | mm | Y coordinate (mm) |
| `z` | `float32` | mm | Z coordinate (drift distance, mm) |
| `r` | `float32` | mm | Radial coordinate sqrt(x^2 + y^2) (mm) |
| `x_err` | `float32` | mm | X position uncertainty (mm) |
| `y_err` | `float32` | mm | Y position uncertainty (mm) |
| `z_err` | `float32` | mm | Z position uncertainty (mm) |
| `xy_chi2` | `float32` | None | Chi-squared for XY reconstruction fit |
| `xy_ndf` | `int16` | None | Degrees of freedom for XY reconstruction |
| `z_quality` | `float32` | None | Z reconstruction quality (0 to 1) |
| `position_goodness` | `float32` | None | Overall position quality (0 to 1) |
| `xy_method` | `<U16` | None | XY reconstruction method (cog, nn, template, or none) |
| `z_method` | `<U16` | None | Z reconstruction method (drift_time, corrected, or none) |
| `drift_time_ns` | `float32` | ns | Drift time in nanoseconds |
| `s2_area` | `float32` | ADC counts | S2 area used for quality checks |
| `s2_n_channels` | `int16` | None | Number of S2 channels used for quality checks |
| `flags` | `uint32` | None | Bit-field reconstruction status flags |
## Usage

### Minimal Example

```python
from waveform_analysis import Context
from waveform_analysis.plugins import profiles

ctx = Context(config={"data_root": "DAQ", "daq_adapter": "vx2730"})
ctx.register(*profiles.cpu_default())
result = ctx.get_data("run_001", "position_reconstruction")
```

示例使用 `run_id="run_001"` 和文档默认运行画像；真实数据路径与配置应以当前实验设置为准。

### Downstream Consumers

- `events`
