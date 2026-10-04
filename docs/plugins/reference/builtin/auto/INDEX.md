# WaveformAnalysis 内置插件参考

> 本页与各插件页面均由当前 `PluginSpec` 和源码事实生成。Auto 画像强调可查阅性；Agent 画像强调执行契约。依赖表中的动态插件使用文档默认画像，不代表所有运行配置下的唯一结果。

## 快速开始

```python
from waveform_analysis import Context
from waveform_analysis.plugins import profiles

ctx = Context(config={"data_root": "DAQ", "daq_adapter": "vx2730"})
ctx.register(*profiles.cpu_default())

# run_id 必须显式传入；目标产物会按 DAG 自动解析依赖并复用 lineage 缓存。
peaks = ctx.get_data("run_001", "peaks")
events = ctx.get_data("run_001", "events")
```

### 执行前预览

```python
preview = ctx.preview_execution("run_001", "events")
print(preview)
```

### 文档默认依赖画像

- Profile：`documentation-default-v1`
- 值：`{"daq_adapter": "vx2730", "use_filtered": false, "wave_source": "records"}`
- 动态依赖页面同时列出 `resolve_depends_on(context, run_id)` 可能读取的配置键；需要真实运行时配置时，应再次预览执行计划。

## 插件总览

| Provides | 插件类 | 类别 | 解析后依赖 | 输出容器 | 执行模式 | 版本 |
| --- | --- | --- | --- | --- | --- | --- |
| [`basic_features`](basic_features.md) | `BasicFeaturesPlugin` | 特征提取 | `records`, `wave_pool` | `structured_array` | `static` | `4.1.0` |
| [`cache_analysis`](cache_analysis.md) | `CacheAnalysisPlugin` | 缓存分析 | - | `dict` | `static` | `0.1.0` |
| [`df`](df.md) | `DataFramePlugin` | 数据导出 | `records`, `basic_features` | `dataframe` | `static` | `1.7.0` |
| [`df_events`](df_events.md) | `GroupedEventsPlugin` | 事件分析 | `df` | `dataframe` | `static` | `0.0.1` |
| [`df_paired`](df_paired.md) | `PairedEventsPlugin` | 事件分析 | `df_events` | `dataframe` | `static` | `0.0.1` |
| [`energy_reconstruction`](energy_reconstruction.md) | `EnergyReconstructionPlugin` | 其他 | `s1_s2_pairs` | `structured_array` | `static` | `0.1.0` |
| [`events`](events.md) | `EventPlugin` | 事件分析 | `s1_s2_pairs`, `position_reconstruction` | `structured_array` | `static` | `0.0.3` |
| [`filtered_waveforms`](filtered_waveforms.md) | `FilteredWaveformsPlugin` | 波形处理 | `st_waveforms` | `structured_array` | `static` | `3.0.0` |
| [`hit`](hit.md) | `HitFinderPlugin` | 特征提取 | `records`, `wave_pool` | `structured_array` | `static` | `3.0.0` |
| [`hit_grouped`](hit_grouped.md) | `HitGroupedPlugin` | 特征提取 | `hit_merged`, `hit_merged_components`, `hit_threshold` | `dataframe` | `static` | `0.5.0` |
| [`hit_merge_clusters`](hit_merge_clusters.md) | `HitMergeClustersPlugin` | 特征提取 | `hit_merged`, `hit_threshold` | `structured_array` | `static` | `1.1.0` |
| [`hit_merged`](hit_merged.md) | `HitMergePlugin` | 特征提取 | `hit_threshold` | `structured_array` | `static` | `2.1.0` |
| [`hit_merged_components`](hit_merged_components.md) | `HitMergedComponentsPlugin` | 特征提取 | `hit_merged`, `hit_threshold` | `structured_array` | `static` | `1.1.0` |
| [`hit_merged_features`](hit_merged_features.md) | `HitMergedFeaturesPlugin` | 特征提取 | `hit_merged`, `hit_merged_components`, `hit_threshold`, `records`, `wave_pool` | `structured_array` | `static` | `1.1.3` |
| [`hit_threshold`](hit_threshold.md) | `ThresholdHitPlugin` | 特征提取 | `records`, `wave_pool`, `records_asymmetry_mask` | `structured_array` | `static` | `1.2.2` |
| [`peak_classification`](peak_classification.md) | `PeakClassificationPlugin` | 特征提取 | `peaks` | `structured_array` | `static` | `1.2.1` |
| [`peaklet_channels`](peaklet_channels.md) | `PeakletChannelsPlugin` | 峰构建 | `peaklets`, `peaklet_components`, `hit_merged`, `hit_merged_components`, `hit_threshold`, `hit_merged_features`, `peaklet_features`, `records`, `wave_pool` | `structured_array` | `static` | `2.0.6` |
| [`peaklet_components`](peaklet_components.md) | `PeakletComponentsPlugin` | 峰构建 | `hit_merged` | `structured_array` | `static` | `1.4.0` |
| [`peaklet_features`](peaklet_features.md) | `PeakletFeaturesPlugin` | 峰构建 | `peaklet_waveforms`, `peaklet_waveform_pool`, `peaklets` | `structured_array` | `static` | `5.0.0` |
| [`peaklet_waveform_pool`](peaklet_waveform_pool.md) | `PeakletWaveformPoolPlugin` | 峰构建 | `peaklet_waveforms` | `array` | `static` | `3.0.0` |
| [`peaklet_waveforms`](peaklet_waveforms.md) | `PeakletWaveformPlugin` | 峰构建 | `peaklets`, `peaklet_components`, `hit_merged`, `hit_merged_components`, `hit_threshold`, `records`, `wave_pool` | `structured_array` | `static` | `2.1.2` |
| [`peaklets`](peaklets.md) | `PeakletPlugin` | 峰构建 | `hit_merged`, `peaklet_components` | `structured_array` | `static` | `1.2.0` |
| [`peaks`](peaks.md) | `PeaksPlugin` | 特征提取 | `peaklets`, `peaklet_features`, `peaklet_channels` | `structured_array` | `static` | `5.0.0` |
| [`position_reconstruction`](position_reconstruction.md) | `PositionReconstructionPlugin` | 其他 | `s1_s2_pairs`, `peaklet_channels` | `structured_array` | `static` | `0.6.0` |
| [`raw_files`](raw_files.md) | `RawFileNamesPlugin` | 数据加载 | - | `list` | `static` | `0.0.2` |
| [`records`](records.md) | `RecordsPlugin` | 记录处理 | `raw_files` | `structured_array` | `static` | `0.14.4` |
| [`records_asymmetry_mask`](records_asymmetry_mask.md) | `RecordsAsymmetryMaskPlugin` | 记录处理 | `records`, `wave_pool` | `array` | `static` | `0.2.0` |
| [`records_detector_mask`](records_detector_mask.md) | `RecordsDetectorMaskPlugin` | 记录处理 | `records`, `records_asymmetry_mask` | `array` | `static` | `0.1.0` |
| [`records_veto_mask`](records_veto_mask.md) | `RecordsVetoMaskPlugin` | 记录处理 | `records`, `records_asymmetry_mask` | `array` | `static` | `0.1.0` |
| [`s1_s2_pair_candidates`](s1_s2_pair_candidates.md) | `S1S2PairCandidatesPlugin` | 事件分析 | `peak_classification`, `peaks` | `structured_array` | `static` | `0.2.0` |
| [`s1_s2_pairs`](s1_s2_pairs.md) | `S1S2PairSelectionPlugin` | 事件分析 | `s1_s2_pair_candidates` | `structured_array` | `static` | `0.3.0` |
| [`st_waveforms`](st_waveforms.md) | `WaveformsPlugin` | 波形处理 | `raw_files` | `structured_array` | `static` | `0.10.0` |
| [`wave_pool`](wave_pool.md) | `WavePoolPlugin` | 波形处理 | `raw_files` | `array` | `static` | `0.14.4` |
| [`wave_pool_filtered`](wave_pool_filtered.md) | `WavePoolFilteredPlugin` | 波形处理 | `records`, `wave_pool` | `array` | `static` | `3.0.0` |
| [`waveform_width`](waveform_width.md) | `WaveformWidthPlugin` | 波形处理 | `hit`, `st_waveforms` | `structured_array` | `static` | `3.0.0` |
| [`waveform_width_integral`](waveform_width_integral.md) | `WaveformWidthIntegralPlugin` | 波形处理 | `records`, `wave_pool` | `structured_array` | `static` | `2.7.0` |

## 按类别浏览

### 数据加载

扫描并组织原始 DAQ 文件，为后续 records 构建提供入口。
| Provides | 插件类 | 解析后依赖 | 页面来源 |
| --- | --- | --- | --- |
| [`raw_files`](raw_files.md) | `RawFileNamesPlugin` | - | `source` |

### 波形处理

构建、筛选或度量波形，并保留 records-backed 的输入关系。
| Provides | 插件类 | 解析后依赖 | 页面来源 |
| --- | --- | --- | --- |
| [`filtered_waveforms`](filtered_waveforms.md) | `FilteredWaveformsPlugin` | `st_waveforms` | `source` |
| [`st_waveforms`](st_waveforms.md) | `WaveformsPlugin` | `raw_files` | `source` |
| [`wave_pool`](wave_pool.md) | `WavePoolPlugin` | `raw_files` | `source` |
| [`wave_pool_filtered`](wave_pool_filtered.md) | `WavePoolFilteredPlugin` | `records`, `wave_pool` | `source` |
| [`waveform_width`](waveform_width.md) | `WaveformWidthPlugin` | `hit`, `st_waveforms` | `source` |
| [`waveform_width_integral`](waveform_width_integral.md) | `WaveformWidthIntegralPlugin` | `records`, `wave_pool` | `source` |

### 峰构建

从 peaklet、通道聚合到 peak 分类，形成峰级分析产物。
| Provides | 插件类 | 解析后依赖 | 页面来源 |
| --- | --- | --- | --- |
| [`peaklet_channels`](peaklet_channels.md) | `PeakletChannelsPlugin` | `peaklets`, `peaklet_components`, `hit_merged`, `hit_merged_components`, `hit_threshold`, `hit_merged_features`, `peaklet_features`, `records`, `wave_pool` | `source` |
| [`peaklet_components`](peaklet_components.md) | `PeakletComponentsPlugin` | `hit_merged` | `source` |
| [`peaklet_features`](peaklet_features.md) | `PeakletFeaturesPlugin` | `peaklet_waveforms`, `peaklet_waveform_pool`, `peaklets` | `source` |
| [`peaklet_waveform_pool`](peaklet_waveform_pool.md) | `PeakletWaveformPoolPlugin` | `peaklet_waveforms` | `source` |
| [`peaklet_waveforms`](peaklet_waveforms.md) | `PeakletWaveformPlugin` | `peaklets`, `peaklet_components`, `hit_merged`, `hit_merged_components`, `hit_threshold`, `records`, `wave_pool` | `source` |
| [`peaklets`](peaklets.md) | `PeakletPlugin` | `hit_merged`, `peaklet_components` | `source` |

### 特征提取

从 hit、peak 或波形计算结构化特征。
| Provides | 插件类 | 解析后依赖 | 页面来源 |
| --- | --- | --- | --- |
| [`basic_features`](basic_features.md) | `BasicFeaturesPlugin` | `records`, `wave_pool` | `source` |
| [`hit`](hit.md) | `HitFinderPlugin` | `records`, `wave_pool` | `source` |
| [`hit_grouped`](hit_grouped.md) | `HitGroupedPlugin` | `hit_merged`, `hit_merged_components`, `hit_threshold` | `source` |
| [`hit_merge_clusters`](hit_merge_clusters.md) | `HitMergeClustersPlugin` | `hit_merged`, `hit_threshold` | `source` |
| [`hit_merged`](hit_merged.md) | `HitMergePlugin` | `hit_threshold` | `published` |
| [`hit_merged_components`](hit_merged_components.md) | `HitMergedComponentsPlugin` | `hit_merged`, `hit_threshold` | `source` |
| [`hit_merged_features`](hit_merged_features.md) | `HitMergedFeaturesPlugin` | `hit_merged`, `hit_merged_components`, `hit_threshold`, `records`, `wave_pool` | `source` |
| [`hit_threshold`](hit_threshold.md) | `ThresholdHitPlugin` | `records`, `wave_pool`, `records_asymmetry_mask` | `source` |
| [`peak_classification`](peak_classification.md) | `PeakClassificationPlugin` | `peaks` | `source` |
| [`peaks`](peaks.md) | `PeaksPlugin` | `peaklets`, `peaklet_features`, `peaklet_channels` | `source` |

### 事件分析

将峰或 hit 组织为事件、配对结果或位置重建结果。
| Provides | 插件类 | 解析后依赖 | 页面来源 |
| --- | --- | --- | --- |
| [`df_events`](df_events.md) | `GroupedEventsPlugin` | `df` | `source` |
| [`df_paired`](df_paired.md) | `PairedEventsPlugin` | `df_events` | `source` |
| [`events`](events.md) | `EventPlugin` | `s1_s2_pairs`, `position_reconstruction` | `source` |
| [`s1_s2_pair_candidates`](s1_s2_pair_candidates.md) | `S1S2PairCandidatesPlugin` | `peak_classification`, `peaks` | `source` |
| [`s1_s2_pairs`](s1_s2_pairs.md) | `S1S2PairSelectionPlugin` | `s1_s2_pair_candidates` | `source` |

### 数据导出

把插件产物整理为分析侧表格或批量输出。
| Provides | 插件类 | 解析后依赖 | 页面来源 |
| --- | --- | --- | --- |
| [`df`](df.md) | `DataFramePlugin` | `records`, `basic_features` | `source` |

### 缓存分析

提供缓存结构与 lineage 的只读诊断信息。
| Provides | 插件类 | 解析后依赖 | 页面来源 |
| --- | --- | --- | --- |
| [`cache_analysis`](cache_analysis.md) | `CacheAnalysisPlugin` | - | `source` |

### 记录处理

构建 records 及其关联的屏蔽、波形访问输入。
| Provides | 插件类 | 解析后依赖 | 页面来源 |
| --- | --- | --- | --- |
| [`records`](records.md) | `RecordsPlugin` | `raw_files` | `source` |
| [`records_asymmetry_mask`](records_asymmetry_mask.md) | `RecordsAsymmetryMaskPlugin` | `records`, `wave_pool` | `source` |
| [`records_detector_mask`](records_detector_mask.md) | `RecordsDetectorMaskPlugin` | `records`, `records_asymmetry_mask` | `source` |
| [`records_veto_mask`](records_veto_mask.md) | `RecordsVetoMaskPlugin` | `records`, `records_asymmetry_mask` | `source` |

### 其他

该类别中的插件按各自页面声明的输入、配置和输出契约运行。
| Provides | 插件类 | 解析后依赖 | 页面来源 |
| --- | --- | --- | --- |
| [`energy_reconstruction`](energy_reconstruction.md) | `EnergyReconstructionPlugin` | `s1_s2_pairs` | `source` |
| [`position_reconstruction`](position_reconstruction.md) | `PositionReconstructionPlugin` | `s1_s2_pairs`, `peaklet_channels` | `source` |


## 常见目标

### 读取峰级结果

```python
peaks = ctx.get_data("run_001", "peaks")
print(peaks.dtype.names)
```

### 读取事件级结果

```python
events = ctx.get_data("run_001", "events")
print(events.dtype.names)
```

### 诊断缓存

```python
cache = ctx.get_data("run_001", "cache_analysis")
```

每个目标的精确依赖、配置键、保存策略和输出字段，以对应插件页为准。

## 自定义插件的最小契约

```python
import numpy as np

from waveform_analysis.plugins import Plugin


class MyPlugin(Plugin):
    provides = "my_output"
    depends_on = ["records"]
    version = "1.0.0"
    output_dtype = np.dtype([("value", "f4")])

    def compute(self, context, run_id, **kwargs):
        records = context.get_data(run_id, "records")
        return np.zeros(len(records), dtype=self.output_dtype)
```

新增或修改插件时，同时检查 `provides`、`depends_on`/`resolve_depends_on`、输出 dtype、配置说明、版本和两套生成文档；发布前运行仓库 `AGENTS.md` 中列出的文档与 schema 闸门。
