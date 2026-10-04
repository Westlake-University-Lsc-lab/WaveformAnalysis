# 插件系统文档

**导航**: [文档中心](../README.md) > 插件系统

WaveformAnalysis 的插件系统文档，包括教程、开发指南和参考资料。

## 站点主入口

- [插件系统与模板 API](PLUGIN_SYSTEM_OVERVIEW.md) - 系统边界、配置、依赖、lineage、生命周期、Bundle 组织、Version 策略与 Plugin Set/Profile 的统一说明
- [内置插件参考](reference/builtin/auto/INDEX.md) - 每个内置产物的依赖、配置和输出字段
- [Jun 与 Junshi 位置重建](position_reconstruction_models.md) - 模型来源、输入标定、配置与验证边界
- 交互式 Plugin DAG - 在离线 HTML 文档的插件系统页面中打开独立 DAG 工具

## 学习路径

### 🚀 入门（初学者）

1. **[插件编写规范](PLUGIN_SYSTEM_OVERVIEW.md)** - 插件编写规范与最佳实践

### 📚 进阶（开发者）

1. **[FilteredWaveformsPlugin](reference/agent/filtered_waveforms.md)** - 配置 BW/SG 波形滤波
2. **[HitFinderPlugin](reference/agent/hit.md)** - 配置峰值检测并读取 `hit` 特征
3. **[流式插件开发指南](guides/STREAMING_PLUGINS_GUIDE.md)** - 开发流式处理插件

### 🔧 开发（高级）

1. **[插件开发完整指南](../development/plugin-development/plugin_guide.md)** - 深入学习插件开发
2. **[适配器系统架构](reference/ADAPTER_SYSTEM_GUIDE.md)** - DAQ 适配器系统与使用指南
3. **[适配器与插件行为](reference/ADAPTER_PLUGIN_BEHAVIOR.md)** - 适配器与插件的交互

## 文档结构

### 📋 [guides/](guides/) - 开发指南（开发者向）

深入的开发指南，适合插件开发者和系统集成者。

| 文档 | 说明 |
|------|------|
| [STREAMING_PLUGINS_GUIDE.md](guides/STREAMING_PLUGINS_GUIDE.md) | 流式插件开发指南 |

### 🔍 [reference/](reference/) - 参考文档

架构详解和参考资料。

| 文档 | 说明 |
|------|------|
| [ADAPTER_SYSTEM_GUIDE.md](reference/ADAPTER_SYSTEM_GUIDE.md) | 适配器系统架构详解 |
| [ADAPTER_PLUGIN_BEHAVIOR.md](reference/ADAPTER_PLUGIN_BEHAVIOR.md) | 适配器与插件行为分析 |
| [filtered_waveforms.md](reference/agent/filtered_waveforms.md) | 波形滤波插件契约、配置与示例 |
| [hit.md](reference/agent/hit.md) | 峰值检测插件契约、配置与示例 |
| [DATA_PRODUCTS.md](../architecture/DATA_PRODUCTS.md) | records 与 wave_pool 共享中间层设计 |
| [builtin/](reference/builtin/) | 内置插件文档 |

## 快速链接

- 🏗️ [插件系统与模板 API](PLUGIN_SYSTEM_OVERVIEW.md) - 当前插件系统的完整事实和开发契约
- 🚀 [快速入门](../user-guide/QUICKSTART_GUIDE.md) - 快速开始
- ⚙️ [配置管理](../features/context/CONFIGURATION.md) - 配置系统

## 相关资源

- [核心架构](../architecture/README.md) - 系统架构文档
- [功能特性](../features/README.md) - 功能特性文档
- [开发指南](../development/README.md) - 开发指南
