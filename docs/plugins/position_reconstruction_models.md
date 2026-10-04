# Jun 与 Junshi 位置重建

`PositionReconstructionPlugin` 的 `xy_method` 可选 `cog`（默认）、`Jun`、`Junshi`。
三者均读取 selected S1–S2 pairs 和 `peaklet_channels`，输出原有
`position_reconstruction` dtype；Z 继续使用漂移时间与配置的漂移速度。

| 方法 | 随包资源 | 输出坐标域 | 来源 |
| --- | --- | --- | --- |
| Jun | v3 LRF CSV、模拟 QE | 40 mm 半径的 1 mm 网格，加局部抛物线插值 | [JW_XiHuTPC_MC](https://github.com/Westlake-University-Lsc-lab/JW_XiHuTPC_MC)，`notebooks/v3.0/v3lib.py` |
| Junshi | uniform / per_pmt 两套纯数组 NN 权重 | 训练半径 39.5 mm；预测可超出此域 | DAQ 共享导出 `/home/wjs/position_reco`，`nn_reco.py` 与 `models/nn_reco_40k_*.joblib` |

Jun 使用七通道条件光分布的最大似然网格扫描。默认 QE 和 LRF 都来自模拟，
`jun_qe` 可提供按模型顺序排列的七个相对 QE。Junshi 保留原输入除以
`max(sum(counts), 1)`、StandardScaler 和 7→128→128→64→2 网络计算；
`junshi_variant` 可选 `uniform`（默认）或 `per_pmt`。推理仅需要 NumPy，
Jun 另使用项目已有的 SciPy；两者按 256 个事件分块。
Junshi 的 QE 差异已包含在训练权重中，输入只做面积/增益换算。

## 输入配置

选择 Jun 或 Junshi 时，必须显式给出：

- `model_channels`：按模型 p0…p6 顺序排列的七组 `[board, channel]`。
- `model_area_per_count`：同顺序的七个正数，输入 `counts = ADC area / scale`。
  使用实测单 PE 面积时得到 PE；全 1 表示保留 ADC 面积作为相对光权重。

模型顺序为中心、右、左、右上、左上、右下、左下。几何配置中的
`detector_geometry.gain` 继续用于 CoG；新模型使用上述显式换算。
七路的相对增益、通道方位及运行期间是否正常采集需要按实验记录确定。
稀疏 `peaklet_channels` 中无 hit 的通道填零；此规则不能补偿失效或停用的 PMT。
同峰同通道多行面积相加，只使用正面积；非有限输入或七路总量为零时跳过 XY。

下面复现历史 Junshi 分析所采用的通道约定、等相对增益与逆时针 30° 旋转。
这套对应关系是历史分析约定，尚未通过独立硬件测量确定：

```python
# st 是已注册默认插件并配置输入数据的 Context。
st.set_config({
    "xy_method": "Junshi",
    "model_channels": [[0, 12], [0, 11], [0, 13], [0, 14], [0, 15], [0, 9], [0, 10]],
    "model_area_per_count": [1.0] * 7,
    "model_rotation_deg": 30.0,
    "junshi_variant": "uniform",
})
positions = st.get_data("00777", "position_reconstruction")

# 使用相同输入约定比较 Jun；默认 QE 是源模拟参数。
st.set_config({"xy_method": "Jun"})
positions_jun = st.get_data("00777", "position_reconstruction")
```

输入阈值 `min_s2_area_for_xy` 保持 ADC 面积单位。模型、通道、换算、旋转、
QE 和 NN 版本选择均进入缓存 lineage。资源随插件发布，更新资源须升级插件版本。

## 结果与验证边界

成功行的 `xy_method` 为 `Jun` / `Junshi`，未重建行为 `none`。
两种源算法只提供 XY；`x_err`、`y_err`、`xy_chi2`、`position_goodness` 为 NaN，
`xy_ndf` 为 0。`FLAG_POSITION_VALID` 表示已计算 XY 和 Z。
边缘标志使用配置探测器半径与模型半径的较小值，再减 `edge_threshold_mm`；
坐标保留原预测值。

提取验证：Jun 对 12 条上游模拟归档事件的 XY 最大差为
2.402×10⁻¹² mm。Junshi 两套数组权重在 14,915 条历史输入和 10 组规范输入上
整批计算与原 sklearn 模型逐值一致；最终分块模块对历史输入的最大差小于
2.843×10⁻¹⁴ mm。历史 uniform 模型与归档结果差小于
7.106×10⁻¹⁵ mm。这些检查验证代码提取与结果复现，位置精度仍需真值或标定数据评估。
模型参数及资源来源见插件 `models/*.json`。
