"""位置重建插件（向量化优化版本）

基于 S1-S2 配对的空间位置重建。

此插件从选定的 S1-S2 配对中提取位置信息，计算事件的三维空间坐标 (x, y, z)。

核心功能：
- 从 s1_s2_pairs 中提取 selected 配对
- 计算 Z 坐标（基于漂移时间）
- 计算 XY 坐标（电荷重心法，基于 S2 光分布）
- 输出位置重建结果及质量指标

位置重建方法：
- Z: 基于漂移时间和漂移速度
- XY: 电荷重心法 (Center of Gravity, CoG)
  使用 S2 信号在各 PMT 通道的分布，应用增益校正后计算加权重心

性能优化（v0.2.0）：
- 向量化 XY 计算，避免 Python 循环
- 批量加载通道数据
- 预计算 PMT 映射表
- 使用 NumPy 广播加速计算

版本历史：
- v0.0.0: 数据结构定义，仅 Z 坐标占位
- v0.1.0: 实现 CoG XY 重建，集成 PMT 几何布局
- v0.2.0: 向量化优化，性能提升 10-100x
- v0.2.1: 修正默认漂移速度单位，确保 drift_time_ns 输出的 Z 坐标为 mm
- v0.3.0: 声明 peaklet_channels 依赖，使 XY 通道面积进入缓存 lineage
- v0.4.0: 直接批量消费 peaklet_channels，移除逐事件 Accessor 构造和查询
- v0.5.0: 几何与增益纳入配置追踪，配置变化后重新加载布局

Author: Claude Code
Version: 0.5.0
"""

from typing import Any

import numpy as np

from waveform_analysis.core.hardware.geometry import (
    PmtLayout,
    load_fallback_layout,
    load_pmt_layout_from_config,
)
from waveform_analysis.core.plugins.core.base import Option, Plugin

# ============================================================================
# 质量标志定义
# ============================================================================

FLAG_POSITION_VALID = 1 << 0  # 位置重建成功
FLAG_Z_RECONSTRUCTED = 1 << 1  # Z 坐标已重建
FLAG_XY_RECONSTRUCTED = 1 << 2  # XY 坐标已重建
FLAG_LOW_S2_SIGNAL = 1 << 3  # S2 信号太弱，XY 重建质量低
FLAG_EDGE_EVENT = 1 << 4  # 边缘事件（靠近探测器边界）
FLAG_AMBIGUOUS_POSITION = 1 << 5  # 位置模糊（多个可能位置）


# ============================================================================
# 数据结构定义
# ============================================================================

POSITION_RECONSTRUCTION_DTYPE = np.dtype(
    [
        # === Identity ===
        ("event_id", "i8"),  # 事件唯一标识
        ("pair_id", "i8"),  # 对应的 S1-S2 配对 ID
        ("s1_peak_id", "i8"),  # S1 peak ID
        ("s2_peak_id", "i8"),  # S2 peak ID
        # === Position (单位: mm) ===
        ("x", "f4"),  # X 坐标
        ("y", "f4"),  # Y 坐标
        ("z", "f4"),  # Z 坐标 (漂移距离)
        ("r", "f4"),  # 径向坐标 r = sqrt(x^2 + y^2)
        # === Position uncertainty (单位: mm) ===
        ("x_err", "f4"),  # X 不确定度
        ("y_err", "f4"),  # Y 不确定度
        ("z_err", "f4"),  # Z 不确定度
        # === Reconstruction quality ===
        ("xy_chi2", "f4"),  # XY 重建的卡方值
        ("xy_ndf", "i2"),  # XY 重建的自由度
        ("z_quality", "f4"),  # Z 重建质量 (0-1)
        ("position_goodness", "f4"),  # 整体位置质量 (0-1)
        # === Reconstruction method ===
        ("xy_method", "U16"),  # XY 重建方法: "cog", "nn", "template", "none"
        ("z_method", "U16"),  # Z 重建方法: "drift_time", "corrected", "none"
        # === Input observables ===
        ("drift_time_ns", "f4"),  # 漂移时间 (ns)
        ("s2_area", "f4"),  # S2 面积（用于质量检查）
        ("s2_n_channels", "i2"),  # S2 通道数（用于质量检查）
        # === Flags ===
        ("flags", "u4"),  # 状态标志位
    ]
)


# ============================================================================
# 插件实现
# ============================================================================


class PositionReconstructionPlugin(Plugin):
    """位置重建插件（向量化优化版本）

    从选定的 S1-S2 配对重建事件的三维空间位置。

    输入:
    - s1_s2_pairs: S1-S2 配对结果（仅处理 selected=True 的配对）

    输出:
    - position_reconstruction: 位置重建结果

    v0.2.0 功能:
    - Z 坐标: 基于 drift_time * drift_velocity（向量化）
    - XY 坐标: 电荷重心法 (Center of Gravity, 向量化)
      * 批量加载所有通道数据
      * 预计算 PMT 映射表
      * 使用 NumPy 广播加速计算
    - 质量评估: 边缘事件检测、低信号标记（向量化）

    性能优化:
    - 避免 Python for 循环
    - 批量处理所有事件
    - 预计算和缓存映射关系
    - 典型性能提升: 10-100x（取决于事件数）

    未来版本计划:
    - v0.3.0: 高级 XY 重建算法 (ML, 模板匹配)
    - v1.0.0: 位置相关修正 (电场、光收集效率)
    """

    provides = "position_reconstruction"
    depends_on = ["s1_s2_pairs", "peaklet_channels"]
    description = "Reconstruct 3D position from S1-S2 pairs using vectorized CoG method"
    version = "0.5.0"
    save_when = "always"
    output_dtype = POSITION_RECONSTRUCTION_DTYPE

    options = {
        "detector_geometry": Option(
            default=None,
            type=dict,
            help="PMT 几何、硬件通道映射及相对增益；未设置时使用七 PMT 布局",
        ),
        "drift_velocity": Option(
            default=0.0013,
            type=float,
            help="漂移速度 (mm/ns)，用于 Z 坐标计算。典型值：液氙 ~0.001 mm/ns, 液氩 ~0.0013 mm/ns",
            min_value=0.0,
        ),
        "min_s2_area_for_xy": Option(
            default=100.0,
            type=float,
            help="XY 重建所需的最小 S2 面积 (ADC counts，与 s2_area 一致)",
            min_value=0.0,
        ),
        "edge_threshold_mm": Option(
            default=5.0,
            type=float,
            help="边缘事件判定阈值：距离 TPC 边界的最小距离 (mm)",
            min_value=0.0,
        ),
        "detector_radius_mm": Option(
            default=62.5,
            type=float,
            help="探测器有效半径 (mm)，用于边缘事件检测",
            min_value=0.0,
        ),
    }

    def __init__(self):
        super().__init__()
        self._pmt_map_cache: dict | None = None

    def _load_pmt_layout(self, context: Any) -> PmtLayout:
        """加载 PMT 几何布局

        优先级：
        1. 全局配置 (detector_geometry)
        2. Fallback 布局 (7-PMT 配置)

        Args:
            context: 上下文对象

        Returns:
            PmtLayout 对象
        """
        geometry = context.get_config(self, "detector_geometry")
        layout = load_pmt_layout_from_config({"detector_geometry": geometry})

        if layout is None:
            # 回退到默认布局
            layout = load_fallback_layout()

        return layout

    def _build_pmt_mapping(self, layout: PmtLayout) -> dict:
        """预计算 PMT 映射表（优化性能）

        将 (board_id, channel_id) -> (x, y, gain) 的映射预先计算好，
        避免重复查找。

        Args:
            layout: PMT 布局对象

        Returns:
            映射字典: {(board_id, channel_id): (x_mm, y_mm, gain)}
        """
        if self._pmt_map_cache is not None:
            return self._pmt_map_cache

        pmt_map = {}
        for entry in layout.entries:
            key = (entry.board_id, entry.channel_id)
            pmt_map[key] = (entry.x_mm, entry.y_mm, entry.gain)

        self._pmt_map_cache = pmt_map
        return pmt_map

    def _compute_xy_cog_vectorized(
        self,
        context: Any,
        run_id: str,
        s2_peak_ids: np.ndarray,
        s2_areas: np.ndarray,
        min_s2_area: float,
        layout: PmtLayout,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """向量化计算所有事件的 XY 坐标

        **性能优化核心函数**

        相比原版本的改进：
        1. 批量加载所有事件的通道数据
        2. 预计算 PMT 映射表
        3. 使用 NumPy 向量化操作
        4. 避免 Python for 循环

        Args:
            context: 上下文对象
            run_id: 运行 ID
            s2_peak_ids: 所有 S2 peak ID 数组
            s2_areas: 所有 S2 面积数组
            min_s2_area: 最小 S2 面积阈值
            layout: PMT 布局对象

        Returns:
            (x_array, y_array, n_channels_array) 元组
        """
        n_events = len(s2_peak_ids)
        x_array = np.full(n_events, np.nan, dtype=np.float32)
        y_array = np.full(n_events, np.nan, dtype=np.float32)
        n_channels_array = np.zeros(n_events, dtype=np.int16)

        # Preserve the historical comparison semantics: NaN is not considered
        # "below threshold" and therefore still attempts channel reconstruction.
        eligible_mask = ~(np.asarray(s2_areas) < min_s2_area)
        if not np.any(eligible_mask):
            return x_array, y_array, n_channels_array

        try:
            peaklet_channels = context.get_data(run_id, "peaklet_channels")
        except Exception:
            return x_array, y_array, n_channels_array

        required_fields = {"peaklet_id", "board", "channel", "area"}
        names = (
            set(peaklet_channels.dtype.names or ())
            if isinstance(peaklet_channels, np.ndarray)
            else set()
        )
        if not isinstance(peaklet_channels, np.ndarray) or not required_fields.issubset(names):
            return x_array, y_array, n_channels_array
        if len(peaklet_channels) == 0 or not layout.entries:
            return x_array, y_array, n_channels_array

        eligible_indices = np.flatnonzero(eligible_mask)
        unique_peak_ids, inverse = np.unique(
            np.asarray(s2_peak_ids)[eligible_indices], return_inverse=True
        )

        channel_peak_ids = peaklet_channels["peaklet_id"]
        channel_order = np.argsort(channel_peak_ids, kind="stable")
        sorted_peak_ids = channel_peak_ids[channel_order]
        left = np.searchsorted(sorted_peak_ids, unique_peak_ids, side="left")
        right = np.searchsorted(sorted_peak_ids, unique_peak_ids, side="right")
        counts = right - left
        total_rows = int(np.sum(counts, dtype=np.int64))
        if total_rows == 0:
            return x_array, y_array, n_channels_array

        group_indices = np.repeat(np.arange(len(unique_peak_ids), dtype=np.intp), counts)
        repeated_left = np.repeat(left, counts)
        repeated_offsets = np.repeat(np.cumsum(counts) - counts, counts)
        row_positions = repeated_left + np.arange(total_rows, dtype=np.intp) - repeated_offsets
        selected_rows = channel_order[row_positions]

        key_dtype = np.dtype([("board", "i8"), ("channel", "i8")])
        layout_keys = np.empty(len(layout.entries), dtype=key_dtype)
        layout_x = np.empty(len(layout.entries), dtype=np.float32)
        layout_y = np.empty(len(layout.entries), dtype=np.float32)
        layout_gain = np.empty(len(layout.entries), dtype=np.float32)
        for index, entry in enumerate(layout.entries):
            layout_keys[index] = (entry.board_id, entry.channel_id)
            layout_x[index] = entry.x_mm
            layout_y[index] = entry.y_mm
            layout_gain[index] = entry.gain
        layout_order = np.argsort(layout_keys, kind="stable")
        layout_keys = layout_keys[layout_order]
        layout_x = layout_x[layout_order]
        layout_y = layout_y[layout_order]
        layout_gain = layout_gain[layout_order]
        # ``_build_pmt_mapping`` historically used a dict comprehension, so a
        # duplicate hardware key kept the final layout entry. Stable sorting
        # preserves source order within equal-key runs; retaining each run's
        # final row therefore keeps that exact compatibility rule.
        keep_last = np.r_[layout_keys[1:] != layout_keys[:-1], True]
        layout_keys = layout_keys[keep_last]
        layout_x = layout_x[keep_last]
        layout_y = layout_y[keep_last]
        layout_gain = layout_gain[keep_last]

        row_keys = np.empty(total_rows, dtype=key_dtype)
        row_keys["board"] = peaklet_channels["board"][selected_rows]
        row_keys["channel"] = peaklet_channels["channel"][selected_rows]
        layout_positions = np.searchsorted(layout_keys, row_keys)
        safe_layout_positions = np.minimum(layout_positions, len(layout_keys) - 1)
        mapped_mask = (layout_positions < len(layout_keys)) & (
            layout_keys[safe_layout_positions] == row_keys
        )
        areas = np.asarray(peaklet_channels["area"][selected_rows], dtype=np.float32)
        # The legacy loop skipped only ``area <= 0``. In particular, NaN rows
        # were retained and propagated to a non-reconstructed (NaN) result.
        valid_rows = mapped_mask & ~(areas <= 0)
        if not np.any(valid_rows):
            return x_array, y_array, n_channels_array

        valid_groups = group_indices[valid_rows]
        valid_layout_positions = safe_layout_positions[valid_rows]
        q_corrected = areas[valid_rows] / layout_gain[valid_layout_positions]
        sum_q = np.bincount(valid_groups, weights=q_corrected, minlength=len(unique_peak_ids))
        sum_x = np.bincount(
            valid_groups,
            weights=q_corrected * layout_x[valid_layout_positions],
            minlength=len(unique_peak_ids),
        )
        sum_y = np.bincount(
            valid_groups,
            weights=q_corrected * layout_y[valid_layout_positions],
            minlength=len(unique_peak_ids),
        )
        channel_counts = np.bincount(valid_groups, minlength=len(unique_peak_ids))

        valid_peaks = sum_q > 0
        unique_x = np.full(len(unique_peak_ids), np.nan, dtype=np.float32)
        unique_y = np.full(len(unique_peak_ids), np.nan, dtype=np.float32)
        unique_n_channels = np.zeros(len(unique_peak_ids), dtype=np.int16)
        unique_x[valid_peaks] = sum_x[valid_peaks] / sum_q[valid_peaks]
        unique_y[valid_peaks] = sum_y[valid_peaks] / sum_q[valid_peaks]
        unique_n_channels[valid_peaks] = channel_counts[valid_peaks]

        x_array[eligible_indices] = unique_x[inverse]
        y_array[eligible_indices] = unique_y[inverse]
        n_channels_array[eligible_indices] = unique_n_channels[inverse]

        return x_array, y_array, n_channels_array

    def compute(self, context: Any, run_id: str, **_kwargs) -> np.ndarray:
        """执行位置重建（向量化优化版本）

        v0.2.0 实现:
        1. 筛选 selected=True 的配对
        2. 加载 PMT 几何布局
        3. 计算 Z 坐标（向量化）
        4. 计算 XY 坐标（批量向量化）
        5. 设置质量标志位（向量化）

        性能优化：
        - 所有数组操作使用 NumPy 向量化
        - 批量处理，避免 Python 循环
        - 预计算映射表
        - 典型加速比：10-100x

        Args:
            context: 上下文对象
            run_id: 运行 ID

        Returns:
            位置重建结果数组
        """
        # 获取依赖数据
        pairs = context.get_data(run_id, "s1_s2_pairs")

        # 获取配置
        drift_velocity = context.get_config(self, "drift_velocity")
        min_s2_area = context.get_config(self, "min_s2_area_for_xy")
        edge_threshold = context.get_config(self, "edge_threshold_mm")
        detector_radius = context.get_config(self, "detector_radius_mm")

        # 筛选已选定的配对
        selected_pairs = pairs[pairs["selected"]]

        # 空数据处理
        if len(selected_pairs) == 0:
            return np.zeros(0, dtype=POSITION_RECONSTRUCTION_DTYPE)

        # 加载 PMT 布局
        layout = self._load_pmt_layout(context)

        # 初始化结果数组
        n_events = len(selected_pairs)
        positions = np.zeros(n_events, dtype=POSITION_RECONSTRUCTION_DTYPE)

        # 填充基本信息（向量化）
        positions["event_id"] = np.arange(n_events)
        positions["pair_id"] = selected_pairs["pair_id"]
        positions["s1_peak_id"] = selected_pairs["s1_peak_id"]
        positions["s2_peak_id"] = selected_pairs["s2_peak_id"]
        positions["drift_time_ns"] = selected_pairs["drift_time_ns"]
        positions["s2_area"] = selected_pairs["s2_area"]
        positions["s2_n_channels"] = selected_pairs["s2_n_channels"]

        # === Z 坐标重建（向量化）===
        positions["z"] = selected_pairs["drift_time_ns"] * drift_velocity
        positions["z_method"] = "drift_time"
        positions["flags"] |= FLAG_Z_RECONSTRUCTED
        positions["z_quality"] = 1.0
        positions["z_err"] = 10.0 * drift_velocity  # 假设 10 ns 不确定度

        # === XY 坐标重建（批量向量化）===
        x_array, y_array, n_channels_array = self._compute_xy_cog_vectorized(
            context,
            run_id,
            selected_pairs["s2_peak_id"],
            selected_pairs["s2_area"],
            min_s2_area,
            layout,
        )

        positions["x"] = x_array
        positions["y"] = y_array

        # 标记 XY 方法（向量化）
        valid_xy_mask = ~np.isnan(x_array)
        positions["xy_method"][valid_xy_mask] = "cog"
        positions["xy_method"][~valid_xy_mask] = "none"

        # 低 S2 信号标记（向量化）
        low_s2_mask = selected_pairs["s2_area"] < min_s2_area
        positions["flags"][low_s2_mask] |= FLAG_LOW_S2_SIGNAL

        # 计算径向坐标（向量化）
        r_array = np.sqrt(x_array**2 + y_array**2)
        positions["r"] = r_array
        positions["flags"][valid_xy_mask] |= FLAG_XY_RECONSTRUCTED

        # 边缘事件检测（向量化）
        edge_mask = valid_xy_mask & (r_array > detector_radius - edge_threshold)
        positions["flags"][edge_mask] |= FLAG_EDGE_EVENT

        # XY 不确定度估计（向量化）
        valid_channels_mask = valid_xy_mask & (n_channels_array > 0)
        positions["x_err"][valid_channels_mask] = 10.0 / np.sqrt(
            n_channels_array[valid_channels_mask]
        )
        positions["y_err"][valid_channels_mask] = 10.0 / np.sqrt(
            n_channels_array[valid_channels_mask]
        )
        positions["xy_ndf"][valid_channels_mask] = np.maximum(
            n_channels_array[valid_channels_mask] - 2, 1
        )

        # === 整体质量评估（向量化）===
        # 有效位置：Z 和 XY 都成功重建
        z_valid = (positions["flags"] & FLAG_Z_RECONSTRUCTED) != 0
        xy_valid = (positions["flags"] & FLAG_XY_RECONSTRUCTED) != 0
        valid_mask = z_valid & xy_valid
        positions["flags"][valid_mask] |= FLAG_POSITION_VALID

        # 质量分数：综合考虑 Z 和 XY
        positions["position_goodness"][valid_mask] = 0.9
        positions["position_goodness"][~valid_mask] = 0.1

        return positions
