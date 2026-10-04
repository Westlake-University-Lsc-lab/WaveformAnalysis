"""peaklet_channels bundle - provides 'peaklet_channels'。"""

from typing import Any

import numba as nb
import numpy as np

from waveform_analysis.core.plugins.builtin.cpu._record_utils import RecordLookup
from waveform_analysis.core.plugins.builtin.cpu._wave_source import (
    WAVE_SOURCE_RECORDS,
    load_wave_input,
    resolve_wave_input_spec,
)
from waveform_analysis.core.plugins.builtin.hit_merged_features.plugin import (
    HitMergedFeaturesPlugin,
)
from waveform_analysis.core.plugins.builtin.shared.canonical_waveform_numba import (
    MAX_CANONICAL_DENSE_SAMPLES_PER_BATCH,
    classify_dense_canonical_groups,
    materialize_dense_canonical_groups,
    reduce_dense_canonical_groups,
)
from waveform_analysis.core.plugins.builtin.shared.waveform_merge import merge_waveform_segments
from waveform_analysis.core.plugins.core.base import Option, Plugin

PEAKLET_CHANNELS_DTYPE = np.dtype(
    [
        ("peaklet_id", "i8"),
        ("board", "i2"),
        ("channel", "i2"),
        ("area", "f4"),
        ("height", "f4"),
        ("n_hits", "i4"),
        ("area_fraction", "f4"),
    ]
)

_CANONICAL_GROUPS_PER_BATCH = 4_096


def _empty_channels() -> np.ndarray:
    return np.zeros(0, dtype=PEAKLET_CHANNELS_DTYPE)


def _validate_peaklet_components(peaklets: np.ndarray, components: np.ndarray) -> None:
    if "component_count" not in (peaklets.dtype.names or ()):
        return

    peaklet_ids = components["peak_id"].astype(np.int64, copy=False)
    invalid = (peaklet_ids < 0) | (peaklet_ids >= len(peaklets))
    if np.any(invalid):
        peaklet_id = int(peaklet_ids[np.flatnonzero(invalid)[0]])
        raise ValueError(
            "peaklet_channels found peaklet_components row with out-of-range "
            f"peak_id={peaklet_id}"
        )
    counts = np.bincount(peaklet_ids, minlength=len(peaklets)).astype(np.int64, copy=False)
    expected = peaklets["component_count"].astype(np.int64, copy=False)
    if not np.array_equal(counts, expected):
        raise ValueError("peaklet_channels found peaklet_components inconsistent with peaklets")


@nb.njit(cache=True, nogil=True)
def _has_dense_identity_merged_indices(merged_indices: np.ndarray) -> bool:
    for index in range(len(merged_indices)):
        if merged_indices[index] != index:
            return False
    return True


@nb.njit(cache=True, nogil=True)
def _keys_are_nondecreasing(
    peaklet_ids: np.ndarray, boards: np.ndarray, channels: np.ndarray
) -> bool:
    for index in range(1, len(peaklet_ids)):
        previous_peaklet = peaklet_ids[index - 1]
        peaklet = peaklet_ids[index]
        if peaklet < previous_peaklet:
            return False
        if peaklet == previous_peaklet:
            previous_board = boards[index - 1]
            board = boards[index]
            if board < previous_board:
                return False
            if board == previous_board and channels[index] < channels[index - 1]:
                return False
    return True


@nb.njit(cache=True, nogil=True)
def _fast_group_structure_is_valid(
    component_peaklet_ids: np.ndarray,
    component_merged_indices: np.ndarray,
    peaklet_component_offsets: np.ndarray,
    peaklet_component_counts: np.ndarray,
    n_features: int,
    feature_valid: np.ndarray,
    feature_boards: np.ndarray,
    feature_channels: np.ndarray,
) -> bool:
    """Check the strict production layout required by the direct CSR path.

    The fast path intentionally rejects rather than repairs malformed input.
    This keeps external/legacy arrays on the existing sorted matching path,
    where all compatibility behavior remains explicit.
    """
    if len(peaklet_component_offsets) != len(peaklet_component_counts):
        return False
    expected_offset = 0
    for peaklet_id in range(len(peaklet_component_offsets)):
        start = peaklet_component_offsets[peaklet_id]
        count = peaklet_component_counts[peaklet_id]
        if (
            start != expected_offset
            or count < 0
            or start < 0
            or start + count > len(component_peaklet_ids)
        ):
            return False
        for component_index in range(start, start + count):
            current_peaklet = component_peaklet_ids[component_index]
            if current_peaklet != peaklet_id:
                return False
            merged_index = component_merged_indices[component_index]
            if merged_index < 0 or merged_index >= n_features:
                return False
        expected_offset += count
    return expected_offset == len(component_peaklet_ids)


@nb.njit(cache=True, nogil=True, parallel=True)
def _count_fast_groups_kernel(
    peaklet_component_offsets: np.ndarray,
    peaklet_component_counts: np.ndarray,
    component_merged_indices: np.ndarray,
    feature_valid: np.ndarray,
    feature_boards: np.ndarray,
    feature_channels: np.ndarray,
    group_counts: np.ndarray,
    member_counts: np.ndarray,
):
    """Count valid members and unique keys independently for every peaklet.

    The temporary key table is local to one peaklet.  This keeps the production
    component CSR untouched while avoiding the global matched/key arrays and
    global lexsort used by the compatibility path.
    """
    for peaklet_id in nb.prange(len(peaklet_component_offsets)):
        start = peaklet_component_offsets[peaklet_id]
        end = start + peaklet_component_counts[peaklet_id]
        local_size = max(1, end - start)
        unique_boards = np.empty(local_size, dtype=np.int16)
        unique_channels = np.empty(local_size, dtype=np.int16)
        n_unique = 0
        members = 0
        for component_index in range(start, end):
            merged_index = component_merged_indices[component_index]
            if feature_valid[merged_index] == 0:
                continue
            board = feature_boards[merged_index]
            channel = feature_channels[merged_index]
            members += 1
            insert_at = 0
            while insert_at < n_unique and (
                unique_boards[insert_at] < board
                or (unique_boards[insert_at] == board and unique_channels[insert_at] < channel)
            ):
                insert_at += 1
            if (
                insert_at < n_unique
                and unique_boards[insert_at] == board
                and unique_channels[insert_at] == channel
            ):
                continue
            for shift in range(n_unique, insert_at, -1):
                unique_boards[shift] = unique_boards[shift - 1]
                unique_channels[shift] = unique_channels[shift - 1]
            unique_boards[insert_at] = board
            unique_channels[insert_at] = channel
            n_unique += 1
        group_counts[peaklet_id] = n_unique
        member_counts[peaklet_id] = members


@nb.njit(cache=True, nogil=True, parallel=True)
def _fill_fast_groups_kernel(
    peaklet_component_offsets: np.ndarray,
    peaklet_component_counts: np.ndarray,
    component_merged_indices: np.ndarray,
    feature_valid: np.ndarray,
    feature_boards: np.ndarray,
    feature_channels: np.ndarray,
    feature_areas: np.ndarray,
    feature_heights: np.ndarray,
    feature_n_hits: np.ndarray,
    group_prefix: np.ndarray,
    member_prefix: np.ndarray,
    group_member_offsets: np.ndarray,
    grouped_merged_indices: np.ndarray,
    out_peaklet_ids: np.ndarray,
    out_boards: np.ndarray,
    out_channels: np.ndarray,
    out_areas: np.ndarray,
    out_heights: np.ndarray,
    out_n_hits: np.ndarray,
):
    """Fill sorted hardware groups and their stable member CSR from prefixes."""
    for peaklet_id in nb.prange(len(peaklet_component_offsets)):
        component_start = peaklet_component_offsets[peaklet_id]
        component_end = component_start + peaklet_component_counts[peaklet_id]
        group_cursor = group_prefix[peaklet_id]
        member_cursor = member_prefix[peaklet_id]
        local_size = max(1, component_end - component_start)
        unique_boards = np.empty(local_size, dtype=np.int16)
        unique_channels = np.empty(local_size, dtype=np.int16)
        n_unique = 0
        for component_index in range(component_start, component_end):
            merged_index = component_merged_indices[component_index]
            if feature_valid[merged_index] == 0:
                continue
            board = feature_boards[merged_index]
            channel = feature_channels[merged_index]
            insert_at = 0
            while insert_at < n_unique and (
                unique_boards[insert_at] < board
                or (unique_boards[insert_at] == board and unique_channels[insert_at] < channel)
            ):
                insert_at += 1
            if (
                insert_at < n_unique
                and unique_boards[insert_at] == board
                and unique_channels[insert_at] == channel
            ):
                continue
            for shift in range(n_unique, insert_at, -1):
                unique_boards[shift] = unique_boards[shift - 1]
                unique_channels[shift] = unique_channels[shift - 1]
            unique_boards[insert_at] = board
            unique_channels[insert_at] = channel
            n_unique += 1

        for local_group in range(n_unique):
            board = unique_boards[local_group]
            channel = unique_channels[local_group]
            output_index = group_cursor + local_group
            group_member_offsets[output_index] = member_cursor
            out_peaklet_ids[output_index] = peaklet_id
            out_boards[output_index] = board
            out_channels[output_index] = channel
            first_member = True
            for component_index in range(component_start, component_end):
                merged_index = component_merged_indices[component_index]
                if feature_valid[merged_index] == 0:
                    continue
                if (
                    feature_boards[merged_index] != board
                    or feature_channels[merged_index] != channel
                ):
                    continue
                grouped_merged_indices[member_cursor] = merged_index
                member_cursor += 1
                area = np.float32(feature_areas[merged_index])
                value = np.float32(feature_heights[merged_index])
                n_hits = np.int32(feature_n_hits[merged_index])
                if first_member:
                    out_areas[output_index] = area
                    out_heights[output_index] = value
                    out_n_hits[output_index] = n_hits
                    first_member = False
                else:
                    out_areas[output_index] = np.float32(out_areas[output_index] + area)
                    if value > out_heights[output_index]:
                        out_heights[output_index] = value
                    out_n_hits[output_index] = np.int32(out_n_hits[output_index] + n_hits)
        group_member_offsets[group_cursor + n_unique] = member_cursor


def _polarity_sign_array(records: np.ndarray) -> np.ndarray:
    sign = np.full(len(records), -1.0, dtype=np.float32)
    names = records.dtype.names or ()
    if "polarity" not in names:
        return sign
    polarity = records["polarity"]
    if polarity.dtype.kind == "S":
        sign[polarity == b"positive"] = 1.0
    elif polarity.dtype.kind == "U":
        sign[polarity == "positive"] = 1.0
    else:
        for index, value in enumerate(polarity):
            if (value.decode("utf-8") if isinstance(value, bytes) else str(value)) == "positive":
                sign[index] = 1.0
    return sign


@nb.njit(cache=True, nogil=True, parallel=True)
def _mark_waveform_rebuild_groups_kernel(
    group_offsets,
    grouped_merged_indices,
    merged_sample_start,
    merged_sample_end,
    merged_time_start,
    merged_time_end,
    rebuild_groups,
    reuse_single_record_features,
):
    """Mark only channel groups whose aggregate features are insufficient.

    A single record-backed merged hit in ADC units can be reused exactly.  Any multi-hit
    group is rebuilt from samples: adding quantized float32 feature areas is
    not bitwise equivalent to the canonical float64 waveform integral.
    """
    n_groups = len(group_offsets) - 1
    for group_index in nb.prange(n_groups):
        start = group_offsets[group_index]
        end = group_offsets[group_index + 1]
        if end <= start:
            rebuild_groups[group_index] = 1
            continue
        if end - start == 1 and reuse_single_record_features:
            merged_index = grouped_merged_indices[start]
            rebuild_groups[group_index] = int(
                merged_sample_start[merged_index] < 0
                or merged_sample_end[merged_index] <= merged_sample_start[merged_index]
            )
            continue

        rebuild_groups[group_index] = 1


@nb.njit(cache=True, nogil=True, parallel=True)
def _fill_fractions_and_validate_kernel(
    output_peaklet_starts: np.ndarray,
    output_peaklet_ids: np.ndarray,
    output_areas: np.ndarray,
    output_fractions: np.ndarray,
    peaklet_areas: np.ndarray,
    area_mismatch: np.ndarray,
    fraction_mismatch: np.ndarray,
):
    """Validate conservation and write fractions in one peaklet-parallel pass."""
    for peaklet_id in nb.prange(len(peaklet_areas)):
        start = output_peaklet_starts[peaklet_id]
        end = output_peaklet_starts[peaklet_id + 1]
        expected_area = peaklet_areas[peaklet_id]
        channel_area = 0.0
        for output_index in range(start, end):
            channel_area += np.float64(output_areas[output_index])
        area_difference = abs(channel_area - expected_area)
        if not (area_difference <= 1e-3 + 1e-5 * abs(expected_area)):
            area_mismatch[peaklet_id] = 1
            continue

        fraction_sum = 0.0
        for output_index in range(start, end):
            if expected_area == 0.0:
                fraction = np.float32(0.0)
            else:
                fraction = np.float32(output_areas[output_index] / expected_area)
            output_fractions[output_index] = fraction
            fraction_sum += np.float64(fraction)
        if expected_area != 0.0:
            fraction_difference = abs(fraction_sum - 1.0)
            if not (fraction_difference <= 1e-3 + 1e-5):
                fraction_mismatch[peaklet_id] = 1


@nb.njit(cache=True, nogil=True, parallel=True)
def _count_group_windows_kernel(
    group_ids: np.ndarray,
    group_offsets: np.ndarray,
    grouped_merged_indices: np.ndarray,
    merged_sample_starts: np.ndarray,
    merged_sample_ends: np.ndarray,
    merged_record_ids: np.ndarray,
    merged_is_single_record: np.ndarray,
    merged_component_offsets: np.ndarray,
    merged_component_counts: np.ndarray,
    component_merged_indices: np.ndarray,
    component_hit_indices: np.ndarray,
    hit_record_ids: np.ndarray,
    hit_starts: np.ndarray,
    hit_ends: np.ndarray,
    window_counts: np.ndarray,
    invalid_groups: np.ndarray,
):
    """Count record windows for each output group and validate CSR slices."""
    for local_group_index in nb.prange(len(group_ids)):
        group_index = group_ids[local_group_index]
        start = group_offsets[group_index]
        end = group_offsets[group_index + 1]
        count = 0
        invalid = 0
        for grouped_index in range(start, end):
            merged_index = grouped_merged_indices[grouped_index]
            sample_start = merged_sample_starts[merged_index]
            sample_end = merged_sample_ends[merged_index]
            if (
                merged_is_single_record[merged_index] != 0
                and sample_start >= 0
                and sample_end > sample_start
            ):
                count += 1
                continue
            component_offset = merged_component_offsets[merged_index]
            component_count = merged_component_counts[merged_index]
            if (
                component_offset < 0
                or component_count <= 0
                or component_offset + component_count > len(component_merged_indices)
            ):
                invalid = 1
                break
            for component_index in range(component_offset, component_offset + component_count):
                if component_merged_indices[component_index] != merged_index:
                    invalid = 1
                    break
                hit_index = component_hit_indices[component_index]
                if hit_index < 0 or hit_index >= len(hit_record_ids):
                    invalid = 1
                    break
            if invalid != 0:
                break
            count += component_count
        if count <= 0:
            invalid = 1
        window_counts[local_group_index] = count
        invalid_groups[local_group_index] = invalid


@nb.njit(cache=True, nogil=True, parallel=True)
def _fill_group_windows_kernel(
    group_ids: np.ndarray,
    group_offsets: np.ndarray,
    grouped_merged_indices: np.ndarray,
    merged_sample_starts: np.ndarray,
    merged_sample_ends: np.ndarray,
    merged_record_ids: np.ndarray,
    merged_is_single_record: np.ndarray,
    merged_component_offsets: np.ndarray,
    merged_component_counts: np.ndarray,
    component_hit_indices: np.ndarray,
    hit_record_ids: np.ndarray,
    hit_starts: np.ndarray,
    hit_ends: np.ndarray,
    output_boards: np.ndarray,
    output_channels: np.ndarray,
    window_offsets: np.ndarray,
    record_ids_out: np.ndarray,
    starts_out: np.ndarray,
    ends_out: np.ndarray,
    boards_out: np.ndarray,
    channels_out: np.ndarray,
):
    """Expand validated group CSR into caller-owned flat window arrays."""
    for local_group_index in nb.prange(len(group_ids)):
        group_index = group_ids[local_group_index]
        cursor = window_offsets[local_group_index]
        start = group_offsets[group_index]
        end = group_offsets[group_index + 1]
        board = output_boards[group_index]
        channel = output_channels[group_index]
        for grouped_index in range(start, end):
            merged_index = grouped_merged_indices[grouped_index]
            sample_start = merged_sample_starts[merged_index]
            sample_end = merged_sample_ends[merged_index]
            if (
                merged_is_single_record[merged_index] != 0
                and sample_start >= 0
                and sample_end > sample_start
            ):
                record_ids_out[cursor] = merged_record_ids[merged_index]
                starts_out[cursor] = sample_start
                ends_out[cursor] = sample_end
                boards_out[cursor] = board
                channels_out[cursor] = channel
                cursor += 1
                continue
            component_offset = merged_component_offsets[merged_index]
            component_count = merged_component_counts[merged_index]
            for component_index in range(component_offset, component_offset + component_count):
                hit_index = component_hit_indices[component_index]
                record_ids_out[cursor] = hit_record_ids[hit_index]
                starts_out[cursor] = hit_starts[hit_index]
                ends_out[cursor] = hit_ends[hit_index]
                boards_out[cursor] = board
                channels_out[cursor] = channel
                cursor += 1


class PeakletChannelsPlugin(Plugin):
    """Reconstruct peaklets into deduplicated per-board/channel contribution rows."""

    provides = "peaklet_channels"
    lineage_virtual = True
    depends_on = [
        "peaklets",
        "peaklet_components",
        "hit_merged",
        "hit_merged_components",
        "hit_threshold",
        "hit_merged_features",
        "peaklet_features",
        "records",
        "wave_pool",
    ]
    description = "Reconstruct deduplicated per-peaklet channel waveform contributions."
    version = "2.0.6"
    output_dtype = PEAKLET_CHANNELS_DTYPE
    save_when = "always"

    options = {
        "wave_source": Option(
            default=WAVE_SOURCE_RECORDS,
            type=str,
            help="波形来源；peaklet_channels 当前正式支持 records。",
        ),
        "use_filtered": Option(
            default=False,
            type=bool,
            help="是否从 wave_pool_filtered 重建通道波形。",
        ),
        "clip_negative_signal": Option(
            default=False,
            type=bool,
            help="是否在通道波形合并与积分前把负采样裁剪为 0。",
        ),
    }

    def resolve_depends_on(self, context: Any, run_id: str | None = None) -> list[str]:
        spec = resolve_wave_input_spec(context, self)
        if not spec.is_records:
            raise ValueError("peaklet_channels currently supports wave_source='records' only")
        return [
            "peaklets",
            "peaklet_components",
            "hit_merged",
            "hit_merged_components",
            "hit_threshold",
            "hit_merged_features",
            "peaklet_features",
            *spec.depends_on,
        ]

    def compute(self, context: Any, run_id: str, **_kwargs) -> np.ndarray:
        peaklets = context.get_data(run_id, "peaklets")
        if not isinstance(peaklets, np.ndarray):
            raise ValueError("peaklet_channels expects peaklets as a structured array")
        if len(peaklets) == 0:
            return _empty_channels()

        components = context.get_data(run_id, "peaklet_components")
        if not isinstance(components, np.ndarray):
            raise ValueError("peaklet_channels expects peaklet_components as a structured array")
        _validate_peaklet_components(peaklets, components)

        features = context.get_data(run_id, "hit_merged_features")
        if not isinstance(features, np.ndarray):
            raise ValueError("peaklet_channels expects hit_merged_features as a structured array")
        peaklet_features = context.get_data(run_id, "peaklet_features")
        if not isinstance(peaklet_features, np.ndarray):
            raise ValueError("peaklet_channels expects peaklet_features as a structured array")

        out, group_offsets, grouped_merged_indices = self._compute_channels(
            peaklets=peaklets,
            components=components,
            features=features,
            peaklet_features=peaklet_features,
            validate=False,
            return_groups=True,
        )
        if len(out) == 0:
            self._validate_and_fill_fractions(out, peaklets, peaklet_features)
            return out

        merged = context.get_data(run_id, "hit_merged")
        if not isinstance(merged, np.ndarray):
            raise ValueError("peaklet_channels requires hit_merged as a structured array")
        rebuild_groups = np.zeros(len(out), dtype=np.uint8)
        _mark_waveform_rebuild_groups_kernel(
            group_offsets,
            grouped_merged_indices,
            merged["sample_start"].astype(np.int64, copy=False),
            merged["sample_end"].astype(np.int64, copy=False),
            merged["time_start"].astype(np.int64, copy=False),
            merged["time_end"].astype(np.int64, copy=False),
            rebuild_groups,
            not bool(context.get_config(HitMergedFeaturesPlugin(), "normalize_to_pe")),
        )
        if np.any(rebuild_groups):
            component_hits = context.get_data(run_id, "hit_merged_components")
            hits = context.get_data(run_id, "hit_threshold")
            if not all(isinstance(value, np.ndarray) for value in (component_hits, hits)):
                raise ValueError("peaklet_channels requires structured hit reconstruction products")
            loaded = load_wave_input(context, self, run_id, needs_records_view=False)
            if not loaded.spec.is_records or loaded.records is None or loaded.wave_pool is None:
                raise ValueError("peaklet_channels currently supports wave_source='records' only")
            unresolved_groups = self._replace_with_numba_canonical_features(
                out=out,
                group_offsets=group_offsets,
                grouped_merged_indices=grouped_merged_indices,
                rebuild_groups=rebuild_groups,
                merged=merged,
                component_hits=component_hits,
                hits=hits,
                records=loaded.records,
                wave_pool=loaded.wave_pool,
                clip_negative_signal=bool(context.get_config(self, "clip_negative_signal")),
            )
            if len(unresolved_groups):
                self._replace_with_waveform_features(
                    out=out,
                    group_offsets=group_offsets,
                    grouped_merged_indices=grouped_merged_indices,
                    rebuild_groups=rebuild_groups,
                    group_indices=unresolved_groups,
                    merged=merged,
                    component_hits=component_hits,
                    hits=hits,
                    records=loaded.records,
                    wave_pool=loaded.wave_pool,
                    clip_negative_signal=bool(context.get_config(self, "clip_negative_signal")),
                )
        self._validate_and_fill_fractions(out, peaklets, peaklet_features)
        return out

    @staticmethod
    def _signal(record: np.void, raw: np.ndarray, clip_negative_signal: bool) -> np.ndarray:
        polarity_value = record["polarity"] if "polarity" in record.dtype.names else "negative"
        polarity = (
            polarity_value.decode("utf-8")
            if isinstance(polarity_value, bytes)
            else str(polarity_value)
        )
        baseline = np.float32(record["baseline"])
        signal = raw.astype(np.float32, copy=False) - baseline
        if polarity != "positive":
            signal = -signal
        if clip_negative_signal:
            signal = np.maximum(signal, np.float32(0.0))
        return signal

    @classmethod
    def _replace_with_numba_canonical_features(
        cls,
        *,
        out: np.ndarray,
        group_offsets: np.ndarray,
        grouped_merged_indices: np.ndarray,
        rebuild_groups: np.ndarray,
        merged: np.ndarray,
        component_hits: np.ndarray,
        hits: np.ndarray,
        records: np.ndarray,
        wave_pool: np.ndarray,
        clip_negative_signal: bool,
    ) -> np.ndarray:
        """Rebuild safe channel groups with the shared dense canonical kernel.

        The CSR expansion is deliberately batched. Groups that cannot be
        expressed as validated record windows, that do not fit the dense time
        grid, or that report a duplicate-sample conflict are returned to the
        existing Python oracle for exact public diagnostics.
        """
        record_lookup = RecordLookup(records)
        component_hit_indices = component_hits["hit_index"].astype(np.int64, copy=False)
        component_merged_indices = component_hits["merged_index"].astype(np.int64, copy=False)
        rec_wave_offsets = records["wave_offset"].astype(np.int64, copy=False)
        rec_event_lengths = records["event_length"].astype(np.int64, copy=False)
        rec_timestamps = records["timestamp"].astype(np.int64, copy=False)
        rec_dts = records["dt"].astype(np.int64, copy=False)
        rec_baselines = records["baseline"].astype(np.float32, copy=False)
        rec_polarity_signs = _polarity_sign_array(records)
        hit_record_ids = hits["record_id"].astype(np.int64, copy=False)
        hit_starts = hits["edge_start"].astype(np.int64, copy=False)
        hit_ends = hits["edge_end"].astype(np.int64, copy=False)

        merged_names = merged.dtype.names or ()
        merged_sample_starts = merged["sample_start"].astype(np.int64, copy=False)
        merged_sample_ends = merged["sample_end"].astype(np.int64, copy=False)
        merged_record_ids = merged["record_id"].astype(np.int64, copy=False)
        merged_is_single_record = (
            merged["is_single_record"].astype(np.int8, copy=False)
            if "is_single_record" in merged_names
            else ((merged_sample_starts >= 0) & (merged_sample_ends > merged_sample_starts)).astype(
                np.int8
            )
        )
        merged_component_offsets = merged["component_offset"].astype(np.int64, copy=False)
        merged_component_counts = merged["component_count"].astype(np.int64, copy=False)

        unresolved_parts: list[np.ndarray] = []
        rebuild_indices = np.flatnonzero(rebuild_groups).astype(np.int64, copy=False)
        for batch_start in range(0, len(rebuild_indices), _CANONICAL_GROUPS_PER_BATCH):
            batch_group_ids = rebuild_indices[
                batch_start : batch_start + _CANONICAL_GROUPS_PER_BATCH
            ]
            window_counts = np.zeros(len(batch_group_ids), dtype=np.int64)
            invalid_groups = np.zeros(len(batch_group_ids), dtype=np.uint8)
            _count_group_windows_kernel(
                batch_group_ids,
                group_offsets,
                grouped_merged_indices,
                merged_sample_starts,
                merged_sample_ends,
                merged_record_ids,
                merged_is_single_record,
                merged_component_offsets,
                merged_component_counts,
                component_merged_indices,
                component_hit_indices,
                hit_record_ids,
                hit_starts,
                hit_ends,
                window_counts,
                invalid_groups,
            )
            invalid_mask = (invalid_groups != 0) | (window_counts <= 0)
            if np.any(invalid_mask):
                unresolved_parts.append(batch_group_ids[invalid_mask])
            safe_local = np.flatnonzero(~invalid_mask)
            if len(safe_local) == 0:
                continue
            safe_group_ids = batch_group_ids[safe_local]
            safe_counts = window_counts[safe_local]
            window_offsets = np.empty(len(safe_counts) + 1, dtype=np.int64)
            window_offsets[0] = 0
            np.cumsum(safe_counts, out=window_offsets[1:])
            n_windows = int(window_offsets[-1])
            record_ids_array = np.empty(n_windows, dtype=np.int64)
            starts_array = np.empty(n_windows, dtype=np.int64)
            ends_array = np.empty(n_windows, dtype=np.int64)
            boards_array = np.empty(n_windows, dtype=np.int16)
            channels_array = np.empty(n_windows, dtype=np.int16)
            _fill_group_windows_kernel(
                safe_group_ids,
                group_offsets,
                grouped_merged_indices,
                merged_sample_starts,
                merged_sample_ends,
                merged_record_ids,
                merged_is_single_record,
                merged_component_offsets,
                merged_component_counts,
                component_hit_indices,
                hit_record_ids,
                hit_starts,
                hit_ends,
                out["board"],
                out["channel"],
                window_offsets,
                record_ids_array,
                starts_array,
                ends_array,
                boards_array,
                channels_array,
            )

            record_indices = record_lookup.get_indices(record_ids_array)
            clipped_starts = np.maximum(starts_array, 0)
            clipped_ends = np.minimum(ends_array, rec_event_lengths[record_indices])
            component_lengths = clipped_ends - clipped_starts
            component_dts = rec_dts[record_indices]
            component_times = rec_timestamps[record_indices] + clipped_starts * component_dts * 1000
            component_ends = component_times + component_lengths * component_dts * 1000
            group_time_starts = np.zeros(len(safe_group_ids), dtype=np.int64)
            group_spans = np.zeros(len(safe_group_ids), dtype=np.int64)
            group_status = np.zeros(len(safe_group_ids), dtype=np.int8)
            classify_dense_canonical_groups(
                window_offsets,
                component_times,
                component_ends,
                component_dts,
                boards_array.astype(np.int16, copy=False),
                channels_array.astype(np.int16, copy=False),
                rec_baselines[record_indices],
                group_time_starts,
                group_spans,
                group_status,
            )
            if np.any(group_status != 0):
                unresolved_parts.append(safe_group_ids[group_status != 0])

            safe_groups = np.flatnonzero(group_status == 0)
            safe_cursor = 0
            while safe_cursor < len(safe_groups):
                cumulative_spans = np.cumsum(group_spans[safe_groups[safe_cursor:]], dtype=np.int64)
                selected_count = int(
                    np.searchsorted(
                        cumulative_spans,
                        MAX_CANONICAL_DENSE_SAMPLES_PER_BATCH,
                        side="right",
                    )
                )
                selected_count = max(selected_count, 1)
                selected_groups = safe_groups[safe_cursor : safe_cursor + selected_count]
                selected_spans = group_spans[selected_groups]
                pool_offsets = np.empty(len(selected_groups) + 1, dtype=np.int64)
                pool_offsets[0] = 0
                np.cumsum(selected_spans, out=pool_offsets[1:])
                # Occupancy is the authoritative initialization bitmap; the
                # materializer writes values before any occupied slot is read.
                # Avoid zero-filling multi-million-sample long windows.
                values = np.empty(int(pool_offsets[-1]), dtype=np.float32)
                occupied = np.zeros(len(values), dtype=np.uint8)
                conflicts = np.zeros(len(selected_groups), dtype=np.uint8)
                materialize_dense_canonical_groups(
                    wave_pool,
                    selected_groups,
                    window_offsets,
                    pool_offsets,
                    group_time_starts[selected_groups],
                    record_indices,
                    clipped_starts,
                    clipped_ends,
                    component_times,
                    component_dts,
                    rec_wave_offsets,
                    rec_baselines,
                    rec_polarity_signs,
                    clip_negative_signal,
                    values,
                    values.view(np.uint32),
                    occupied,
                    conflicts,
                )
                reduced_areas = np.zeros(len(selected_groups), dtype=np.float64)
                reduced_heights = np.zeros(len(selected_groups), dtype=np.float32)
                has_samples = np.zeros(len(selected_groups), dtype=np.uint8)
                reduce_dense_canonical_groups(
                    values,
                    pool_offsets,
                    occupied,
                    reduced_areas,
                    reduced_heights,
                    has_samples,
                )
                unsafe = (conflicts != 0) | (has_samples == 0)
                if np.any(unsafe):
                    unresolved_parts.append(safe_group_ids[selected_groups[unsafe]])
                safe_reduced = np.flatnonzero(~unsafe)
                for local_index in safe_reduced:
                    output_group_index = int(safe_group_ids[selected_groups[local_index]])
                    out[output_group_index]["area"] = np.float32(reduced_areas[local_index])
                    out[output_group_index]["height"] = reduced_heights[local_index]
                safe_cursor += selected_count

        if not unresolved_parts:
            return np.zeros(0, dtype=np.int64)
        return np.unique(np.concatenate(unresolved_parts).astype(np.int64, copy=False))

    @classmethod
    def _replace_with_waveform_features(
        cls,
        *,
        out: np.ndarray,
        group_offsets: np.ndarray,
        grouped_merged_indices: np.ndarray,
        rebuild_groups: np.ndarray,
        group_indices: np.ndarray | None = None,
        merged: np.ndarray,
        component_hits: np.ndarray,
        hits: np.ndarray,
        records: np.ndarray,
        wave_pool: np.ndarray,
        clip_negative_signal: bool,
    ) -> None:
        """Rebuild only the channel groups that can contain shared samples."""
        record_lookup = RecordLookup(records)
        hit_indices = component_hits["hit_index"].astype(np.int64, copy=False)

        if group_indices is None:
            group_indices = np.flatnonzero(rebuild_groups)
        for group_index in group_indices:
            out_row = out[group_index]
            peaklet_id = int(out_row["peaklet_id"])
            board = int(out_row["board"])
            channel = int(out_row["channel"])
            group_start = int(group_offsets[group_index])
            group_end = int(group_offsets[group_index + 1])
            segments: list[dict[str, Any]] = []
            for grouped_index in range(group_start, group_end):
                merged_index = int(grouped_merged_indices[grouped_index])
                merged_row = merged[merged_index]
                sample_start = int(merged_row["sample_start"])
                sample_end = int(merged_row["sample_end"])
                is_single = (
                    bool(merged_row["is_single_record"])
                    if "is_single_record" in merged.dtype.names
                    else sample_start >= 0 and sample_end > sample_start
                )
                component_offset = int(merged_row["component_offset"])
                component_count = int(merged_row["component_count"])
                if (
                    component_count > 0
                    and component_offset >= 0
                    and component_offset + component_count <= len(hit_indices)
                    and np.all(
                        component_hits["merged_index"][
                            component_offset : component_offset + component_count
                        ]
                        == merged_index
                    )
                ):
                    merged_hit_indices = hit_indices[
                        component_offset : component_offset + component_count
                    ]
                else:
                    # Compatibility fallback for older/incomplete merged rows.
                    # Production rows use the O(1) CSR slice above.
                    merged_hit_indices = hit_indices[component_hits["merged_index"] == merged_index]
                windows = (
                    [(int(merged_row["record_id"]), sample_start, sample_end)]
                    if is_single and sample_start >= 0 and sample_end > sample_start
                    else [
                        (
                            int(hits[hit_index]["record_id"]),
                            int(hits[hit_index]["edge_start"]),
                            int(hits[hit_index]["edge_end"]),
                        )
                        for hit_index in merged_hit_indices
                    ]
                )
                for record_id, start, end in windows:
                    record_index = int(record_lookup.get_indices(np.array([record_id]))[0])
                    record = records[record_index]
                    clipped_start = max(0, start)
                    clipped_end = min(int(record["event_length"]), end)
                    if clipped_end <= clipped_start:
                        continue
                    offset = int(record["wave_offset"])
                    signal = cls._signal(
                        record,
                        wave_pool[offset + clipped_start : offset + clipped_end],
                        clip_negative_signal,
                    )
                    dt_ns = int(record["dt"])
                    dt_ps = dt_ns * 1000
                    abs_time_ps = (
                        int(record["timestamp"])
                        + np.arange(clipped_start, clipped_end, dtype=np.int64) * dt_ps
                    )
                    segments.append(
                        {
                            "waveform": signal,
                            "abs_time_ps": abs_time_ps,
                            "dt": dt_ns,
                            "board": board,
                            "channel": channel,
                            "record_id": record_id,
                            "merged_index": merged_index,
                        }
                    )

            channel_wave = merge_waveform_segments(
                segments,
                sum_channels=False,
                dense=False,
                context=(
                    f"peaklet_channels peaklet_id={peaklet_id}, board={board}, channel={channel}"
                ),
            )["waveform"]
            if len(channel_wave) == 0:
                raise ValueError(
                    "peaklet_channels could not reconstruct waveform for "
                    f"peaklet_id={peaklet_id}, board={board}, channel={channel}"
                )
            out_row["area"] = np.sum(channel_wave, dtype=np.float64)
            out_row["height"] = np.max(channel_wave)

    @staticmethod
    def _validate_and_fill_fractions(
        out: np.ndarray, peaklets: np.ndarray, peaklet_features: np.ndarray
    ) -> None:
        feature_ids = peaklet_features["peak_id"].astype(np.int64, copy=False)
        if len(peaklet_features) == len(peaklets) and _has_dense_identity_merged_indices(
            feature_ids
        ):
            # Production peaklet_features is dense by peak_id.  Keep the
            # native float32 field as a view instead of materializing a full
            # float64 area table; the Numba kernel promotes each value to
            # float64 for the same conservation arithmetic.
            area_by_peaklet = peaklet_features["area"].astype(np.float32, copy=False)
        else:
            area_by_peaklet = np.zeros(len(peaklets), dtype=np.float64)
            valid_features = (feature_ids >= 0) & (feature_ids < len(peaklets))
            area_by_peaklet[feature_ids[valid_features]] = peaklet_features["area"][
                valid_features
            ].astype(np.float64, copy=False)
        out_ids = out["peaklet_id"].astype(np.int64, copy=False)
        if np.any((out_ids < 0) | (out_ids >= len(peaklets))):
            # Keep the historical indexing failure shape for malformed
            # external products instead of silently dropping an output row.
            _ = area_by_peaklet[out_ids]
        output_peaklet_starts = np.searchsorted(
            out_ids, np.arange(len(peaklets) + 1, dtype=np.int64), side="left"
        )
        area_mismatch = np.zeros(len(peaklets), dtype=np.uint8)
        fraction_mismatch = np.zeros(len(peaklets), dtype=np.uint8)
        _fill_fractions_and_validate_kernel(
            output_peaklet_starts,
            out_ids,
            out["area"],
            out["area_fraction"],
            area_by_peaklet,
            area_mismatch,
            fraction_mismatch,
        )
        mismatch = area_mismatch != 0
        if np.any(mismatch):
            peaklet_id = int(np.flatnonzero(mismatch)[0])
            start = int(output_peaklet_starts[peaklet_id])
            end = int(output_peaklet_starts[peaklet_id + 1])
            channel_area = np.sum(out["area"][start:end], dtype=np.float64)
            raise ValueError(
                "peaklet_channels area conservation failed for "
                f"peaklet_id={peaklet_id}: channel_area={channel_area} "
                f"!= peak_area={area_by_peaklet[peaklet_id]}"
            )
        nonzero_area = area_by_peaklet != 0.0
        fraction_mismatch = (fraction_mismatch != 0) & nonzero_area
        if np.any(fraction_mismatch):
            peaklet_id = int(np.flatnonzero(fraction_mismatch)[0])
            start = int(output_peaklet_starts[peaklet_id])
            end = int(output_peaklet_starts[peaklet_id + 1])
            fraction_sum = np.sum(out["area_fraction"][start:end], dtype=np.float64)
            raise ValueError(
                "peaklet_channels fraction conservation failed for "
                f"peaklet_id={peaklet_id}: fraction_sum={fraction_sum}"
            )

    def _compute_channels(
        self,
        *,
        peaklets: np.ndarray,
        components: np.ndarray,
        features: np.ndarray,
        peaklet_features: np.ndarray,
        validate: bool = True,
        return_groups: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
        if len(components) == 0 or len(features) == 0:
            empty = _empty_channels()
            if return_groups:
                return empty, np.zeros(1, dtype=np.int64), np.zeros(0, dtype=np.int64)
            return empty

        # Production peaklet_components is a contiguous CSR slice per
        # peaklet, and hit_merged_features is dense by merged_index.  When
        # both invariants (including stable (peaklet, board, channel) order)
        # hold, the two-stage Numba path writes output/member CSR directly.
        component_names = components.dtype.names or ()
        peaklet_names = peaklets.dtype.names or ()
        feature_names = features.dtype.names or ()
        if (
            {
                "peak_id",
                "merged_index",
            }.issubset(component_names)
            and {
                "component_offset",
                "component_count",
            }.issubset(peaklet_names)
            and {
                "merged_index",
                "board",
                "channel",
                "area",
                "height",
                "n_hits",
                "valid",
            }.issubset(feature_names)
        ):
            feature_merged = features["merged_index"].astype(np.int64, copy=False)
            if _has_dense_identity_merged_indices(feature_merged):
                peaklet_component_offsets = peaklets["component_offset"].astype(
                    np.int64, copy=False
                )
                peaklet_component_counts = peaklets["component_count"].astype(np.int64, copy=False)
                # Keep the native i1/i2 feature fields.  Widening every
                # board/channel to int64 here costs hundreds of MiB on the
                # production feature cache and does not improve the Numba
                # comparisons or the i2 output fields.
                feature_valid = features["valid"]
                feature_boards = features["board"]
                feature_channels = features["channel"]
                if _fast_group_structure_is_valid(
                    components["peak_id"].astype(np.int64, copy=False),
                    components["merged_index"].astype(np.int64, copy=False),
                    peaklet_component_offsets,
                    peaklet_component_counts,
                    len(features),
                    feature_valid,
                    feature_boards,
                    feature_channels,
                ):
                    group_counts = np.zeros(len(peaklets), dtype=np.int64)
                    member_counts = np.zeros(len(peaklets), dtype=np.int64)
                    _count_fast_groups_kernel(
                        peaklet_component_offsets,
                        peaklet_component_counts,
                        components["merged_index"].astype(np.int64, copy=False),
                        feature_valid,
                        feature_boards,
                        feature_channels,
                        group_counts,
                        member_counts,
                    )
                    group_prefix = np.empty(len(peaklets) + 1, dtype=np.int64)
                    member_prefix = np.empty(len(peaklets) + 1, dtype=np.int64)
                    group_prefix[0] = 0
                    member_prefix[0] = 0
                    np.cumsum(group_counts, out=group_prefix[1:])
                    np.cumsum(member_counts, out=member_prefix[1:])
                    n_groups = int(group_prefix[-1])
                    n_members = int(member_prefix[-1])
                    out = np.zeros(n_groups, dtype=PEAKLET_CHANNELS_DTYPE)
                    group_member_offsets = np.empty(n_groups + 1, dtype=np.int64)
                    grouped_merged_indices = np.empty(n_members, dtype=np.int64)
                    _fill_fast_groups_kernel(
                        peaklet_component_offsets,
                        peaklet_component_counts,
                        components["merged_index"].astype(np.int64, copy=False),
                        feature_valid,
                        feature_boards,
                        feature_channels,
                        features["area"].astype(np.float32, copy=False),
                        features["height"].astype(np.float32, copy=False),
                        features["n_hits"].astype(np.int32, copy=False),
                        group_prefix,
                        member_prefix,
                        group_member_offsets,
                        grouped_merged_indices,
                        out["peaklet_id"],
                        out["board"],
                        out["channel"],
                        out["area"],
                        out["height"],
                        out["n_hits"],
                    )
                    # Match NumPy's pairwise floating-point reduction exactly
                    # for the public area field.  The Numba fill kernel keeps
                    # the direct CSR path allocation-free for matching/key
                    # arrays; this bounded member-area view is the only
                    # compatibility reduction materialization.
                    if n_groups:
                        member_areas = features["area"][grouped_merged_indices]
                        out["area"] = np.add.reduceat(
                            member_areas, group_member_offsets[:-1]
                        ).astype(np.float32, copy=False)
                    if validate:
                        self._validate_and_fill_fractions(out, peaklets, peaklet_features)
                    if return_groups:
                        return out, group_member_offsets, grouped_merged_indices
                    return out

        component_merged = components["merged_index"].astype(np.int64, copy=False)
        feature_merged = features["merged_index"].astype(np.int64, copy=False)
        if _has_dense_identity_merged_indices(feature_merged):
            matched = (component_merged >= 0) & (component_merged < len(features))
            matched_positions = component_merged[matched]
            valid_matched = features["valid"][matched_positions] != 0
            component_positions = np.flatnonzero(matched)
            matched = np.zeros(len(components), dtype=bool)
            matched[component_positions[valid_matched]] = True
            matched_features = features[component_merged[matched]]
        else:
            valid_features = features[features["valid"] != 0]
            if len(valid_features) == 0:
                empty = _empty_channels()
                if return_groups:
                    return empty, np.zeros(1, dtype=np.int64), np.zeros(0, dtype=np.int64)
                return empty
            valid_merged = valid_features["merged_index"].astype(np.int64, copy=False)
            feature_order = np.argsort(valid_merged, kind="mergesort")
            sorted_merged = valid_merged[feature_order]
            matched_pos = np.searchsorted(sorted_merged, component_merged, side="right") - 1
            matched = matched_pos >= 0
            matched[matched] &= sorted_merged[matched_pos[matched]] == component_merged[matched]
            matched_features = valid_features[feature_order[matched_pos[matched]]]
        if not np.any(matched):
            empty = _empty_channels()
            if return_groups:
                return empty, np.zeros(1, dtype=np.int64), np.zeros(0, dtype=np.int64)
            return empty

        peaklet_ids = components["peak_id"][matched].astype(np.int64, copy=False)
        boards = matched_features["board"].astype(np.int64, copy=False)
        channels = matched_features["channel"].astype(np.int64, copy=False)
        grouped_merged_indices = component_merged[matched]
        if _keys_are_nondecreasing(peaklet_ids, boards, channels):
            areas = matched_features["area"].astype(np.float32, copy=False)
            heights = matched_features["height"].astype(np.float32, copy=False)
            n_hits = matched_features["n_hits"].astype(np.int32, copy=False)
        else:
            group_order = np.lexsort((channels, boards, peaklet_ids))
            peaklet_ids = peaklet_ids[group_order]
            boards = boards[group_order]
            channels = channels[group_order]
            areas = matched_features["area"][group_order].astype(np.float32, copy=False)
            heights = matched_features["height"][group_order].astype(np.float32, copy=False)
            n_hits = matched_features["n_hits"][group_order].astype(np.int32, copy=False)
            grouped_merged_indices = grouped_merged_indices[group_order]

        group_start_mask = np.r_[
            True,
            (peaklet_ids[1:] != peaklet_ids[:-1])
            | (boards[1:] != boards[:-1])
            | (channels[1:] != channels[:-1]),
        ]
        group_starts = np.flatnonzero(group_start_mask)

        out = np.zeros(len(group_starts), dtype=PEAKLET_CHANNELS_DTYPE)
        out["peaklet_id"] = peaklet_ids[group_starts]
        out["board"] = boards[group_starts]
        out["channel"] = channels[group_starts]
        out["area"] = np.add.reduceat(areas, group_starts).astype(np.float32, copy=False)
        out["height"] = np.maximum.reduceat(heights, group_starts).astype(np.float32, copy=False)
        out["n_hits"] = np.add.reduceat(n_hits, group_starts).astype(np.int32, copy=False)

        group_offsets = np.empty(len(group_starts) + 1, dtype=np.int64)
        group_offsets[:-1] = group_starts
        group_offsets[-1] = len(grouped_merged_indices)

        if validate:
            self._validate_and_fill_fractions(out, peaklets, peaklet_features)
        if return_groups:
            return out, group_offsets, grouped_merged_indices
        return out


__all__ = ["PEAKLET_CHANNELS_DTYPE", "PeakletChannelsPlugin"]
