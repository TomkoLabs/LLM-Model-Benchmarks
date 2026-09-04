from __future__ import annotations

import math
from typing import Any, Mapping


STREAMING_TPS_SEMANTICS = (
    "nonempty_visible_content_chunks_per_benchmark_wall_second"
)
NON_STREAMING_TPS_SEMANTICS = (
    "server_reported_completion_tokens_per_benchmark_wall_second"
)


def _nonnegative_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number) or number < 0:
        return None
    return number


def _stats_mean(value: Any) -> float | None:
    if not isinstance(value, Mapping):
        return None
    return _nonnegative_number(value.get("mean"))


def estimated_visible_generation_chunks_per_second(
    itl_stats: Any,
) -> float | None:
    """Return the observed visible-content chunk cadence from mean ITL.

    Pinned Infermark records one interval between successive non-empty
    ``delta.content`` events. Those events are transport chunks, not proven
    tokenizer tokens, and ``delta.reasoning_content`` events are ignored.
    """

    mean_itl = _stats_mean(itl_stats)
    if mean_itl is None or mean_itl <= 0:
        return None
    return 1.0 / mean_itl


def streaming_metric_aliases(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Add accurately named aliases for pinned Infermark streaming fields."""

    ttft = metrics.get("ttft_seconds_c1")
    itl = metrics.get("itl_seconds_c1")
    return {
        "infermark_measurement_mode": "streaming",
        "tokens_per_second_semantics": STREAMING_TPS_SEMANTICS,
        "generation_tokens_per_second_c1": None,
        "visible_output_chunks_per_second_c1": metrics.get(
            "tokens_per_second_c1"
        ),
        "time_to_first_visible_chunk_seconds_c1": ttft,
        "visible_output_inter_chunk_latency_seconds_c1": itl,
        "estimated_visible_generation_chunks_per_second_c1": (
            estimated_visible_generation_chunks_per_second(itl)
        ),
    }


def non_streaming_metric_aliases(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Describe pinned Infermark's non-streaming usage-based throughput."""

    return {
        "infermark_measurement_mode": "non_streaming",
        "tokens_per_second_semantics": NON_STREAMING_TPS_SEMANTICS,
        "generation_tokens_per_second_c1": None,
        "end_to_end_output_tokens_per_second_c1": metrics.get(
            "tokens_per_second_c1"
        ),
    }


def performance_summary(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Build a display-only c1 summary, including historical result fallback."""

    actual_generation = _nonnegative_number(
        metrics.get("generation_tokens_per_second_c1")
    )
    visible_generation = _nonnegative_number(
        metrics.get("estimated_visible_generation_chunks_per_second_c1")
    )
    if visible_generation is None:
        visible_generation = estimated_visible_generation_chunks_per_second(
            metrics.get("visible_output_inter_chunk_latency_seconds_c1")
            or metrics.get("itl_seconds_c1")
        )

    ttft = (
        metrics.get("time_to_first_visible_chunk_seconds_c1")
        or metrics.get("ttft_seconds_c1")
    )
    visible_throughput = _nonnegative_number(
        metrics.get("visible_output_chunks_per_second_c1")
    )
    if visible_throughput is None and (
        metrics.get("infermark_measurement_mode") == "streaming"
        or isinstance(metrics.get("itl_seconds_c1"), Mapping)
    ):
        visible_throughput = _nonnegative_number(
            metrics.get("tokens_per_second_c1")
        )

    return {
        "generation_tokens_per_second": actual_generation,
        "generation_tokens_per_second_source": (
            "legacy-result-token-count-source"
            if actual_generation is not None
            else None
        ),
        "prefill_tokens_per_second": None,
        "mean_time_to_first_token_seconds": None,
        "estimated_visible_generation_chunks_per_second": visible_generation,
        "mean_time_to_first_visible_chunk_seconds": _stats_mean(ttft),
        "end_to_end_visible_output_chunks_per_second": visible_throughput,
    }
