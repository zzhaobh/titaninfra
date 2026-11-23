"""TitanInfra profiling API backed by HolisticTraceAnalysis (HTA).

Install with pip install HolisticTraceAnalysis. The wrapper exposes temporal
breakdown, communication-compute overlap, GPU-kernel ranking, idle-time
breakdown, and formatted text reports as pandas DataFrames.
"""

import logging

logger = logging.getLogger(__name__)


class TraceAnalysis:
    """Provide HTA-backed trace analysis as pandas DataFrames."""

    def __init__(self, trace_dir: str) -> None:
        """Initialize the analyzer from a directory of rank trace JSON files."""
        try:
            from hta.trace_analysis import TraceAnalysis as _HTAAnalysis
        except ImportError as e:
            raise ImportError(
                "HolisticTraceAnalysis not installed.\n"
                "Install: pip install HolisticTraceAnalysis\n"
                "Repo: https://github.com/facebookresearch/HolisticTraceAnalysis"
            ) from e

        self._trace_dir = trace_dir
        self._analyzer = _HTAAnalysis(trace_dir=trace_dir)
        logger.info(f"TraceAnalysis initialized: {trace_dir}")

    def get_temporal_breakdown(self, ranks: list[int] | None = None):
        """Return compute, communication, and idle percentages per rank."""
        df = self._analyzer.get_temporal_breakdown(visualize=False)
        if ranks is not None:
            df = df[df["rank"].isin(ranks)]
        return df

    def get_comm_comp_overlap(self, ranks: list[int] | None = None):
        """Return communication-compute overlap percentages per rank."""
        df = self._analyzer.get_comm_comp_overlap(visualize=False)
        if ranks is not None:
            df = df[df["rank"].isin(ranks)]
        return df

    def get_gpu_kernel_breakdown(self, top_k: int = 10):
        """Return the top_k GPU kernels by total CUDA time."""
        df = self._analyzer.get_gpu_kernel_breakdown(visualize=False)
        return df.head(top_k)

    def get_idle_time_breakdown(self, ranks: list[int] | None = None):
        """Return idle-time categories per rank, optionally filtered."""
        result = self._analyzer.get_idle_time_breakdown(visualize=False)
        df = result[0] if isinstance(result, tuple) else result
        if ranks is not None:
            df = df[df["rank"].isin(ranks)]
        return df

    def generate_report(self, ranks: list[int] | None = None, top_k: int = 10) -> str:
        """Generate a formatted report covering all analysis dimensions."""
        lines = []
        sep = "=" * 60

        lines.append(sep)
        lines.append("1. Temporal Breakdown (compute / comm / idle per rank)")
        lines.append(sep)
        lines.append(self.get_temporal_breakdown(ranks=ranks).to_string(index=False))

        lines.append(f"\n{sep}")
        lines.append("2. Communication-Computation Overlap")
        lines.append(sep)
        lines.append(self.get_comm_comp_overlap(ranks=ranks).to_string(index=False))

        lines.append(f"\n{sep}")
        lines.append(f"3. Top-{top_k} GPU Kernels (by total CUDA time)")
        lines.append(sep)
        lines.append(self.get_gpu_kernel_breakdown(top_k=top_k).to_string(index=False))

        lines.append(f"\n{sep}")
        lines.append("4. Idle Time Breakdown")
        lines.append(sep)
        lines.append(self.get_idle_time_breakdown(ranks=ranks).to_string(index=False))

        lines.append(f"\n{sep}")
        lines.append("Analysis complete.")
        lines.append(sep)

        return "\n".join(lines)
