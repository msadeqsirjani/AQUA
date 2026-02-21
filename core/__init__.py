# Shared quantization primitives
from .hessian_sensitivity import compute_hessian_trace

# AQUA: Outlier-Aware Distribution-Guided Mixed-Precision
from .aqua import (
    IntQuantizer as AQUAIntQuantizer,
    FpQuantizer as AQUAFpQuantizer,
    DTYPE_CATALOG, ALL_DTYPE_NAMES, resolve_dtypes,
    AQUAConv2d, AQUALinear,
    replace_with_aqua,
    get_aqua_layer_stats, compute_per_layer_quant_error,
    collect_quantizer_params, count_aqua_layers,
    get_avg_bits, get_dtype_counts,
    AQUATrainer,
    analyze_layer_distributions, recommend_format,
    compute_fisher_diagonal, compute_outlier_masks,
    compute_dtype_errors, build_bops_table, solve_dtype_assignment,
    compute_energy as aqua_compute_energy,
    dtype_entropy as aqua_dtype_entropy,
    format_entropy as aqua_format_entropy,
)
