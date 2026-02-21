# Shared quantization primitives
from .jacob_fake_quant import JacobFakeQuantize
from .hessian_sensitivity import compute_hessian_trace

# Jacob QAT (Jacob et al., CVPR 2018)
from .jacob_qat import QConv2d, QLinear, fold_bn_into_conv, prepare_model_for_qat, JacobQATTrainer

# HAQ (Wang et al., CVPR 2019)
from .haq import DDPGAgent, HAQEnvironment
from .haq.haq_quantize_utils import QConv2d as HAQQConv2d, QLinear as HAQQLinear

# Edge-MPQ (Zhao et al., IEEE TC 2024)
from .edge_mpq import apply_mixed_precision, MPQConv2d, MPQLinear, solve_bit_assignment

# ASQ (Zhou et al., arXiv 2025)
from .asq import ASQActivationQuantizer, POSTWeightQuantizer, LSQQuantizer
from .asq import replace_with_asq, replace_with_lsq, ASQConv2d, ASQLinear

# SDP (ACM TODAES, 2022)
from .sdp import SDPQuantizer, replace_with_sdp, SDPConv2d, SDPLinear

# AdaPT (Kummer et al., arXiv 2021)
from .adapt import FixedPointQuantizer, AdaPTPrecisionController

# LCPAQ (arXiv 2402.17706, 2024)
from .lcpaq import build_cost_table, greedy_pareto_selection, compute_quant_errors
from .lcpaq import proxy_hyperparameter_search

# EfficientQAT (Chen et al., ACL 2025)
from .efficient_qat import (
    BlockAPTrainer, EQATConv2d, EQATLinear,
    LearnableQuantizer, set_quant_state, split_resnet_into_blocks,
    E2EQPTrainer,
)

# AQUA v2: Outlier-Aware Distribution-Guided Mixed-Precision
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
