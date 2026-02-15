"""
EQAT 3-term loss — Eq. 9 from the paper.

L_total = L_CE(y_quant, y_true)
        + α · KL(y_quant ‖ y_full)
        + β_epoch · E_total({q})

Progressive energy penalty (Algorithm 1):
  β_epoch = β · min(1.0, epoch / warmup_epochs)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EQATLoss(nn.Module):
    def __init__(self, alpha: float = 0.95, beta: float = 0.01,
                 warmup_epochs: int = 10):
        super().__init__()
        self.alpha          = alpha
        self.beta           = beta
        self.warmup_epochs  = warmup_epochs

    def beta_now(self, epoch: int) -> float:
        return self.beta * min(1.0, epoch / max(1, self.warmup_epochs))

    def forward(self, y_quant: torch.Tensor, y_full: torch.Tensor,
                y_true: torch.Tensor, e_norm: torch.Tensor,
                epoch: int):
        """
        Args:
            y_quant : logits from quantized forward
            y_full  : logits from full-precision forward (detached)
            y_true  : ground-truth labels
            e_norm  : normalised energy scalar (differentiable)
            epoch   : current epoch (1-indexed)

        Returns:
            total loss, dict of components
        """
        l_ce = F.cross_entropy(y_quant, y_true)

        # KL(y_quant ‖ y_full) — Eq. 9: F.kl_div(log_Q, P) = KL(P‖Q)
        # so to get KL(y_quant‖y_full) set P=p_quant, Q=p_full
        log_p_full = F.log_softmax(y_full,  dim=1)
        p_quant    = F.softmax(y_quant, dim=1)
        l_kl       = F.kl_div(log_p_full, p_quant, reduction='batchmean')

        beta  = self.beta_now(epoch)
        total = l_ce + self.alpha * l_kl + beta * e_norm

        return total, {
            'L_CE'    : l_ce.item(),
            'L_KL'    : l_kl.item(),
            'E_norm'  : e_norm.item(),
            'beta'    : beta,
        }
