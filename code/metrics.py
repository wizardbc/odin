from typing import Dict, Optional, Tuple

import torch
from torch import Tensor

from torchmetrics.classification.precision_recall_curve import BinaryPrecisionRecallCurve
from torchmetrics.functional.classification.roc import _binary_roc_compute
from torchmetrics.functional.classification.precision_recall_curve import _binary_precision_recall_curve_compute
from torchmetrics.utilities.data import dim_zero_cat

from tqdm.auto import tqdm
import matplotlib.pyplot as plt


class BinaryMetrics(BinaryPrecisionRecallCurve):
  is_differentiable: bool = False
  higher_is_better: Optional[bool] = None
  full_state_update: bool = False

  def compute_prc_in(self) -> Tuple[Tensor, Tensor, Tensor]:
    if self.thresholds is None:
      state = [dim_zero_cat(self.preds), dim_zero_cat(self.target)]
    else:
      state = self.confmat
    return _binary_precision_recall_curve_compute(state, self.thresholds)

  def compute_prc_out(self) -> Tuple[Tensor, Tensor, Tensor]:
    if self.thresholds is None:
      state = [1-dim_zero_cat(self.preds), 1-dim_zero_cat(self.target)]
    else:
      state = self.confmat.flip((1,2))
    return _binary_precision_recall_curve_compute(state, self.thresholds)

  def compute_roc(self) -> Tuple[Tensor, Tensor, Tensor]:
    if self.thresholds is None:
      state = [dim_zero_cat(self.preds), dim_zero_cat(self.target)]
    else:
      state = self.confmat
    return _binary_roc_compute(state, self.thresholds)

  def compute(self) -> Tuple[Tuple, Tuple, Tuple]:
    """Compute ROC, PRC_In, PRC_Out.
    """
    return {
      'ROC': self.compute_roc(),
      'PRC_In': self.compute_prc_in(),
      'PRC_Out': self.compute_prc_out(),
    }

def runner(cls):
  class Runner(cls):
    """Metrics seen in ODIN paper.
    https://github.com/facebookresearch/odin/blob/main/code/calMetric.py
    """
    def __init__(self, *args, **kwargs):
      super(self.__class__, self).__init__(*args, **kwargs)
      self.metrics = BinaryMetrics().to(self.device)

    def fpr_at_95(self) -> float:
      fpr, tpr, _ = self.metrics.compute_roc()
      return fpr[tpr >= 0.95].min()

    def detection_err(self) -> float:
      fpr, tpr, _ = self.metrics.compute_roc()
      return ((1-tpr+fpr)/2).min()

    def auroc(self, plot:bool=False, **plot_kwargs) -> float:
      fpr, tpr, _ = self.metrics.compute_roc()
      if plot:
        plt.plot(fpr.cpu(), tpr.cpu(), **plot_kwargs)
      return torch.trapz(tpr, fpr)

    def aupr_in(self, plot:bool=False, **plot_kwargs) -> float:
      p, r, _ = self.metrics.compute_prc_in()
      if plot:
        plt.plot(r.cpu(), p.cpu(), **plot_kwargs)
      return -torch.trapz(p, r)

    def aupr_out(self, plot:bool=False, **plot_kwargs) -> float:
      p, r, _ = self.metrics.compute_prc_out()
      if plot:
        plt.plot(r.cpu(), p.cpu(), **plot_kwargs)
      return -torch.trapz(p, r)

    def compute_metrics(self) -> Dict[str, float]:
      res = self.metrics.compute()
      
      fpr, tpr, _ = res['ROC']
      p_in, r_in, _ = res['PRC_In']
      p_out, r_out, _ = res['PRC_Out']
      
      metrics = ({
        'FPR@95': fpr[tpr >= 0.95].min().item(),
        'DErr': ((1-tpr+fpr)/2).min().item(),
        'AUROC': torch.trapz(tpr, fpr).item(),
        'AUPR_In': -torch.trapz(p_in, r_in).item(),
        'AUPR_Out': -torch.trapz(p_out, r_out).item(),
      })

      return metrics

    def run_over_dl(self, dataloader, ood:bool=False, prog_bar:bool=True, postfix:bool=False):
      if prog_bar:
        pbar = tqdm(dataloader, desc='Out-of-dist' if ood else 'In-dist')
      else:
        pbar = dataloader
      for imgs, _ in pbar:
        preds = self.forward(imgs).max(dim=-1)[0]
        targets = torch.tensor([int(not ood)]*preds.shape[0], device=self.device)
        self.metrics(preds, targets)
        if prog_bar and postfix:
          pbar.set_postfix(self.compute_metrics())

    def run(self, in_dl, out_dl, prog_bar:bool=True, postfix:bool=False):
      self.metrics.reset()
      self.run_over_dl(in_dl, ood=False, prog_bar=prog_bar, postfix=postfix)
      self.run_over_dl(out_dl, ood=True, prog_bar=prog_bar, postfix=postfix)
      return self.compute_metrics()

  return Runner