# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmagic.registry import MODELS
from .loss_wrapper import masked_loss

from pytorch_wavelets import DWTForward

_reduction_modes = ['none', 'mean', 'sum']


@masked_loss
def l1_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L1 loss.

    Args:
        pred (Tensor): Prediction Tensor with shape (n, c, h, w).
        target ([type]): Target Tensor with shape (n, c, h, w).

    Returns:
        Tensor: Calculated L1 loss.
    """
    return F.l1_loss(pred, target, reduction='none')


@masked_loss
def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MSE loss.

    Args:
        pred (Tensor): Prediction Tensor with shape (n, c, h, w).
        target ([type]): Target Tensor with shape (n, c, h, w).

    Returns:
        Tensor: Calculated MSE loss.
    """
    return F.mse_loss(pred, target, reduction='none')

@masked_loss
def patch_loss0(pred: torch.Tensor,
                     target: torch.Tensor,
                     eps: float = 1e-12) -> torch.Tensor:
    """Charbonnier loss.

    Args:
        pred (Tensor): Prediction Tensor with shape (n, c, h, w).
        target ([type]): Target Tensor with shape (n, c, h, w).
        eps (float): A value used to control the curvature near zero.
            Default: 1e-12.

    Returns:
        Tensor: Calculated Charbonnier loss.
    """
    # breakpoint()
    h, w = pred.shape[3], pred.shape[4]     # 256*256
    pred = pred.view(-1, 3, h, w)
    target = target.view(-1, 3, h, w)
    patch_size = h//4
    pred_patches = pred.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    pred = pred_patches.contiguous().view(-1, 3, patch_size, patch_size)
    target_patches = target.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    target = target_patches.contiguous().view(-1, 3, patch_size, patch_size)

    DWT2 = DWTForward(J=1, wave='haar', mode='reflect').to(pred.device)
    LLp, Hcp = DWT2(pred)
    LHp, HLp, HHp = Hcp[0][:, :, 0, :, :], Hcp[0][:, :, 1, :, :], Hcp[0][:, :, 2, :, :]
    LLt, Hct = DWT2(target)
    LHt, HLt, HHt = Hct[0][:, :, 0, :, :], Hct[0][:, :, 1, :, :], Hct[0][:, :, 2, :, :]
    loss_LH = torch.sqrt((LHp - LHt)**2 + eps)
    loss_HL = torch.sqrt((HLp - HLt) ** 2 + eps)
    loss_HH = torch.sqrt((HHp - HHt) ** 2 + eps)
    loss_LH_mean = torch.mean(loss_LH, dim=1)
    loss_HL_mean = torch.mean(loss_HL, dim=1)
    loss_HH_mean = torch.mean(loss_HH, dim=1)
    loss_mean = loss_LH_mean + loss_HL_mean + loss_HH_mean
    loss_max = torch.max(loss_mean, dim=1)

    return loss_LH + loss_HL + loss_HH + loss_max

# @masked_loss
def patch_loss(pred: torch.Tensor,
                     target: torch.Tensor,
                     eps: float = 1e-12) -> torch.Tensor:
    """Charbonnier loss.

    Args:
        pred (Tensor): Prediction Tensor with shape (n, c, h, w).
        target ([type]): Target Tensor with shape (n, c, h, w).
        eps (float): A value used to control the curvature near zero.
            Default: 1e-12.

    Returns:
        Tensor: Calculated Charbonnier loss.
    """
    # breakpoint()
    h, w = pred.shape[3], pred.shape[4]     # 256*256
    pred = pred.view(-1, 3, h, w)
    target = target.view(-1, 3, h, w)
    patch_size = h//4
    pred_patches = pred.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    pred = pred_patches.contiguous().view(-1, 3, patch_size, patch_size)
    target_patches = target.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    target = target_patches.contiguous().view(-1, 3, patch_size, patch_size)

    DWT2 = DWTForward(J=1, wave='haar', mode='reflect').to(pred.device)
    LLp, Hcp = DWT2(pred)
    LHp, HLp, HHp = Hcp[0][:, :, 0, :, :], Hcp[0][:, :, 1, :, :], Hcp[0][:, :, 2, :, :]
    LLt, Hct = DWT2(target)
    LHt, HLt, HHt = Hct[0][:, :, 0, :, :], Hct[0][:, :, 1, :, :], Hct[0][:, :, 2, :, :]
    loss_LH = torch.sqrt((LHp - LHt)**2 + eps)
    loss_HL = torch.sqrt((HLp - HLt) ** 2 + eps)
    loss_HH = torch.sqrt((HHp - HHt) ** 2 + eps)
    loss_LH_mean = torch.mean(loss_LH, dim=1)
    loss_HL_mean = torch.mean(loss_HL, dim=1)
    loss_HH_mean = torch.mean(loss_HH, dim=1)
    loss_mean = loss_LH_mean + loss_HL_mean + loss_HH_mean
    loss_max = 0
    for i in range(loss_mean.shape[0]):
        loss_max += torch.max(loss_mean[i]).item()
    loss_max = loss_max/loss_mean.shape[0]

    return torch.mean(loss_LH + loss_HL + loss_HH) + loss_max

@masked_loss
def wav_loss(pred: torch.Tensor,
                     target: torch.Tensor,
                     eps: float = 1e-12) -> torch.Tensor:
    """Charbonnier loss.

    Args:
        pred (Tensor): Prediction Tensor with shape (n, c, h, w).
        target ([type]): Target Tensor with shape (n, c, h, w).
        eps (float): A value used to control the curvature near zero.
            Default: 1e-12.

    Returns:
        Tensor: Calculated Charbonnier loss.
    """
    # breakpoint()
    h, w = pred.shape[3], pred.shape[4]
    pred = pred.view(-1, 3, h, w)
    target = target.view(-1, 3, h, w)
    DWT2 = DWTForward(J=1, wave='haar', mode='reflect').to(pred.device)
    LLp, Hcp = DWT2(pred)
    LHp, HLp, HHp = Hcp[0][:, :, 0, :, :], Hcp[0][:, :, 1, :, :], Hcp[0][:, :, 2, :, :]
    LLt, Hct = DWT2(target)
    LHt, HLt, HHt = Hct[0][:, :, 0, :, :], Hct[0][:, :, 1, :, :], Hct[0][:, :, 2, :, :]
    loss_LH = torch.sqrt((LHp - LHt)**2 + eps)
    loss_HL = torch.sqrt((HLp - HLt) ** 2 + eps)
    loss_HH = torch.sqrt((HHp - HHt) ** 2 + eps)
    return loss_LH + loss_HL + loss_HH

@masked_loss
def charbonnier_loss(pred: torch.Tensor,
            target: torch.Tensor,
            eps: float = 1e-12) -> torch.Tensor:
    """Charbonnier loss.

    Args:
        pred (Tensor): Prediction Tensor with shape (n, c, h, w).
        target ([type]): Target Tensor with shape (n, c, h, w).
        eps (float): A value used to control the curvature near zero.
            Default: 1e-12.

    Returns:
        Tensor: Calculated Charbonnier loss.
    """

    return torch.sqrt((pred - target)**2 + eps)


def tv_loss(input: torch.Tensor) -> torch.Tensor:
    """L2 total variation loss, as in Mahendran et al."""
    input = F.pad(input, (0, 1, 0, 1), 'replicate')
    x_diff = input[..., :-1, 1:] - input[..., :-1, :-1]
    y_diff = input[..., 1:, :-1] - input[..., :-1, :-1]
    return (x_diff**2 + y_diff**2).mean([1, 2, 3])


@MODELS.register_module()
class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
        sample_wise (bool): Whether calculate the loss sample-wise. This
            argument only takes effect when `reduction` is 'mean' and `weight`
            (argument of `forward()`) is not None. It will first reduce loss
            with 'mean' per-sample, and then it means over all the samples.
            Default: False.
    """

    def __init__(self,
                 loss_weight: float = 1.0,
                 reduction: str = 'mean',
                 sample_wise: bool = False) -> None:
        super().__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.sample_wise = sample_wise

    def forward(self,
                pred: torch.Tensor,
                target: torch.Tensor,
                weight: Optional[torch.Tensor] = None,
                **kwargs) -> torch.Tensor:
        """Forward Function.

        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * l1_loss(
            pred,
            target,
            weight,
            reduction=self.reduction,
            sample_wise=self.sample_wise)


@MODELS.register_module()
class MSELoss(nn.Module):
    """MSE (L2) loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
        sample_wise (bool): Whether calculate the loss sample-wise. This
            argument only takes effect when `reduction` is 'mean' and `weight`
            (argument of `forward()`) is not None. It will first reduces loss
            with 'mean' per-sample, and then it means over all the samples.
            Default: False.
    """

    def __init__(self,
                 loss_weight: float = 1.0,
                 reduction: str = 'mean',
                 sample_wise: bool = False) -> None:
        super().__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.sample_wise = sample_wise

    def forward(self,
                pred: torch.Tensor,
                target: torch.Tensor,
                weight: Optional[torch.Tensor] = None,
                **kwargs) -> torch.Tensor:
        """Forward Function.

        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * mse_loss(
            pred,
            target,
            weight,
            reduction=self.reduction,
            sample_wise=self.sample_wise)


@MODELS.register_module()
class CharbonnierLoss(nn.Module):
    """Charbonnier loss (one variant of Robust L1Loss, a differentiable variant
    of L1Loss).

    Described in "Deep Laplacian Pyramid Networks for Fast and Accurate
        Super-Resolution".

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
        sample_wise (bool): Whether calculate the loss sample-wise. This
            argument only takes effect when `reduction` is 'mean' and `weight`
            (argument of `forward()`) is not None. It will first reduces loss
            with 'mean' per-sample, and then it means over all the samples.
            Default: False.
        eps (float): A value used to control the curvature near zero.
            Default: 1e-12.
    """

    def __init__(self,
                 loss_weight: float = 1.0,
                 reduction: str = 'mean',
                 sample_wise: bool = False,
                 eps: float = 1e-12) -> None:
        super().__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.sample_wise = sample_wise
        self.eps = eps

    def forward(self,
                pred: torch.Tensor,
                target: torch.Tensor,
                weight: Optional[torch.Tensor] = None,
                **kwargs) -> torch.Tensor:
        """Forward Function.

        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * charbonnier_loss(
            pred,
            target,
            weight,
            eps=self.eps,
            reduction=self.reduction,
            sample_wise=self.sample_wise)

@MODELS.register_module()
class WCLoss(nn.Module):
    """Charbonnier loss (one variant of Robust L1Loss, a differentiable variant
    of L1Loss).

    Described in "Deep Laplacian Pyramid Networks for Fast and Accurate
        Super-Resolution".

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
        sample_wise (bool): Whether calculate the loss sample-wise. This
            argument only takes effect when `reduction` is 'mean' and `weight`
            (argument of `forward()`) is not None. It will first reduces loss
            with 'mean' per-sample, and then it means over all the samples.
            Default: False.
        eps (float): A value used to control the curvature near zero.
            Default: 1e-12.
    """

    def __init__(self,
                 loss_weight: float = 1.0,
                 reduction: str = 'mean',
                 sample_wise: bool = False,
                 eps: float = 1e-12) -> None:
        super().__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.sample_wise = sample_wise
        self.eps = eps

    def forward(self,
                pred: torch.Tensor,
                target: torch.Tensor,
                weight: Optional[torch.Tensor] = None,
                **kwargs) -> torch.Tensor:
        """Forward Function.

        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        wavloss = wav_loss(pred, target, eps=self.eps)
        rgbloss = charbonnier_loss(
            pred,
            target,
            weight,
            eps=self.eps,
            reduction=self.reduction,
            sample_wise=self.sample_wise)
        return 0.8*wavloss + 0.2*rgbloss

@MODELS.register_module()
class WCPatchLoss(nn.Module):
    """Charbonnier loss (one variant of Robust L1Loss, a differentiable variant
    of L1Loss).

    Described in "Deep Laplacian Pyramid Networks for Fast and Accurate
        Super-Resolution".

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
        sample_wise (bool): Whether calculate the loss sample-wise. This
            argument only takes effect when `reduction` is 'mean' and `weight`
            (argument of `forward()`) is not None. It will first reduces loss
            with 'mean' per-sample, and then it means over all the samples.
            Default: False.
        eps (float): A value used to control the curvature near zero.
            Default: 1e-12.
    """

    def __init__(self,
                 loss_weight: float = 1.0,
                 reduction: str = 'mean',
                 sample_wise: bool = False,
                 eps: float = 1e-12) -> None:
        super().__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.sample_wise = sample_wise
        self.eps = eps

    def forward(self,
                pred: torch.Tensor,
                target: torch.Tensor,
                weight: Optional[torch.Tensor] = None,
                **kwargs) -> torch.Tensor:
        """Forward Function.

        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        patchloss = patch_loss(pred, target, eps=self.eps)
        rgbloss = charbonnier_loss(
            pred,
            target,
            weight,
            eps=self.eps,
            reduction=self.reduction,
            sample_wise=self.sample_wise)
        return 0.5*patchloss + 0.5*rgbloss

@MODELS.register_module()
class MaskedTVLoss(L1Loss):
    """Masked TV loss.

    Args:
        loss_weight (float, optional): Loss weight. Defaults to 1.0.
    """

    def __init__(self, loss_weight: float = 1.0) -> None:
        super().__init__(loss_weight=loss_weight)

    def forward(self,
                pred: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward function.

        Args:
            pred (torch.Tensor): Tensor with shape of (n, c, h, w).
            mask (torch.Tensor, optional): Tensor with shape of (n, 1, h, w).
                Defaults to None.

        Returns:
            [type]: [description]
        """
        y_diff = super().forward(
            pred[:, :, :-1, :], pred[:, :, 1:, :], weight=mask[:, :, :-1, :])
        x_diff = super().forward(
            pred[:, :, :, :-1], pred[:, :, :, 1:], weight=mask[:, :, :, :-1])

        loss = x_diff + y_diff

        return loss


@MODELS.register_module()
class PSNRLoss(nn.Module):
    """PSNR Loss in "HINet: Half Instance Normalization Network for Image
    Restoration".

    Args:
        loss_weight (float, optional): Loss weight. Defaults to 1.0.
        reduction: reduction for PSNR. Can only be mean here.
        toY: change to calculate the PSNR of Y channel in YCbCr format
    """

    def __init__(self, loss_weight: float = 1.0, toY: bool = False) -> None:
        super(PSNRLoss, self).__init__()
        self.loss_weight = loss_weight
        import numpy as np
        self.scale = 10 / np.log(10)
        self.toY = toY
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True

    def forward(self, pred: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        assert len(pred.size()) == 4

        return self.loss_weight * self.scale * torch.log((
            (pred - target)**2).mean(dim=(1, 2, 3)) + 1e-8).mean()
