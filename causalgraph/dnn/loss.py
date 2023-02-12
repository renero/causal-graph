#
# Implementations of loss functions for DNNs.
# Reject using the one coming from CDT, because it is not working properly.
# Reject using the one at https://github.com/ZongxianLee/MMD_Loss.Pytorch
# Accepted using https://github.com/KevinMusgrave/pytorch-adapt
#

import numpy as np
import torch

from pytorch_metric_learning.distances import LpDistance
from pytorch_metric_learning.utils import common_functions as pml_cf
from typing import List, Union


def default(x, default_x, kwargs=None, condition=None):
    if condition is None:
        condition = lambda x: x is None
    if kwargs is not None:
        if isinstance(kwargs, dict):
            return default_x(**kwargs) if condition(x) else x
        if isinstance(kwargs, list):
            return default_x(*kwargs) if condition(x) else x
    return default_x if condition(x) else x


def split_half(x, dim):
    d = x.shape[dim] // 2
    return torch.split(x, d, dim=dim)


def num_elements_minus_diag(x):
    n = x.shape[0]
    return n * (n - 1)


def get_kernel_scales(low=-8, high=8, num_kernels=33, base=2.0):
    return torch.from_numpy(np.logspace(low, high, num=num_kernels, base=base))


def _mmd_dist_mats(x, y, dist_func):
    xx = dist_func(x, x)
    yy = dist_func(y, y)
    zz = dist_func(x, y)

    with torch.no_grad():
        # https://arxiv.org/pdf/1409.6041.pdf
        # https://arxiv.org/pdf/1707.07269.pdf
        scale = -1.0 / torch.median(xx)

    return xx, yy, zz, scale


def get_mmd_dist_mats(x, y, dist_func):
    if isinstance(x, (list, tuple)):
        xx, yy, zz, scale = [], [], [], []
        for i in range(len(x)):
            _xx, _yy, _zz, _scale = _mmd_dist_mats(x[i], y[i], dist_func)
            xx.append(_xx)
            yy.append(_yy)
            zz.append(_zz)
            scale.append(_scale)
        return xx, yy, zz, scale
    else:
        return _mmd_dist_mats(x, y, dist_func)


def get_default_kernel_weights(scale):
    if torch.is_tensor(scale) and torch.numel(scale) > 1:
        return torch.ones_like(scale) / len(scale)
    else:
        return 1


def _mmd_quadratic(x, scale, weights):
    return torch.sum(torch.exp(x.unsqueeze(2) * scale) * weights, dim=2)


def get_mmd_quadratic(xx, yy, zz, scale, weights=None):
    # https://jmlr.csail.mit.edu/papers/volume13/gretton12a/gretton12a.pdf
    # https://arxiv.org/pdf/1502.02791.pdf
    is_joint_mmd = isinstance(xx, (list, tuple))

    if is_joint_mmd:
        xx_prod, yy_prod, zz_prod = 1, 1, 1
        for i in range(len(xx)):
            curr_weights = default(weights, get_default_kernel_weights(scale[i]))
            xx_prod *= _mmd_quadratic(xx[i], scale[i], curr_weights)
            yy_prod *= _mmd_quadratic(yy[i], scale[i], curr_weights)
            zz_prod *= _mmd_quadratic(zz[i], scale[i], curr_weights)
        xx_prod.fill_diagonal_(0)
        yy_prod.fill_diagonal_(0)
        xx, yy, zz = xx_prod, yy_prod, zz_prod
    else:
        weights = default(weights, get_default_kernel_weights(scale))
        xx = _mmd_quadratic(xx, scale, weights).fill_diagonal_(0)
        yy = _mmd_quadratic(yy, scale, weights).fill_diagonal_(0)
        zz = _mmd_quadratic(zz, scale, weights)

    xx_scaler = 1.0 / num_elements_minus_diag(xx)
    yy_scaler = 1.0 / num_elements_minus_diag(yy)
    return xx_scaler * torch.sum(xx) + yy_scaler * torch.sum(yy) - 2 * torch.mean(zz)


def _mmd_linear(x, i, j, scale, weights):
    return torch.sum(torch.exp(x[i, j] * scale) * weights, dim=0)


def _mmd_linear_helper(xx, yy, zz, scale, weights):
    B = xx.shape[0]
    idx_range = torch.arange(0, B // 2, device=xx.device)
    s1 = idx_range * 2
    s2 = s1 + 1

    if scale.ndim == 0:
        scale = scale.unsqueeze(0)
    scale = scale.unsqueeze(1)
    weights = default(weights, get_default_kernel_weights(scale))

    loss1 = _mmd_linear(xx, s1, s2, scale, weights)
    loss2 = _mmd_linear(yy, s1, s2, scale, weights)
    loss3 = _mmd_linear(zz, s1, s2, scale, weights)
    loss4 = _mmd_linear(zz, s2, s1, scale, weights)

    return loss1, loss2, loss3, loss4


def get_mmd_linear(xx, yy, zz, scale, weights=None):
    # https://jmlr.csail.mit.edu/papers/volume13/gretton12a/gretton12a.pdf
    # https://arxiv.org/pdf/1502.02791.pdf
    is_joint_mmd = isinstance(xx, (list, tuple))
    B = xx[0].shape[0] if is_joint_mmd else xx.shape[0]

    if is_joint_mmd:
        product_list = [1, 1, 1, 1]
        for i in range(len(xx)):
            curr_kernels = _mmd_linear_helper(xx[i], yy[i], zz[i], scale[i], weights)
            product_list = [a * b for a, b in zip(product_list, curr_kernels)]
        loss1, loss2, loss3, loss4 = [torch.sum(a) for a in product_list]
    else:
        loss1, loss2, loss3, loss4 = [
            torch.sum(a) for a in _mmd_linear_helper(xx, yy, zz, scale, weights)
        ]

    loss = loss1 + loss2 - loss3 - loss4
    return torch.sum(loss) / float(B // 2)


class MMDLoss(torch.nn.Module):
    """
    Implementation of

    - [Learning Transferable Features with Deep Adaptation Networks](https://arxiv.org/abs/1502.02791)
    - [Deep Transfer Learning with Joint Adaptation Networks](https://arxiv.org/abs/1605.06636).
    """

    def __init__(
        self, kernel_scales: Union[float, torch.Tensor] = 1, mmd_type: str = "linear"
    ):
        """
        Arguments:
            kernel_scales: The kernel bandwidth is scaled by this amount.
                If a tensor, then multiple kernel bandwidths are used.
            mmd_type: 'linear' or 'quadratic'. 'linear' uses the linear estimate of MK-MMD.
        """
        super().__init__()
        self.kernel_scales = kernel_scales
        self.dist_func = LpDistance(normalize_embeddings=False, p=2, power=2)
        self.mmd_type = mmd_type
        if mmd_type == "linear":
            self.mmd_func = get_mmd_linear
        elif mmd_type == "quadratic":
            self.mmd_func = get_mmd_quadratic
        else:
            raise ValueError("mmd_type must be either linear or quadratic")

    # input can be embeddings or list of embeddings
    def forward(
        self,
        x: Union[torch.Tensor, List[torch.Tensor]],
        y: Union[torch.Tensor, List[torch.Tensor]],
    ) -> torch.Tensor:
        """
        Arguments:
            x: features or a list of features from one domain.
            y: features or a list of features from the other domain.
        Returns:
            MMD if the inputs are tensors, and Joint MMD (JMMD) if the inputs are lists of tensors.
        """
        xx, yy, zz, scale = get_mmd_dist_mats(x, y, self.dist_func)
        if torch.is_tensor(self.kernel_scales):
            s = scale[0] if isinstance(scale, (list, tuple)) else scale
            self.kernel_scales = pml_cf.to_device(self.kernel_scales, s, dtype=s.dtype)

        if isinstance(scale, (list, tuple)):
            for i in range(len(scale)):
                scale[i] = scale[i] * self.kernel_scales
        else:
            scale = scale * self.kernel_scales

        return self.mmd_func(xx, yy, zz, scale)
