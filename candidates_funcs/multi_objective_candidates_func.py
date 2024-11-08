from typing import Optional
import torch
from botorch.utils.transforms import normalize, unnormalize
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.multi_objective.box_decompositions.non_dominated import FastNondominatedPartitioning
from botorch.acquisition.multi_objective.monte_carlo import qExpectedHypervolumeImprovement
from botorch.acquisition.multi_objective.logei import qLogExpectedHypervolumeImprovement
from botorch.models.utils.gpytorch_modules import get_matern_kernel_with_gamma_prior


def ehvi(
    train_x: torch.Tensor,
    train_obj: torch.Tensor,
    train_con: torch.Tensor,
    bounds: torch.Tensor,
    pending_x: torch.Tensor,
):
    """Expected Hiper-volume Improvement.

    カーネルはパラメータの事前分布にガンマ分布を指定したMatarnカーネルを利用

    Args:
        train_x (torch.Tensor): 観測データのパラメータ (n, x_dim)
        train_obj (torch.Tensor): 観測データの目的関数の評価値 (n, obj_dim)
        train_con (torch.Tensor): aa
        bounds (torch.Tensor): aa
        pending_x (torch.Tensor): aa

    """
    train_x = normalize(train_x, bounds=bounds)
    # 目的変数ごとにGPを学習
    Y_dim = train_obj.size(-1)
    models = []
    for i in range(Y_dim):
        y = train_obj[:, i].unsqueeze(1)
        covar_module = get_matern_kernel_with_gamma_prior(ard_num_dims=train_x.shape[-1])
        models.append(SingleTaskGP(train_x, y, outcome_transform=Standardize(m=1), covar_module=covar_module))
    model = ModelListGP(*models)
    mll = SumMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    # 獲得関数
    with torch.no_grad():
        pred = model.posterior(train_x).mean
    ref_point = torch.min(train_obj.squeeze(), 0).values - torch.abs(torch.min(train_obj.squeeze(), 0).values) * 0.1
    partitioning = FastNondominatedPartitioning(ref_point=ref_point, Y=pred)
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([128]))
    acq_func = qExpectedHypervolumeImprovement(
        model=model,
        ref_point=ref_point,
        partitioning=partitioning,
        sampler=sampler,
    )

    standard_bounds = torch.zeros_like(bounds)
    standard_bounds[1] = 1
    candidates, _ = optimize_acqf(
        acq_function=acq_func,
        bounds=standard_bounds,
        q=1,
        num_restarts=10,
        raw_samples=512,
        options={'batch_limit': 5, 'maxiter': 200},
        sequential=True,
    )
    candidates = unnormalize(candidates.detach(), bounds=bounds)

    return candidates


def log_ehvi(
    train_x: torch.Tensor,
    train_obj: torch.Tensor,
    train_con: torch.Tensor,
    bounds: torch.Tensor,
    pending_x: torch.Tensor,
):
    """Expected Hiper-volume Improvement.

    カーネルはパラメータの事前分布にガンマ分布を指定したMatarnカーネルを利用

    Args:
        train_x (torch.Tensor): 観測データのパラメータ (n, x_dim)
        train_obj (torch.Tensor): 観測データの目的関数の評価値 (n, obj_dim)
        train_con (torch.Tensor): aa
        bounds (torch.Tensor): aa
        pending_x (torch.Tensor): aa

    """
    train_x = normalize(train_x, bounds=bounds)
    # 目的変数ごとにGPを学習
    Y_dim = train_obj.size(-1)
    models = []
    for i in range(Y_dim):
        y = train_obj[:, i].unsqueeze(1)
        covar_module = get_matern_kernel_with_gamma_prior(ard_num_dims=train_x.shape[-1])
        models.append(SingleTaskGP(train_x, y, outcome_transform=Standardize(m=1), covar_module=covar_module))
    model = ModelListGP(*models)
    mll = SumMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    # 獲得関数
    with torch.no_grad():
        pred = model.posterior(train_x).mean
    ref_point = torch.min(train_obj.squeeze(), 0).values - torch.abs(torch.min(train_obj.squeeze(), 0).values) * 0.1
    partitioning = FastNondominatedPartitioning(ref_point=ref_point, Y=pred)
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([128]))
    acq_func = qLogExpectedHypervolumeImprovement(
        model=model,
        ref_point=ref_point,
        partitioning=partitioning,
        sampler=sampler,
    )

    standard_bounds = torch.zeros_like(bounds)
    standard_bounds[1] = 1
    candidates, _ = optimize_acqf(
        acq_function=acq_func,
        bounds=standard_bounds,
        q=1,
        num_restarts=10,
        raw_samples=512,
        options={'batch_limit': 5, 'maxiter': 200},
        sequential=True,
    )
    candidates = unnormalize(candidates.detach(), bounds=bounds)

    return candidates
