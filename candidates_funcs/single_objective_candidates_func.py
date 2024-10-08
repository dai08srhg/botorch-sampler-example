from typing import Optional
import torch
from botorch.fit import fit_gpytorch_mll, fit_fully_bayesian_model_nuts
from botorch.optim import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.transforms import normalize, unnormalize
from botorch.models import SingleTaskGP
from botorch.models.model import Model
from botorch.models.transforms.outcome import Standardize
from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from botorch.models.utils.gpytorch_modules import get_matern_kernel_with_gamma_prior
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.acquisition.analytic import ExpectedImprovement, LogExpectedImprovement
from botorch.acquisition.logei import qLogExpectedImprovement
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel
from gpytorch.priors.torch_priors import GammaPrior, LogNormalPrior
from botorch.acquisition.objective import PosteriorTransform
from botorch.acquisition.analytic import AnalyticAcquisitionFunction


##############
# 自作獲得関数
#############
class LCB(AnalyticAcquisitionFunction):
    """
    Lower Confidence Bound
    """

    def __init__(self, model: Model, maximize: bool = True, beta: float = 0.5, posterior_transform=None) -> None:
        super().__init__(model=model, posterior_transform=posterior_transform)
        self.beta = beta
        self.maximize = maximize

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        mean, sigma = self._mean_and_sigma(X)
        lcb = -(mean - self.beta * torch.sqrt(sigma))
        if self.maximize:
            lcb = -lcb
        return lcb


class ThompsonSampling(AnalyticAcquisitionFunction):
    """
    トンプソンサンプリング獲得関数
    """

    def __init__(
        self,
        model: Model,
        bounds: torch.Tensor,
        delta: int = 10,
        posterior_transform: Optional[PosteriorTransform] = None,
    ) -> None:
        super().__init__(model=model, posterior_transform=posterior_transform)

        n_dim = bounds.shape[-1]
        grids_for_each_dim = []
        for i in range(n_dim):
            grids_for_each_dim.append(torch.linspace(bounds[0, i], bounds[1, i], delta))
        X = torch.cartesian_prod(*grids_for_each_dim)  # 値域のグリッドを作成

        # 事後分布生成
        posterior = self.model.posterior(X=X, posterior_transform=self.posterior_transform)
        # 事後分布からリサンプリング
        Y = posterior.rsample()

        self.candidates = X[torch.argmax(Y)]  # GPからのサンプリングされた関数上での最適点を取得

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return -((X - self.candidates)**2).sum(axis=(1, 2))


##################
# candidates_func
##################
def ei(train_x: torch.Tensor, train_obj: torch.Tensor, train_con, bounds, pending_x):
    """Expected Improvementのモンテカルロ獲得関数
    v0.12.0であるため, SingleTaskGPではdimension-scaled log-normal hyperparameter priorsがデフォルト

    Args:
        train_x (torch.Tensor): 観測データのパラメータ (n, x_dim)
        train_obj (torch.Tensor): 観測データの目的関数の評価値 (n, obj_dim)
    """
    train_x = normalize(train_x, bounds=bounds)
    model = SingleTaskGP(train_x, train_obj, outcome_transform=Standardize(m=train_obj.size(-1)))

    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([128]))
    acq_func = qExpectedImprovement(model=model, best_f=train_obj.max(), sampler=sampler)

    standard_bounds = torch.zeros_like(bounds)
    standard_bounds[1] = 1

    candidates, _ = optimize_acqf(acq_function=acq_func,
                                  bounds=standard_bounds,
                                  q=1,
                                  num_restarts=10,
                                  raw_samples=512,
                                  options={
                                      "batch_limit": 5,
                                      "maxiter": 200
                                  },
                                  sequential=True)
    candidates = unnormalize(candidates.detach(), bounds=bounds)
    return candidates


def log_ei(train_x: torch.Tensor, train_obj: torch.Tensor, train_con, bounds, pending_x):
    """Log Expected Improvementのモンテカルロ獲得関数
    v0.12.0であるため, SingleTaskGPではdimension-scaled log-normal hyperparameter priorsがデフォルト

    Args:
        train_x (torch.Tensor): 観測データのパラメータ (n, x_dim)
        train_obj (torch.Tensor): 観測データの目的関数の評価値 (n, obj_dim)
    """
    train_x = normalize(train_x, bounds=bounds)
    model = SingleTaskGP(train_x, train_obj, outcome_transform=Standardize(m=train_obj.size(-1)))

    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([128]))
    acq_func = qLogExpectedImprovement(model=model, best_f=train_obj.max(), sampler=sampler)

    standard_bounds = torch.zeros_like(bounds)
    standard_bounds[1] = 1

    candidates, _ = optimize_acqf(acq_function=acq_func,
                                  bounds=standard_bounds,
                                  q=1,
                                  num_restarts=10,
                                  raw_samples=512,
                                  options={
                                      "batch_limit": 5,
                                      "maxiter": 200
                                  },
                                  sequential=True)
    candidates = unnormalize(candidates.detach(), bounds=bounds)

    return candidates


def ei_gammma_prior(train_x: torch.Tensor, train_obj: torch.Tensor, train_con, bounds, pending_x):
    """Expected Improvementのモンテカルロ獲得関数
    MaternKernelのlengthscaleにGamma分布を用いたEI
    """
    train_x = normalize(train_x, bounds=bounds)
    covar_module = get_matern_kernel_with_gamma_prior(ard_num_dims=train_x.shape[-1])
    model = SingleTaskGP(train_x,
                         train_obj,
                         outcome_transform=Standardize(m=train_obj.size(-1)),
                         covar_module=covar_module)

    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([128]))
    acq_func = qExpectedImprovement(model=model, best_f=train_obj.max(), sampler=sampler)

    standard_bounds = torch.zeros_like(bounds)
    standard_bounds[1] = 1

    candidates, _ = optimize_acqf(acq_function=acq_func,
                                  bounds=standard_bounds,
                                  q=1,
                                  num_restarts=10,
                                  raw_samples=512,
                                  options={
                                      "batch_limit": 5,
                                      "maxiter": 200
                                  },
                                  sequential=True)
    candidates = unnormalize(candidates.detach(), bounds=bounds)
    return candidates


def log_ei_gammma_prior(train_x: torch.Tensor, train_obj: torch.Tensor, train_con, bounds, pending_x):
    """Expected Improvementのモンテカルロ獲得関数
    MaternKernelのlengthscaleにGamma分布を用いたLogEI
    """
    train_x = normalize(train_x, bounds=bounds)
    covar_module = get_matern_kernel_with_gamma_prior(ard_num_dims=train_x.shape[-1])
    model = SingleTaskGP(train_x,
                         train_obj,
                         outcome_transform=Standardize(m=train_obj.size(-1)),
                         covar_module=covar_module)

    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([128]))
    acq_func = qLogExpectedImprovement(model=model, best_f=train_obj.max(), sampler=sampler)

    standard_bounds = torch.zeros_like(bounds)
    standard_bounds[1] = 1

    candidates, _ = optimize_acqf(acq_function=acq_func,
                                  bounds=standard_bounds,
                                  q=1,
                                  num_restarts=10,
                                  raw_samples=512,
                                  options={
                                      "batch_limit": 5,
                                      "maxiter": 200
                                  },
                                  sequential=True)
    candidates = unnormalize(candidates.detach(), bounds=bounds)
    return candidates


def lcb(train_x: torch.Tensor, train_obj: torch.Tensor, train_con, bounds, pending_x):
    """Lower Confidence Bound (LCB)"""
    train_x = normalize(train_x, bounds=bounds)
    model = SingleTaskGP(train_x, train_obj, outcome_transform=Standardize(m=train_obj.size(-1)))

    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    # beta = torch.log(torch.Tensor([train_x.size()[0]]))[0]  # LCBのハイパラ
    acq_func = LCB(model=model, maximize=True)  # 獲得関数に自作のLCBを利用

    standard_bounds = torch.zeros_like(bounds)
    standard_bounds[1] = 1

    candidates, _ = optimize_acqf(acq_function=acq_func,
                                  bounds=standard_bounds,
                                  q=1,
                                  num_restarts=10,
                                  raw_samples=512,
                                  options={
                                      "batch_limit": 5,
                                      "maxiter": 200
                                  },
                                  sequential=True)
    candidates = unnormalize(candidates.detach(), bounds=bounds)
    return candidates


def saas_ei(train_x: torch.Tensor, train_obj: torch.Tensor, train_con, bounds, pending_x):
    """SAAS + EI
    """
    train_x = normalize(train_x, bounds=bounds)
    model = SaasFullyBayesianSingleTaskGP(train_x,
                                          train_obj,
                                          train_Yvar=torch.full_like(train_obj, 1e-6),
                                          outcome_transform=Standardize(m=train_obj.size(-1)))
    fit_fully_bayesian_model_nuts(model, warmup_steps=256, num_samples=128, thinning=16, disable_progbar=True)

    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([128]))
    acq_func = qExpectedImprovement(model=model, best_f=train_obj.max(), sampler=sampler)

    standard_bounds = torch.zeros_like(bounds)
    standard_bounds[1] = 1

    candidates, _ = optimize_acqf(acq_function=acq_func,
                                  bounds=standard_bounds,
                                  q=1,
                                  num_restarts=10,
                                  raw_samples=512,
                                  options={
                                      "batch_limit": 5,
                                      "maxiter": 200
                                  },
                                  sequential=True)
    candidates = unnormalize(candidates.detach(), bounds=bounds)
    return candidates


def thompson_sampling(train_x: torch.Tensor, train_obj: torch.Tensor, train_con, bounds, pending_x):
    """トンプソンサンプリング
    """
    train_x = normalize(train_x, bounds=bounds)
    model = SingleTaskGP(train_x, train_obj, outcome_transform=Standardize(m=train_obj.size(-1)))

    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    acq_func = ThompsonSampling(model=model, bounds=bounds)

    standard_bounds = torch.zeros_like(bounds)
    standard_bounds[1] = 1

    candidates, _ = optimize_acqf(acq_function=acq_func,
                                  bounds=standard_bounds,
                                  q=1,
                                  num_restarts=10,
                                  raw_samples=512,
                                  options={
                                      "batch_limit": 5,
                                      "maxiter": 200
                                  },
                                  sequential=True)
    candidates = unnormalize(candidates.detach(), bounds=bounds)
    return candidates
