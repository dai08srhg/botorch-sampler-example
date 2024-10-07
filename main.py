import os
import numpy as np
import optuna
from enum import Enum
from tqdm import tqdm
import polars as pl
from test_functions.single_objective import Hartmann6, StyblinskiTang, FiveWellPotentioal, Hartmann6Cat2
from candidates_funcs.single_objective_candidates_func import (ei, log_ei, lcb, saas_ei, ei_gammma_prior,
                                                               log_ei_gammma_prior)

TargetFunction = Hartmann6 | StyblinskiTang | FiveWellPotentioal


class SamplerName(str, Enum):
    """
    最適化バージョン
    """
    TPE = 'TPE'
    EIGammaPrior = 'EI with GammaPrior'
    LogEIGammaPrior = 'LogEI with GammaPrior'
    EI = 'EI'
    LogEI = 'LogEI'
    LCB = 'LCB'
    SaasEI = 'SAAS+EI'


class Optimizer:

    def __init__(self, sampler_name: SamplerName):
        if sampler_name == SamplerName.TPE:
            self.sampler = optuna.samplers.TPESampler()
        elif sampler_name == SamplerName.LCB:
            self.sampler = optuna.integration.BoTorchSampler(candidates_func=lcb)
        elif sampler_name == SamplerName.EIGammaPrior:
            self.sampler = optuna.integration.BoTorchSampler(candidates_func=ei_gammma_prior)
        elif sampler_name == SamplerName.LogEIGammaPrior:
            self.sampler = optuna.integration.BoTorchSampler(candidates_func=log_ei_gammma_prior)
        elif sampler_name == SamplerName.EI:
            self.sampler = optuna.integration.BoTorchSampler(candidates_func=ei)
        elif sampler_name == SamplerName.LogEI:
            self.sampler = optuna.integration.BoTorchSampler(candidates_func=log_ei)
        elif sampler_name == SamplerName.SaasEI:
            self.sampler = optuna.integration.BoTorchSampler(candidates_func=saas_ei)
        else:
            pass

    def _set_samples(self, Xs: np.ndarray, ys: np.ndarray, distributions: dict):
        """studyに観測データを登録
        ※ Tell_and_Askのインターフェースを利用

        Args:
            Xs (np.ndarray): shape=(n, x_dim)
            ys (np.ndarray): shape=(n, y_dim)
            distributions (Dict[str, optuna.distributions]): 探索空間
        """
        features = list(distributions.keys())
        for X, y in zip(Xs, ys):
            params = {}
            for feature, x in zip(features, X):
                params[feature] = x
            trial = optuna.trial.create_trial(params=params, distributions=distributions, value=y[0])
            self.study.add_trial(trial)

    def create_study(self):
        """
        """
        self.study = optuna.create_study(direction='minimize', sampler=self.sampler)

    def get_candidate(self, Xs: np.ndarray, ys: np.ndarray, distributions: dict):
        """
        Args:
            Xs (np.ndarray): shape=(n, x_dim)
            ys (np.ndarray): shape=(n, y_dim)
            distributions (Dict[str, optuna.distributions]): 探索空間
        """
        self._set_samples(Xs, ys, distributions)
        trial = self.study.ask()
        new_X = []
        for feature, dist in distributions.items():
            if type(dist) is optuna.distributions.FloatDistribution:
                new_X.append(trial.suggest_float(feature, dist.low, dist.high))
            elif type(dist) is optuna.distributions.CategoricalDistribution:
                new_X.append(trial.suggest_categorical(feature, dist.choices))
        new_X = np.array(new_X)
        return new_X.reshape(1, new_X.shape[0])


def run_optimization(func: TargetFunction,
                     X_init: np.ndarray,
                     y_init: np.ndarray,
                     sampler_name: SamplerName,
                     iters: int = 100):
    """
    探索実行
    """
    sampler = Optimizer(sampler_name)
    Xs = X_init.copy()
    ys = y_init.copy()

    distributions = func.distributions
    for i in tqdm(range(iters)):
        sampler.create_study()
        new_X = sampler.get_candidate(Xs, ys, distributions)
        new_y = func.f(new_X)
        Xs = np.concatenate([Xs, new_X])
        ys = np.concatenate([ys, new_y])
    return ys


if __name__ == '__main__':
    optuna.logging.disable_default_handler()

    # 関数定義
    # exp_name = 'StyblinskiTang8'
    # f = StyblinskiTang(dim=8)
    # exp_name = 'StyblinskiTang40'
    # f = StyblinskiTang(dim=40)
    # exp_name = 'Hartmann6'
    # f = Hartmann6()
    # exp_name = 'FiveWellPotentioal'
    # f = FiveWellPotentioal()
    exp_name = 'Hartmann6Cat2'
    f = Hartmann6Cat2()

    os.makedirs(f'exp_result/{exp_name}', exist_ok=True)
    print(f'Run experiment: {exp_name}')

    EXP_NUM = 10  # 実験回数
    SERCH_NUM = 100  # 観測回数
    INIT_NUM = 10  # 初期点の数
    use_methods = [SamplerName.TPE, SamplerName.LogEIGammaPrior, SamplerName.LogEI]

    for j in range(1, EXP_NUM + 1):
        print(f'Start trial:{j}')
        serch_fs = {}

        # 初期点ランダムに10点
        X_init = f.random_x()
        y_init = f.f(X_init)
        for i in range(INIT_NUM - 1):
            X = f.random_x()
            y = f.f(X)
            X_init = np.concatenate([X_init, X])
            y_init = np.concatenate([y_init, y])

        # ランダム探索
        ys_random = y_init.copy()
        for i in range(SERCH_NUM):
            new_y = f.f(f.random_x())
            ys_random = np.concatenate([ys_random, new_y])
        serch_fs['Random'] = ys_random.squeeze()

        # 各手法で探索
        for method in use_methods:
            print(f'Start optimization using {method.value}')
            ys = run_optimization(f, X_init, y_init, method, SERCH_NUM)
            serch_fs[method.value] = ys.squeeze()

        # 探索結果を格納
        df = pl.DataFrame(serch_fs)
        df.write_csv(f'exp_result/{exp_name}/run_{j}.csv')
