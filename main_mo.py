# 多目的ベイズ最適化の実行
import optuna
import os
import numpy as np
import polars as pl
from enum import Enum
from tqdm import tqdm
from test_functions.multi_objective import BraninCurrin, Hartmann6Obj2
from candidates_funcs.multi_objective_candidates_func import ehvi, log_ehvi


TargetFunction = BraninCurrin


class SamplerName(str, Enum):
    """最適化バージョン."""

    MOTPE = 'MOTPE'
    EHVI = 'EHVI'
    LogEHVI = 'LogEHVI'


class Optimizer:
    """最適化クラス."""

    def __init__(self, sampler_name: SamplerName):
        if sampler_name == SamplerName.MOTPE:
            self.sampler = optuna.samplers.TPESampler()
        elif sampler_name == SamplerName.EHVI:
            self.sampler = optuna.integration.BoTorchSampler(candidates_func=ehvi)
        elif sampler_name == SamplerName.LogEHVI:
            self.sampler = optuna.integration.BoTorchSampler(candidates_func=log_ehvi)
        else:
            pass

    def _set_samples(self, Xs: np.ndarray, ys: np.ndarray, distributions: dict):
        """studyに観測データを登録.

        ※ Tell_and_Askのインターフェースを利用.

        Args:
            Xs (np.ndarray): shape=(n, x_dim).
            ys (np.ndarray): shape=(n, y_dim).
            distributions (Dict[str, optuna.distributions]): 探索空間

        """
        features = list(distributions.keys())
        for X, y in zip(Xs, ys):
            params = {}
            for feature, x in zip(features, X):
                params[feature] = x
            trial = optuna.trial.create_trial(params=params, distributions=distributions, values=[y[0], y[1]])
            self.study.add_trial(trial)

    def create_study(self, directions):
        """.

        Args:
            directions (list[str]): 探索方向

        """
        self.study = optuna.create_study(directions=directions, sampler=self.sampler)

    def get_candidate(self, Xs: np.ndarray, ys: np.ndarray, distributions: dict):
        """候補点を取得.

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


def run_optimization(
    func: TargetFunction,
    X_init: np.ndarray,
    y_init: np.ndarray,
    sampler_name: SamplerName,
    directions: list[str],
    iters: int = 100,
):
    """探索を実行."""
    sampler = Optimizer(sampler_name)
    Xs = X_init.copy()
    ys = y_init.copy()

    distributions = func.distributions
    for _ in tqdm(range(iters)):
        sampler.create_study(directions)
        new_X = sampler.get_candidate(Xs, ys, distributions)
        new_y = func.f(new_X)
        Xs = np.concatenate([Xs, new_X])
        ys = np.concatenate([ys, new_y])
    return ys


if __name__ == '__main__':
    optuna.logging.disable_default_handler()

    ## 実験定義
    exp_name = 'BraninCurrin'
    os.makedirs(f'exp_result/{exp_name}', exist_ok=True)
    print(f'Run experiment: {exp_name}')
    use_methods = [SamplerName.MOTPE, SamplerName.EHVI, SamplerName.LogEHVI]

    EXP_NUM = 1  # 実験回数
    SERCH_NUM = 30  # 観測回数
    INIT_NUM = 10  # 初期点の数

    # タスク設定
    if exp_name == 'BraninCurrin':
        target_f = BraninCurrin()
    elif exp_name == 'Hartmann6Obj2':
        target_f = Hartmann6Obj2()
    directions = target_f.task

    for j in range(1, EXP_NUM + 1):
        print(f'Start trial:{j}')
        serch_fs = {}

        # 初期点ランダムに10点
        X_init = target_f.random_x()
        y_init = target_f.f(X_init)
        for _ in range(INIT_NUM - 1):
            X = target_f.random_x()
            y = target_f.f(X)
            X_init = np.concatenate([X_init, X])
            y_init = np.concatenate([y_init, y])

        # ランダム探索
        os.makedirs(f'exp_result/{exp_name}/random', exist_ok=True)
        ys_random = y_init.copy()
        for _ in range(SERCH_NUM):
            new_y = target_f.f(target_f.random_x())
            ys_random = np.concatenate([ys_random, new_y])
        # 結果保存
        records = {}
        for i in range(ys_random.shape[-1]):
            records[f'y{i}'] = ys_random[:, i]
        df = pl.DataFrame(records)
        df.write_csv(f'exp_result/{exp_name}/random/run_{j}.csv')

        # 各手法で探索
        for method in use_methods:
            print(f'Start optimization using {method.value}')
            os.makedirs(f'exp_result/{exp_name}/{method.value}', exist_ok=True)
            ys = run_optimization(target_f, X_init, y_init, method, directions, SERCH_NUM)
            # 結果の保存
            records = {}
            for i in range(ys_random.shape[-1]):
                records[f'y{i}'] = ys[:, i]
            df = pl.DataFrame(records)
            df.write_csv(f'exp_result/{exp_name}/{method.value}/run_{j}.csv')
