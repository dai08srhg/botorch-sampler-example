import os
import numpy as np
import optuna
from enum import Enum
from tqdm import tqdm
import polars as pl
from test_functions.single_objective import Hartmann6, StyblinskiTang, FiveWellPotentioal
from candidates_funcs.single_objective_candidates_func import ei, log_ei, lcb, saas_ei


class SamplerName(str, Enum):
    """
    最適化バージョン
    """
    TPE = 'tpe'
    LCB = 'lcb'
    EI = 'ei'
    LogEI = 'log_ei'
    SaasEI = 'SAAS+EI'


class Optimizer:

    def __init__(self, sampler_name: SamplerName):
        if sampler_name == SamplerName.TPE:
            self.sampler = optuna.samplers.TPESampler()
        elif sampler_name == SamplerName.LCB:
            self.sampler = optuna.integration.BoTorchSampler(candidates_func=lcb)
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


if __name__ == '__main__':
    optuna.logging.disable_default_handler()

    # 関数定義
    exp_name = 'StyblinskiTang40'
    f = StyblinskiTang(dim=40)
    distributions = f.distributions

    os.makedirs(f'exp_result/{exp_name}', exist_ok=True)

    for j in range(1, 11):
        print(f'Run experiment {j}')
        # 初期点ランダムに10点
        X_init = f.random_x()
        y_init = f.f(X_init)
        for i in range(9):
            X = f.random_x()
            y = f.f(X)
            X_init = np.concatenate([X_init, X])
            y_init = np.concatenate([y_init, y])

        # サンプラー定義
        ys_random = y_init.copy()
        tpe_sampler = Optimizer(SamplerName.TPE)
        Xs_tpe = X_init.copy()
        ys_tpe = y_init.copy()

        ei_sampler = Optimizer(SamplerName.EI)
        Xs_ei = X_init.copy()
        ys_ei = y_init.copy()

        logei_sampler = Optimizer(SamplerName.LogEI)
        Xs_logei = X_init.copy()
        ys_logei = y_init.copy()

        saas_sampler = Optimizer(SamplerName.SaasEI)
        Xs_saas = X_init.copy()
        ys_saas = y_init.copy()
        '''
        lcb_sampler = Optimizer(SamplerName.LCB)
        Xs_lcb = X_init.copy()
        ys_lcb = y_init.copy()
        '''

        # 探索
        ITER = 100
        for i in tqdm(range(ITER)):
            # ランダム探索
            new_y = f.f(f.random_x())
            ys_random = np.concatenate([ys_random, new_y])

            # TPEでの探索
            tpe_sampler.create_study()
            new_X = tpe_sampler.get_candidate(Xs_tpe, ys_tpe, distributions)
            new_y = f.f(new_X)
            Xs_tpe = np.concatenate([Xs_tpe, new_X])
            ys_tpe = np.concatenate([ys_tpe, new_y])

            # EIでの探索
            ei_sampler.create_study()
            new_X = ei_sampler.get_candidate(Xs_ei, ys_ei, distributions)
            new_y = f.f(new_X)
            Xs_ei = np.concatenate([Xs_ei, new_X])
            ys_ei = np.concatenate([ys_ei, new_y])

            # LogEIでの探索
            logei_sampler.create_study()
            new_X = logei_sampler.get_candidate(Xs_logei, ys_logei, distributions)
            new_y = f.f(new_X)
            Xs_logei = np.concatenate([Xs_logei, new_X])
            ys_logei = np.concatenate([ys_logei, new_y])

            # SAAS + EI
            saas_sampler.create_study()
            new_X = saas_sampler.get_candidate(Xs_saas, ys_saas, distributions)
            new_y = f.f(new_X)
            Xs_saas = np.concatenate([Xs_saas, new_X])
            ys_saas = np.concatenate([ys_saas, new_y])
            '''
            # LCBで探索
            lcb_sampler.create_study()
            new_X = lcb_sampler.get_candidate(Xs_lcb, ys_lcb, distributions)
            new_y = f.f(new_X)
            Xs_lcb = np.concatenate([Xs_lcb, new_X])
            ys_lcb = np.concatenate([ys_lcb, new_y])
            '''

        # 探索結果を格納
        df = pl.DataFrame({
            'Random': ys_random.squeeze(),
            'TPE': ys_tpe.squeeze(),
            'EI': ys_ei.squeeze(),
            'LogEI': ys_logei.squeeze(),
            'SAAS+EI': ys_saas.squeeze(),
            # 'LCB': ys_lcb.squeeze()
        })
        df.write_csv(f'exp_result/{exp_name}/run_{j}.csv')
