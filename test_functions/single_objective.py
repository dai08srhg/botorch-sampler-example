import random
import numpy as np
from optuna.distributions import FloatDistribution, CategoricalDistribution


class InputError(Exception):
    """."""


class Hartmann6:
    """Hartmann 6-Dimensional function. Based on the following MATLAB code.

    https://www.sfu.ca/~ssurjano/hart6.html
    """

    def __init__(self, sd=0):
        self.sd = sd
        self.distributions = {f'x{d}': FloatDistribution(0.0, 1.0) for d in range(6)}
        self.min_f = -3.32237

    def f(self, xx: np.ndarray) -> np.ndarray:
        """f.

        Args:
            xx (np.ndarray): 入力. xx.shape=(1, 6)

        Returns:
            np.ndarray: 出力. shape=(1, 1)

        """
        if xx.shape != (1, 6):
            raise InputError('入力次元エラー. shape=(1, 6) is required')

        n = xx.shape[0]
        y = np.zeros(n)
        for i in range(n):
            alpha = np.array([1.0, 1.2, 3.0, 3.2])
            A = np.array(
                [
                    [10, 3, 17, 3.5, 1.7, 8],
                    [0.05, 10, 17, 0.1, 8, 14],
                    [3, 3.5, 1.7, 10, 17, 8],
                    [17, 8, 0.05, 10, 0.1, 14],
                ]
            )
            P = 1e-4 * np.array(
                [
                    [1312, 1696, 5569, 124, 8283, 5886],
                    [2329, 4135, 8307, 3736, 1004, 9991],
                    [2348, 1451, 3522, 2883, 3047, 6650],
                    [4047, 8828, 8732, 5743, 1091, 381],
                ]
            )

            outer = 0
            for ii in range(4):
                inner = 0
                for jj in range(6):
                    xj = xx[i, jj]
                    Aij = A[ii, jj]
                    Pij = P[ii, jj]
                    inner = inner + Aij * (xj - Pij) ** 2
                new = alpha[ii] * np.exp(-inner)
                outer = outer + new
            y[i] = -outer

        if self.sd == 0:
            noise = np.zeros(n)
        else:
            noise = np.random.normal(0, self.sd, n)

        return (y + noise).reshape(1, 1)

    def random_x(self) -> np.ndarray:
        """入力空間の点をランダムに1点取得."""
        x = np.random.uniform(low=0.0, high=1.0, size=6)
        x = x.reshape(1, x.shape[0])
        return x


class Hartmann6Cat2:
    """Hartmann 6-Dimensional functionをベースとし, 1,4次元目をカテゴリカル変数に変換.

    ※ 離散化して順序をシャッフル (順序関係の意味をなくす)
    """

    def __init__(self, sd=0):
        self.sd = sd
        self.distributions = {f'x{d}': FloatDistribution(0.0, 1.0) for d in range(6)}
        self.distributions['x0'] = CategoricalDistribution([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        self.distributions['x3'] = CategoricalDistribution([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])

        self.x0_map = {1: 0.6, 2: 0.5, 3: 0.2, 4: 0.0, 5: 0.4, 6: 0.3, 7: 0.9, 8: 0.8, 9: 0.1, 10: 0.7}
        self.x3_map = {
            1: 0.66,
            2: 0.73,
            3: 0.26,
            4: 1.0,
            5: 0.8,
            6: 0.6,
            7: 0.53,
            8: 0.86,
            9: 0.13,
            10: 0.33,
            11: 0.06,
            12: 0.93,
            13: 0.4,
            14: 0.2,
            15: 0.46,
        }

    def f(self, xx: np.ndarray) -> np.ndarray:
        """."""
        xx = np.copy(xx)
        if xx.shape != (1, 6):
            raise InputError('入力次元エラー. shape=(1, 6) is required')

        # カテゴリ変数の次元を置換
        xx[0][0] = self.x0_map[xx[0][0]]
        xx[0][3] = self.x3_map[xx[0][3]]

        n = xx.shape[0]
        y = np.zeros(n)
        for i in range(n):
            alpha = np.array([1.0, 1.2, 3.0, 3.2])
            A = np.array(
                [
                    [10, 3, 17, 3.5, 1.7, 8],
                    [0.05, 10, 17, 0.1, 8, 14],
                    [3, 3.5, 1.7, 10, 17, 8],
                    [17, 8, 0.05, 10, 0.1, 14],
                ]
            )
            P = 1e-4 * np.array(
                [
                    [1312, 1696, 5569, 124, 8283, 5886],
                    [2329, 4135, 8307, 3736, 1004, 9991],
                    [2348, 1451, 3522, 2883, 3047, 6650],
                    [4047, 8828, 8732, 5743, 1091, 381],
                ]
            )
            outer = 0
            for ii in range(4):
                inner = 0
                for jj in range(6):
                    xj = xx[i, jj]
                    Aij = A[ii, jj]
                    Pij = P[ii, jj]
                    inner = inner + Aij * (xj - Pij) ** 2
                new = alpha[ii] * np.exp(-inner)
                outer = outer + new
            y[i] = -outer
        if self.sd == 0:
            noise = np.zeros(n)
        else:
            noise = np.random.normal(0, self.sd, n)

        return (y + noise).reshape(1, 1)

    def random_x(self):
        """."""
        x = np.random.uniform(low=0.0, high=1.0, size=6)
        x[0] = random.choice(list(self.x0_map.keys()))
        x[3] = random.choice(list(self.x3_map.keys()))
        return x.reshape(1, x.shape[0])


class StyblinskiTang:
    """https://www.sfu.ca/~ssurjano/stybtang.html ."""

    def __init__(self, dim=2, sd=0):
        self.dim = dim
        self.distributions = {f'x{d}': FloatDistribution(-5.0, 5.0) for d in range(dim)}
        self.min_f = -39.16599 * dim

    def f(self, xx: np.ndarray):
        """f.

        Args:
            xx (np.ndarray): 入力. xx.shape=(1, x_dim)

        Returns:
            np.ndarray: 出力. shape=(1, 1)

        """
        if xx.shape != (1, self.dim):
            raise InputError(f'入力次元エラー. shape=(1, {self.dim}) is required')
        xx_ = np.squeeze(xx)

        a = (1 / 2) * sum([x**4 - 16 * x**2 + 5 * x for x in xx_])
        return np.array([a]).reshape(1, 1)

    def random_x(self):
        """."""
        x = np.random.uniform(low=-5.0, high=5.0, size=self.dim)
        return x.reshape(1, x.shape[0])


class FiveWellPotentioal:

    def __init__(self, sd=0):
        """初期化."""
        self.distributions = {f'x{d}': FloatDistribution(-20.0, 20.0) for d in range(2)}
        self.min_f = -1.4616

    def f(self, xx: np.ndarray):
        """f.

        Args:
            xx (np.ndarray): 入力. xx.shape=(1, 2)

        Returns:
            np.ndarray: 出力. shape=(1,1)

        """
        if xx.shape != (1, 2):
            raise InputError('入力次元エラー. shape=(1, 2) is required')
        xx_ = np.squeeze(xx)
        f = (1 + 0.0001 * (xx_[0] ** 2 + xx_[1] ** 2) ** (1.2)) * (
            1
            - (1 / (1 + 0.05 * (xx_[0] ** 2 + (xx_[1] - 10) ** 2)))
            - (1 / (1 + 0.05 * ((xx_[0] - 10) ** 2 + xx_[1] ** 2)))
            - (1.5 / (1 + 0.03 * ((xx_[0] + 10) ** 2 + xx_[1] ** 2)))
            - (2 / (1 + 0.05 * ((xx_[0] - 5) ** 2 + (xx_[1] + 10) ** 2)))
            - (1 / (1 + 0.1 * ((xx_[0] + 5) ** 2 + (xx_[1] + 10) ** 2)))
        )
        return np.array(f).reshape(1, 1)

    def random_x(self):
        """ """
        x = np.random.uniform(low=-20.0, high=20.0, size=2)
        return x.reshape(1, x.shape[0])


class Ackley:
    """ """
