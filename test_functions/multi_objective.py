import random
import numpy as np
import torch
from botorch.test_functions.multi_objective import BraninCurrin as br
from optuna.distributions import FloatDistribution
from test_functions.single_objective import Hartmann6


class InputError(Exception):
    """."""


class BraninCurrin:
    """BraninCurri Function.

    目的変数2つ
     - Branin Function: https://www.sfu.ca/~ssurjano/branin.html
     - Currin Function: https://www.sfu.ca/~ssurjano/curretal88exp.html

    Args:
        task (list[str]): 探索方向
        reference_point (np.array): HVの参照点

    """

    def __init__(self) -> None:
        pass
        tkwargs = {
            'dtype': torch.double,
            'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        }
        self.task = ['maximize', 'maximize']
        self.reference_point = np.array([-350, -15])
        self.problem = br(negate=True).to(**tkwargs)
        self.distributions = {f'x{d}': FloatDistribution(0.0, 1.0) for d in range(2)}

    def f(self, xx: np.ndarray) -> np.ndarray:
        """f.

        Args:
            xx (np.ndarray): 入力. xx.shape=(1, 2)

        Returns:
            np.ndarray: 出力. shape=(1, 2)

        """
        if xx.shape != (1, 2):
            raise InputError('入力次元エラー. shape=(1, 2) is required')

        xx = torch.from_numpy(xx)
        f = self.problem(xx).numpy().copy()
        return f

    def random_x(self) -> np.ndarray:
        """入力空間の点をランダムに1点取得."""
        x = np.random.uniform(low=0.0, high=1.0, size=2)
        x = x.reshape(1, x.shape[0])
        return x


class Hartmann6Obj2:
    """Hartmann6を編集したテスト関数."""

    def __init__(self) -> None:
        self.hart_f = Hartmann6()
        self.task = ['maximize', 'maximize']
        self.reference_point = np.array([0, 0])
        self.distributions = self.hart_f.distributions

    def f(self, xx: np.ndarray) -> np.ndarray:
        """f.

        Args:
            xx (np.ndarray): 入力. xx.shape=(1, 6)

        Returns:
            np.ndarray: 出力. shape=(1, 2)

        """
        y1 = self.hart_f.f(xx)
        y2 = y1**2 + 10
        return np.concatenate([y1, y2], axis=1)

    def random_x(self) -> np.ndarray:
        return self.hart_f.random_x()


f = Hartmann6Obj2()
print(f.f(f.random_x()))
