import polars as pl
import matplotlib.pyplot as plt
import japanize_matplotlib
import numpy as np
from pymoo.indicators.hv import HV
from test_functions.multi_objective import BraninCurrin, Hartmann6Obj2


def hyper_volume(y_obj: np.array, directions: list[str], reference_point=None):
    """パレート超体積を計算.

    pymooのget_performance_indicatorを使って計算
    pymooは最小化前提でのHVなので, 最大化の場合は符号を反転する.

    Args:
        y_obj: shape=(n, y_dim)
        directions (list[str]): 最適化方向
        reference_point: aa

    """
    if y_obj.shape[-1] != len(directions):
        raise Exception('Not match dim_size')

    sign = np.array([-1.0 if directions[i] == 'maximize' else 1.0 for i in range(y_obj.shape[-1])])
    y_obj = y_obj * sign
    if reference_point is None:
        reference_point = np.max(y_obj, axis=0) + np.min(y_obj, axis=0) * 0.1  # 観測データ上の最大値よりも少し大きい点
    else:
        reference_point = reference_point * sign
    f = HV(ref_point=reference_point)
    hv = f(y_obj)
    return hv


if __name__ == '__main__':
    exp_name = 'BraninCurrin'
    methods = ['random', 'MOTPE', 'EHVI', 'LogEHVI']

    ### タスク設定
    if exp_name == 'BraninCurrin':
        target_f = BraninCurrin()
    elif exp_name == 'Hartmann6Obj2':
        target_f = Hartmann6Obj2()
    directions = target_f.task
    reference_point = target_f.reference_point

    # 探索の様子をプロット
    fig = plt.figure(figsize=(25, 10))
    for i in range(len(methods)):
        method = methods[i]
        df_ = pl.read_csv(f'exp_result/{exp_name}/{method}/run_1.csv')
        plt.subplot(2, int(len(methods) / 2), i + 1)
        plt.title(f'{method}')
        plt.scatter(df_['y0'], df_['y1'], marker='o')
    plt.tight_layout()
    plt.savefig(f'exp_result/{exp_name}/all.png')

    # Hiper-volumeの推移プロット
    fig = plt.figure(figsize=(10, 6))
    for i in range(len(methods)):
        method = methods[i]
        df = pl.read_csv(f'exp_result/{exp_name}/{method}/run_1.csv')
        hvs = []
        for i in range(10, len(df_)):
            df_ = df[:i]
            ys_ = df_.to_numpy()
            hv = hyper_volume(ys_, directions, reference_point=reference_point)
            hvs.append(hv)
        plt.plot(range(len(hvs)), hvs, label=method)
    plt.tight_layout()
    plt.legend(loc='upper left')
    plt.savefig(f'exp_result/{exp_name}/hv.png')
