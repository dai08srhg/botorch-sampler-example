import polars as pl
import matplotlib.pyplot as plt
import japanize_matplotlib
import numpy as np
from collections import defaultdict


def get_best_ys(ys):
    """ """
    best_y = np.inf
    best_ys = []
    for y in ys:
        if y < best_y:
            best_y = y
        best_ys.append(best_y)
    return best_ys


if __name__ == '__main__':
    exp_name = 'Hartmann6Cat2'

    col2best_ys = {}

    # 全施行プロット
    fig = plt.figure(figsize=(22, 12))
    fig.suptitle(f'{exp_name}')
    for j in range(1, 11):
        df = pl.read_csv(f'exp_result/{exp_name}/run_{j}.csv')
        for col in df.columns:
            ys = df[col].to_list()
            ys = ys[9:]
            plt.subplot(3, 4, j)
            plt.scatter(range(len(ys)), ys, marker='.', label=f'{col}')

            best_ys = get_best_ys(ys)
            plt.plot(range(len(best_ys)), best_ys, label=f'{col}(best)')

            best_ys_ = col2best_ys.get(col, [])
            best_ys_.append(best_ys)
            col2best_ys[col] = best_ys_

    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(f'exp_result/{exp_name}/{exp_name}_all.png')

    # 平均パフォーマンスをプロット
    fig = plt.figure(figsize=(8, 5))
    fig.suptitle(f'{exp_name}', fontsize=18)
    for col, best_yss in col2best_ys.items():
        performance = np.array(best_yss)
        performance = np.mean(performance, axis=0)

        plt.plot(range(len(performance)), performance, label=f'{col}(best)')
    plt.ylabel('best_f')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(f'exp_result/{exp_name}/{exp_name}_performance.png')
