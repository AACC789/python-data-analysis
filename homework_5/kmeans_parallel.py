"""K-means++ 初始化 + 多线程分配步。运行：python kmeans_parallel.py
依赖：numpy pandas scikit-learn threadpoolctl matplotlib。所有路径默认相对于本文件。
"""
from __future__ import annotations
import argparse
import json
import os
import platform
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, silhouette_score
from threadpoolctl import threadpool_limits
import matplotlib
matplotlib.use("Agg")  # 自动保存图片，兼容无图形界面的环境
import matplotlib.pyplot as plt

FEATURES = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']


def validate(x, centers):
    if x.ndim != 2 or centers.ndim != 2 or x.shape[1] != centers.shape[1]:
        raise ValueError('样本和中心必须为特征维数一致的二维数组')
    if len(x) == 0 or len(centers) == 0:
        raise ValueError('样本和中心不能为空')
    if not np.isfinite(x).all() or not np.isfinite(centers).all():
        raise ValueError('数据必须为有限数值')


def kmeans_plus_plus(x, k, seed=42):
    """按距离平方 D² 概率抽样，手动实现经典 K-means++。"""
    if x.ndim != 2 or len(x) == 0 or not np.isfinite(x).all():
        raise ValueError('数据必须是非空、有限数值二维数组')
    if not 1 <= k <= len(np.unique(x, axis=0)):
        raise ValueError('k 必须介于 1 与不同样本的数量之间')
    rng = np.random.default_rng(seed)
    centers = [x[rng.integers(len(x))].copy()]
    nearest = np.sum((x - centers[0]) ** 2, axis=1)
    for _ in range(1, k):
        idx = rng.choice(len(x), p=nearest / nearest.sum())
        centers.append(x[idx].copy())
        nearest = np.minimum(nearest, np.sum((x - centers[-1]) ** 2, axis=1))
    return np.asarray(centers)


def assign_block(x_block, centers):
    """线程只读取输入，返回自己的标签和距离，不修改共享数组。"""
    # 逐中心计算，避免创建 n×k×d 三维临时数组。
    distances = np.empty((len(x_block), len(centers)), dtype=np.float64)
    for j, center in enumerate(centers):
        diff = x_block - center
        distances[:, j] = np.einsum('ij,ij->i', diff, diff)
    labels = np.argmin(distances, axis=1)  # 相同距离时选编号最小的中心
    return labels, distances[np.arange(len(x_block)), labels]


def assign_parallel(x, centers, executor, workers):
    """按样本行均分任务；结果按提交顺序合并。executor 在各轮之间复用。"""
    if workers < 1:
        raise ValueError('线程数必须为正数')
    boundaries = np.linspace(0, len(x), min(workers, len(x)) + 1, dtype=int)
    futures = [executor.submit(assign_block, x[a:b], centers)
               for a, b in zip(boundaries[:-1], boundaries[1:])]
    parts = [future.result() for future in futures]  # 等待全部任务完成再更新中心
    return (np.concatenate([part[0] for part in parts]),
            np.concatenate([part[1] for part in parts]))


def fit_kmeans(x, initial_centers, workers=4, max_iter=300):
    validate(x, initial_centers)
    if workers < 1 or max_iter < 1:
        raise ValueError('workers 和 max_iter 必须为正数')
    centers = initial_centers.copy()
    previous = None
    converged = False
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for iteration in range(1, max_iter + 1):
            labels, distances = assign_parallel(x, centers, pool, workers)
            if previous is not None and np.array_equal(labels, previous):
                converged = True
                break
            new_centers = centers.copy()
            for j in range(len(centers)):
                members = x[labels == j]
                if len(members):
                    new_centers[j] = members.mean(axis=0)
                # 空簇保留原中心，避免求空数组均值产生 NaN。
            centers = new_centers
            previous = labels.copy()
        # 达到迭代上限时，也确保返回标签属于最终中心。
        labels, distances = assign_parallel(x, centers, pool, workers)
    return centers, labels, float(distances.sum()), iteration, converged


def benchmark(x, centers, repeats=9):
    """只测一次分配步，包含任务提交、等待及结果合并；不含线程池创建。"""
    rows = []
    for factor in (1, 2000):
        # 扩展数据仅用于性能测试，不能视为新增的真实观测。
        data = np.tile(x, (factor, 1))
        expected_labels, expected_distances = assign_block(data, centers)
        for workers in (1, 2, 4, 8):
            pool = ThreadPoolExecutor(max_workers=workers) if workers > 1 else None
            try:
                def run():
                    return (assign_block(data, centers) if pool is None else
                            assign_parallel(data, centers, pool, workers))
                run()  # 预热
                elapsed = []
                for _ in range(repeats):
                    start = time.perf_counter()
                    labels, distances = run()
                    elapsed.append((time.perf_counter() - start) * 1000)
                if not np.array_equal(labels, expected_labels):
                    raise AssertionError('多线程标签与单线程不同')
                np.testing.assert_allclose(distances, expected_distances, rtol=0, atol=0)
                rows.append({'samples': len(data), 'threads': workers,
                             'median_ms': float(np.median(elapsed)), 'repeats': repeats,
                             'equal_to_serial': True})
            finally:
                if pool is not None:
                    pool.shutdown()
    return rows



def save_plots(df, labels, centers_cm, timings, out):
    """保存三张图；用英文标注，避免部分电脑缺少中文字体。"""
    palette = ['#2563eb', '#f59e0b', '#10b981']
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False,
                         'axes.spines.right': False, 'figure.facecolor': 'white'})
    x = df['PetalLengthCm'].to_numpy()
    y = df['PetalWidthCm'].to_numpy()

    def cluster_scatter(ax, show_centers=True):
        for j in range(len(centers_cm)):
            mask = labels == j
            ax.scatter(x[mask], y[mask], s=42, color=palette[j % len(palette)],
                       alpha=0.75, edgecolors='white', linewidths=0.4,
                       label=f'Cluster {j} (n={mask.sum()})')
        if show_centers:
            ax.scatter(centers_cm[:, 2], centers_cm[:, 3], marker='X',
                       s=210, c='black', edgecolors='white', linewidths=1,
                       label='Cluster centers', zorder=5)
        ax.set(xlabel='Petal length (cm)', ylabel='Petal width (cm)')
        ax.grid(alpha=0.15)
        ax.legend(fontsize=9)

    fig, ax = plt.subplots(figsize=(8, 5.5), constrained_layout=True)
    cluster_scatter(ax)
    ax.set_title('K-means++ clustering: petal feature view', pad=12)
    fig.savefig(out / 'cluster_scatter.png', dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True,
                             constrained_layout=True)
    for j, species in enumerate(sorted(df['Species'].unique())):
        mask = df['Species'].to_numpy() == species
        axes[0].scatter(x[mask], y[mask], s=42, color=palette[j % len(palette)],
                        alpha=0.75, edgecolors='white', linewidths=0.4,
                        label=species.replace('Iris-', ''))
    axes[0].set(title='Ground-truth species', xlabel='Petal length (cm)',
                ylabel='Petal width (cm)')
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.15)
    cluster_scatter(axes[1])
    axes[1].set_title('Predicted clusters')
    fig.savefig(out / 'species_comparison.png', dpi=200)
    plt.close(fig)

    data = pd.DataFrame(timings)
    sizes = sorted(data['samples'].unique())
    fig, axes = plt.subplots(1, len(sizes), figsize=(12, 4.8),
                             squeeze=False, constrained_layout=True)
    for ax, size in zip(axes[0], sizes):
        part = data[data['samples'] == size].sort_values('threads')
        bars = ax.bar([str(t) for t in part['threads']], part['median_ms'],
                      color=['#64748b', '#60a5fa', '#2563eb', '#1e40af'])
        ax.bar_label(bars, fmt='%.3f', padding=4, fontsize=9)
        ax.set(title=f'{size:,} samples', xlabel='Threads', ylabel='Median time (ms)')
        ax.set_ylim(0, part['median_ms'].max() * 1.23)
        ax.grid(axis='y', alpha=0.15)
        ax.set_axisbelow(True)
    fig.suptitle('Assignment step timing (9 repeats; separate y-axis scales)')
    fig.savefig(out / 'thread_timing.png', dpi=200)
    plt.close(fig)
    return [out / name for name in ('cluster_scatter.png',
                                    'species_comparison.png', 'thread_timing.png')]


def main():
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', type=Path, default=base / 'data' / 'Iris.csv')
    parser.add_argument('--out', type=Path, default=base / 'results')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers 必须为正数')
    if not args.csv.is_file():
        parser.error(f'找不到数据文件：{args.csv}。请将 data 文件夹与本程序放在同一目录，或使用 --csv 指定路径。')
    df = pd.read_csv(args.csv)
    x_raw = df[FEATURES].to_numpy(dtype=np.float64)
    if not np.isfinite(x_raw).all():
        raise ValueError('特征存在缺失值或无穷值，请先清洗数据')
    mean, std = x_raw.mean(axis=0), x_raw.std(axis=0)
    std = np.where(std == 0, 1.0, std)
    x = np.ascontiguousarray((x_raw - mean) / std)
    initial = kmeans_plus_plus(x, 3, 42)
    # 控制底层数值库线程，避免把隐式并行误认为本程序线程池的效果。
    with threadpool_limits(limits=1):
        centers, labels, sse, iterations, converged = fit_kmeans(x, initial, args.workers)
        serial = fit_kmeans(x, initial, 1)
        np.testing.assert_array_equal(labels, serial[1])
        np.testing.assert_allclose(centers, serial[0], rtol=0, atol=0)
        # 额外检查非整除分块，以及线程数大于样本数的情况。
        with ThreadPoolExecutor(max_workers=8) as pool:
            for small in (x[:7], x[:2]):
                parallel = assign_parallel(small, initial, pool, 8)
                reference = assign_block(small, initial)
                np.testing.assert_array_equal(parallel[0], reference[0])
                np.testing.assert_allclose(parallel[1], reference[1], rtol=0, atol=0)
        timings = benchmark(x, centers)
    args.out.mkdir(parents=True, exist_ok=True)
    df.assign(cluster=labels).to_csv(args.out / 'cluster_labels.csv', index=False)
    cross = pd.crosstab(pd.Series(labels, name='cluster'), df.Species)
    cross.to_csv(args.out / 'cluster_species.csv')
    pd.DataFrame(centers * std + mean, columns=FEATURES).to_csv(args.out / 'centers_cm.csv', index_label='cluster')
    pd.DataFrame(timings).to_csv(args.out / 'benchmark.csv', index=False)
    result = {'samples': len(df), 'missing_cells': int(df.isna().sum().sum()),
              'duplicate_features': int(df.duplicated(subset=FEATURES).sum()),
              'seed': 42, 'k': 3, 'workers': args.workers,
              'iterations': iterations, 'converged': converged, 'sse': sse,
              'silhouette': float(silhouette_score(x, labels)),
              'adjusted_rand_index': float(adjusted_rand_score(df.Species, labels)),
              'serial_parallel_equal': True,
              'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                              'pandas': pd.__version__, 'cpu': platform.machine(),
                              'logical_cpu_count': os.cpu_count()},
              'benchmark': timings}
    (args.out / 'metrics.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(cross)
    image_paths = save_plots(df, labels, centers * std + mean, timings, args.out)
    print('\n图片已保存到：')
    for path in image_paths:
        print(path.resolve())


if __name__ == '__main__':
    main()
