"""
题目：使用 Elastic Net 对 Kaggle 上的 Netflix 数据做分析。
数据集：Netflix Movies and TV Shows（netflix_titles.csv）
来源：https://www.kaggle.com/datasets/shivamb/netflix-shows

分析目标：利用电影的上映年份、演员人数、制作国家、内容分级和题材，
分析并预测电影时长（分钟）。仅选择电影，避免把剧集季数与分钟混用。

安装依赖：python -m pip install numpy pandas scikit-learn matplotlib
运行方式：python netflix_elastic_net_A.py
将 netflix_titles.csv 放在脚本同目录即可离线运行；缺少文件时自动联网下载。
本题的 rating 是内容分级（PG、TV-MA 等），不是用户评分。
"""
from pathlib import Path
from urllib.request import Request, urlopen
import sys

try:
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.compose import ColumnTransformer
    from sklearn.dummy import DummyRegressor
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import ElasticNet
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import GridSearchCV, KFold, train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
except ModuleNotFoundError as error:
    raise SystemExit(
        f"缺少依赖：{error.name}\n请在当前 Python 环境安装：\n"
        f'"{sys.executable}" -m pip install numpy pandas scikit-learn matplotlib'
    ) from error

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "netflix_titles.csv"
OUTPUT = ROOT / "netflix_result_A"
# 以下为上述 Kaggle 数据的公开课程镜像，下载后检查必要字段。
CSV_URL = "https://raw.githubusercontent.com/fralfaro/MAT281/main/docs/labs/data/netflix_titles.csv"


def read_data():
    if not DATA_PATH.exists():
        print("正在下载 netflix_titles.csv……", flush=True)
        try:
            request = Request(CSV_URL, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(request, timeout=60) as response:
                content = response.read()
            if not content.startswith(b"show_id,type,title"):
                raise ValueError("下载内容不是预期的 Netflix CSV")
            DATA_PATH.write_bytes(content)
        except Exception as error:
            raise SystemExit(
                "下载失败。请从 Kaggle 下载 netflix_titles.csv，放在脚本同目录后重试。\n"
                f"具体原因：{error}"
            ) from error
    data = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    data.columns = data.columns.str.strip()
    required = {"type", "title", "duration", "release_year", "rating", "cast", "country", "listed_in"}
    if not required.issubset(data.columns):
        raise ValueError(f"数据字段不符，缺少：{sorted(required - set(data.columns))}")
    return data


def split_labels(text):
    """把国家或题材拆成多标签，所有标签均保留，不只取第一项。"""
    labels = [part.strip() for part in str(text).split(",") if part.strip()]
    return labels or ["Unknown"]


def make_samples(data):
    subset = data.loc[data["type"].astype(str).str.strip().eq("Movie")].copy()
    # 仅解析形如 90 min 的值；目标为空或单位错误的记录删除，不填补目标。
    subset["minutes"] = pd.to_numeric(
        subset["duration"].astype("string").str.extract(r"^\s*(\d+)\s+min\s*$", expand=False),
        errors="coerce",
    )
    subset = subset.dropna(subset=["minutes"])
    subset = subset.loc[subset["minutes"] > 0].drop_duplicates(subset=["title", "release_year"])
    X = pd.DataFrame(index=subset.index)
    X["release_year"] = pd.to_numeric(subset["release_year"], errors="coerce")
    X["cast_count"] = subset["cast"].map(
        lambda value: np.nan if pd.isna(value) or not str(value).strip() else len(split_labels(value))
    )
    for column in ["rating", "country", "listed_in"]:
        X[column] = subset[column].fillna("Unknown").astype(str).str.strip().replace("", "Unknown")
    if len(X) < 30:
        raise ValueError("有效电影记录不足 30 条，请确认读取的是正确的完整数据。")
    # duration/minutes 不出现在 X 中，以防目标泄漏；ID、名称和简介不用于回归。
    return X, subset["minutes"], subset["title"]


def describe_data(data, y):
    print("\n一、数据概况")
    print(f"原始数据：{data.shape[0]} 条记录，{data.shape[1]} 个字段")
    print(data["type"].value_counts().to_string())
    print("各列缺失数：\n" + data.isna().sum().to_string())
    print(f"清理后有效电影：{len(y)} 部")
    print("电影时长（分钟）描述统计：\n" + y.describe().round(2).to_string())
    data["type"].value_counts().to_csv(OUTPUT / "content_counts.csv", header=["count"])
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    data["type"].value_counts().plot.bar(ax=axes[0], rot=0, color=["#235789", "#ed6a5a"])
    axes[0].set(title="Netflix content types", ylabel="Count")
    axes[1].hist(y, bins=35, color="#235789", edgecolor="white")
    axes[1].set(title="Movie duration distribution", xlabel="Duration (minutes)", ylabel="Count")
    figure.tight_layout()
    figure.savefig(OUTPUT / "data_overview.png", dpi=150)
    plt.close(figure)


def model_pipeline():
    # 编码器只在各训练折上拟合，验证/测试中的新类别不会造成报错。
    preprocess = ColumnTransformer([
        ("number", SimpleImputer(strategy="median"), ["release_year", "cast_count"]),
        ("rating", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["rating"]),
        ("country", CountVectorizer(tokenizer=split_labels, token_pattern=None,
                                    lowercase=False, binary=True), "country"),
        ("genre", CountVectorizer(tokenizer=split_labels, token_pattern=None,
                                  lowercase=False, binary=True), "listed_in"),
    ], sparse_threshold=0)
    # 连续变量与编码后的特征均标准化，再惩罚系数；该步骤也在每一折内拟合。
    return Pipeline([
        ("prepare", preprocess),
        ("scale", StandardScaler()),
        ("elastic", ElasticNet(max_iter=20000, tol=1e-4)),
    ])


def analyse(X, y, titles):
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("\n二、Elastic Net 建模")
    print(f"训练集 {len(y_train)} 条；测试集 {len(y_test)} 条。正在进行五折交叉验证……", flush=True)
    # alpha 控制总惩罚强度；l1_ratio 控制 L1 惩罚在混合惩罚中的比例。
    search = GridSearchCV(
        model_pipeline(),
        {"elastic__alpha": np.logspace(-2, 1, 8), "elastic__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9]},
        cv=KFold(5, shuffle=True, random_state=42),
        scoring="neg_root_mean_squared_error", n_jobs=2,
    )
    search.fit(x_train, y_train)
    fitted = search.best_estimator_
    predicted = fitted.predict(x_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, predicted)))
    mae = float(mean_absolute_error(y_test, predicted))
    r2 = float(r2_score(y_test, predicted))
    # 均值预测是简单参考线，用来判断模型是否提供了实际预测信息。
    baseline = DummyRegressor(strategy="mean").fit(x_train, y_train).predict(x_test)
    base_rmse = float(np.sqrt(mean_squared_error(y_test, baseline)))
    print(f"最佳参数：alpha={search.best_params_['elastic__alpha']:.5f}，"
          f"l1_ratio={search.best_params_['elastic__l1_ratio']:.1f}")
    print(f"交叉验证 RMSE：{-search.best_score_:.3f} 分钟")
    print(f"测试 RMSE：{rmse:.3f} 分钟；MAE：{mae:.3f} 分钟；R²：{r2:.3f}")
    print(f"训练均值预测的测试 RMSE：{base_rmse:.3f} 分钟")
    pd.DataFrame([{"RMSE": rmse, "MAE": mae, "R2": r2, "baseline_RMSE": base_rmse,
                   "CV_RMSE": -search.best_score_, **search.best_params_}]).to_csv(OUTPUT / "metrics.csv", index=False)
    pd.DataFrame(search.cv_results_).to_csv(OUTPUT / "cross_validation.csv", index=False)
    pd.DataFrame({"title": titles.loc[y_test.index], "actual_minutes": y_test,
                  "predicted_minutes": predicted}).to_csv(OUTPUT / "predictions.csv", index=False)

    names = fitted.named_steps["prepare"].get_feature_names_out()
    coefficients = fitted.named_steps["elastic"].coef_
    table = pd.DataFrame({"feature": names, "coefficient": coefficients})
    table["absolute_coefficient"] = table["coefficient"].abs()
    table = table.sort_values("absolute_coefficient", ascending=False)
    table.to_csv(OUTPUT / "coefficients.csv", index=False)
    kept = int(np.sum(np.abs(coefficients) > 1e-8))
    print("\n三、特征与结果分析")
    print(f"共有 {len(coefficients)} 个编码后特征，其中 {kept} 个系数非零。")
    print("绝对系数最大的十个特征：\n" + table.head(10).to_string(index=False, float_format=lambda z: f"{z:.3f}"))
    print("模型优于均值预测参考线。" if rmse < base_rmse else "模型未优于均值参考线，现有特征的预测信息有限。")
    print(f"平均绝对预测误差约为 {mae:.2f} 分钟。正系数表示在其他特征固定时预测时长增加，负系数表示减少。")
    print("系数作用在标准化后的特征上；题材和国家存在相关性，结果表示统计关联，不能作为因果结论。")
    print("数据是历史内容目录，本实验不预测用户评分、播放量或推荐效果；单次划分结果不能推广到所有新电影。")

    figure, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_test, predicted, s=12, alpha=0.4, color="#235789")
    bounds = [min(y_test.min(), predicted.min()), max(y_test.max(), predicted.max())]
    ax.plot(bounds, bounds, "r--", label="Ideal prediction")
    ax.set(xlabel="Actual duration (minutes)", ylabel="Predicted duration (minutes)", title="Elastic Net test predictions")
    ax.legend()
    figure.tight_layout()
    figure.savefig(OUTPUT / "prediction_scatter.png", dpi=150)
    plt.close(figure)


def main():
    OUTPUT.mkdir(exist_ok=True)
    data = read_data()
    X, y, titles = make_samples(data)
    describe_data(data, y)
    analyse(X, y, titles)
    print(f"\n分析完成，表格与图像保存到：{OUTPUT}")


if __name__ == "__main__":
    main()
