"""作业4 · 实践题：用 Hitters / Boston 房价数据对比 Ridge、Lasso、Elastic Net。"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

URLS = {
    "Hitters": "https://raw.githubusercontent.com/selva86/datasets/master/Hitters.csv",
    "Boston": "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv",
}


def load_data(name):
    """同目录下存在 CSV 时优先读本地，否则联网读取并缓存。"""
    local = Path(__file__).resolve().parent / (name + ".csv")
    if local.exists():
        df = pd.read_csv(local)
    else:
        try:
            df = pd.read_csv(URLS[name])
            df.to_csv(local, index=False)
        except Exception as exc:
            raise RuntimeError(f"无法下载 {name}；请将对应 CSV 放在 {local}") from exc
    df.columns = df.columns.str.strip()
    target = "Salary" if name == "Hitters" else "medv"
    target = next((c for c in df if c.lower() == target.lower()), None)
    if target is None:
        raise ValueError(f"{name} 中未找到目标列 Salary/medv：{list(df.columns)}")
    df[target] = pd.to_numeric(df[target], errors="coerce")
    df = df.dropna(subset=[target])
    X = df.drop(columns=[target])
    # 非官方 CSV 可能带有导出索引，不将其当作特征。
    X = X.loc[:, ~X.columns.str.match(r"^Unnamed:")]
    return X, df[target]


def evaluate(name):
    X, y = load_data(name)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    numeric = X.select_dtypes(include=np.number).columns.tolist()
    categorical = X.select_dtypes(exclude=np.number).columns.tolist()
    numeric_pipe = Pipeline([("fill", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categories_pipe = Pipeline([("fill", SimpleImputer(strategy="most_frequent")),
                                ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    preprocess = ColumnTransformer([("num", numeric_pipe, numeric), ("cat", categories_pipe, categorical)])
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    models = {
        "Ridge": (Ridge(), {"model__alpha": np.logspace(-2, 4, 18)}),
        "Lasso": (Lasso(max_iter=100000), {"model__alpha": np.logspace(-2, 3, 18)}),
        "Elastic Net": (ElasticNet(max_iter=100000),
                        {"model__alpha": np.logspace(-2, 3, 14), "model__l1_ratio": [0.2, 0.5, 0.8]}),
    }
    print(f"\n{name}: 清理后 n={len(y)}，训练={len(y_train)}，测试={len(y_test)}；目标单位："
          + ("千美元" if name == "Hitters" else "千美元（MEDV）"))
    rows = []
    for label, (estimator, grid) in models.items():
        search = GridSearchCV(Pipeline([("prep", preprocess), ("model", estimator)]),
                              grid, scoring="neg_root_mean_squared_error", cv=cv, n_jobs=-1)
        search.fit(X_train, y_train)
        prediction = search.predict(X_test)
        coef = search.best_estimator_.named_steps["model"].coef_
        rows.append({"模型": label, "CV_RMSE": -search.best_score_,
                     "测试_RMSE": np.sqrt(mean_squared_error(y_test, prediction)),
                     "测试_MAE": mean_absolute_error(y_test, prediction),
                     "测试_R2": r2_score(y_test, prediction),
                     "零系数数": int(np.sum(np.isclose(coef, 0, atol=1e-8))),
                     "特征数": len(coef), "最佳参数": str(search.best_params_)})
    result = pd.DataFrame(rows).sort_values("测试_RMSE")
    print(result.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    best = result.iloc[0]
    print(f"结果分析：在本次测试划分中，{best['模型']} 的 RMSE 最低，"
          f"为 {best['测试_RMSE']:.3f}；其 {best['特征数']} 个系数中"
          f"有 {best['零系数数']} 个为零。此结果只对应本次划分。")
    result.to_csv(Path(__file__).resolve().parent / f"result_{name.lower()}_A.csv", index=False)


if __name__ == "__main__":
    for dataset in URLS:
        evaluate(dataset)
