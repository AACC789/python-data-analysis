"""Spambase 垃圾邮件分类：线性核与 RBF 核 SVM。
运行：python svm_spambase.py
依赖：pip install numpy pandas scikit-learn matplotlib
数据优先读取脚本旁 data/spambase.data；不存在时从 UCI 下载。
"""
from pathlib import Path
import io
import json
import platform
import urllib.request
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report, ConfusionMatrixDisplay,
                             RocCurveDisplay)
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

BASE = Path(__file__).resolve().parent
DATA_URL = "https://archive.ics.uci.edu/static/public/94/spambase.zip"
SEED = 42
WORDS = "make address all 3d our over remove internet order mail receive will people report addresses free business email you credit your font 000 money hp hpl george 650 lab labs telnet 857 data 415 85 technology 1999 parts pm direct cs meeting original project re edu table conference".split()
FEATURES = (["word_freq_" + w for w in WORDS]
            + ["char_freq_" + c for c in [";", "(", "[", "!", "$", "#"]]
            + ["capital_run_length_average", "capital_run_length_longest",
               "capital_run_length_total"])


def load_data():
    folder = BASE / "data"
    folder.mkdir(exist_ok=True)
    path = folder / "spambase.data"
    if not path.exists():
        archive = folder / "spambase.zip"
        try:
            content = archive.read_bytes() if archive.exists() else urllib.request.urlopen(DATA_URL, timeout=60).read()
            with zipfile.ZipFile(io.BytesIO(content)) as z:
                member = next(n for n in z.namelist() if n.endswith("spambase.data"))
                path.write_bytes(z.read(member))
        except Exception as exc:
            raise RuntimeError("下载失败。请从 UCI 下载 Spambase，将 spambase.data 放入脚本旁的 data 文件夹后重试。") from exc
    df = pd.read_csv(path, header=None)
    if df.shape[1] != 58:
        raise ValueError("数据必须含 57 个特征及最后一列标签，共 58 列。")
    df.columns = FEATURES + ["spam"]
    if not df.spam.isin([0, 1]).all():
        raise ValueError("标签必须为 0 或 1。")
    if not np.isfinite(df.to_numpy(dtype=float)).all():
        raise ValueError("数据存在缺失值或无穷值，请先检查数据。")
    return df


def evaluate(model, X, y):
    pred = model.predict(X)
    scores = model.decision_function(X) if hasattr(model, "decision_function") else model.predict_proba(X)[:, 1]
    return {"accuracy": accuracy_score(y, pred),
            "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0),
            "f1": f1_score(y, pred, zero_division=0),
            "roc_auc": roc_auc_score(y, scores)}


def main():
    out = BASE / "results"
    out.mkdir(exist_ok=True)
    raw = load_data()
    # 删除标签相互矛盾的相同特征组，再去掉完全重复记录。
    # 确保同一特征向量不会同时落入训练集和测试集。
    conflicts = raw.groupby(FEATURES, dropna=False)["spam"].transform("nunique").gt(1)
    unambiguous = raw.loc[~conflicts]
    clean = unambiguous.drop_duplicates().reset_index(drop=True)
    X, y = clean[FEATURES], clean.spam.astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y)
    audit = {"raw_samples": len(raw), "features": len(FEATURES),
             "missing_values": int(raw.isna().sum().sum()),
             "raw_class_counts": {str(k): int(v) for k, v in raw.spam.value_counts().items()},
             "raw_exact_duplicate_rows": int(raw.duplicated().sum()),
             "conflicting_rows_removed": int(conflicts.sum()),
             "duplicate_rows_removed_after_conflicts": int(unambiguous.duplicated().sum()),
             "clean_samples": len(clean),
             "clean_class_counts": {str(k): int(v) for k, v in y.value_counts().items()},
             "train_samples": len(y_train), "test_samples": len(y_test),
             "train_class_counts": {str(k): int(v) for k, v in y_train.value_counts().items()},
             "test_class_counts": {str(k): int(v) for k, v in y_test.value_counts().items()}}
    # scaler 在 Pipeline 内：每折只对该折训练部分拟合。
    pipe = Pipeline([("scale", StandardScaler()), ("svm", SVC(cache_size=512))])
    grid = [{"svm__kernel": ["linear"], "svm__C": [0.1, 1, 10]},
            {"svm__kernel": ["rbf"], "svm__C": [0.1, 1, 10],
             "svm__gamma": ["scale", 0.001, 0.01, 0.1]}]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    search = GridSearchCV(pipe, grid, scoring={"f1": "f1", "accuracy": "accuracy", "roc_auc": "roc_auc"},
                          refit="f1", cv=cv, n_jobs=1, error_score="raise")
    print("开始 5 折交叉验证，共 15 组参数、75 次拟合……", flush=True)
    search.fit(X_train, y_train)
    pd.DataFrame(search.cv_results_).to_csv(out / "cv_results.csv", index=False)
    # 各核函数的候选模型均只根据训练集交叉验证 F1 选出。
    cvframe = pd.DataFrame(search.cv_results_)
    selected = {}
    summaries = []
    for kernel in ["linear", "rbf"]:
        indices = [i for i, p in enumerate(search.cv_results_["params"]) if p["svm__kernel"] == kernel]
        index = max(indices, key=lambda i: search.cv_results_["mean_test_f1"][i])
        params = search.cv_results_["params"][index]
        candidate = Pipeline([("scale", StandardScaler()), ("svm", SVC(cache_size=512))]).set_params(**params)
        candidate.fit(X_train, y_train)
        selected[kernel] = candidate
        summaries.append({"model": kernel, "cv_f1": float(cvframe.loc[index, "mean_test_f1"]),
                          "cv_f1_std": float(cvframe.loc[index, "std_test_f1"]),
                          **evaluate(candidate, X_test, y_test), "params": json.dumps(params)})
    dummy = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    summaries.insert(0, {"model": "majority_baseline", "cv_f1": None, "cv_f1_std": None,
                         **evaluate(dummy, X_test, y_test), "params": "most_frequent"})
    table = pd.DataFrame(summaries)
    table.to_csv(out / "model_comparison.csv", index=False)
    best = search.best_estimator_
    pred = best.predict(X_test)
    cm = confusion_matrix(y_test, pred)
    report = classification_report(y_test, pred, target_names=["Non-spam", "Spam"], digits=4, zero_division=0)
    (out / "classification_report.txt").write_text(report, encoding="utf-8")
    pd.DataFrame({"original_row": X_test.index, "true_label": y_test.to_numpy(),
                  "predicted_label": pred, "decision_score": best.decision_function(X_test)}).to_csv(out / "predictions.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].bar(["Non-spam", "Spam"], [int((y == 0).sum()), int((y == 1).sum())], color=["#397aab", "#ed9458"])
    axes[0].set_title("Class distribution (cleaned)")
    axes[0].set_ylabel("Emails")
    ConfusionMatrixDisplay(cm, display_labels=["Non-spam", "Spam"]).plot(ax=axes[1], colorbar=False, cmap="Blues")
    axes[1].set_title("Best SVM: confusion matrix")
    for name, model in selected.items():
        RocCurveDisplay.from_estimator(model, X_test, y_test, ax=axes[2], name=name)
    axes[2].plot([0, 1], [0, 1], "k--", alpha=0.5)
    axes[2].set_title("ROC curves (held-out test)")
    fig.tight_layout()
    fig.savefig(out / "classification_results.png", dpi=160)
    plt.close(fig)
    result = {"data_url": DATA_URL, "seed": SEED, "audit": audit,
              "best_params": search.best_params_, "best_cv_f1": search.best_score_,
              "best_test_metrics": evaluate(best, X_test, y_test),
              "confusion_matrix": cm.tolist(),
              "support_vectors": best.named_steps["svm"].n_support_.tolist(),
              "versions": {"python": platform.python_version(), "numpy": np.__version__,
                           "pandas": pd.__version__, "scikit-learn": sklearn.__version__, "matplotlib": matplotlib.__version__}}
    (out / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(table.to_string(index=False))
    print(report)
    print("结果已保存至：", out)


if __name__ == "__main__":
    main()
