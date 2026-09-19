"""Carseats 数据集多元线性回归分析。

任务：以 Sales 为响应变量，以 Price、Income、Advertising 和 ShelveLoc
为解释变量，输出模型报告、ShelveLoc 基准组及系数解释，并计算 VIF。

需要的第三方库：pandas、statsmodels。
"""

from pathlib import Path
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor


def load_carseats() -> pd.DataFrame:
    """优先读取同目录 Carseats.csv，否则从公开数据源下载。"""
    local_file = Path(__file__).with_name("Carseats.csv")
    if local_file.exists():
        return pd.read_csv(local_file)

    urls = [
        "https://www.statlearning.com/s/Carseats.csv",
        "https://vincentarelbundock.github.io/Rdatasets/csv/ISLR/Carseats.csv",
    ]
    last_error = None
    for url in urls:
        try:
            data = pd.read_csv(url)
            if "Sales" in data.columns:
                return data
        except Exception as exc:  # 尝试下一个数据源
            last_error = exc

    raise RuntimeError(
        "无法加载 Carseats 数据集。请下载 Carseats.csv 并放到本 Python 文件同目录。"
    ) from last_error


def main() -> None:
    data = load_carseats()

    required = ["Sales", "Price", "Income", "Advertising", "ShelveLoc"]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"数据集中缺少字段：{missing}")

    df = data[required].dropna().copy()

    # 明确类别顺序，使 Bad 成为 ShelveLoc 的基准组。
    df["ShelveLoc"] = pd.Categorical(
        df["ShelveLoc"], categories=["Bad", "Medium", "Good"], ordered=False
    )

    formula = (
        "Sales ~ Price + Income + Advertising "
        "+ C(ShelveLoc, Treatment(reference='Bad'))"
    )
    model = smf.ols(formula=formula, data=df).fit()

    print("=" * 70)
    print("一、模型拟合报告")
    print("=" * 70)
    print(model.summary())

    print("\n" + "=" * 70)
    print("二、ShelveLoc 基准组与 Good 系数解释")
    print("=" * 70)
    print("ShelveLoc 的基准组是 Bad（较差货架位置）。")

    good_name = "C(ShelveLoc, Treatment(reference='Bad'))[T.Good]"
    good_coef = model.params[good_name]
    print(f"ShelveLoc[Good] 的估计系数为：{good_coef:.4f}")
    print(
        "商业含义：在 Price、Income 和 Advertising 保持不变的条件下，"
        f"货架位置为 Good 的商店，其预测销售额平均比 Bad 位置高 "
        f"{good_coef:.4f} 个 Sales 单位。Carseats 中 Sales 以千件计，"
        f"因此约多销售 {good_coef * 1000:.0f} 件。"
    )

    # 按回归模型相同方式生成哑变量，drop_first=True 删除 Bad 基准组，
    # 从而避免“虚拟变量陷阱”。
    x = pd.get_dummies(
        df[["Price", "Income", "Advertising", "ShelveLoc"]],
        columns=["ShelveLoc"],
        drop_first=True,
        dtype=float,
    )

    vif_result = pd.DataFrame(
        {
            "变量": x.columns,
            "VIF": [
                variance_inflation_factor(x.values, i)
                for i in range(x.shape[1])
            ],
        }
    )

    print("\n" + "=" * 70)
    print("三、方差膨胀因子（VIF）")
    print("=" * 70)
    print(vif_result.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    max_vif = vif_result["VIF"].max()
    if max_vif < 5:
        conclusion = "各变量 VIF 均小于 5，不存在明显的多重共线性风险。"
    elif max_vif < 10:
        conclusion = "部分变量 VIF 位于 5～10，可能存在一定多重共线性，需要关注。"
    else:
        conclusion = "存在 VIF 不小于 10 的变量，模型可能有较严重的多重共线性。"

    print(f"\n判断：{conclusion}")
    print("说明：常数项不参与 VIF 判断；分类变量以非基准组哑变量表示。")


if __name__ == "__main__":
    main()
