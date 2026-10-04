## 作业1 · 推导题：证明正交设计下 $\widehat{\beta}_{\mathrm{Ridge}}=\frac{1}{1+\lambda}\widehat{\beta}_{\mathrm{OLS}}$

普通最小二乘法的正规方程为 $X^TX\hat\beta_{OLS}=X^Ty$。当 $X^TX=I$ 时，$\hat\beta_{OLS}=X^Ty$。岭回归目标函数对 $\beta$ 求导并令其等于零，得到

$$
-2X^T(y-X\beta)+2\lambda\beta=0,
\qquad (X^TX+\lambda I)\hat\beta_{Ridge}=X^Ty.
$$

代入 $X^TX=I$，即 $(1+\lambda)I\hat\beta_{Ridge}=X^Ty=\hat\beta_{OLS}$，故

$$
\boxed{\hat\beta_{Ridge}=\frac{1}{1+\lambda}\hat\beta_{OLS}}.
$$

因此在此特殊条件下，每个斜率系数被同一比例收缩；$\lambda=0$ 时两者相同，$\lambda$ 越大，岭回归系数越靠近零。**仅有“列间正交”并不足以得到统一的 $1/(1+\lambda)$ 比例。**若 $X^TX=\operatorname{diag}(d_1,\ldots,d_p)$，则 $\hat\beta_{Ridge,j}=d_j/(d_j+\lambda)\hat\beta_{OLS,j}$；只有所有 $d_j=1$ 才恢复题目中的等式。

## 作业3 · 分析题：为什么正则化前必须 Z-score 标准化？不做的后果？

Z-score 对第 $j$ 个特征作 $z_{ij}=(x_{ij}-\bar x_j)/s_j$。Ridge 惩罚 $\lambda\sum_j\beta_j^2$，Lasso 惩罚 $\lambda\sum_j|\beta_j|$，Elastic Net 同时含二者。惩罚直接作用在**系数数值**上，而系数数值依赖特征的计量单位。例如同一个面积用平方米或平方厘米表示，系数会相差一万倍；在不改变实际预测意义的情况下，两种表示却受到不同强度的惩罚。标准化使连续特征在训练集上的尺度可比，让调参主要反映预测与收缩之间的取舍。

如果不标准化，数值范围大的变量往往只需较小的系数就能改变预测，受到的惩罚较轻；数值范围小的变量为了产生相似影响需要较大的系数，容易被过度压缩，Lasso 甚至可能错误地将其置零。结果可能包括变量选择随单位改变、交叉验证选出的参数不稳定、系数解释失真。标准化无法消除特征之间的相关性，也不能保证某个模型总是最优。

实际步骤应先分训练集和测试集，再**仅用训练集拟合**缺失值填补器和标准化器；交叉验证的每一折也须在该折训练部分重新计算均值与标准差。因此在编程实现时应把标准化放入交叉验证的每一折训练流程中。截距一般不参加惩罚；类别变量如使用独热编码，应说明其 0/1 尺度与 Z-score 后的连续变量不同。

