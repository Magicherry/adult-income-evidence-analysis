# Adult Income 数据集的三角证据分析

# 摘要

本项目将 UCI Adult / Census Income 数据集视为一个数据分析问题，而不只是一个预测任务。我们关注四个问题：哪些变量能区分两类收入人群，特征之间如何关联，哪些信号在稀疏约束下依然保留，以及分类边界究竟主要是线性的，还是存在值得关注的非线性。为回答这些问题，本文结合了聚焦式探索分析、连续变量的类条件高斯摘要、互信息（MI）、L1 正则 Logistic 回归、小规模交互项检验、SVM 核函数比较，以及轻量级稳健性检查。结果表明，连续特征中区分能力最强的是 `education_num`、`age` 和 `hours_per_week`；相对地，`capital_gain` 和 `capital_loss` 明显偏离高斯分布，在高斯摘要下表现较弱。MI 显示 `marital_status`-`relationship` 与 `workclass`-`occupation` 等特征对存在较强依赖，但筛选出的数值交互项并未带来有意义的预测提升。稀疏 Logistic 模型在测试集上取得 `0.897` 的 ROC-AUC，并稳定保留 `marital_status`、`education_num`、`capital_gain_log1p`、`hours_per_week` 和 `age`。在 SVM 比较中，二次多项式核表现最好，ROC-AUC 为 `0.905`；线性核与 RBF 核则基本持平，分别为 `0.897` 和 `0.896`。总体来看，这个问题的大部分结构已经可以由相对简单的模型解释，同时存在有限的低阶非线性，若干强依赖特征之间也表现出明显冗余。

# 1. 引言

Adult 收入数据集很适合作为我们课程项目的数据分析对象，因为它同时包含连续变量和类别变量，类别不平衡也比较明显，而且适合从多个角度展开分析。与其只比较“哪个分类器得分更高”，本项目更关心数据本身的结构，以及不同分析方法是否会指向一致的结论。

本文围绕以下四个研究问题展开：

1. 数据的底层分布是什么样的，尤其是连续特征在不同收入类别之间的分布差异如何？
2. 特征之间如何相互依赖，哪些依赖关系可能提示有意义的交互？
3. 哪些特征携带了真实的预测信号，哪些更弱、冗余，或者更接近噪声？
4. 收入分类边界主要是线性的，还是存在值得关注的非线性结构？

一个特征单独看可能很有区分力，但放进多变量模型后却可能变得冗余；两个特征之间存在强依赖，也不一定意味着它们之间存在有价值的交互。因此，本文采用多种互补方法，并把最终结论建立在多方面证据的综合之上，而不是依赖单一模型结果。

# 2. 数据集与预处理

原始数据集包含 `48,842` 条记录，使用标准 Adult/Census Income 字段。目标变量为二分类标签 `income_gt_50k`，由原始 `income` 字段是否为 `>50K` 转换而来。

根据项目设定，两个输入特征被移除：

- `education`，因为它与 `education_num` 信息重复
- `fnlwgt`，按项目要求删除

后续分析保留的预测变量包括：

- 连续变量：`age`、`education_num`、`capital_gain`、`capital_loss`、`hours_per_week`
- 类别变量：`workclass`、`marital_status`、`occupation`、`relationship`、`race`、`gender`、`native_country`

在建模阶段，`capital_gain` 和 `capital_loss` 还会使用 `log1p` 变换形式，`native_country` 则被折叠为 `native_country_grouped`。

主划分采用分层 80/20 训练测试划分，随机种子为 `42`。稳健性检查使用 `7`、`42` 和 `99` 三个种子。所有调参都严格限制在训练集内部完成。

有一个预处理细节需要单独说明。`missing_value_summary.csv` 报告缺失值数量为零，但保存下来的类别汇总结果中，`workclass`、`occupation` 等字段里的 `?` 仍以显式类别出现。因此，本文按已保存输出的实际表现来描述结果，而不预设缺失值处理在整个流程中都被一致执行。

相比之下，`native_country` 的处理更清楚。完整数据中，`United-States` 大约占 `89.7%`，而折叠后特征与标签之间的 MI 只有 `0.0006`（见 `native_country_relevance.csv` 与 `feature_label_mi.csv`）。和主要的人口统计、家庭结构及工作相关变量相比，它提供的信号较弱。

# 3. 实验设计

整体实验设计采用“三角验证”思路：每个阶段从不同角度观察同一个问题，任何单一结果都不被视为充分证据。

- 聚焦式 EDA 用来确认类别比例、概括主要特征模式，并标记零膨胀、重复主值等明显分布问题。
- 类条件高斯分析为连续变量提供一个简洁的描述性视角。这里并不是声称数据服从高斯分布，而是借此比较均值、方差、重叠程度和单变量区分能力。
- 互信息分析同时衡量特征之间的依赖关系和特征对标签的相关性，并加入第二套离散化方案作为小规模敏感性检查。
- L1 Logistic 回归用于观察稀疏条件下的预测信号，并考察哪些特征组能在正则路径上持续保留。
- 交互项验证只测试一小组由 MI 结果和领域合理性共同筛出的数值交互。
- 线性、多项式和 RBF SVM 则用于判断非线性决策边界在实际中是否重要。
- 稳健性检查考察结论对随机种子、正则化强度和 MI 离散化方式的敏感程度。

# 4. 结果

## 4.1 聚焦式探索分析

训练集存在明显的类别不平衡：`29,724` 条样本（`76.1%`）属于 `<=50K`，`9,349` 条样本（`23.9%`）属于 `>50K`（见 `class_balance_summary.csv` 与 `class_balance.png`）。因此，后续比较更依赖 ROC-AUC 和 F1，而不是只看准确率。

![图1. 训练集类别分布。](../outputs/figures/class_balance.png)

*图1. 训练集中的类别分布。较高收入类别在整个分析中始终是少数类。*

探索分析显示出三个对后续解释有直接影响的分布特征（见 `continuous_feature_summary.csv`、`continuous_feature_flags.csv` 与 `continuous_by_income_grid.png`）：

- `capital_gain` 极其稀疏，训练集中有 `91.8%` 为 0。
- `capital_loss` 更稀疏，训练集中有 `95.2%` 为 0。
- `hours_per_week` 存在明显主值，训练集中有 `46.5%` 的样本集中在同一个取值。

这些现象解释了为什么均值和方差对 `education_num` 和 `age` 更有解释力，而对资本相关变量则帮助有限；它们同时也说明了后续进行 `log1p` 变换的必要性。

类别变量的汇总也已经显示出明显差异（见 `categorical_frequency_summary.csv` 与 `categorical_frequency_grid.png`）：

- `Married-civ-spouse` 的高收入比例为 `44.8%`，而 `Never-married` 只有 `4.4%`。
- `Husband` 和 `Wife` 的高收入比例分别为 `45.0%` 和 `47.5%`，而 `Own-child` 只有 `1.6%`。
- `Male` 的高收入比例为 `30.4%`，`Female` 为 `10.9%`。
- `Self-emp-inc` 的高收入比例达到 `54.9%`，明显高于 `Private` 的 `21.8%`。
- `Exec-managerial` 和 `Prof-specialty` 的高收入比例分别为 `48.2%` 和 `44.6%`，而 `Other-service` 只有 `4.0%`。

![图2. 不同收入类别下的连续特征分布。](../outputs/figures/continuous_by_income_grid.png)

*图2. 连续特征在不同收入类别下的分布对比。`education_num`、`age` 和 `hours_per_week` 有明显位移，而 `capital_gain` 与 `capital_loss` 则主要表现为大量零值和长右尾。*

![图3. 关键类别变量频数汇总。](../outputs/figures/categorical_frequency_grid.png)

*图3. 选定类别变量的分布及其高收入比例。家庭结构与职业相关变量在正式建模前就已经呈现出明显差异。*

## 4.2 类条件分布分析

高斯摘要按单变量 ROC-AUC 和分布重叠程度对连续特征进行排序（见 `gaussian_fit_summary.csv`、`ranked_continuous_features.csv`、`gaussian_fit_overlays.png` 与 `continuous_separation_ranking.png`）。

| 特征 | `<=50K` 均值 | `>50K` 均值 | 单变量 ROC-AUC | Cohen's d | 解释 |
| --- | --- | --- | --- | --- | --- |
| `education_num` | 9.600 | 11.603 | 0.716 | 0.825 | 区分能力最强的连续特征 |
| `age` | 36.919 | 44.350 | 0.682 | 0.556 | 有中等分离能力，但仍有明显重叠 |
| `hours_per_week` | 38.892 | 45.434 | 0.671 | 0.541 | 中等分离，但两类样本都集中在 40 小时附近 |
| `capital_gain` | 147.696 | 3949.978 | 0.588 | 0.532 | 高度偏态且零膨胀，在高斯视角下较弱 |
| `capital_loss` | 55.346 | 199.978 | 0.535 | 0.358 | 高度偏态，在高斯视角下较弱 |

`education_num` 是最强的连续区分变量，而且优势比较明显。它在两类中的中位数也差异清楚（在 `continuous_feature_summary.csv` 中分别为 `9` 和 `12`），说明它的作用并不只是均值偏移。

`age` 和 `hours_per_week` 也有一定区分力，但分离并不彻底。高收入样本整体更年长、工作时间更长，不过两类之间的重叠仍然较大，尤其是工作时长，两类的中位数都还是 `40`。

资本变量则与其他连续变量区别较大。两类在 `capital_gain` 和 `capital_loss` 上的中位数都为 `0`，但高收入类的均值明显更大。这种模式符合“少量极大值抬高均值”的情况。高斯摘要能反映出它们与其他变量的差异，但显然不是描述这类变量形状的理想工具。

![图4. 连续特征的高斯拟合覆盖图。](../outputs/figures/gaussian_fit_overlays.png)

*图4. 经验直方图与类条件高斯曲线的对比。对 `education_num`、`age` 和 `hours_per_week`，高斯视角还有一定解释力；对资本变量则明显不够贴切。*

![图5. 连续特征分离能力排序。](../outputs/figures/continuous_separation_ranking.png)

*图5. 按单变量 ROC-AUC 排序的连续特征。`education_num` 是最强的边际连续区分变量。*

## 4.3 特征依赖分析

MI 分析区分了两件容易混在一起的事：特征之间的依赖关系，以及特征对标签的直接相关性（见 `feature_label_mi.csv`、`top_feature_pairs.csv`、`mi_heatmap.png` 与 `top_mi_pairs.png`）。

与标签 MI 最高的特征如下：

| 特征 | 特征-标签 MI |
| --- | --- |
| `relationship` | 0.116 |
| `marital_status` | 0.111 |
| `age` | 0.064 |
| `occupation` | 0.063 |
| `education_num` | 0.062 |
| `capital_gain` | 0.054 |

最强的特征对依赖如下：

| 特征对 | Pairwise MI |
| --- | --- |
| `marital_status` x `relationship` | 0.725 |
| `workclass` x `occupation` | 0.329 |
| `relationship` x `gender` | 0.270 |
| `age` x `marital_status` | 0.233 |
| `education_num` x `occupation` | 0.213 |

这些高 MI 特征对很有信息量，但不能直接等同于“值得加入的交互项”。例如，`marital_status` 和 `relationship` 都在描述家庭结构，因此它们之间极高的 MI 更像是在说明信息重叠，而不是存在一个有价值的乘性交互。`workclass` 和 `occupation` 也有类似情况。

因此，保存下来的交互候选项主要聚焦在数值变量之间：

- `age` x `hours_per_week`
- `age` x `education_num`
- `education_num` x `hours_per_week`
- `education_num` x `capital_gain`
- `capital_gain` x `hours_per_week`

离散化敏感性检查非常稳定（见 `mi_sensitivity_summary.csv`）：两套方案下 top-10 MI 特征对重合度为 `10 / 10`，候选交互项集合重合度为 `5 / 5`。在本文采用的检查范围内，依赖关系排序相当稳固。

![图6. 特征间互信息热力图。](../outputs/figures/mi_heatmap.png)

*图6. 保留分析特征之间的 MI 热力图。最强依赖主要集中在家庭结构和工作背景相关变量之间，而不在数值交互候选项之间。*

![图7. 最高互信息特征对。](../outputs/figures/top_mi_pairs.png)

*图7. 互信息最高的特征对。强依赖关系确实存在，但后续结果表明，“强依赖”并不自动等于“值得加入预测交互项”。*

## 4.4 稀疏 Logistic 回归中的信号与噪声

L1 Logistic 回归主要回答两个问题：稀疏线性模型本身能达到怎样的效果，以及哪些特征组会在正则化变化时持续保留（见 `logistic_cv_summary.csv`、`logistic_baseline_metrics.csv`、`stable_features_summary.csv` 与 `logistic_coefficient_paths.png`）。

基线测试集指标如下：

| Accuracy | Precision | Recall | F1 | ROC-AUC | 最优 `C` |
| --- | --- | --- | --- | --- | --- |
| 0.848 | 0.723 | 0.589 | 0.649 | 0.897 | 1.0 |

交叉验证 ROC-AUC 从最强正则化（`C = 0.001`）下的 `0.887` 上升到 `C >= 0.1` 时大约 `0.900` 的平台区，而 `C = 0.1` 到 `C = 10` 之间的差异只出现在小数点后第四位。这说明只要允许中等程度的稀疏性，整体结论并不依赖某一个非常精确的正则化强度。

在完整 `C` 网格上最稳定的特征组如下：

| 特征组 | 在正则路径上的非零比例 | 平均绝对组系数 |
| --- | --- | --- |
| `marital_status` | 1.000 | 3.359 |
| `education_num` | 1.000 | 0.741 |
| `capital_gain_log1p` | 1.000 | 0.475 |
| `hours_per_week` | 1.000 | 0.328 |
| `age` | 1.000 | 0.317 |
| `capital_loss_log1p` | 1.000 | 0.220 |

还有一些特征组也有用，但稳定性稍弱：

- `occupation`：`88.9%`
- `relationship`、`workclass`、`gender`：各为 `77.8%`
- `native_country_grouped`、`race`：各为 `66.7%`

最强信号并不是平均分散在所有变量里，而是集中在少数核心特征组上。`marital_status`、`education_num`、`age`、工作强度变量以及资本相关变量构成了最清楚的主干。

另外，一些在高斯摘要下看起来较弱的变量，在经过变换后依然对多变量模型有贡献。`capital_gain_log1p` 和 `capital_loss_log1p` 在整个正则路径中都保留下来，说明“分布描述得好不好”和“预测上有没有用”是两回事。

同时，高 MI 并不保证在存在冗余时依然稳定。`relationship` 的特征-标签 MI 最高，但在稀疏 Logistic 中，`marital_status` 作为特征组更稳定。更合理的解释是，两者编码了大量重叠信息，而模型在被迫保持稀疏时更倾向于依赖其中一种表示。

![图8. L1 Logistic 系数路径。](../outputs/figures/logistic_coefficient_paths.png)

*图8. L1 正则路径下的系数变化。只有少数特征组在大部分路径上都保持活跃。*

## 4.5 交互项验证

交互阶段检验的是：一小组有针对性的数值交互项，是否能在主效应 Logistic 模型之外提供额外收益（见 `tested_interactions.csv`、`interaction_evidence_table.csv` 与 `interaction_delta_cv_auc.png`）。

| 交互项 | Pairwise MI | CV ROC-AUC 变化 | 测试集 ROC-AUC 变化 | 证据代码 |
| --- | --- | --- | --- | --- |
| `age` x `hours_per_week` | 0.0657 | +0.000174 | +0.000210 | unstable |
| `age` x `education_num` | 0.0517 | -0.000026 | -0.000012 | unstable |
| `education_num` x `hours_per_week` | 0.0265 | -0.000043 | -0.000003 | unstable |
| `education_num` x `capital_gain` | 0.0160 | -0.000025 | +0.000057 | unstable |
| `capital_gain` x `hours_per_week` | 0.0077 | +0.000042 | -0.000066 | unstable |

关键在于，这些变化都非常小。交叉验证 ROC-AUC 的最大正向变化只有 `+0.000174`，测试集上的最大正向变化也只有 `+0.000210`，不足以支撑“有实质性提升”的判断。

没有任何交互项强到足以形成最终增强模型，这也是为什么 `interaction_model_metrics.csv` 中 `interaction_augmented` 一行是 `NaN`，而不是一个保留下来的最终模型结果。

因此这一阶段的结论是：MI 筛出的候选项看起来合理，但在这组有限且可解释的测试中，没有任何显式数值交互能够稳定超越主效应模型。

![图9. 各交互项对 CV ROC-AUC 的增量影响。](../outputs/figures/interaction_delta_cv_auc.png)

*图9. 加入每个候选交互项后，交叉验证 ROC-AUC 的变化都非常小，因此最终没有保留交互增强模型。*

## 4.6 基于 SVM 的决策边界分析

SVM 比较为判断边界形状提供了最直接的证据（见 `svm_tuning_summary.csv`、`svm_comparison.csv`、`svm_kernel_comparison.png` 与 `final_model_comparison_table.csv`）。

| 模型族 | 最优调参设置 | CV ROC-AUC | 测试集 ROC-AUC | 测试集 F1 |
| --- | --- | --- | --- | --- |
| 线性 SVM | `C = 0.1` | 0.893 | 0.897 | 0.644 |
| 多项式 SVM | `degree = 2`, `C = 1.0`, `coef0 = 1.0` | 0.898 | 0.905 | 0.670 |
| RBF SVM | `C = 3.0`, `gamma = 0.05` | 0.892 | 0.896 | 0.670 |

- 多项式核在测试集上的 ROC-AUC 比线性核高约 `0.0078`（`0.905` 对 `0.897`）。
- RBF 核并没有超过线性基线，测试集 ROC-AUC 反而略低，为 `0.896`。
- 在多项式核内部，最佳配置是二次模型；保存下来的三次模型在交叉验证中明显更差（`0.883` 和 `0.883`），低于最佳二次配置的 `0.898`。

这说明数据里存在一定的低阶非线性，但并没有表现出广泛而复杂的非线性结构。也就是说，纯线性边界并不能完整解释问题，但也没有证据表明必须依赖高度灵活的核函数。

这一结果也和前面的交互项结论相吻合。多项式核可能在完整编码后的特征空间里捕捉到了更广泛的低阶效应，包括平方项以及本文没有显式测试到的交互，而不是仅仅那五个数值乘积项。

![图10. SVM 核函数在测试集上的比较。](../outputs/figures/svm_kernel_comparison.png)

*图10. 不同 SVM 模型族在测试集上的 ROC-AUC。多项式核表现最好，而 RBF 并没有优于线性基线。*

## 4.7 稳健性检查

### 正则化稳定性

Logistic 的 CV 曲线在 `C >= 0.1` 后非常平坦（见 `logistic_cv_summary.csv`）：

- `C = 0.1`：平均 CV ROC-AUC `0.899822`
- `C = 0.3162`：平均 CV ROC-AUC `0.900060`
- `C = 1.0`：平均 CV ROC-AUC `0.900099`
- `C = 10.0`：平均 CV ROC-AUC `0.900054`

这进一步说明，Logistic 的整体结论并不依赖某一个精确的正则化强度。

### MI 离散化稳定性

保存的 `mi_sensitivity_summary.csv` 表显示，两套离散化方案下 top-10 特征对完全重合，候选交互项集合也完全重合：

- Top-10 特征对重合度：`10 / 10`
- 候选交互项重合度：`5 / 5`

在本项目范围内，MI 给出的依赖结构是最稳定的一类结果。

### 随机种子稳定性

不同种子下测试集 ROC-AUC 如下：

| 模型 | Seed 7 | Seed 42 | Seed 99 | 范围 |
| --- | --- | --- | --- | --- |
| L1 Logistic | 0.901 | 0.897 | 0.901 | 0.897 到 0.901 |
| 线性 SVM | 0.900 | 0.897 | 0.901 | 0.897 到 0.901 |
| 多项式 SVM | 0.906 | 0.905 | 0.906 | 0.905 到 0.906 |

多项式核在所有保存下来的种子上都优于线性核：

- Seed `7`：+`0.0063`
- Seed `42`：+`0.0078`
- Seed `99`：+`0.0045`

这加强了边界形状结论的可信度。与此同时，保存下来的稳健性表格（`robust_feature_group_frequency.csv`、`robustness_boundary_consistency.csv`、`robustness_mi_overlap.csv`）也说明，稳定的是整体结论和总体性能模式，而不是每个特征组的精确排序。因此，本文可以合理声称“主要结论稳定”，但不应过度强调细粒度排序的稳定性。

![图11. 不同随机种子下的稳健性表现。](../outputs/figures/robustness_metric_ranges.png)

*图11. 关键模型在不同随机种子下的 ROC-AUC 稳定性。多项式 SVM 始终优于线性 SVM。*

# 5. 讨论

综合这些结果，可以得到一幅相对一致的图景。`education_num`、`age`、`hours_per_week` 以及家庭结构相关变量，在描述性分析、MI 排名和稀疏 Logistic 模型中都表现出较强信号。资本变量则形成了一个有代表性的对照：它们在高斯摘要下因为零膨胀和偏态而显得较弱，但经过变换后却在整个正则路径中持续保留。类似地，`marital_status`-`relationship` 和 `workclass`-`occupation` 这样的高 MI 特征对，更像是在反映重叠信息或共享背景，而不是提示缺失了重要交互项，因为显式测试的数值交互几乎没有带来增益。SVM 结果也与这一图景一致。线性模型已经有不错表现，但二次多项式核稳定优于线性和 RBF，这表明数据中存在有限的低阶非线性，而不是普遍而复杂的非线性结构。

# 6. 对研究问题的最终回答

## 1. 数据的底层分布是什么样的，尤其是连续特征在不同收入类别之间的分布差异如何？

连续特征的表现并不一致。`education_num`、`age` 和 `hours_per_week` 在高收入类别上整体偏高，表现出最明显的类别分离。`capital_gain` 和 `capital_loss` 则由大量零值和长右尾主导，因此明显非高斯，在简单边际高斯摘要下区分能力较弱。类别变量方面，`marital_status`、`relationship`、`occupation` 和 `gender` 也表现出明显差异。

## 2. 特征之间如何相互依赖，哪些依赖关系提示了可能有意义的交互？

特征之间确实存在较强依赖，尤其是 `marital_status`-`relationship`、`workclass`-`occupation` 和 `relationship`-`gender`。这些结果在两套保存下来的离散化方案下都很稳定。但后续交互项验证表明，筛出的数值交互项并没有带来有意义的预测提升。因此，更稳妥的结论是：数据中存在真实的依赖结构，但大多数依赖并不应直接解释为“值得加入的预测交互项”。

## 3. 哪些特征携带了真实的预测信号，哪些较弱、冗余或更像噪声？

最强且最稳健的信号来自 `marital_status`、`education_num`、`capital_gain_log1p`、`hours_per_week`、`age` 和 `capital_loss_log1p`，这些特征在完整的 L1 正则路径中都稳定存活。`occupation`、`relationship`、`workclass` 和 `gender` 也有作用，但稳定性较低。`race` 和 `native_country_grouped` 相对较弱。MI 和 L1 的对比还说明，家庭结构变量之间存在明显冗余，尤其是 `relationship` 和 `marital_status`。

## 4. 收入决策边界主要是线性的，还是存在有意义的非线性结构？

决策边界并非纯线性，但也不是强烈、广泛的非线性。线性模型已经表现良好，测试集 ROC-AUC 约为 `0.897`。二次多项式 SVM 将其提升到 `0.905`，而 RBF 核并没有超过线性基线。因此，最合理的结论是：这个问题包含有限的低阶非线性结构，而不是高复杂度的一般非线性边界。

# 7. 局限性

- 高斯分析本质上只是描述性工具。它适合做排序和比较，但并不适合 `capital_gain` 和 `capital_loss` 这类零膨胀变量。
- 保存下来的缺失值处理证据并不一致。预处理规范写明 `?` 应被视为缺失值，但实际类别输出中它仍作为显式类别出现。
- 交互项阶段刻意收缩了搜索范围，只测试了五个数值交互，而没有覆盖所有可能的数值、类别或混合交互。
- SVM 比较有助于判断边界形状，但核函数表现本身并不能指出具体是哪类非线性项在起作用。
- 所有结论都具有数据集特异性，不应直接外推到其他场景。

# 8. 结论

Adult 收入数据在两类之间确实存在清晰差异，但并不是所有差异在稀疏多变量模型中都同样重要。`education_num`、`age`、工作强度变量、资本相关变量以及家庭结构变量构成了最强的信号核心。互信息揭示了明显的依赖结构，但这些依赖很少转化为有用的显式交互项。线性模型已经解释了问题的大部分结构，而二次多项式 SVM 的稳定优势则说明数据中存在有限的低阶非线性，而不是广泛而复杂的非线性结构。稳健性检查也支持同样的整体结论。
