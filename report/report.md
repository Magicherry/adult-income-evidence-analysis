# Triangulated Evidence Analysis of the Adult Income Dataset

# Abstract

This project examines the UCI Adult / Census Income dataset as a data analysis problem, not just a prediction benchmark. The main questions are which variables separate the two income groups, how the features relate to one another, which signals remain after sparsity is enforced, and whether the classification boundary is mostly linear or meaningfully nonlinear. To answer these questions, the workflow combines focused exploratory analysis, class-conditional Gaussian summaries for continuous variables, mutual information (MI), L1-regularized logistic regression, a small interaction test, SVM kernel comparisons, and lightweight robustness checks. The clearest continuous separators are `education_num`, `age`, and `hours_per_week`. By contrast, `capital_gain` and `capital_loss` are strongly non-Gaussian and look weak under a Gaussian summary. MI highlights strong dependencies such as `marital_status`-`relationship` and `workclass`-`occupation`, but the shortlisted numeric interactions do not improve prediction in a meaningful way. The sparse logistic model reaches a held-out ROC-AUC of `0.897` and consistently keeps `marital_status`, `education_num`, `capital_gain_log1p`, `hours_per_week`, and `age`. Among the SVMs, a degree-2 polynomial kernel performs best with ROC-AUC `0.905`, while linear and RBF kernels are essentially tied at `0.897` and `0.896`. Overall, the results suggest that most of the structure in this problem is already captured by a relatively simple model, with some limited low-order nonlinearity and clear redundancy among several dependent features.

# 1. Introduction

The Adult income dataset is a good fit for our course project because it combines continuous and categorical variables, has visible class imbalance, and supports several different kinds of analysis. Rather than asking only which classifier gives the best score, this project focuses on the structure of the data itself and on whether different methods tell a consistent story.

The analysis is organized around four research questions:

1. What do the underlying distributions of the data look like, especially for continuous features across income classes?
2. How do features depend on each other, and which dependencies suggest meaningful interactions?
3. Which features carry genuine predictive signal, and which appear weak, redundant, or noisy?
4. Is the income decision boundary mainly linear, or does it show meaningful nonlinear structure?

A feature can look useful on its own and still become redundant in a multivariate model. A strong dependency between two features may reflect overlap rather than a useful interaction. For that reason, the project uses several complementary methods and treats the final interpretation as a synthesis of evidence rather than a conclusion drawn from one model alone.

# 2. Dataset and Preprocessing

The raw dataset contains `48,842` rows with the standard Adult/Census Income fields. The target is the binary label `income_gt_50k`, derived from whether the original `income` field is `>50K`.

Two input features were removed by design:

- `education`, because it duplicates `education_num`
- `fnlwgt`, following the project specification

The retained predictors are:

- Continuous: `age`, `education_num`, `capital_gain`, `capital_loss`, `hours_per_week`
- Categorical: `workclass`, `marital_status`, `occupation`, `relationship`, `race`, `gender`, `native_country`

For modeling, `capital_gain` and `capital_loss` are also represented through `log1p` transforms, and `native_country` is collapsed into `native_country_grouped`.

The main split is a stratified 80/20 train/test split with seed `42`. The robustness checks reuse seeds `7`, `42`, and `99`. All model tuning is restricted to the training split.

One preprocessing detail needs to be stated clearly. `missing_value_summary.csv` reports zero missing values, but the saved categorical summaries still show `?` as an explicit category in fields such as `workclass` and `occupation`. This report therefore describes the saved outputs as they actually appear, rather than assuming the intended missing-value handling was applied everywhere in the pipeline.

The handling of `native_country` is more straightforward. In the full dataset, roughly `89.7%` of rows fall under `United-States`, and the grouped feature has label MI of only `0.0006` (`native_country_relevance.csv`, `feature_label_mi.csv`). Relative to the main demographic and work-related variables, it contributes little signal.

# 3. Experimental Design

The overall design is triangulated: every stage gives a different view of the dataset, and no single result is treated as complete evidence by itself.

- Focused EDA is used to establish class balance, summarize the main feature patterns, and flag obvious shape issues such as zero inflation or dominant repeated values.
- Class-conditional Gaussian analysis provides a simple descriptive view of the continuous features. It is used for comparison, not as a claim that the data are actually Gaussian.
- Mutual information analysis measures both feature-feature dependence and feature-label relevance. A second discretization scheme is included as a small sensitivity check.
- L1 logistic regression is used to study sparse predictive signal and to see which feature groups survive across a regularization path.
- Interaction validation tests a small set of numeric interactions selected from MI evidence and domain plausibility.
- Linear, polynomial, and RBF SVMs are compared to assess whether nonlinear decision boundaries matter in practice.
- Robustness checks examine how stable the main conclusions are across random seeds, regularization strength, and MI discretization choices.

# 4. Results

## 4.1 Focused Exploratory Analysis

The training split is noticeably imbalanced: `29,724` rows (`76.1%`) are `<=50K`, while `9,349` rows (`23.9%`) are `>50K` (`class_balance_summary.csv`, `class_balance.png`). Because of this imbalance, later comparisons rely more on ROC-AUC and F1 than on accuracy alone.

![Figure 1. Training split class balance.](../outputs/figures/class_balance.png)

*Figure 1. Class balance in the training split. The higher-income class remains the minority throughout the analysis.*

The exploratory summaries point to three distributional issues that matter later (`continuous_feature_summary.csv`, `continuous_feature_flags.csv`, `continuous_by_income_grid.png`):

- `capital_gain` is extremely sparse, with `91.8%` zeros in the training split.
- `capital_loss` is even sparser, with `95.2%` zeros.
- `hours_per_week` has a dominant repeated value, with `46.5%` of training rows at one value.

These patterns help explain why mean-and-variance summaries are more informative for `education_num` and `age` than for the capital variables. They also motivate the later `log1p` transforms.

The categorical summaries already show strong differences across income groups (`categorical_frequency_summary.csv`, `categorical_frequency_grid.png`):

- `Married-civ-spouse` has a `44.8%` positive-income rate, while `Never-married` has only `4.4%`.
- `Husband` and `Wife` have positive-income rates of `45.0%` and `47.5%`, while `Own-child` is only `1.6%`.
- `Male` has a `30.4%` positive-income rate versus `10.9%` for `Female`.
- `Self-emp-inc` has a `54.9%` positive-income rate, much higher than `Private` at `21.8%`.
- `Exec-managerial` and `Prof-specialty` have positive-income rates of `48.2%` and `44.6%`, while `Other-service` is only `4.0%`.

![Figure 2. Continuous feature distributions by income class.](../outputs/figures/continuous_by_income_grid.png)

*Figure 2. Continuous feature distributions across income classes. `education_num`, `age`, and `hours_per_week` shift visibly, while `capital_gain` and `capital_loss` are dominated by zeros and long right tails.*

![Figure 3. Key categorical frequency summaries.](../outputs/figures/categorical_frequency_grid.png)

*Figure 3. Selected categorical distributions with positive-income rates. Household structure and occupation-related variables already show large differences before formal modeling.*

## 4.2 Class-Conditional Distribution Analysis

The Gaussian summary ranks the continuous features by single-feature ROC-AUC and overlap statistics (`gaussian_fit_summary.csv`, `ranked_continuous_features.csv`, `gaussian_fit_overlays.png`, `continuous_separation_ranking.png`).

| Feature | Mean `<=50K` | Mean `>50K` | Single-feature ROC-AUC | Cohen's d | Interpretation |
| --- | --- | --- | --- | --- | --- |
| `education_num` | 9.600 | 11.603 | 0.716 | 0.825 | clearest continuous separator |
| `age` | 36.919 | 44.350 | 0.682 | 0.556 | moderate separation with visible overlap |
| `hours_per_week` | 38.892 | 45.434 | 0.671 | 0.541 | moderate separation, but both classes still center around 40 hours |
| `capital_gain` | 147.696 | 3949.978 | 0.588 | 0.532 | highly skewed and zero-inflated; weak under the Gaussian lens |
| `capital_loss` | 55.346 | 199.978 | 0.535 | 0.358 | highly skewed and weak under the Gaussian lens |

`education_num` is clearly the strongest continuous discriminator. Its class medians also differ visibly (`9` versus `12` in `continuous_feature_summary.csv`), so the pattern is not just a small mean shift.

`age` and `hours_per_week` show meaningful but incomplete separation. Higher-income observations tend to be older and work more hours, but the class overlap is still substantial, especially for work hours where both groups have a median of `40`.

The capital variables behave differently from the rest. Both classes have median `0` for `capital_gain` and `capital_loss`, but the higher-income class has much larger means. That pattern is consistent with rare but very large positive values. The Gaussian view captures part of the story, but it is clearly a poor summary of their full shape.

![Figure 4. Gaussian overlays for continuous features.](../outputs/figures/gaussian_fit_overlays.png)

*Figure 4. Empirical histograms with fitted class-conditional Gaussian curves. The Gaussian view is fairly reasonable for `education_num`, `age`, and `hours_per_week`, but much less so for the capital variables.*

![Figure 5. Continuous feature separation ranking.](../outputs/figures/continuous_separation_ranking.png)

*Figure 5. Ranking of continuous features by single-feature ROC-AUC. `education_num` is the strongest marginal continuous separator.*

## 4.3 Feature Dependency Analysis

The MI analysis separates two related but distinct ideas: dependence between features and direct relevance to the label (`feature_label_mi.csv`, `top_feature_pairs.csv`, `mi_heatmap.png`, `top_mi_pairs.png`).

The highest feature-label MI values are:

| Feature | Feature-label MI |
| --- | --- |
| `relationship` | 0.116 |
| `marital_status` | 0.111 |
| `age` | 0.064 |
| `occupation` | 0.063 |
| `education_num` | 0.062 |
| `capital_gain` | 0.054 |

The strongest feature-feature dependencies are:

| Feature pair | Pairwise MI |
| --- | --- |
| `marital_status` x `relationship` | 0.725 |
| `workclass` x `occupation` | 0.329 |
| `relationship` x `gender` | 0.270 |
| `age` x `marital_status` | 0.233 |
| `education_num` x `occupation` | 0.213 |

These strong pairs are informative, but they should not automatically be read as interaction candidates. For example, `marital_status` and `relationship` are both tied to household structure, so their high MI is more likely telling us that they overlap heavily than that they form a useful multiplicative effect. A similar point applies to `workclass` and `occupation`.

The saved interaction shortlist therefore focuses on numeric pairs:

- `age` x `hours_per_week`
- `age` x `education_num`
- `education_num` x `hours_per_week`
- `education_num` x `capital_gain`
- `capital_gain` x `hours_per_week`

The discretization sensitivity check is very stable (`mi_sensitivity_summary.csv`): the top-10 MI pairs overlap `10 / 10` across the two schemes, and the candidate interaction set overlaps `5 / 5`. Within the scope of this check, the dependency ranking is quite robust.

![Figure 6. Pairwise mutual information heatmap.](../outputs/figures/mi_heatmap.png)

*Figure 6. MI heatmap across the retained analysis features. The strongest dependencies are concentrated in household-structure and work-context variables rather than in the numeric interaction shortlist.*

![Figure 7. Top feature-feature MI pairs.](../outputs/figures/top_mi_pairs.png)

*Figure 7. Highest-MI feature pairs. Strong dependence is present, but later results show that dependence alone does not justify adding explicit interaction terms.*

## 4.4 Signal vs Noise from Sparse Logistic Regression

The L1 logistic regression results help answer two questions at once: how well a sparse linear model performs, and which feature groups keep contributing as regularization changes (`logistic_cv_summary.csv`, `logistic_baseline_metrics.csv`, `stable_features_summary.csv`, `logistic_coefficient_paths.png`).

The held-out baseline metrics are:

| Accuracy | Precision | Recall | F1 | ROC-AUC | Best `C` |
| --- | --- | --- | --- | --- | --- |
| 0.848 | 0.723 | 0.589 | 0.649 | 0.897 | 1.0 |

Cross-validated ROC-AUC rises from `0.887` at the strongest regularization (`C = 0.001`) to a plateau around `0.900` once `C` reaches `0.1`. From `C = 0.1` through `C = 10`, the scores differ only in the fourth decimal place. In practice, that means the overall logistic result is not very sensitive to the exact regularization level once moderate sparsity is allowed.

The most stable feature groups across the full `C` grid are:

| Feature group | Nonzero share across grid | Mean absolute group coefficient |
| --- | --- | --- |
| `marital_status` | 1.000 | 3.359 |
| `education_num` | 1.000 | 0.741 |
| `capital_gain_log1p` | 1.000 | 0.475 |
| `hours_per_week` | 1.000 | 0.328 |
| `age` | 1.000 | 0.317 |
| `capital_loss_log1p` | 1.000 | 0.220 |

Several other groups remain useful, but less consistently:

- `occupation`: nonzero in `88.9%` of grid points
- `relationship`, `workclass`, `gender`: each nonzero in `77.8%`
- `native_country_grouped`, `race`: each nonzero in `66.7%`

The strongest signal is fairly concentrated rather than spread evenly across the full feature set. `marital_status`, `education_num`, `age`, work intensity, and the capital-related variables form the main core.

Some variables that looked weak under the Gaussian summary still matter in the multivariate model after transformation. `capital_gain_log1p` and `capital_loss_log1p` survive the full regularization path even though the raw variables were poor Gaussian fits.

High MI does not guarantee stability once redundancy is present. `relationship` has the highest feature-label MI, but `marital_status` is the more stable grouped feature in the sparse logistic model. The most likely explanation is that they carry overlapping information and the model can rely more heavily on one grouped representation than the other.

![Figure 8. L1 logistic coefficient paths.](../outputs/figures/logistic_coefficient_paths.png)

*Figure 8. Coefficient paths across the L1 regularization grid. A relatively small set of feature groups remains active across most of the path.*

## 4.5 Interaction Validation

The interaction stage asks whether a small set of targeted numeric interactions improves on the main-effects logistic model (`tested_interactions.csv`, `interaction_evidence_table.csv`, `interaction_delta_cv_auc.png`).

| Interaction | Pairwise MI | Delta CV ROC-AUC | Delta test ROC-AUC | Evidence code |
| --- | --- | --- | --- | --- |
| `age` x `hours_per_week` | 0.0657 | +0.000174 | +0.000210 | unstable |
| `age` x `education_num` | 0.0517 | -0.000026 | -0.000012 | unstable |
| `education_num` x `hours_per_week` | 0.0265 | -0.000043 | -0.000003 | unstable |
| `education_num` x `capital_gain` | 0.0160 | -0.000025 | +0.000057 | unstable |
| `capital_gain` x `hours_per_week` | 0.0077 | +0.000042 | -0.000066 | unstable |

The key point is that every change is tiny. The largest positive change in cross-validated ROC-AUC is only `+0.000174`, and the largest positive change on the test set is only `+0.000210`. Those gains are too small to support any claim of a meaningful improvement.

No interaction survives strongly enough to justify a final augmented model, which is why `interaction_model_metrics.csv` reports `NaN` for the `interaction_augmented` row instead of a retained fitted model.

So the conclusion here is that the MI screen produced plausible candidates, but within this limited and reasonable test set, none of the explicit numeric interactions added robust predictive value beyond the main effects.

![Figure 9. Incremental interaction effects on CV ROC-AUC.](../outputs/figures/interaction_delta_cv_auc.png)

*Figure 9. Change in cross-validated ROC-AUC after adding each tested interaction. All effects are very small, so no interaction-augmented final model was kept.*

## 4.6 Decision Boundary Analysis with SVM

The SVM comparison provides the clearest evidence about boundary shape (`svm_tuning_summary.csv`, `svm_comparison.csv`, `svm_kernel_comparison.png`, `final_model_comparison_table.csv`).

| Model family | Best tuning setting | CV ROC-AUC | Test ROC-AUC | Test F1 |
| --- | --- | --- | --- | --- |
| Linear SVM | `C = 0.1` | 0.893 | 0.897 | 0.644 |
| Polynomial SVM | `degree = 2`, `C = 1.0`, `coef0 = 1.0` | 0.898 | 0.905 | 0.670 |
| RBF SVM | `C = 3.0`, `gamma = 0.05` | 0.892 | 0.896 | 0.670 |

- The polynomial kernel improves on the linear kernel by about `0.0078` ROC-AUC on the held-out test set (`0.905` versus `0.897`).
- The RBF kernel does not improve on the linear baseline; its test ROC-AUC is slightly lower at `0.896`.
- Within the polynomial family, degree `2` works best. The saved degree-3 settings are clearly worse in cross-validation (`0.883` and `0.883`) than the best degree-2 result (`0.898`).

This points to limited low-order nonlinearity rather than broad nonlinear complexity. A purely linear boundary is not the full story, but neither is there evidence that a highly flexible kernel is necessary.

This interpretation also fits the earlier interaction results. The polynomial kernel may be capturing a wider range of low-order effects across the encoded feature space, including squared terms and interactions that were not covered by the five explicit numeric products tested earlier.

![Figure 10. Held-out SVM kernel comparison.](../outputs/figures/svm_kernel_comparison.png)

*Figure 10. Held-out ROC-AUC by SVM family. The polynomial kernel performs best, while RBF does not improve on the linear baseline.*

## 4.7 Robustness Checks

### Regularization stability

The logistic CV curve is very flat once `C` reaches `0.1` (`logistic_cv_summary.csv`):

- `C = 0.1`: mean CV ROC-AUC `0.899822`
- `C = 0.3162`: mean CV ROC-AUC `0.900060`
- `C = 1.0`: mean CV ROC-AUC `0.900099`
- `C = 10.0`: mean CV ROC-AUC `0.900054`

This supports the earlier point that the logistic conclusion is not very sensitive to the exact regularization choice.

### MI discretization stability

The saved `mi_sensitivity_summary.csv` table shows complete overlap in the top-10 feature pairs and complete overlap in the candidate interaction set across the two discretization schemes:

- Top-10 pair overlap: `10 / 10`
- Candidate interaction overlap: `5 / 5`

Within this project, the MI dependency picture is one of the more stable results.

### Seed stability

The saved test ROC-AUC values across seeds are:

| Model | Seed 7 | Seed 42 | Seed 99 | Range |
| --- | --- | --- | --- | --- |
| L1 logistic | 0.901 | 0.897 | 0.901 | 0.897 to 0.901 |
| Linear SVM | 0.900 | 0.897 | 0.901 | 0.897 to 0.901 |
| Polynomial SVM | 0.906 | 0.905 | 0.906 | 0.905 to 0.906 |

The polynomial kernel stays ahead of the linear kernel at every saved seed:

- Seed `7`: +`0.0063`
- Seed `42`: +`0.0078`
- Seed `99`: +`0.0045`

That strengthens the main boundary-shape conclusion. At the same time, the saved robustness tables (`robust_feature_group_frequency.csv`, `robustness_boundary_consistency.csv`, `robustness_mi_overlap.csv`) suggest that the broad conclusions are more stable than the exact ordering of every feature group, so the report should be careful not to overclaim fine-grained stability.

![Figure 11. Robustness across split seeds.](../outputs/figures/robustness_metric_ranges.png)

*Figure 11. ROC-AUC stability across random seeds for the key models. The ordering of polynomial SVM above linear SVM is consistent across the saved robustness runs.*

# 5. Discussion

Taken together, the methods point to a fairly consistent picture. `education_num`, `age`, `hours_per_week`, and the household-structure variables carry the clearest signal across the descriptive summaries, MI rankings, and sparse logistic model. The capital variables are a useful contrast: they look weak under the Gaussian summary because of zero inflation and skew, but their transformed versions remain active across the full regularization path. Strong MI pairs such as `marital_status`-`relationship` and `workclass`-`occupation` mostly look like overlap or shared context rather than missing interaction terms, since the tested numeric interactions add almost no lift. The SVM results also fit this picture. Linear models already perform well, but the degree-2 polynomial kernel improves consistently over both the linear and RBF alternatives, which points to modest low-order nonlinearity rather than strong general nonlinear complexity.

# 6. Final Answers to the Research Questions

## 1. What do the underlying distributions of the data look like, especially for continuous features across income classes?

The continuous features do not all behave the same way. `education_num`, `age`, and `hours_per_week` show the clearest class separation, with higher-income observations shifted upward on average. `capital_gain` and `capital_loss` are dominated by zeros and long right tails, so they are strongly non-Gaussian and look weak under a simple marginal Gaussian summary. Among the categorical variables, the largest class differences appear in `marital_status`, `relationship`, `occupation`, and `gender`.

## 2. How do features depend on each other, and which dependencies suggest meaningful interactions?

Several feature pairs are strongly dependent, especially `marital_status`-`relationship`, `workclass`-`occupation`, and `relationship`-`gender`. These results are stable across the two saved discretization schemes. However, the later interaction stage shows that the shortlisted numeric interactions do not produce meaningful predictive lift. The stronger conclusion is that the dataset contains real dependency structure, but most of it should not be interpreted as evidence for useful explicit interaction terms.

## 3. Which features carry genuine predictive signal, and which appear weak, redundant, or noisy?

The strongest robust signals are `marital_status`, `education_num`, `capital_gain_log1p`, `hours_per_week`, `age`, and `capital_loss_log1p`, all of which survive the full L1 regularization path. `occupation`, `relationship`, `workclass`, and `gender` remain useful but less stable. `race` and `native_country_grouped` are relatively weak. The comparison between MI and L1 also suggests redundancy among the household-structure variables, especially `relationship` and `marital_status`.

## 4. Is the income decision boundary mainly linear, or does it show meaningful nonlinear structure?

The boundary is not purely linear, but it is also not strongly nonlinear in a broad sense. Linear models already perform well, with test ROC-AUC around `0.897`. A degree-2 polynomial SVM improves this to `0.905`, while the RBF kernel does not beat the linear baseline. The most defensible conclusion is that the problem contains modest low-order nonlinear structure rather than strong high-complexity nonlinearity.

# 7. Limitations

- The Gaussian analysis is only a descriptive lens. It is useful for comparison, but it is a poor fit for zero-inflated variables such as `capital_gain` and `capital_loss`.
- The saved missing-value artifacts are inconsistent. The preprocessing specification describes `?` handling, but the realized categorical summaries still show `?` as explicit categories.
- The interaction stage is deliberately narrow. It tests five numeric interactions, not the full space of possible mixed or categorical interactions.
- The SVM comparison is informative about boundary shape, but it does not identify which exact nonlinear terms matter.
- All conclusions are specific to the Adult dataset and should not be generalized beyond this setting without further evidence.

# 8. Conclusion

The Adult income dataset shows clear differences between the two income classes, but only part of that structure remains central in sparse multivariate models. `education_num`, `age`, work intensity, capital-related variables, and household-structure features make up the strongest signal core. MI reveals substantial dependency structure, yet those dependencies rarely translate into useful explicit interaction terms. Linear models already explain much of the problem, while a degree-2 polynomial SVM performs consistently better, pointing to limited low-order nonlinearity rather than broad nonlinear complexity. The robustness checks support the same overall picture.
