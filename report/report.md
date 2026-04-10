# Triangulated Evidence Analysis of the Adult Income Dataset

# Abstract

This project studies the UCI Adult / Census Income dataset as an analysis problem rather than a pure prediction task. The goal is to understand which variables visibly separate the two income classes, how features depend on each other, which signals remain robust inside sparse discriminative models, and whether the income decision boundary is mainly linear or meaningfully nonlinear. The saved workflow combines focused exploratory analysis, class-conditional Gaussian summaries for continuous variables, mutual information (MI) for dependency analysis, L1-regularized logistic regression, a small interaction-validation stage, SVM kernel comparisons, and lightweight robustness checks. The strongest continuous separators are `education_num`, `age`, and `hours_per_week`, while `capital_gain` and `capital_loss` are highly non-Gaussian and only weakly separating under a Gaussian lens. MI identifies strong dependencies such as `marital_status`-`relationship` and `workclass`-`occupation`, but the shortlisted numeric interactions do not produce meaningful predictive gains. Sparse logistic regression reaches a held-out ROC-AUC of `0.897` and consistently retains `marital_status`, `education_num`, `capital_gain_log1p`, `hours_per_week`, and `age`. In the SVM comparison, a degree-2 polynomial kernel performs best (ROC-AUC `0.905`), while linear and RBF kernels are essentially tied (`0.897` and `0.896`). Taken together, the results support a mostly well-explained problem with limited low-order nonlinearity, clear redundancy among some dependent features, and a smaller core of robust signal than the full feature set initially suggests.

# 1. Introduction

The Adult income dataset is a useful course-project setting since it mixes continuous and categorical variables, contains visible class imbalance, and supports several complementary kinds of analysis. Instead of asking only which classifier predicts best, this project asks what the data structure itself looks like and how different analytical lenses agree or disagree.

The project is organized around four research questions:

1. What do the underlying distributions of the data look like, especially for continuous features across income classes?
2. How do features depend on each other, and which dependencies suggest meaningful interactions?
3. Which features carry genuine predictive signal, and which appear weak, redundant, or noisy?
4. Is the income decision boundary mainly linear, or does it show meaningful nonlinear structure?

Questions above are deliberately linked. A feature may separate the classes marginally but still be redundant in a multivariate model. A strong dependency pair may reflect overlap or redundancy rather than a useful predictive interaction. A nonlinear kernel may improve even when a small set of hand-built interaction terms does not. The goal of the project is therefore to build a coherent story from multiple kinds of evidence rather than to rely on a single modeling result.

# 2. Dataset and Preprocessing

The raw dataset contains `48,842` rows and the standard Adult/Census Income fields. The label is the binary income outcome `income_gt_50k`, derived from whether the original `income` field is `>50K`.

Two input features were removed by design:

- `education`, because it duplicates `education_num`
- `fnlwgt`, following the project specification

The retained predictors used across the analysis were:

- Continuous: `age`, `education_num`, `capital_gain`, `capital_loss`, `hours_per_week`
- Categorical: `workclass`, `marital_status`, `occupation`, `relationship`, `race`, `gender`, `native_country`

For modeling, `capital_gain` and `capital_loss` were also represented as `log1p` transforms, and `native_country` was collapsed into `native_country_grouped`.

The main split is a stratified 80/20 train/test split with seed `42`, and the saved robustness checks reuse seeds `7`, `42`, and `99`. All tuning is done on the training split only.

The preprocessing record contains one important caveat. `missing_value_summary.csv` reports zero missing values, but the realized categorical summaries still show `?` as an explicit category for at least `workclass` and `occupation`. The report therefore describes the saved outputs as they actually behave: the missing marker appears to have remained a category in at least part of the analysis pipeline rather than being serialized as null.

The treatment of `native_country` is more consistent. The grouped feature has a raw share of roughly `89.7%` for `United-States` in the full dataset, and its grouped MI with the label is only `0.0006` (`native_country_relevance.csv`, `feature_label_mi.csv`). That makes it a weak source of signal relative to the main demographic and work-related variables.

# 3. Experimental Design

Each experiment was chosen to answer one of the research questions directly. The design follows a triangulated approach: each phase addresses a different part of the problem, and no single result is considered sufficient on its own.

- Focused EDA establishes the class balance, the main descriptive differences between groups, and any obvious shape problems such as zero inflation or dominant repeated values.
- Class-conditional Gaussian analysis provides a simple probabilistic lens on continuous features. It is not used as a claim that the data are actually Gaussian; it is used to compare means, variances, overlap, and single-feature discrimination.
- Mutual information analysis measures both feature-feature dependency and feature-label relevance. A second discretization scheme serves as a lightweight sensitivity check.
- L1 logistic regression addresses the signal-vs-noise question by asking which feature groups survive across a regularization path rather than only at one chosen model.
- Interaction validation tests a small number of candidate numeric interactions selected from MI and domain plausibility. This phase is intentionally narrow.
- Linear, polynomial, and RBF SVMs test whether nonlinear decision boundaries matter in practice, and the kernel comparison is interpreted jointly with the earlier interaction evidence.
- Robustness checks examine stability across random seeds, regularization strength, and MI discretization.


# 4. Results

## 4.1 Focused Exploratory Analysis

The training split remains substantially imbalanced: `29,724` rows (`76.1%`) are `<=50K` and `9,349` rows (`23.9%`) are `>50K` (`class_balance_summary.csv`, `class_balance.png`). This matters because accuracy alone would overstate performance, so later comparisons rely more heavily on ROC-AUC and F1.

![Figure 1. Training split class balance.](../outputs/figures/class_balance.png)

*Figure 1. Class balance in the training split. The higher-income class is the minority class throughout the analysis.*

The focused EDA reveals three shape issues that matter for later interpretation (`continuous_feature_summary.csv`, `continuous_feature_flags.csv`, `continuous_by_income_grid.png`):

- `capital_gain` is extremely sparse, with `91.8%` zeros in the training split.
- `capital_loss` is even more sparse, with `95.2%` zeros.
- `hours_per_week` has a dominant repeated value, with `46.5%` of training rows at a single value.

These are not cosmetic observations. They explain why mean-and-variance summaries are more useful for `education_num` and `age` than for the capital variables, and they also motivate the later `log1p` transforms.

The categorical summaries already suggest that the outcome is associated with household structure, work context, and education-related occupation patterns (`categorical_frequency_summary.csv`, `categorical_frequency_grid.png`):

- `Married-civ-spouse` has a `44.8%` positive-income rate, while `Never-married` has only `4.4%`.
- `Husband` and `Wife` have positive-income rates of `45.0%` and `47.5%`, while `Own-child` is only `1.6%`.
- `Male` has a `30.4%` positive-income rate versus `10.9%` for `Female`.
- `Self-emp-inc` has a `54.9%` positive-income rate, much higher than `Private` at `21.8%`.
- `Exec-managerial` and `Prof-specialty` have positive-income rates of `48.2%` and `44.6%`, while `Other-service` is only `4.0%`.

The main role of EDA here is therefore to establish which variables are likely to matter later and which variables have distributional shapes that need careful interpretation.

![Figure 2. Continuous feature distributions by income class.](../outputs/figures/continuous_by_income_grid.png)

*Figure 2. Continuous feature distributions across income classes. `education_num`, `age`, and `hours_per_week` show visible shifts, while `capital_gain` and `capital_loss` are dominated by zeros and long right tails.*

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

Three patterns matter most.

First, `education_num` is the strongest continuous discriminator by a clear margin. Its class medians also differ visibly (`9` versus `12` in `continuous_feature_summary.csv`), so its advantage is not only a mean effect.

Second, `age` and `hours_per_week` show meaningful but not clean separation. Their higher-income means are larger, but the classes still overlap substantially, especially for work hours where both classes have a median of `40`.

Third, the capital variables behave very differently from the others. Both classes have median `0` for `capital_gain` and `capital_loss`, yet the higher-income class has much larger means. That is exactly what one would expect from rare but large positive values. The Gaussian artifacts correctly treat them as non-Gaussian. Under this lens they are weak separators, but that does not imply they are useless overall; it only means the Gaussian summary is a poor description of their shape.

![Figure 4. Gaussian overlays for continuous features.](../outputs/figures/gaussian_fit_overlays.png)

*Figure 4. Empirical histograms with fitted class-conditional Gaussian curves. The Gaussian lens is reasonably informative for `education_num`, `age`, and `hours_per_week`, but much less appropriate for the capital variables.*

![Figure 5. Continuous feature separation ranking.](../outputs/figures/continuous_separation_ranking.png)

*Figure 5. Ranking of continuous features by single-feature ROC-AUC. `education_num` is the strongest marginal continuous separator.*

## 4.3 Feature Dependency Analysis

The MI analysis separates two ideas that are often conflated: dependency between features and direct relevance to the label (`feature_label_mi.csv`, `top_feature_pairs.csv`, `mi_heatmap.png`, `top_mi_pairs.png`).

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

These high-MI pairs are informative, but not all of them are interaction candidates. For example, `marital_status` and `relationship` are semantically related household-structure variables, so their very high MI likely reflects overlapping information rather than a productive multiplicative interaction. The same is true, in a different way, for `workclass` and `occupation`.

The saved candidate interaction shortlist focuses on numeric pairs instead:

- `age` x `hours_per_week`
- `age` x `education_num`
- `education_num` x `hours_per_week`
- `education_num` x `capital_gain`
- `capital_gain` x `hours_per_week`

The discretization sensitivity check is notably stable (`mi_sensitivity_summary.csv`): the top-10 MI pairs overlap `10 / 10` across the two schemes, and the candidate interaction set overlaps `5 / 5`. That makes the dependency ranking fairly robust within the limited sensitivity scope used here.

![Figure 6. Pairwise mutual information heatmap.](../outputs/figures/mi_heatmap.png)

*Figure 6. MI heatmap across the retained analysis features. The strongest cells are concentrated among household-structure and work-context variables rather than among the numeric interaction shortlist.*

![Figure 7. Top feature-feature MI pairs.](../outputs/figures/top_mi_pairs.png)

*Figure 7. Highest-MI feature pairs. Strong dependence is present, but later results show that dependence alone is not enough to justify predictive interaction terms.*

## 4.4 Signal vs Noise from Sparse Logistic Regression

The L1 logistic regression results answer two separate questions: how well a sparse linear model performs, and which feature groups remain useful as regularization changes (`logistic_cv_summary.csv`, `logistic_baseline_metrics.csv`, `stable_features_summary.csv`, `logistic_coefficient_paths.png`).

The held-out baseline metrics are:

| Accuracy | Precision | Recall | F1 | ROC-AUC | Best `C` |
| --- | --- | --- | --- | --- | --- |
| 0.848 | 0.723 | 0.589 | 0.649 | 0.897 | 1.0 |

Cross-validated ROC-AUC rises from `0.887` at the strongest regularization (`C = 0.001`) to a plateau near `0.900` once `C` reaches `0.1`, and the values from `C = 0.1` through `C = 10` differ only in the fourth decimal place. That suggests the main predictive story is not highly sensitive to the exact regularization level once moderate sparsity is allowed.

The most stable feature groups across the full `C` grid are:

| Feature group | Nonzero share across grid | Mean absolute group coefficient |
| --- | --- | --- |
| `marital_status` | 1.000 | 3.359 |
| `education_num` | 1.000 | 0.741 |
| `capital_gain_log1p` | 1.000 | 0.475 |
| `hours_per_week` | 1.000 | 0.328 |
| `age` | 1.000 | 0.317 |
| `capital_loss_log1p` | 1.000 | 0.220 |

Several additional groups remain fairly stable but not completely universal:

- `occupation`: nonzero in `88.9%` of grid points
- `relationship`, `workclass`, `gender`: each nonzero in `77.8%`
- `native_country_grouped`, `race`: each nonzero in `66.7%`

This pattern yields three substantive conclusions.

First, the strongest robust signal is not diffuse. `marital_status`, `education_num`, `age`, work-intensity, and capital-related variables form the clearest core.

Second, some variables that looked weak under the Gaussian lens remain useful in the multivariate model after transformation. In particular, `capital_gain_log1p` and `capital_loss_log1p` survive the full regularization path even though the raw features were poor Gaussian fits. This is a good example of why distributional analysis and predictive analysis answer different questions.

Third, high MI does not guarantee stability once redundancy is introduced. `relationship` has the highest feature-label MI, but `marital_status` is the more stable grouped feature in the logistic model. That suggests these two variables share overlapping information, and the sparse model can lean more heavily on one of them.

![Figure 8. L1 logistic coefficient paths.](../outputs/figures/logistic_coefficient_paths.png)

*Figure 8. Coefficient paths across the L1 regularization grid. A small set of feature groups remains active across almost the full path, supporting the robust-signal interpretation.*

## 4.5 Interaction Validation

The interaction stage tests whether a small set of targeted numeric interactions adds value beyond the main-effects logistic model (`tested_interactions.csv`, `interaction_evidence_table.csv`, `interaction_delta_cv_auc.png`).

| Interaction | Pairwise MI | Delta CV ROC-AUC | Delta test ROC-AUC | Evidence code |
| --- | --- | --- | --- | --- |
| `age` x `hours_per_week` | 0.0657 | +0.000174 | +0.000210 | unstable |
| `age` x `education_num` | 0.0517 | -0.000026 | -0.000012 | unstable |
| `education_num` x `hours_per_week` | 0.0265 | -0.000043 | -0.000003 | unstable |
| `education_num` x `capital_gain` | 0.0160 | -0.000025 | +0.000057 | unstable |
| `capital_gain` x `hours_per_week` | 0.0077 | +0.000042 | -0.000066 | unstable |

Two facts are decisive here.

First, the predictive changes are tiny. Even the largest positive change in cross-validated ROC-AUC is only `+0.000174`, and the largest positive test change is only `+0.000210`. Those are too small to justify claiming a meaningful improvement.

Second, no interaction survived screening strongly enough to form a final augmented model. That is why `interaction_model_metrics.csv` contains `NaN` values for the `interaction_augmented` row rather than a fitted final model.

This phase therefore supports a restrained conclusion: the saved dependency analysis did identify plausible candidate interactions, but within this small and defensible test set, none provided robust predictive value beyond the main effects.

![Figure 9. Incremental interaction effects on CV ROC-AUC.](../outputs/figures/interaction_delta_cv_auc.png)

*Figure 9. Change in cross-validated ROC-AUC from adding each tested interaction. All changes are very small, which is why no interaction-augmented final model was retained.*

## 4.6 Decision Boundary Analysis with SVM

The SVM comparison provides the main evidence about boundary shape (`svm_tuning_summary.csv`, `svm_comparison.csv`, `svm_kernel_comparison.png`, `final_model_comparison_table.csv`).

| Model family | Best tuning setting | CV ROC-AUC | Test ROC-AUC | Test F1 |
| --- | --- | --- | --- | --- |
| Linear SVM | `C = 0.1` | 0.893 | 0.897 | 0.644 |
| Polynomial SVM | `degree = 2`, `C = 1.0`, `coef0 = 1.0` | 0.898 | 0.905 | 0.670 |
| RBF SVM | `C = 3.0`, `gamma = 0.05` | 0.892 | 0.896 | 0.670 |

The most important comparison is not simply “which model won,” but how the pattern of wins should be interpreted.

- The polynomial kernel improves on the linear kernel by about `0.0078` ROC-AUC on the held-out test set (`0.905` versus `0.897`).
- The RBF kernel does not improve on the linear kernel. Its test ROC-AUC is slightly lower (`0.896`).
- Within the polynomial family, the best setting is a degree-2 model. The saved degree-3 settings are materially worse in cross-validation (`0.883` and `0.883`) than the best degree-2 setting (`0.898`).

Taken together, that pattern suggests limited low-order nonlinearity rather than strong general nonlinear complexity. A purely linear boundary is not the whole story, but neither is there evidence that a highly flexible kernel is necessary.

This result also fits the earlier interaction evidence in a nuanced way. The polynomial kernel improvement is consistent with some low-order nonlinear structure, but the manually selected numeric product terms did not help. The most plausible interpretation is not that the two analyses contradict each other. Rather, the polynomial kernel may be capturing a broader set of low-order effects, including squared terms and interactions spread across the full encoded feature space, while the explicit interaction search tested only five numeric products.

![Figure 10. Held-out SVM kernel comparison.](../outputs/figures/svm_kernel_comparison.png)

*Figure 10. Held-out ROC-AUC by SVM family. The polynomial kernel performs best, while RBF does not improve on the linear baseline.*

## 4.7 Robustness Checks

The robustness phase combines three kinds of stability evidence: regularization stability, MI discretization stability, and split-seed stability.

### Regularization stability

The logistic CV curve is very flat once `C` reaches `0.1` (`logistic_cv_summary.csv`):

- `C = 0.1`: mean CV ROC-AUC `0.899822`
- `C = 0.3162`: mean CV ROC-AUC `0.900060`
- `C = 1.0`: mean CV ROC-AUC `0.900099`
- `C = 10.0`: mean CV ROC-AUC `0.900054`

This indicates that the overall logistic conclusion is not highly sensitive to the exact chosen regularization strength.

### MI discretization stability

The saved `mi_sensitivity_summary.csv` table shows complete overlap in the top-10 feature pairs and complete overlap in the candidate interaction set across the two discretization schemes:

- Top-10 pair overlap: `10 / 10`
- Candidate interaction overlap: `5 / 5`

That makes the dependency story one of the more stable parts of the project.

### Seed stability

The saved test ROC-AUC values across seeds are:

| Model | Seed 7 | Seed 42 | Seed 99 | Range |
| --- | --- | --- | --- | --- |
| L1 logistic | 0.901 | 0.897 | 0.901 | 0.897 to 0.901 |
| Linear SVM | 0.900 | 0.897 | 0.901 | 0.897 to 0.901 |
| Polynomial SVM | 0.906 | 0.905 | 0.906 | 0.905 to 0.906 |

The polynomial kernel remains better than the linear kernel at every saved seed:

- Seed `7`: +`0.0063`
- Seed `42`: +`0.0078`
- Seed `99`: +`0.0045`

This supports the direction of the main boundary-shape conclusion. At the same time, the saved robustness tables (`robust_feature_group_frequency.csv`, `robustness_boundary_consistency.csv`, `robustness_mi_overlap.csv`) suggest that exact feature-importance rankings are less central than the headline performance metrics, so the project should claim stability for the broad conclusions rather than for a perfectly fixed ordering of every feature group.

![Figure 11. Robustness across split seeds.](../outputs/figures/robustness_metric_ranges.png)

*Figure 11. ROC-AUC stability across random seeds for the key models. The ordering of polynomial SVM above linear SVM is consistent across the saved robustness runs.*

# 5. Discussion

The main contribution of the project is not any single metric table but the way different methods narrow the interpretation.

The strongest agreement appears around `education_num`, `age`, `hours_per_week`, and household-structure variables. `education_num` is the clearest continuous separator under the Gaussian summary, has substantial feature-label MI, and survives the full logistic regularization path. `age` and `hours_per_week` follow a similar pattern, though with more overlap. `marital_status` and `relationship` dominate the categorical dependence story, and at least one of them remains central under sparsity.

The clearest example of productive disagreement is the capital variables. Under the Gaussian analysis, `capital_gain` and `capital_loss` are poor fits and weak marginal separators because both classes are dominated by zeros. Under the sparse logistic model, however, the transformed capital features survive across the entire regularization path. This is a useful lesson: a feature can be badly summarized by a Gaussian approximation and still be a real predictive contributor after transformation and in combination with other features.

The dependency analysis also becomes more informative when compared against later phases. `marital_status`-`relationship` and `workclass`-`occupation` are strong MI pairs, but they do not imply that explicit interaction terms are needed. In fact, the tested numeric interactions fail to produce meaningful lift. This shows that strong dependence can signal redundancy or shared context rather than additive predictive value from cross-terms.

The SVM results help resolve the linear-versus-nonlinear question. A linear model is already competitive: the sparse logistic model and the linear SVM both reach test ROC-AUC values near `0.897`. That rules out the claim that the problem requires highly complex nonlinear modeling. But the polynomial SVM improves consistently over both linear and RBF alternatives, which argues against a strictly linear interpretation. The best evidence is therefore for modest low-order nonlinearity, not strong nonlinear complexity.

The different methods also disagree in a useful way about feature importance ranking. `relationship` has the highest MI with the label, but `marital_status` is more stable across the logistic regularization path. This suggests overlap rather than contradiction: both variables encode related household structure, but the sparse discriminative model can rely more heavily on one grouped representation when forced to simplify.

Overall, the triangulated design works as intended. The probabilistic summary, MI analysis, sparse linear model, explicit interaction test, and kernel comparison do not say exactly the same thing, but together they produce a tighter and more believable story than any of them could alone.

# 6. Final Answers to the Research Questions

## 1. What do the underlying distributions of the data look like, especially for continuous features across income classes?

The continuous features are heterogeneous. `education_num`, `age`, and `hours_per_week` show the clearest class separation, with higher-income observations shifted upward on average. `capital_gain` and `capital_loss` are dominated by zeros and heavy right tails, so they are visibly non-Gaussian and only weakly separating under a simple marginal Gaussian analysis. The categorical summaries also show strong class differences for `marital_status`, `relationship`, `occupation`, and `gender`.

## 2. How do features depend on each other, and which dependencies suggest meaningful interactions?

Several feature pairs are strongly dependent, especially `marital_status`-`relationship`, `workclass`-`occupation`, and `relationship`-`gender`. These dependencies are stable across the two saved discretization schemes. However, the later interaction stage shows that the shortlisted numeric interactions do not produce meaningful predictive lift. The strongest conclusion is therefore that the dataset contains real dependency structure, but most of it should not be interpreted as evidence for useful explicit interaction terms.

## 3. Which features carry genuine predictive signal, and which appear weak, redundant, or noisy?

The strongest robust signals are `marital_status`, `education_num`, `capital_gain_log1p`, `hours_per_week`, `age`, and `capital_loss_log1p`, all of which survive the full L1 regularization path. `occupation`, `relationship`, `workclass`, and `gender` remain useful but somewhat less stable. `race` and `native_country_grouped` are comparatively weak. The comparison between MI and L1 also suggests redundancy among some household-structure variables, especially `relationship` and `marital_status`.

## 4. Is the income decision boundary mainly linear, or does it show meaningful nonlinear structure?

The boundary is not purely linear, but it is also not strongly nonlinear in a general sense. Linear models already perform well, with test ROC-AUC around `0.897`. A degree-2 polynomial SVM improves this to `0.905`, while the RBF kernel does not beat the linear baseline. The most defensible conclusion is that the problem contains modest low-order nonlinear structure rather than strong high-complexity nonlinear behavior.

# 7. Limitations

- The Gaussian analysis is intentionally a descriptive lens. It is useful for ranking and comparison, but it is a poor fit for zero-inflated variables such as `capital_gain` and `capital_loss`.
- The saved missing-value artifacts are inconsistent. The preprocessing specification describes `?` handling, but the realized categorical summaries still show `?` as explicit categories. That means the report must interpret the saved outputs rather than assume the intended pipeline was fully realized.
- The interaction stage is deliberately narrow. It tests five numeric interactions, not all possible mixed or categorical interactions, so a failure to improve does not prove that no useful interaction exists anywhere in the feature space.
- The SVM comparison is informative about boundary shape, but kernel performance does not identify which exact nonlinear terms matter.
- All conclusions are dataset-specific. They describe structure in the Adult dataset and should not be generalized beyond this setting without additional evidence.

# 8. Conclusion

The Adult income dataset contains clear descriptive differences across the two income classes, but only a subset of those differences remain central after multivariate sparse modeling. `education_num`, `age`, work-intensity, capital-related variables, and household-structure features form the strongest signal core. Mutual information reveals substantial dependency structure, yet those dependencies rarely translate into useful explicit interaction terms. Linear models already explain much of the problem, but a degree-2 polynomial SVM performs consistently better, indicating limited low-order nonlinearity rather than strong general nonlinear complexity. The robustness checks reinforce the broad conclusions: the main performance patterns and the main MI rankings are stable, while finer-grained feature rankings should be interpreted more cautiously.
