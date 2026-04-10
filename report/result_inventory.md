# Result Inventory

## Artifact Availability

The project currently has a complete saved artifact set for the notebook-driven experiment workflow.

- Figures found: 12 files in `outputs/figures/`
- Tables found: 30 files in `outputs/tables/`
- Metrics files found: 5 files in `outputs/metrics/`
- Interpretation notes found: 0 files in `outputs/text/` because that directory does not exist in the current version
- Logs found: 0 files in `outputs/logs/` because that directory is no longer created
- Report files currently present: `report.md`, `report_zh.md`, `result_inventory.md`

Major expected artifacts were present:

- `audit_summary.csv`
- `continuous_feature_flags.csv`
- `mi_sensitivity_summary.csv`
- `interaction_evidence_table.csv`
- `final_model_comparison_table.csv`
- `robust_feature_group_frequency.csv`
- `robustness_metric_ranges.csv`
- `robustness_boundary_consistency.csv`
- `robustness_mi_overlap.csv`
- `final_summary_table.csv`

## Current Caveats

- `missing_value_summary.csv` reports zero missing values because raw `?` markers were preserved as category values rather than serialized as nulls.
- `interaction_model_metrics.csv` includes an `interaction_augmented` row with `NaN` metrics. In the current pipeline this means no tested interaction survived screening strongly enough to build a final augmented model.
- Several current outputs were introduced after the earlier report draft, so older prose references to `outputs/text/*.md` interpretation notes are no longer aligned with the notebook artifact set.

## Result Inventory Table

| Experiment | Input files used for write-up | Output files found | Main usable findings |
| --- | --- | --- | --- |
| Data audit and preprocessing | `outputs/tables/data_dictionary.csv`, `outputs/tables/retained_features.csv`, `outputs/tables/missing_value_summary.csv`, `outputs/tables/native_country_relevance.csv`, `outputs/tables/audit_summary.csv` | All found | Dataset has 48,842 rows; the main split seed is 42; `education` and `fnlwgt` were removed; `native_country` is retained in grouped form; `United-States` accounts for 89.7% of rows. |
| Focused EDA | `outputs/tables/class_balance_summary.csv`, `outputs/tables/continuous_feature_summary.csv`, `outputs/tables/categorical_frequency_summary.csv`, `outputs/tables/continuous_feature_flags.csv`, `outputs/figures/class_balance.png`, `outputs/figures/continuous_by_income_grid.png`, `outputs/figures/categorical_frequency_grid.png` | All found | Training split is imbalanced (`<=50K` 76.1%, `>50K` 23.9%); continuous feature flags are saved as structured codes in `continuous_feature_flags.csv`; categorical frequency summaries remain available for group-level inspection. |
| Class-conditional Gaussian analysis | `outputs/tables/gaussian_fit_summary.csv`, `outputs/tables/ranked_continuous_features.csv`, `outputs/figures/gaussian_fit_overlays.png`, `outputs/figures/continuous_separation_ranking.png` | All found | `education_num`, `age`, and `hours_per_week` have the highest single-feature ROC-AUC values; `capital_gain` and `capital_loss` are flagged as `non_gaussian_weak`. |
| Mutual information analysis | `outputs/tables/mi_matrix.csv`, `outputs/tables/feature_label_mi.csv`, `outputs/tables/top_feature_pairs.csv`, `outputs/tables/candidate_interactions.csv`, `outputs/tables/top_feature_pairs_sensitivity.csv`, `outputs/tables/candidate_interactions_sensitivity.csv`, `outputs/tables/mi_sensitivity_summary.csv`, `outputs/figures/mi_heatmap.png`, `outputs/figures/top_mi_pairs.png`, `outputs/figures/feature_label_mi_ranking.png` | All found | Top feature-label MI comes from `relationship`, `marital_status`, `age`, `occupation`, and `education_num`; overlap counts across baseline and sensitivity discretizations are saved in `mi_sensitivity_summary.csv`. |
| Sparse logistic regression | `outputs/metrics/logistic_cv_summary.csv`, `outputs/metrics/logistic_baseline_metrics.csv`, `outputs/tables/coefficient_path.csv`, `outputs/tables/stable_features_summary.csv`, `outputs/figures/logistic_coefficient_paths.png` | All found | Best saved model uses `C = 1.0`; test ROC-AUC is about 0.897; the most stable feature groups are `marital_status`, `education_num`, `capital_gain_log1p`, `hours_per_week`, and `age`. |
| Interaction validation | `outputs/tables/tested_interactions.csv`, `outputs/tables/interaction_evidence_table.csv`, `outputs/tables/interaction_survival_summary.csv`, `outputs/metrics/interaction_model_metrics.csv`, `outputs/figures/interaction_delta_cv_auc.png` | All found | Five numeric interaction candidates were tested; all rows in `interaction_evidence_table.csv` are currently labeled `unstable`; no final interaction-augmented model was selected. |
| SVM boundary analysis | `outputs/tables/svm_tuning_summary.csv`, `outputs/metrics/svm_comparison.csv`, `outputs/tables/final_model_comparison_table.csv`, `outputs/figures/svm_kernel_comparison.png` | All found | Polynomial SVM performed best on held-out ROC-AUC (about 0.905); linear SVM is close; RBF does not improve over polynomial in the saved run. |
| Robustness checks | `outputs/metrics/robustness_summary.csv`, `outputs/tables/robust_feature_group_frequency.csv`, `outputs/tables/robustness_metric_ranges.csv`, `outputs/tables/robustness_boundary_consistency.csv`, `outputs/tables/robustness_mi_overlap.csv`, `outputs/figures/robustness_metric_ranges.png` | All found | Logistic and SVM metrics are stable across seeds `7`, `42`, and `99`; cross-seed feature-group frequencies are saved explicitly; MI overlap and SVM boundary consistency are now stored as tables instead of narrative notes. |
| Cross-method summary | `outputs/tables/final_summary_table.csv` | All found | The saved summary table identifies `education_num` as the top continuous separator, `relationship` as the top MI feature, `marital_status` as the top logistic feature group, and `poly` as the best held-out SVM family. |

## Proposed Report Structure

The current artifact set supports the following report structure:

1. Introduction: motivation, dataset, and research questions.
2. Dataset and Preprocessing: retained features, split protocol, grouped `native_country`, and the `?`-marker caveat.
3. Experimental Design: focused EDA, Gaussian analysis, MI, sparse logistic regression, interaction validation, SVM comparison, and robustness checks.
4. Results:
   - Data audit and preprocessing outputs
   - Focused exploratory analysis
   - Class-conditional distribution analysis
   - Mutual information and candidate interaction screening
   - Sparse logistic signal
   - Interaction validation
   - SVM family comparison
   - Robustness checks
   - Cross-method summary table
5. Discussion: agreement and disagreement across methods.
6. Conclusion.
