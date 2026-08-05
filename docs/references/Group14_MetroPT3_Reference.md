# Reference Report — Group 14, EDS 6340 (MetroPT-3 Failure Prediction)

> **Purpose of this file:** a complete, image-free text reconstruction of Group 14's project report,
> structured so it can be pasted into a chat alongside your own project for a point-by-point comparison.
> All numbers are transcribed from the figure outputs in the original PDF (confusion matrices,
> classification reports, hyperparameter printouts). Where the original captions were internally
> inconsistent, this is flagged with **[CAPTION DISCREPANCY]**. Analytical caveats added on top of the
> report's own claims are marked **[REVIEWER NOTE]** so you can separate "what they said" from
> "what an evaluator might say."

---

## HOW TO USE THIS FILE FOR COMPARISON

Paste this file, then paste (or attach) your own project, then use a prompt like one of these:

- *"Compare my project to the Group 14 reference above. Go section by section: problem framing,
  labeling, preprocessing, feature selection, model choice, tuning, metrics, and validity of results.
  For each, say where mine is stronger, weaker, or equivalent, and why."*
- *"Using the Group 14 reference, build a comparison table: for every model we both trained, put their
  metric next to mine and flag any gap larger than 0.02."*
- *"Group 14 got perfect scores. Audit my methodology against theirs for the same leakage risks listed
  in the reference. Am I making the same mistakes?"*
- *"Act as a course grader. Given the Group 14 reference as the baseline submission, grade my project
  relative to it and justify the grade."*

A **Comparison Checklist** is provided at the very end — hand it to the model as the rubric.

---

## 1. PROBLEM & DATASET

- **Task:** binary classification — predict Air Production Unit (APU) failure in metro trains.
- **Dataset:** MetroPT-3. Real compressor APU sensor readings (pressure, temperature, motor current,
  air-intake valves).
- **Size:** 1,516,948 instances; 15 features.
- **Type:** tabular, multivariate, time-series.
- **Target `status`:** 0 = healthy/running, 1 = failure/maintenance needed.
- **15 feature columns referenced:** TP2, TP3, H1, DV_pressure, Reservoirs, Oil_temperature,
  Motor_current, COMP, DV_eletric, Towers, MPG, LPS, Pressure_switch, Oil_level, Caudal_impulses.

### Labeling method (critical — this drives everything downstream)
- Original data is **unlabeled**.
- Labels were created from a **maintenance timeline table** (4 failure windows, all "Air leak /
  High stress"):
  - #1: 4/18/2020 00:00 → 4/18/2020 23:59
  - #2: 5/29/2020 23:30 → 5/30/2020 06:00
  - #3: 6/5/2020 10:00 → 6/7/2020 14:30
  - #4: 7/15/2020 14:30 → 7/15/2020 19:00
- **Every timestamp inside a maintenance window was labeled 1; everything else 0.**

> **[REVIEWER NOTE — labeling]** This labels the *failure/maintenance period itself*, not the
> *lead-up* to failure. A predictive-maintenance model should forecast failure **before** it happens
> (a warning horizon). Labeling the window means the model learns "what a failure looks like while it's
> happening," which is (a) trivially separable from normal operation and (b) not actually predictive.
> This is the single biggest methodological question to compare against your own labeling scheme.

---

## 2. EXPLORATORY DATA ANALYSIS

- **Correlation:** target `status` reported as highly correlated with TP2, H1, DV_pressure,
  Oil_temperature, Motor_current, COMP, DV_electric, MPG. Report emphasizes "strong linear
  relationships" throughout — this claim is later used to justify Lasso and linear models.
- **Outliers:** boxplots after preprocessing showed **no outliers present** (per report).

> **[REVIEWER NOTE — EDA]** "Strong linear correlation with the target" on a labeled-window setup is
> partly a symptom of the labeling problem: during a failure window several sensors sit at abnormal
> levels, so they correlate almost perfectly with the label. High correlation here is evidence *for*
> the leakage concern, not just evidence the problem is easy.

---

## 3. PREPROCESSING PIPELINE

Steps the team performed:
1. **Removed unnecessary columns** (noise reduction).
2. **Reformatted timestamp** to a standard format.
3. **Created `status` label** from the maintenance table (see §1).
4. **Subsampling to balance classes** — original ≈ 1,500,000 negatives vs ≈ 30,000 positives
   (roughly 50:1 imbalance). Subsampled to a balanced set.
5. **Outlier removal** via IQR analysis (iterative).

Steps skipped because the data provider had already done them:
- Data segmentation
- Normalization
- Feature extraction

> **[REVIEWER NOTE — preprocessing]** Two things to check against your work:
> 1. **Order of operations / leakage:** if subsampling or scaling happened *before* the train/test
>    split, statistics leak from test into train. The report doesn't state the split happened first.
> 2. **Subsampling a time series:** randomly subsampling adjacent, near-identical time-series rows and
>    then doing a random train/test split puts almost-duplicate rows on both sides of the split. This
>    alone can manufacture near-perfect scores. Did you split by **time** (train on earlier windows,
>    test on later) or randomly? This is a key differentiator.

---

## 4. FEATURE SELECTION

- **Method chosen:** Lasso (L1) via **LassoCV**.
- **Justification given:** strong linearity among variables; Lasso induces sparsity and handles
  high-dimensional/linear structure well.
- **Alternatives considered:** KNN-based importance, correlation coefficients.
- **Measured effect (on SVM):**

| Metric | Before feature selection | After feature selection |
|---|---|---|
| Train time | 27.45 s | 19.59 s |
| Predict time | 0.68 s | 0.5 s |
| Recall (class 0) | baseline | "slightly increased" |
| Accuracy | ~0.99 | ~0.99 (unchanged) |

- SVM confusion matrix **before** FS: TN 5918, FP 135, FN 28, TP 5808
- SVM confusion matrix **after** FS: TN 5921, FP 132, FN 28, TP 5808

> **[REVIEWER NOTE — feature selection]** Feature selection here mainly bought **speed**, not accuracy
> (accuracy was already saturated). Fair and honestly reported. Compare: did your feature selection
> change accuracy meaningfully, and did you report the before/after the same way?

---

## 5. MODELS — FULL RESULTS

> Confusion matrix convention below: **[TN, FP, FN, TP]** where rows = true label (0 then 1),
> columns = predicted label (0 then 1). Support = number of samples per class in the test set.

### 5.1 Logistic Regression
*(Report titles this "Linear Regression" but correctly implements **logistic** regression and explains why linear regression is inappropriate for classification.)*
- **Tuning:** ROC-curve threshold search. Thresholds tried: 0.2, 0.4, 0.6, 0.8. Best AUC at
  **threshold = 0.6** (AUC ≈ 0.98470). Other AUCs: 0.2→0.97984, 0.4→0.98107, 0.8→0.98311.
- **Accuracy:** 0.98
- Class 0: precision 0.99, recall 0.98, F1 0.98 (support 6026)
- Class 1: precision 0.98, recall 0.99, F1 0.98 (support 5863)
- **Confusion matrix:** TN 5881, FP 145, FN 48, TP 5815

### 5.2 Naive Bayes
Three variants tested; **Gaussian** selected (features are continuous).

| Variant | Accuracy | Precision (0/1) | Recall (0/1) | Confusion [TN, FP, FN, TP] |
|---|---|---|---|---|
| Bernoulli (report says "Binomial") | 0.93 | 0.99 / 0.87 | 0.86 / 1.00 | 5195, 858, 29, 5807 |
| Multinomial | 0.92 | 1.00 / 0.87 | 0.86 / 1.00 | 5179, 874, 24, 5812 |
| **Gaussian (chosen)** | **0.95** | 1.00 / 0.91 | 0.90 / 1.00 | 5477, 576, 25, 5811 |

> **[CAPTION DISCREPANCY]** Figure captions in the report swap some NB confusion matrices/labels
> (e.g. a matrix captioned "Gaussian confusion matrix" sits under KNN text). The **accuracy and
> classification-report numbers above are the reliable ones**; treat the caption-to-image mapping as
> unreliable in the original.

### 5.3 K-Nearest Neighbors (KNN)
- **Tuning:** swept k; evaluated MSE and accuracy. **Optimal k = 3** (accuracy peaked ~0.9945, error
  lowest at k=3, both degrading as k grew).
- **Accuracy:** 0.99
- Class 0: precision 1.00, recall 0.99; Class 1: precision 0.99, recall 1.00 (support 6026 / 5863)
- **Before tuning CM:** TN 5969, FP 57, FN 16, TP 5847
- **After tuning (k=3) CM:** TN 5978, FP 48, FN 14, TP 5849

### 5.4 Random Forest
- **Tuning:** Out-of-Bag (OOB) score + GridSearch (also a RandomizedSearch pass).
- **OOB score:** 99.67%
- **Best params (OOB run):** n_estimators=100, max_depth=20, min_samples_split=2,
  min_samples_leaf=1, max_features=10
- **Best params (GridSearch run):** n_estimators=200, min_samples_split=5, min_samples_leaf=1,
  max_features=10, max_depth=20
- **Accuracy:** ≈ 1.00 (precision/recall/F1 all 1.00)
- **Confusion matrix (OOB):** TN 9011, FP 20, FN 23, TP 8780 (support 9031 / 8803)

> **[CAPTION DISCREPANCY]** The "Random Forest Confusion Matrix with Grid Search" figure shows the
> same 9011/20/23/8780 values as the OOB matrix.

### 5.5 Support Vector Machine (SVM)
- **Kernels compared:** linear vs nonlinear (RBF). **Linear chosen** (matches the "linear dataset"
  narrative; simpler, effectively equal accuracy).
- **C tuning:** grid search over C; accuracy rose monotonically with C. **C = 1000 chosen** as a
  compute/accuracy compromise.
- **Accuracy:** 0.99 (all variants)
- **Linear kernel CM:** TN 8741, FP 195, FN 48, TP 8850 (support 8936 / 8898)
- **Nonlinear kernel CM:** TN 8855, FP 81, FN 51, TP 8847
- **Linear + C=1000 CM:** TN 8872, FP 64, FN 41, TP 8857

### 5.6 Bidirectional Feature Elimination (wrapper method)
- Applied bidirectional (forward+backward) elimination; discussed bias–variance tradeoff.
- **Result:** essentially identical to Lasso; small accuracy bump, reduced train/predict time.
- **Before CM:** TN 6004, FP 22, FN 24, TP 5839 (accuracy ≈ 1.00)
- **After CM:** TN 6007, FP 19, FN 21, TP 5842 (accuracy ≈ 1.00)

> **[CAPTION DISCREPANCY]** These figures are captioned "Linear SVM performance" but the printout
> header says "Random Forest Classifier." The classifier identity here is ambiguous in the original.

### 5.7 XGBoost
- Combined with Lasso-selected features.
- **Accuracy:** 100%
- **Confusion matrix:** TN 5948, FP 0, FN 0, TP 5941 — **zero misclassifications** (support 5948 / 5941)

### 5.8 Extreme Learning Machine (ELM)
- Single hidden layer, random input weights, analytically solved output weights.
- **Tuning:** swept hidden units; **40 hidden units** chosen. Accuracy vs units rose steeply to ~10
  units then plateaued near 1.000.
- **Accuracy:** 100%
- **Confusion matrix:** TN 5967, FP 0, FN 0, TP 5922 — **zero misclassifications**

### 5.9 Neural Network
- Architecture: **2 hidden layers, 32 units each** (report text says 32; one figure caption implies a
  generic diagram).
- **Accuracy:** 1.00
- **Confusion matrix:** TN 5907, FP 0, FN 0, TP 5982 — **zero misclassifications**

### 5.10 Ensemble
- **Method:** average the predictions of the top 3 models (XGBoost + ELM + Neural Network).
- **Accuracy:** 1.00
- **Confusion matrix:** TN 5948, FP 0, FN 0, TP 5941 — **zero misclassifications**

---

## 6. FINAL COMPARISON TABLE (their Table 9.1)

| Model | Precision | Recall | F1-Score | Accuracy | Notable errors (FP+FN) |
|---|---|---|---|---|---|
| Logistic Regression | 0.98 | 0.98 | 0.98 | 0.98 | 193 |
| Naive Bayes (Gaussian) | 0.95 | 0.95 | 0.95 | 0.95 | 601 |
| KNN (k=3) | 0.99 | 0.99 | 0.99 | 0.99 | 62 |
| SVM (linear, C=1000) | 0.99 | 0.99 | 0.99 | 0.99 | 105 |
| Random Forest | 1.00 | 1.00 | 1.00 | ~1.00 | 43 |
| XGBoost | 1.00 | 1.00 | 1.00 | 1.00 | 0 |
| Extreme Learning Machine | 1.00 | 1.00 | 1.00 | 1.00 | 0 |
| Neural Network | 1.00 | 1.00 | 1.00 | 1.00 | 0 |
| Ensemble | 1.00 | 1.00 | 1.00 | 1.00 | 0 |

*(Error counts derived from the confusion matrices above; test-set sizes vary by model because splits
were re-run per model — another thing to check against your fixed-split setup.)*

---

## 7. THEIR STATED CONCLUSIONS

- Dataset is highly linear → linear decision boundaries separate classes well → favors logistic
  regression and linear SVM.
- Strong feature correlations penalize Naive Bayes (violates its independence assumption) → its lowest
  score (0.95).
- Scaling + feature selection improved accuracy and efficiency.
- The 3 advanced models + ensemble reached perfect scores; ELM praised for training stability
  ("converges to global maximum").
- Overall framed as a success across the board.

---

## 8. INDEPENDENT VALIDITY AUDIT (use this to pressure-test both projects)

> These are **not** in the original report. They're the questions a strong reviewer / hiring manager /
> grader would ask. Run your own project through the same list.

1. **Perfect scores are a warning, not a win.** Four models at exactly 1.00 with all-zero off-diagonals
   on real sensor data almost always indicates **data leakage or train/test contamination**, not
   genuine skill. A credible predictive-maintenance result usually looks like 0.85–0.97 with a real
   precision/recall tradeoff.
2. **Label leakage (most likely culprit).** Labeling the maintenance window itself (not a pre-failure
   horizon) lets the model separate "mid-failure sensor state" from "normal" — easy but not predictive.
3. **Temporal leakage from random splitting.** Time-series rows are autocorrelated; adjacent rows are
   near-duplicates. A random split (vs a chronological split) puts near-identical rows in both train
   and test. Subsampling before splitting worsens this.
4. **Per-model re-splitting.** Support counts differ across models (e.g. 6026/5863 vs 8936/8898 vs
   9031/8803), implying different splits/sizes per model. That makes the final comparison table
   not strictly apples-to-apples.
5. **No horizon / no lead-time analysis.** True predictive maintenance reports *how far in advance*
   failure is caught. This is absent.
6. **Metric choice.** On (originally) imbalanced failure data, accuracy is a weak headline metric.
   Precision/recall/PR-AUC on the **positive (failure)** class, at the original imbalance, is more
   honest than balanced-subsample accuracy.
7. **No cross-validation of the final comparison.** Single held-out numbers, no variance/error bars.

---

## 9. COMPARISON CHECKLIST (hand this to the model as the rubric)

For each row, the model should state: **[Group 14]** vs **[Mine]** vs **[Who's stronger + why]**.

- [ ] **Problem framing** — is failure predicted *ahead* of time, or just detected during the window?
- [ ] **Labeling scheme** — window-labeling vs pre-failure horizon labeling.
- [ ] **Train/test split** — random vs chronological/time-based; done before or after subsampling/scaling.
- [ ] **Class imbalance handling** — subsample vs class weights vs SMOTE vs thresholding; evaluated at
      balanced or original imbalance.
- [ ] **Leakage controls** — dedup of adjacent rows, group-aware splits, no future info in features.
- [ ] **Feature selection** — method, and whether it changed accuracy or only speed.
- [ ] **Model coverage** — which models each project trained (overlap set for direct metric comparison).
- [ ] **Hyperparameter tuning** — method (grid/random/OOB/ROC), and whether tuning used a proper
      validation set (not the test set).
- [ ] **Metrics reported** — accuracy only vs precision/recall/F1/PR-AUC/ROC-AUC on the failure class.
- [ ] **Evaluation rigor** — single split vs k-fold; variance/error bars; consistent test set across models.
- [ ] **Result plausibility** — are near-perfect scores investigated for leakage, or accepted at face value?
- [ ] **Reproducibility** — fixed seeds, documented split, single pipeline vs per-model re-runs.
- [ ] **Conclusions** — supported by the evidence and appropriately hedged?

---

## 10. SOURCE LINKS (from their report)

- GitHub: https://github.com/harveyphm/MetroPT-3-Anomaly-Detection
- (Presentation link in original was a private SharePoint URL — likely not publicly accessible.)
