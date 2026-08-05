# Mahalanobis Score — Review & Action Items

## 1. Is the function OK?

Mostly, but with issues:

- **`md = md = ...`** — harmless double-assignment typo. Clean it up.
- **Silent singular-covariance bug (the real one).** `np.cov(train, rowvar=False)` on a rank-deficient matrix + `pinv` does **not** error — it returns a garbage inverse. The function "runs" but the distances are meaningless when `cov` is singular. Add a condition-number guard so it warns instead of silently emitting noise.
- **No shape guard.** `mu`/`cov` come from train but are never stored/reused. If the scoring feature matrix ever differs from train in column order or count, `diff = Xn - mu` will misalign or broadcast-fail. Add a shape assertion.

## 2. Are the first results understandable, or a problem?

There's a contradiction that must be resolved before trusting anything:

- The two code blocks (on `X` vs `X_maha`) print the **identical** healthy/fail means: `healthy=30.78 fail=6.43`. So dropping `regime_locked` **did not change the score**. Condition number stayed `2.2e21`, rank still `21/22` — the covariance is **still singular**. One dummy drop did not cure the collinearity.
- That printed ordering is **inverted** (healthy > fail), which would give AUC < 0.5 on failures — yet the eval table shows Mahalanobis AUC **0.64–0.75**. These cannot describe the same score array.
- **Action:** confirm the eval consumed the same `scores["mahalanobis"]` you printed. Right now the printout says *inverted* and the eval says *weakly-correct* — one of them is reading a different vector. Don't accept "it earned a seat" until this reconciles.
- **Sub-0.5 AUCs are not "mediocre."** `degraded`: Mahalanobis `0.302`, iforest `0.248` — both **below 0.5** means they rank degraded-vs-healthy the **wrong way** on that target. That's anti-correlation, i.e. a signal something is off, not just weak detection.

## 3. Work to improve it

Dropping one dummy can't fix **structural** rank deficiency (multiple collinear blocks: occupancy fractions summing to 1, one-hot groups, `_missing` indicators). Options, roughly in priority order:

1. **Diagnose the null space.** `U, s, Vt = np.linalg.svd(cov)`; inspect `Vt[-1]` (near-zero singular vector) — it names exactly which columns are collinear. Fix the real redundancy instead of guessing.
2. **Shrinkage covariance** — highest-leverage single change. Replace raw `np.cov` + `pinv` with `sklearn.covariance.LedoitWolf` or `OAS`: a well-conditioned, invertible estimate built for this regime. Lets you keep all features.
3. **Drop all-but-one per collinear group** systematically: one occupancy fraction, one level per one-hot, and decide whether `_missing` indicators belong in a distance metric at all (they usually don't).
4. **PCA-whiten to the retained rank**, then Euclidean distance in whitened space — equivalent to Mahalanobis but explicitly discards null directions instead of `pinv` doing it badly.
5. **Robust location/scatter** (`sklearn.covariance.MinCovDet`) if train is contaminated with outliers distorting `mu`/`cov` — plausible in an anomaly-detection setting.

**Bottom line:** LedoitWolf is the recommended first move. Re-run the comparison on a *correctly conditioned* covariance. If Mahalanobis still lands ~0.7 while zmax is 0.86, the original "drop it" call is directionally fine — but make that decision on a valid run, which you don't have yet.
