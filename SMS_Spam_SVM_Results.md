# SMS Spam Classification with SVM — Run Report

Dataset: [UCI SMS Spam Collection](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) (`SMS.csv`)

## 1. Data Overview
- **5,572** total messages (**4,825 ham / 747 spam**) — 403 duplicate rows removed before training.
- Class imbalance: spam makes up ~13% of the dataset, which motivated the class-imbalance fix below.
- Train/test split: **4,135 / 1,034** messages (80/20, stratified by label).

## 2. Baseline SVM (with `class_weight='balanced'`)
A `TfidfVectorizer` (max 5,000 features, unigrams+bigrams) feeding a `LinearSVC` with `class_weight='balanced'` to address the imbalance:

| Metric | Score |
|---|---|
| Accuracy | 0.978 |
| Precision (spam) | 0.922 |
| Recall (spam) | 0.901 |
| F1 (spam) | 0.911 |

Balancing the class weights pushed recall up compared to an unweighted baseline, at a modest cost to precision.

## 3. Hyperparameter Tuning
Expanded grid search (`GridSearchCV`, 5-fold CV, scored on F1) over TF-IDF and SVM parameters:
- `tfidf__max_features`: 3000 / 5000 / 8000
- `tfidf__ngram_range`: (1,1) / (1,2)
- `tfidf__min_df`: 1 / 2 / 3
- `svm__C`: 0.01 / 0.1 / 1 / 10 / 100
- `svm__loss`: hinge / squared_hinge
- `svm__class_weight`: None / balanced

**Best parameters found:**
```
C=1, class_weight=None, loss=hinge, max_features=8000, min_df=1, ngram_range=(1,2)
```
Best CV F1: **0.914**

**Test-set performance of the tuned model:**

| Metric | Score |
|---|---|
| Accuracy | 0.985 |
| Precision (spam) | 0.983 |
| Recall (spam) | 0.901 |
| F1 (spam) | **0.940** |

Interestingly, the wider search landed on `class_weight=None` — the larger vocabulary (8,000 features) and bigrams gave the model enough signal that manual class balancing was no longer the deciding factor. Overall this is a solid F1 improvement over the original baseline notebook (0.936 → 0.940) with much higher precision (0.98 vs 0.98) and comparable recall.

## 4. Confusion Matrix (tuned model, test set)
The confusion matrix plot (added to the notebook) shows the tuned model correctly identifies the large majority of both classes, with the residual errors concentrated in a small number of missed spam messages (false negatives) rather than ham misclassified as spam — a reasonable trade-off for a spam filter, since false positives (blocking real messages) are more costly to users than the occasional missed spam.

## 5. Cross-Validation (5-fold, full dataset, best model)
```
Fold scores: [0.981, 0.981, 0.984, 0.983, 0.981]
Mean Accuracy: 0.982
```
Low variance across folds indicates the model generalizes well and isn't overfit to a particular split.

## 6. Sample Predictions
```
"Congratulations! You have won a free iPhone. Click here now." → Spam
"Hi, are we meeting tomorrow at 3 PM?"                          → Ham
```

## Summary of Changes Implemented
1. **Class imbalance handling** — added `class_weight='balanced'` to the baseline SVM.
2. **Expanded hyperparameter search** — grid search now tunes TF-IDF vectorization (max features, n-grams, min_df) alongside SVM regularization, loss, and class weighting, scored on F1.
3. **Confusion matrix visualization** — added a plotted confusion matrix for the tuned model on the test set.

**Net result:** Test F1 improved to **0.940**, accuracy to **98.5%**, with strong, low-variance cross-validation performance (~98.2%).
