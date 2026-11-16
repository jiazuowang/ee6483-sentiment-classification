# Product Review Sentiment Classification

An EE6483 course-project implementation comparing sparse linear models and a compact neural baseline for binary sentiment classification.

**19 configurations · grouped cross-validation · class imbalance · feature and resource comparisons**

The selected class-weighted SVM achieved **91.42% accuracy** and **0.8357 macro F1** on a separate, labelled holdout of 1,480 reviews.

## My implementation

This repository presents my implementation work within a group course project. The course instructor supplied the dataset. My work focuses on the classifier pipelines, experiment runner, grouped cross-validation, parameter selection, and feature/resource comparisons. The original group report and other members' identities are not distributed here.

- Built 19 configurations spanning Naive Bayes, logistic regression, linear SVMs, and a compact MLP.
- Compared word unigrams, word bigrams, character n-grams, and class weighting.
- Kept normalized duplicate reviews together across development and holdout partitions.
- Selected the final model using development macro F1 before evaluating it on the holdout.
- Recorded fold results, model selection, class-wise metrics, and sparse-feature storage costs.

Implementation: [experiment runner](code/run_experiments.py) and [prediction entry point](code/predict.py).

## Evaluation design

The supplied dataset contains 7,401 labelled training reviews and 1,851 unlabelled test reviews. The training labels are imbalanced: 6,319 positive and 1,082 negative.

1. Normalize case and whitespace to identify duplicate-text groups.
2. Reserve the first split from a shuffled, stratified five-fold group split as the holdout: 5,921 development rows and 1,480 holdout rows.
3. Compare configurations using three grouped development folds. Fit vocabulary and IDF within each training fold.
4. Select by mean macro F1, using accuracy to break ties.
5. Evaluate the locked selection on the holdout, then refit on all labelled data for test predictions.

The audit found 219 extra normalized duplicate training rows and 101 test rows whose normalized text also appeared in training. Test labels were unavailable and were not used for selection.

## Recorded results

| Configuration | Development CV macro F1 | Sample SD across 3 folds |
| --- | ---: | ---: |
| Naive Bayes, word unigrams | 0.4604 | 0.0002 |
| SVM, word unigrams, C=1 | 0.7986 | 0.0105 |
| Balanced logistic regression, words, C=2 | 0.8332 | 0.0154 |
| Balanced SVM, words, C=0.25 | 0.8365 | 0.0139 |
| **Balanced SVM, words + characters, C=0.25** | **0.8436** | **0.0136** |
| Compact MLP, learning rate 0.003 | 0.7921 | 0.0171 |

Full records: [all configurations](results/cv_summary.csv), [individual folds](results/cv_folds.csv), [selection](results/selection.json).

| Selected model: labelled holdout | Result |
| --- | ---: |
| Accuracy | 91.42% |
| Macro F1 | 0.8357 |
| Negative-class F1 | 0.7221 |
| Positive-class F1 | 0.9493 |
| Always-positive baseline accuracy | 85.41% |
| Group-bootstrap 95% accuracy interval | 89.84%-92.87% |

Confusion matrix (rows = true labels, columns = predicted labels):

| | Predicted negative | Predicted positive |
| --- | ---: | ---: |
| True negative | 165 | 51 |
| True positive | 76 | 1,188 |

The confidence interval resamples normalized-text groups. It does not capture model-selection, split, or domain-transfer uncertainty. See [evaluation record](results/evaluation.json).

## Engineering trade-offs and limits

- Adding character features to the balanced word SVM raises CV macro F1 from 0.8365 to 0.8436, while transformed feature storage rises from 4.95 to 33.37 MiB. This modest gain should be weighed against the extra memory; the fold variability overlaps.
- Storage measurements cover the transformed arrays, not peak process memory. Recorded timings are specific to the original machine and two-thread numerical configuration; they are not deployment latency benchmarks.
- Negative-class F1 is substantially below positive-class F1. Accuracy alone would hide this gap.
- Grouping blocks normalized exact duplicates, but cannot rule out near duplicates or shared products/authors because those IDs are absent.
- The MLP's internal early-stopping split is not group-aware and occurs after feature fitting. External CV and holdout partitions remain separate; the neural baseline has this additional limitation.

Additional records: [resource comparison](results/resource_comparison.csv), [MLP seed sensitivity](results/mlp_seed_results.csv), [aggregate data audit](results/data_audit.json).

## Run locally

The original data and fitted model are not included. Use only data you are authorized to access; see [data requirements](data/README.md).

The original environment is recorded in `environment.yml` and `requirements.txt`:

```bash
conda env create -f environment.yml
conda activate ee6483
python code/run_experiments.py
python code/predict.py
```

Run from the repository root. The experiment script writes local results, figures, a fitted model, and predictions. Existing files under `results/` can be overwritten; make a separate copy if you want to preserve the recorded results. Optional `--resume` reuses `results/cv_folds.csv`; use it only for an interrupted run with identical data, splits, configuration, and environment. Do not use it to mix newly generated folds with the archived records.

To check pipeline construction on artificial text without the course data:

```bash
python -m unittest discover -s tests -v
```

This smoke check does not reproduce the course scores. The supplied numerical records were checked against the local prediction/fold artifacts during portfolio preparation; the full 19-configuration experiment was not retrained for this publication.

## Data and attribution

The course instructor supplied the dataset. This repository includes implementation code and aggregate experimental records. Raw reviews, submission files, row-level predictions, fitted vocabularies/models and the original group report are excluded. Course data remains subject to its original access and reuse conditions.

Classifiers and feature extraction use [scikit-learn](https://scikit-learn.org/). My work covers pipeline implementation and experimental comparison.
