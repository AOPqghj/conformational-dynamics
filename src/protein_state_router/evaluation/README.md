# Evaluation module

Evaluation converts held-out logits into interpretable probabilities, metrics,
plots, and portable run artifacts. Keep the three phases separate: fit model
weights on training data, fit calibration on validation predictions, and report
the final performance only on the held-out test split.

## What is provided

- `metrics.classification_metrics(labels, probabilities)` reports sample count,
  prevalence, AUPRC, AUROC (when both classes exist), Brier score, expected
  calibration error, MCC, balanced accuracy, F1, and recall at precision ≥0.8.
- `calibration.PlattCalibrator` re-exports the validation-only calibrator from
  the models module.
- `bootstrap.bootstrap_interval(...)` creates a 95% nonparametric bootstrap
  interval for a supplied metric. It skips resamples with only one class.
- `plots.reliability_diagram(...)` returns a Matplotlib figure showing observed
  frequency against predicted probability in ten bins.
- `plots.classification_figures(...)` returns compact ROC, precision-recall, and
  confusion-matrix figures for a held-out prediction vector.
- `reports.write_report(run_dir, predictions, metrics, manifest)` writes
  `predictions.parquet`, `metrics.json`, and `manifest.json`.

The smoke pipeline adds raw and calibrated fusion probabilities, individual
branch logits, availability flags, label/source/split fields, and catalog IDs to
its prediction table so errors can be audited after training.
