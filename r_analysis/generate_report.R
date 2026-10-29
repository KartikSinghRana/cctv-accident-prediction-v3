if (!exists("project_dir", inherits = TRUE)) {
  project_dir <- normalizePath(getwd(), winslash = "/", mustWork = TRUE)
}

analysis_dir <- file.path(project_dir, "r_analysis")
input_csv <- file.path(analysis_dir, "predictions.csv")
classes <- c("No_Accident", "Minor_Accident", "Severe_Accident")
probability_columns <- paste0("prob_", classes)

if (!file.exists(input_csv)) {
  stop("Predictions CSV not found: ", input_csv, "\nRun the notebook's R export cell first.")
}

predictions <- read.csv(input_csv, stringsAsFactors = FALSE, check.names = FALSE)
required_columns <- c("file", "true", "pred", probability_columns)
missing_columns <- setdiff(required_columns, names(predictions))
if (length(missing_columns)) {
  stop("predictions.csv is missing required columns: ", paste(missing_columns, collapse = ", "))
}
if (!nrow(predictions)) {
  stop("predictions.csv contains no test predictions.")
}
if (anyNA(predictions[, required_columns])) {
  stop("predictions.csv contains missing values in required columns.")
}
if (anyDuplicated(predictions$file)) {
  stop("predictions.csv contains duplicate filenames.")
}
if (any(!predictions$true %in% classes) || any(!predictions$pred %in% classes)) {
  stop("The true and pred columns must contain only: ", paste(classes, collapse = ", "))
}

probabilities <- as.matrix(predictions[, probability_columns, drop = FALSE])
storage.mode(probabilities) <- "numeric"
if (any(!is.finite(probabilities)) || any(probabilities < 0 | probabilities > 1)) {
  stop("Class probabilities must be finite values between 0 and 1.")
}
if (any(abs(rowSums(probabilities) - 1) > 1e-3)) {
  stop("Class probabilities must sum to 1 for every prediction.")
}

actual <- factor(predictions$true, levels = classes)
predicted <- factor(predictions$pred, levels = classes)
confusion <- table(Actual = actual, Predicted = predicted)
true_positive <- diag(confusion)
support <- rowSums(confusion)
predicted_count <- colSums(confusion)

divide_or_zero <- function(numerator, denominator) {
  result <- numeric(length(numerator))
  valid <- denominator != 0
  result[valid] <- numerator[valid] / denominator[valid]
  result
}

metrics_from_indices <- function(indices) {
  matrix <- table(
    factor(actual[indices], levels = classes),
    factor(predicted[indices], levels = classes)
  )
  true_positive <- diag(matrix)
  support <- rowSums(matrix)
  predicted_count <- colSums(matrix)
  precision <- divide_or_zero(true_positive, predicted_count)
  recall <- divide_or_zero(true_positive, support)
  f1 <- divide_or_zero(2 * precision * recall, precision + recall)
  c(accuracy = sum(true_positive) / sum(matrix), macro_f1 = mean(f1))
}

precision <- divide_or_zero(true_positive, predicted_count)
recall <- divide_or_zero(true_positive, support)
f1 <- divide_or_zero(2 * precision * recall, precision + recall)
specificity <- vapply(seq_along(classes), function(index) {
  true_negative <- sum(confusion) - support[index] - predicted_count[index] + true_positive[index]
  divide_or_zero(true_negative, sum(confusion) - support[index])
}, numeric(1))

roc_curve <- function(labels, scores) {
  positive_count <- sum(labels)
  negative_count <- length(labels) - positive_count
  if (!positive_count || !negative_count) {
    return(list(fpr = NA_real_, tpr = NA_real_, auc = NA_real_))
  }
  ordering <- order(scores, decreasing = TRUE)
  sorted_scores <- scores[ordering]
  sorted_labels <- labels[ordering]
  group_starts <- which(!duplicated(sorted_scores))
  group_sizes <- diff(c(group_starts, length(sorted_labels) + 1))
  group_positive <- vapply(seq_along(group_starts), function(index) {
    start <- group_starts[index]
    sum(sorted_labels[start:(start + group_sizes[index] - 1)])
  }, numeric(1))
  group_negative <- group_sizes - group_positive
  tpr <- c(0, cumsum(group_positive) / positive_count, 1)
  fpr <- c(0, cumsum(group_negative) / negative_count, 1)
  list(
    fpr = fpr,
    tpr = tpr,
    auc = sum(diff(fpr) * (head(tpr, -1) + tail(tpr, -1)) / 2)
  )
}

curves <- lapply(seq_along(classes), function(index) {
  roc_curve(actual == classes[index], probabilities[, index])
})
auc <- vapply(curves, `[[`, numeric(1), "auc")
class_metrics <- data.frame(
  class = classes,
  support = as.integer(support),
  precision = precision,
  recall = recall,
  specificity = specificity,
  f1 = f1,
  auc_ovr = auc,
  stringsAsFactors = FALSE
)

class_indices <- lapply(classes, function(class_name) which(actual == class_name))
if (any(lengths(class_indices) == 0)) {
  stop("The test predictions must contain at least one sample for each class.")
}
set.seed(20261029)
bootstrap_metrics <- replicate(1000, {
  selected <- unlist(lapply(class_indices, function(indices) {
    sample(indices, size = length(indices), replace = TRUE)
  }), use.names = FALSE)
  metrics_from_indices(selected)
})
accuracy_interval <- quantile(bootstrap_metrics["accuracy", ], c(0.025, 0.975), names = FALSE)
macro_f1_interval <- quantile(bootstrap_metrics["macro_f1", ], c(0.025, 0.975), names = FALSE)

one_hot <- sapply(classes, function(class_name) as.numeric(actual == class_name))
multiclass_brier <- mean(rowSums((probabilities - one_hot)^2))
summary_metrics <- data.frame(
  metric = c(
    "samples", "accuracy", "accuracy_bootstrap_95ci_lower",
    "accuracy_bootstrap_95ci_upper", "macro_precision", "macro_recall",
    "macro_f1", "macro_f1_bootstrap_95ci_lower",
    "macro_f1_bootstrap_95ci_upper", "weighted_f1", "macro_auc_ovr",
    "multiclass_brier_score"
  ),
  value = c(
    nrow(predictions),
    sum(diag(confusion)) / sum(confusion),
    accuracy_interval,
    mean(precision),
    mean(recall),
    mean(f1),
    macro_f1_interval,
    weighted.mean(f1, support),
    mean(auc, na.rm = TRUE),
    multiclass_brier
  ),
  stringsAsFactors = FALSE
)

row_indices <- seq_len(nrow(predictions))
predicted_confidence <- probabilities[cbind(
  row_indices,
  match(predictions$pred, classes)
)]
true_class_probability <- probabilities[cbind(
  row_indices,
  match(predictions$true, classes)
)]
top_two_margin <- apply(probabilities, 1, function(row) {
  sorted <- sort(row, decreasing = TRUE)
  sorted[1] - sorted[2]
})
correct <- predictions$true == predictions$pred
error_analysis <- data.frame(
  file = predictions$file,
  true = predictions$true,
  pred = predictions$pred,
  correct = correct,
  predicted_confidence = predicted_confidence,
  true_class_probability = true_class_probability,
  top_two_margin = top_two_margin,
  error_type = ifelse(
    correct,
    "Correct",
    ifelse(predicted_confidence >= 0.8, "High-confidence error", "Other error")
  ),
  stringsAsFactors = FALSE
)

group_analysis <- do.call(rbind, lapply(classes, function(class_name) {
  class_rows <- which(predictions$true == class_name)
  class_errors <- class_rows[!correct[class_rows]]
  most_common_error <- if (!length(class_errors)) {
    "None"
  } else {
    error_pairs <- paste(
      predictions$true[class_errors],
      "->",
      predictions$pred[class_errors]
    )
    names(sort(table(error_pairs), decreasing = TRUE))[1]
  }
  data.frame(
    group_id = class_name,
    samples = length(class_rows),
    errors = length(class_errors),
    error_rate = length(class_errors) / length(class_rows),
    most_common_error = most_common_error,
    stringsAsFactors = FALSE
  )
}))

write.csv(class_metrics, file.path(analysis_dir, "metrics_by_class.csv"), row.names = FALSE)
write.csv(summary_metrics, file.path(analysis_dir, "summary_metrics.csv"), row.names = FALSE)
write.csv(as.data.frame.matrix(confusion), file.path(analysis_dir, "confusion_matrix.csv"))
write.csv(error_analysis, file.path(analysis_dir, "error_analysis.csv"), row.names = FALSE)
write.csv(group_analysis, file.path(analysis_dir, "group_analysis.csv"), row.names = FALSE)

plot_classes <- c("No accident", "Minor", "Severe")
plot_colors <- c("#2563eb", "#d97706", "#dc2626")
png(
  file.path(analysis_dir, "evaluation_report.png"),
  width = 1800,
  height = 1400,
  res = 150
)
par(mfrow = c(2, 2), mar = c(5, 5, 4, 2))

par(mar = c(9, 8, 4, 2))
plot.new()
plot.window(xlim = c(0, length(classes)), ylim = c(0, length(classes)), asp = 1)
title("Test confusion matrix", xlab = "Predicted class", ylab = "Actual class")
heat_colors <- colorRampPalette(c("#f7fbff", "#2171b5"))(100)
max_count <- max(confusion)
for (row in seq_along(classes)) {
  for (column in seq_along(classes)) {
    value <- confusion[row, column]
    color_index <- if (max_count == 0) 1 else max(1, ceiling(value / max_count * 100))
    y <- length(classes) - row
    rect(column - 1, y, column, y + 1, col = heat_colors[color_index], border = "white")
    text(column - 0.5, y + 0.5, labels = value)
  }
}
axis(1, at = seq_along(classes) - 0.5, labels = plot_classes, las = 2, cex.axis = 0.85)
axis(2, at = seq_along(classes) - 0.5, labels = rev(plot_classes), las = 2, cex.axis = 0.85)

par(mar = c(5, 5, 4, 2))
plot(
  c(0, 1), c(0, 1), type = "n", xlim = c(0, 1), ylim = c(0, 1),
  xaxs = "i", yaxs = "i",
  xlab = "False positive rate", ylab = "True positive rate",
  main = "One-vs-rest ROC"
)
abline(0, 1, col = "gray60", lty = 2)
for (index in seq_along(classes)) {
  lines(curves[[index]]$fpr, curves[[index]]$tpr, col = plot_colors[index], lwd = 2)
}
legend(
  "bottomright",
  legend = paste0(plot_classes, " (AUC ", sprintf("%.3f", auc), ")"),
  col = plot_colors, lwd = 2, cex = 0.8
)

par(mar = c(9, 5, 4, 2))
barplot(
  rbind(precision, recall, f1),
  beside = TRUE,
  names.arg = plot_classes,
  col = c("#2563eb", "#0f766e", "#d97706"),
  ylim = c(0, 1.2),
  las = 2,
  ylab = "Score",
  main = "Per-class performance"
)
legend(
  "top",
  legend = c("Precision", "Recall", "F1"),
  fill = c("#2563eb", "#0f766e", "#d97706"),
  horiz = TRUE,
  bty = "n",
  cex = 0.8
)

par(mar = c(5, 5, 4, 2))
confidence_groups <- list(
  Correct = predicted_confidence[correct],
  Incorrect = predicted_confidence[!correct]
)
confidence_groups <- confidence_groups[lengths(confidence_groups) > 0]
boxplot(
  confidence_groups,
  ylim = c(0, 1),
  ylab = "Predicted-class confidence",
  main = "Confidence by outcome",
  col = c("#93c5fd", "#fca5a5")[seq_along(confidence_groups)]
)
dev.off()
file.copy(
  file.path(analysis_dir, "evaluation_report.png"),
  file.path(analysis_dir, "r_evaluation.png"),
  overwrite = TRUE
)

cat("R evaluation complete for", nrow(predictions), "test predictions.\n")
cat(
  "Accuracy:",
  sprintf("%.2f%%", 100 * summary_metrics$value[summary_metrics$metric == "accuracy"]),
  "\n"
)
cat("Report and CSV outputs saved under:", analysis_dir, "\n")
