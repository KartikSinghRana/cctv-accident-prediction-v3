import io
import secrets
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image, UnidentifiedImageError

from model_inference import (
    CLASS_NAMES,
    load_trained_model,
    predict_with_explanations,
)


PROJECT_DIR = Path(__file__).resolve().parent
ANALYSIS_DIR = PROJECT_DIR / "r_analysis"
TEST_DIR = PROJECT_DIR / "data" / "processed" / "test"
CHECKPOINT_PATH = PROJECT_DIR / "checkpoints" / "best_model.pth"
CLASS_COLORS = {
    "No_Accident": "#2563eb",
    "Minor_Accident": "#d97706",
    "Severe_Accident": "#dc2626",
}
PLOT_TEMPLATE = "plotly_white"
PROBABILITY_COLUMNS = [f"prob_{class_name}" for class_name in CLASS_NAMES]

st.set_page_config(
    page_title="RoadSafe | Model Insights",
    page_icon=":material/analytics:",
    layout="wide",
    initial_sidebar_state="expanded",
)


def display_class(class_name: str) -> str:
    return class_name.replace("_", " ")


def safe_test_image_path(class_name: str, filename: str) -> Path:
    if class_name not in CLASS_NAMES or Path(filename).name != filename:
        raise ValueError("The evaluation data contains an invalid test image path.")
    image_path = (TEST_DIR / class_name / filename).resolve()
    try:
        image_path.relative_to(TEST_DIR.resolve())
    except ValueError as error:
        raise ValueError("The evaluation data contains an unsafe test image path.") from error
    if not image_path.is_file():
        raise FileNotFoundError(
            f"Test image is missing: {image_path}. Restore the original "
            "`data/processed/test` folder."
        )
    return image_path


@st.cache_resource(show_spinner="Loading the original trained severity model...")
def get_trained_model(checkpoint_path: str, checkpoint_mtime: int):
    del checkpoint_mtime
    return load_trained_model(checkpoint_path)


@st.cache_data(show_spinner=False)
def load_evaluation(directory: str) -> dict[str, object]:
    root = Path(directory)
    required = (
        "summary_metrics.csv",
        "metrics_by_class.csv",
        "confusion_matrix.csv",
        "error_analysis.csv",
        "group_analysis.csv",
        "evaluation_report.png",
    )
    missing = [name for name in required if not (root / name).is_file()]
    prediction_path = root / "predictions.csv"
    if not prediction_path.is_file():
        missing.append("predictions.csv")
    if missing:
        raise FileNotFoundError(
            "Missing R evaluation output(s): "
            + ", ".join(missing)
            + ". Run `Rscript rscript.R` from the project folder first."
        )

    summary_frame = pd.read_csv(root / "summary_metrics.csv")
    if not {"metric", "value"}.issubset(summary_frame.columns):
        raise ValueError("summary_metrics.csv must contain metric and value columns.")
    summary = dict(zip(summary_frame["metric"], summary_frame["value"]))
    classes = pd.read_csv(root / "metrics_by_class.csv")
    errors = pd.read_csv(root / "error_analysis.csv")
    groups = pd.read_csv(root / "group_analysis.csv")
    confusion = pd.read_csv(root / "confusion_matrix.csv", index_col=0)
    predictions = pd.read_csv(prediction_path)

    required_prediction_columns = {"file", "true", "pred", *PROBABILITY_COLUMNS}
    for label, frame, columns in (
        ("predictions.csv", predictions, required_prediction_columns),
        (
            "metrics_by_class.csv",
            classes,
            {"class", "support", "precision", "recall", "specificity", "f1", "auc_ovr"},
        ),
        (
            "error_analysis.csv",
            errors,
            {
                "file",
                "true",
                "pred",
                "correct",
                "predicted_confidence",
                "true_class_probability",
                "top_two_margin",
                "error_type",
            },
        ),
    ):
        absent = columns.difference(frame.columns)
        if absent:
            raise ValueError(f"{label} is missing columns: {', '.join(sorted(absent))}.")

    if predictions.empty or predictions["file"].duplicated().any():
        raise ValueError("predictions.csv must contain unique held-out test images.")
    if set(predictions["true"]) != set(CLASS_NAMES):
        raise ValueError("predictions.csv must contain all three actual severity classes.")
    if not set(predictions["pred"]).issubset(CLASS_NAMES):
        raise ValueError("predictions.csv contains an unsupported predicted class.")
    if set(classes["class"]) != set(CLASS_NAMES):
        raise ValueError("metrics_by_class.csv must contain all three severity classes.")
    errors["correct"] = errors["correct"].astype(str).str.lower().map(
        {"true": True, "false": False}
    )
    if errors["correct"].isna().any():
        raise ValueError("error_analysis.csv contains invalid correct values.")
    for column in ("predicted_confidence", "true_class_probability", "top_two_margin"):
        errors[column] = pd.to_numeric(errors[column], errors="coerce")
        if errors[column].isna().any():
            raise ValueError(f"error_analysis.csv contains invalid {column} values.")

    return {
        "summary": summary,
        "classes": classes,
        "errors": errors,
        "groups": groups,
        "confusion": confusion,
        "predictions": predictions,
        "report_path": root / "evaluation_report.png",
    }


@st.cache_data(show_spinner="Generating Grad-CAM for the held-out example...", max_entries=18)
def load_test_example(
    filename: str,
    true_class: str,
    checkpoint_path: str,
    checkpoint_mtime: int,
) -> dict[str, object]:
    image_path = safe_test_image_path(true_class, filename)
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    model, device = get_trained_model(checkpoint_path, checkpoint_mtime)
    return predict_with_explanations(model, device, image)


def choose_random_examples(
    predictions: pd.DataFrame,
    previous: dict[str, str] | None = None,
) -> dict[str, str]:
    previous = previous or {}
    selected: dict[str, str] = {}
    for class_name in CLASS_NAMES:
        candidates = predictions.loc[
            predictions["true"] == class_name, "file"
        ].astype(str).tolist()
        available = [filename for filename in candidates if filename != previous.get(class_name)]
        if not available:
            available = candidates
        while available:
            filename = secrets.choice(available)
            try:
                safe_test_image_path(class_name, filename)
            except FileNotFoundError:
                available.remove(filename)
                continue
            selected[class_name] = filename
            break
        if class_name not in selected:
            raise FileNotFoundError(
                f"No source test images are available for {display_class(class_name)}."
            )
    return selected


try:
    data = load_evaluation(str(ANALYSIS_DIR))
except (OSError, ValueError, pd.errors.ParserError) as error:
    st.error(f"Could not load original-project R results: {error}")
    st.stop()

summary = data["summary"]
class_metrics = data["classes"]
all_errors = data["errors"]
confusion = data["confusion"]
predictions = data["predictions"]
group_analysis = data["groups"]

with st.container(horizontal=True):
    st.metric("Test images", f"{int(float(summary['samples'])):,}", border=True)
    st.metric(
        "Accuracy",
        f"{float(summary['accuracy']):.1%}",
        help=(
            "Stratified image-bootstrap 95% CI: "
            f"{float(summary['accuracy_bootstrap_95ci_lower']):.1%}–"
            f"{float(summary['accuracy_bootstrap_95ci_upper']):.1%}"
        ),
        border=True,
    )
    st.metric(
        "Macro F1",
        f"{float(summary['macro_f1']):.3f}",
        help=(
            "Stratified image-bootstrap 95% CI: "
            f"{float(summary['macro_f1_bootstrap_95ci_lower']):.3f}–"
            f"{float(summary['macro_f1_bootstrap_95ci_upper']):.3f}"
        ),
        border=True,
    )
    st.metric("Weighted F1", f"{float(summary['weighted_f1']):.3f}", border=True)
    st.metric("Macro ROC AUC", f"{float(summary['macro_auc_ovr']):.3f}", border=True)

with st.sidebar:
    st.markdown("## Explore results")
    st.caption("Filter predictions across the original held-out test set.")
    selected_actual = st.multiselect("Actual class", CLASS_NAMES, default=CLASS_NAMES)
    selected_predicted = st.multiselect("Predicted class", CLASS_NAMES, default=CLASS_NAMES)
    outcome_filter = st.selectbox(
        "Prediction outcome",
        ["All predictions", "Errors only", "High-confidence errors", "Correct predictions"],
    )
    confidence_floor = st.slider(
        "Minimum predicted confidence",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.05,
        format="%.2f",
    )
    path_search = st.text_input("Find image filename", placeholder="e.g. Minor_Accident_0001.jpg")
    st.divider()
    st.caption("Original checkpoint · EfficientNetV2-S · 224 × 224")

filtered = all_errors[
    all_errors["true"].isin(selected_actual)
    & all_errors["pred"].isin(selected_predicted)
    & (all_errors["predicted_confidence"] >= confidence_floor)
].copy()
if outcome_filter == "Errors only":
    filtered = filtered[~filtered["correct"]]
elif outcome_filter == "High-confidence errors":
    filtered = filtered[filtered["error_type"] == "High-confidence error"]
elif outcome_filter == "Correct predictions":
    filtered = filtered[filtered["correct"]]
if path_search.strip():
    filtered = filtered[
        filtered["file"].str.contains(path_search.strip(), case=False, na=False)
    ]

overview_tab, errors_tab, groups_tab, report_tab, predict_tab = st.tabs(
    ["Overview", "Error explorer", "Group analysis", "R report", "Predict & explain"]
)

with overview_tab:
    st.title("Road accident severity")
    st.caption(
        "Held-out test performance and image-level model diagnostics for the original "
        "three-class dataset. Metrics are generated by the reproducible R evaluation."
    )
    st.subheader("Held-out examples with Grad-CAM")
    st.caption(
        "Each column shows one randomly selected test image from its actual class and "
        "the model's prediction for that image."
    )
    gallery_ready = all(
        any(
            (TEST_DIR / class_name / filename).is_file()
            for filename in predictions.loc[
                predictions["true"] == class_name, "file"
            ].astype(str)
        )
        for class_name in CLASS_NAMES
    )
    if not gallery_ready:
        st.info(
            "The original test images are not included in this repository. Restore "
            "`data/processed/test/<class>/` to display examples; image uploads and "
            "the R report remain available without the dataset."
        )
    else:
        if "example_paths" not in st.session_state:
            st.session_state.example_paths = choose_random_examples(predictions)
        checkpoint_mtime = CHECKPOINT_PATH.stat().st_mtime_ns
        if st.button(
            "Show another random set",
            help="Choose a new test image from each actual class.",
        ):
            st.session_state.example_paths = choose_random_examples(
                predictions,
                st.session_state.example_paths,
            )

        example_columns = st.columns(3, gap="medium")
        for column, class_name in zip(example_columns, CLASS_NAMES):
            filename = st.session_state.example_paths[class_name]
            with column:
                st.markdown(f"#### {display_class(class_name)}")
                try:
                    example = load_test_example(
                        filename,
                        class_name,
                        str(CHECKPOINT_PATH),
                        checkpoint_mtime,
                    )
                except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
                    st.error(f"Could not generate this example: {error}")
                    continue
                original_col, gradcam_col = st.columns(2, gap="small")
                with original_col:
                    st.image(
                        example["display_image"],
                        caption="Test image",
                        alt=f"Held-out {display_class(class_name)} example",
                        width="stretch",
                    )
                with gradcam_col:
                    st.image(
                        example["gradcam_overlay"],
                        caption="Grad-CAM",
                        alt=f"Grad-CAM for {filename}",
                        width="stretch",
                    )
                st.caption(f"Actual: {display_class(class_name)}")
                st.markdown(f"**Prediction:** {display_class(example['predicted_class'])}")
                st.caption(f"Confidence: {example['confidence']:.1%}")
                st.caption(f"Test image: {filename}")

    st.subheader("Test-set performance")
    left, right = st.columns(2, gap="large")
    with left:
        confusion_plot = go.Figure(
            data=go.Heatmap(
                z=confusion.to_numpy(),
                x=[display_class(value) for value in confusion.columns],
                y=[display_class(value) for value in confusion.index],
                colorscale=[[0, "#eff6ff"], [0.55, "#60a5fa"], [1, "#1d4ed8"]],
                text=confusion.to_numpy(),
                texttemplate="%{text}",
                hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Images: %{z}<extra></extra>",
            )
        )
        confusion_plot.update_layout(
            template=PLOT_TEMPLATE,
            height=390,
            xaxis_title="Predicted class",
            yaxis_title="Actual class",
            margin={"l": 10, "r": 10, "t": 15, "b": 10},
        )
        st.plotly_chart(
            confusion_plot,
            width="stretch",
            alt="Confusion matrix showing actual and predicted severity classes",
        )
    with right:
        class_long = class_metrics.melt(
            id_vars="class",
            value_vars=["precision", "recall", "f1", "auc_ovr"],
            var_name="Metric",
            value_name="Score",
        )
        class_long["class"] = class_long["class"].map(display_class)
        class_long["Metric"] = class_long["Metric"].replace(
            {"auc_ovr": "ROC AUC", "f1": "F1"}
        )
        class_plot = px.bar(
            class_long,
            x="class",
            y="Score",
            color="Metric",
            barmode="group",
            range_y=[0, 1],
            color_discrete_sequence=["#1d4ed8", "#0f766e", "#d97706", "#7c3aed"],
            template=PLOT_TEMPLATE,
        )
        class_plot.update_layout(
            height=390,
            xaxis_title=None,
            yaxis_title="Score",
            legend_title=None,
            margin={"l": 10, "r": 10, "t": 15, "b": 10},
        )
        st.plotly_chart(
            class_plot,
            width="stretch",
            alt="Precision, recall, F1, and ROC AUC for each accident class",
        )

    st.subheader("Per-class metrics")
    display_metrics = class_metrics.rename(
        columns={
            "class": "Class",
            "support": "Test support",
            "precision": "Precision",
            "recall": "Recall",
            "specificity": "Specificity",
            "f1": "F1",
            "auc_ovr": "ROC AUC",
        }
    ).copy()
    display_metrics["Class"] = display_metrics["Class"].map(display_class)
    st.dataframe(
        display_metrics.style.format(
            {
                "Test support": "{:,.0f}",
                "Precision": "{:.3f}",
                "Recall": "{:.3f}",
                "Specificity": "{:.3f}",
                "F1": "{:.3f}",
                "ROC AUC": "{:.3f}",
            }
        ),
        hide_index=True,
        width="stretch",
        alt="Per-class held-out test metrics",
    )

with errors_tab:
    st.subheader("Find patterns in individual mistakes")
    error_count = int((~all_errors["correct"]).sum())
    high_confidence_count = int(
        (all_errors["error_type"] == "High-confidence error").sum()
    )
    metric_columns = st.columns(3)
    metric_columns[0].metric("Misclassified images", f"{error_count:,}")
    metric_columns[1].metric("High-confidence errors", f"{high_confidence_count:,}")
    metric_columns[2].metric("Rows matching filters", f"{len(filtered):,}")

    error_pairs = (
        all_errors.loc[~all_errors["correct"]]
        .groupby(["true", "pred"], as_index=False)
        .size()
        .sort_values("size", ascending=False)
    )
    if not error_pairs.empty:
        error_pairs["Confusion"] = (
            error_pairs["true"].map(display_class)
            + " → "
            + error_pairs["pred"].map(display_class)
        )
        pair_plot = px.bar(
            error_pairs.head(10),
            x="Confusion",
            y="size",
            color="true",
            color_discrete_map=CLASS_COLORS,
            labels={"size": "Misclassified images", "true": "Actual class"},
            template=PLOT_TEMPLATE,
        )
        pair_plot.update_layout(
            height=330,
            showlegend=False,
            xaxis_title=None,
            margin={"l": 10, "r": 10, "t": 15, "b": 10},
        )
        st.plotly_chart(
            pair_plot,
            width="stretch",
            alt="Most common actual-to-predicted error pairs",
        )

    st.markdown(f"#### Matching predictions · {len(filtered):,} rows")
    st.dataframe(
        filtered[
            [
                "file",
                "true",
                "pred",
                "error_type",
                "predicted_confidence",
                "true_class_probability",
                "top_two_margin",
            ]
        ].style.format(
            {
                "predicted_confidence": "{:.3f}",
                "true_class_probability": "{:.3f}",
                "top_two_margin": "{:.3f}",
            }
        ),
        hide_index=True,
        width="stretch",
        height=420,
        alt="Filtered original test predictions, confidence, and error details",
    )
    st.download_button(
        "Download filtered predictions as CSV",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="filtered_error_analysis.csv",
        mime="text/csv",
    )
    with st.expander("Preview the most confident errors"):
        preview_errors = all_errors.loc[~all_errors["correct"]].nlargest(
            6, "predicted_confidence"
        )
        if not TEST_DIR.is_dir():
            st.info("Original test images are not present, so image previews are unavailable.")
        elif preview_errors.empty:
            st.info("No misclassified test images are available.")
        else:
            preview_columns = st.columns(3)
            for index, (_, row) in enumerate(preview_errors.iterrows()):
                try:
                    image_path = safe_test_image_path(str(row["true"]), str(row["file"]))
                except (OSError, ValueError):
                    continue
                with preview_columns[index % 3]:
                    st.image(
                        str(image_path),
                        width="stretch",
                        alt=f"Actual {row['true']}, predicted {row['pred']}",
                    )
                    st.caption(
                        f"Actual: {display_class(row['true'])} · "
                        f"Predicted: {display_class(row['pred'])} · "
                        f"Confidence: {row['predicted_confidence']:.1%}"
                    )

with groups_tab:
    st.subheader("How do errors vary across classes?")
    st.info(
        "The original dataset does not include verified source-video or scene IDs. "
        "This view therefore groups test errors by actual class; it does not claim "
        "that errors cluster by video, camera, or conditions."
    )
    if group_analysis.empty:
        st.warning("No class-level summary is available. Re-run the R evaluator.")
    else:
        group_plot = px.bar(
            group_analysis,
            x="group_id",
            y="error_rate",
            color="error_rate",
            color_continuous_scale=["#38bdf8", "#fbbf24", "#fb7185"],
            hover_data=["samples", "errors", "most_common_error"],
            labels={
                "group_id": "Actual class",
                "error_rate": "Error rate",
                "samples": "Images",
                "errors": "Errors",
            },
            template=PLOT_TEMPLATE,
        )
        group_plot.update_layout(
            height=400,
            yaxis={"range": [0, 1], "tickformat": ".0%"},
            xaxis_title=None,
            coloraxis_showscale=False,
            margin={"l": 10, "r": 10, "t": 15, "b": 10},
        )
        st.plotly_chart(
            group_plot,
            width="stretch",
            alt="Test error rate grouped by actual accident class",
        )
        st.dataframe(
            group_analysis.sort_values("error_rate", ascending=False),
            hide_index=True,
            width="stretch",
            alt="Class-level error rates and most common mistakes",
        )
        st.download_button(
            "Download class error summary",
            data=group_analysis.to_csv(index=False).encode("utf-8"),
            file_name="class_error_summary.csv",
            mime="text/csv",
        )

with report_tab:
    st.subheader("Reproducible R evaluation report")
    st.image(
        str(data["report_path"]),
        caption=(
            "R-generated plots: confusion matrix, one-vs-rest ROC, per-class scores, "
            "and prediction confidence."
        ),
        width="stretch",
    )
    st.markdown("#### Summary metrics")
    summary_frame = pd.DataFrame(
        [
            {"Metric": metric.replace("_", " ").title(), "Value": value}
            for metric, value in summary.items()
        ]
    )
    st.dataframe(
        summary_frame,
        hide_index=True,
        width="stretch",
        alt="Summary metrics generated by the base-R evaluation script",
    )
    st.download_button(
        "Download summary metrics",
        data=summary_frame.to_csv(index=False).encode("utf-8"),
        file_name="summary_metrics.csv",
        mime="text/csv",
    )

with predict_tab:
    st.subheader("Predict and explain an image")
    st.caption(
        "Upload a road-scene image. The original trained checkpoint returns class "
        "probabilities, Grad-CAM, and input-gradient saliency."
    )
    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png", "webp"],
        max_upload_size=20,
    )
    if uploaded_file is not None:
        if uploaded_file.size > 20 * 1024 * 1024:
            st.error("Please upload an image smaller than 20 MB.")
        else:
            try:
                with Image.open(io.BytesIO(uploaded_file.getvalue())) as source:
                    uploaded_image = source.convert("RGB")
            except (UnidentifiedImageError, OSError, ValueError) as error:
                st.error(f"Could not read the uploaded image: {error}")
            else:
                try:
                    model, device = get_trained_model(
                        str(CHECKPOINT_PATH),
                        CHECKPOINT_PATH.stat().st_mtime_ns,
                    )
                    with st.spinner("Running the original model and generating explanations..."):
                        result = predict_with_explanations(model, device, uploaded_image)
                except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
                    st.error(f"Could not generate a prediction: {error}")
                else:
                    st.success(
                        f"Prediction: {display_class(result['predicted_class'])} "
                        f"({result['confidence']:.1%} confidence)"
                    )
                    image_columns = st.columns(3, gap="medium")
                    for column, image_key, caption in zip(
                        image_columns,
                        ("display_image", "gradcam_overlay", "saliency_overlay"),
                        ("Uploaded image", "Grad-CAM", "Input-gradient saliency"),
                    ):
                        with column:
                            st.image(
                                result[image_key],
                                caption=caption,
                                alt=f"{caption} for the uploaded road image",
                                width="stretch",
                            )
                    probabilities = pd.DataFrame(
                        [
                            {
                                "Class": display_class(class_name),
                                "Probability": result["probabilities"][class_name],
                            }
                            for class_name in CLASS_NAMES
                        ]
                    )
                    probability_plot = px.bar(
                        probabilities,
                        x="Probability",
                        y="Class",
                        orientation="h",
                        color="Class",
                        color_discrete_map={
                            display_class(class_name): CLASS_COLORS[class_name]
                            for class_name in CLASS_NAMES
                        },
                        range_x=[0, 1],
                        template=PLOT_TEMPLATE,
                    )
                    probability_plot.update_layout(
                        height=260,
                        showlegend=False,
                        xaxis_tickformat=".0%",
                        margin={"l": 10, "r": 10, "t": 10, "b": 10},
                    )
                    st.plotly_chart(
                        probability_plot,
                        width="stretch",
                        alt="Model class probabilities for the uploaded image",
                    )

st.divider()
st.caption(
    "Explanations are visual diagnostics, not causal evidence. Performance estimates "
    "refer only to the saved test split and do not establish deployment performance."
)
