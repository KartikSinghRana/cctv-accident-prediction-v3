# Road Accident Severity Dashboard

This dashboard wraps the original three-class CNN and its existing held-out
test predictions. It does not retrain the model or use the larger CCTV dataset.

## Run the R evaluation

From this project directory, run:

```powershell
& "C:\Program Files\R\R-4.6.1\bin\Rscript.exe" rscript.R
```

If `Rscript` is already on `PATH`, you can use `Rscript rscript.R` instead.

The base-R evaluator reads `r_analysis/predictions.csv` and writes the summary
metrics, per-class metrics, confusion matrix, per-image error analysis,
class-level error summary, and `r_analysis/evaluation_report.png`.
It also refreshes `r_analysis/r_evaluation.png` for compatibility with the
original report filename.

## Run the Streamlit dashboard

Use the same Python environment that has the original model dependencies
installed. If needed, install the dashboard dependencies:

```powershell
python -m pip install -r requirements-dashboard.txt
```

Then start Streamlit from this directory:

```powershell
streamlit run streamlit_app.py
```

The dashboard uses `checkpoints/best_model.pth` and the existing images under
`data/processed/test/` when available. It provides a light-themed overview,
randomly selected test images with Grad-CAM, individual error exploration, a
transparent class-level error analysis, the R-generated report, and
uploaded-image prediction with Grad-CAM and input-gradient saliency. The
dataset files are intentionally excluded from this repository; without them,
the dashboard still displays the report and supports image uploads, while
test-image previews are hidden.

To restore the local image gallery, place the original processed test images
at `data/processed/test/<class>/`, retaining the filenames in
`r_analysis/predictions.csv`. The original dataset inputs expected by the
notebook are under `dataset3/`; dataset contents are not redistributed here.
The checkpoint is included for immediate inference; GitHub may warn that it
exceeds its recommended individual file size.

The output-free notebook copy is `road_accident_severity_cnn_v2_source.ipynb`.
The original executed notebook is retained locally but excluded from the repo
because its saved outputs embed many original dataset images.

The original dataset does not provide verified source-video identifiers, so
the Group analysis tab summarizes errors by actual class rather than making
unsupported video-level claims. Explanation maps are visual diagnostics, not
causal evidence.
