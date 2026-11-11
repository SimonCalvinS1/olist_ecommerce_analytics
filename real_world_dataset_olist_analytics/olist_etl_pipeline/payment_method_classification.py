from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = SCRIPT_DIR.parent / "olist_data" / "olist_order_payments_dataset.csv"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "model_outputs"
FEATURE_COLUMNS = [
    "payment_sequential",
    "payment_installments",
    "payment_value",
]
SPLIT_RATIOS = (0.8, 0.1, 0.1)
RANDOM_STATE = 42


def stratified_split(
    labels: np.ndarray,
    ratios: tuple[float, float, float] = SPLIT_RATIOS,
    random_state: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split each class proportionally, retaining classes with at least 3 rows."""
    rng = np.random.default_rng(random_state)
    split_indices: list[list[int]] = [[], [], []]

    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        exact_counts = np.asarray(ratios) * len(indices)
        counts = np.floor(exact_counts).astype(int)
        for split_index in np.argsort(-(exact_counts - counts))[: len(indices) - counts.sum()]:
            counts[split_index] += 1

        if len(indices) >= len(ratios):
            for split_index in np.flatnonzero(counts == 0):
                donor = int(np.argmax(counts))
                counts[donor] -= 1
                counts[split_index] += 1

        start = 0
        for split_index, count in enumerate(counts):
            split_indices[split_index].extend(indices[start : start + count].tolist())
            start += count

    result = []
    for indices in split_indices:
        rng.shuffle(indices)
        result.append(np.asarray(indices, dtype=int))
    return result[0], result[1], result[2]


def build_models(class_count: int) -> dict[str, object]:
    return {
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "XGBoost": XGBClassifier(
            objective="multi:softprob",
            num_class=class_count,
            eval_metric="mlogloss",
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            tree_method="hist",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
    }


def normalize_shap_values(values: object) -> np.ndarray:
    if isinstance(values, list):
        return np.stack(values, axis=-1)
    if hasattr(values, "values"):
        values = values.values
    result = np.asarray(values)
    if result.ndim == 2:
        result = result[:, :, np.newaxis]
    if result.ndim != 3:
        raise ValueError(f"Unexpected SHAP values shape: {result.shape}")
    return result


def save_shap_summary(
    shap_values: np.ndarray, class_names: np.ndarray, output_path: Path
) -> None:
    class_importance = np.abs(shap_values).mean(axis=0)
    feature_importance = class_importance.sum(axis=1)
    feature_order = np.argsort(feature_importance)[::-1]
    positions = np.arange(len(feature_order))
    left = np.zeros(len(feature_order))
    colors = plt.get_cmap("tab10")(np.arange(len(class_names)))
    figure, axis = plt.subplots(figsize=(11, 6))
    for class_index, class_name in enumerate(class_names):
        values = class_importance[feature_order, class_index]
        axis.barh(
            positions,
            values,
            left=left,
            color=colors[class_index],
            label=class_name,
            height=0.65,
        )
        left += values

    axis.set_yticks(positions)
    axis.set_yticklabels([FEATURE_COLUMNS[index] for index in feature_order])
    axis.invert_yaxis()
    axis.set_xlabel("Mean absolute SHAP value (average impact magnitude)")
    axis.set_title("Feature impact by payment type")
    axis.legend(title="Payment type", loc="center left", bbox_to_anchor=(1.02, 0.5))
    figure.tight_layout(rect=(0, 0, 0.78, 1))
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def save_shap_artifacts(
    model: object,
    X_test: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_encoder: LabelEncoder,
    row_indices: np.ndarray,
    output_dir: Path,
) -> None:
    sample_count = min(1000, len(X_test))
    sample = X_test.iloc[:sample_count]
    sample_row_indices = row_indices[:sample_count]
    explainer = shap.TreeExplainer(model)
    shap_values = normalize_shap_values(explainer.shap_values(sample))
    expected_values = np.asarray(explainer.expected_value).reshape(-1)
    class_count = len(label_encoder.classes_)

    if shap_values.shape[1] != len(FEATURE_COLUMNS):
        raise ValueError(f"SHAP feature count does not match model inputs: {shap_values.shape}")
    if shap_values.shape[2] != class_count:
        raise ValueError(f"SHAP class count does not match target classes: {shap_values.shape}")

    importance = np.abs(shap_values).mean(axis=(0, 2))
    pd.DataFrame(
        {"feature": FEATURE_COLUMNS, "mean_absolute_shap_value": importance}
    ).sort_values("mean_absolute_shap_value", ascending=False).to_csv(
        output_dir / "shap_feature_importance.csv", index=False
    )

    attribution_data: dict[str, np.ndarray] = {
        "source_row_index": sample_row_indices,
        "actual_payment_type": label_encoder.inverse_transform(y_true[:sample_count]),
        "predicted_payment_type": label_encoder.inverse_transform(y_pred[:sample_count]),
    }
    for class_index, class_name in enumerate(label_encoder.classes_):
        if len(expected_values) == 1:
            base_value = np.repeat(expected_values[0], sample_count)
        else:
            base_value = np.repeat(expected_values[class_index], sample_count)
        attribution_data[f"base_value__{class_name}"] = base_value
        for feature_index, feature_name in enumerate(FEATURE_COLUMNS):
            attribution_data[f"shap__{class_name}__{feature_name}"] = shap_values[
                :, feature_index, class_index
            ]
    pd.DataFrame(attribution_data).to_csv(output_dir / "shap_test_attributions.csv", index=False)

    save_shap_summary(shap_values, label_encoder.classes_, output_dir / "shap_summary.png")


def run_pipeline(data_path: Path, output_dir: Path) -> dict[str, object]:
    data = pd.read_csv(data_path)
    required_columns = {*FEATURE_COLUMNS, "payment_type"}
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing_columns)}")
    if data[list(required_columns)].isna().any().any():
        raise ValueError("Feature and target columns must not contain missing values.")

    X = data[FEATURE_COLUMNS]
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(data["payment_type"].astype(str))
    train_indices, validation_indices, test_indices = stratified_split(y)

    X_train, y_train = X.iloc[train_indices], y[train_indices]
    X_validation, y_validation = X.iloc[validation_indices], y[validation_indices]
    X_test, y_test = X.iloc[test_indices], y[test_indices]

    validation_results: dict[str, dict[str, float]] = {}
    models = build_models(len(label_encoder.classes_))
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        validation_predictions = model.predict(X_validation)
        validation_results[model_name] = {
            "macro_f1": float(
                f1_score(y_validation, validation_predictions, average="macro", zero_division=0)
            ),
            "accuracy": float(accuracy_score(y_validation, validation_predictions)),
        }

    best_model_name = max(
        validation_results,
        key=lambda name: (
            validation_results[name]["macro_f1"],
            validation_results[name]["accuracy"],
        ),
    )
    best_model = models[best_model_name]
    test_predictions = best_model.predict(X_test)
    report = classification_report(
        y_test,
        test_predictions,
        labels=np.arange(len(label_encoder.classes_)),
        target_names=label_encoder.classes_,
        output_dict=True,
        zero_division=0,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = pd.DataFrame(
        {
            "source_row_index": test_indices,
            "actual_payment_type": label_encoder.inverse_transform(y_test),
            "predicted_payment_type": label_encoder.inverse_transform(test_predictions),
        }
    )
    predictions.to_csv(output_dir / "test_predictions.csv", index=False)
    metrics: dict[str, object] = {
        "target": "payment_type",
        "features": FEATURE_COLUMNS,
        "split_sizes": {
            "train": len(train_indices),
            "validate": len(validation_indices),
            "test": len(test_indices),
        },
        "class_counts": data["payment_type"].value_counts().to_dict(),
        "validation_metrics": validation_results,
        "selected_model": best_model_name,
        "test_metrics": {
            "accuracy": float(accuracy_score(y_test, test_predictions)),
            "macro_f1": float(
                f1_score(y_test, test_predictions, average="macro", zero_division=0)
            ),
            "classification_report": report,
        },
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)

    save_shap_artifacts(
        best_model,
        X_test,
        y_test,
        test_predictions,
        label_encoder,
        test_indices,
        output_dir,
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare Random Forest and XGBoost for Olist payment type prediction."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH, help="Input CSV path.")
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for metrics and predictions."
    )
    args = parser.parse_args()

    metrics = run_pipeline(args.data, args.output_dir)
    print(f"Selected model: {metrics['selected_model']}")
    print(f"Validation metrics: {metrics['validation_metrics']}")
    print(f"Test metrics: {metrics['test_metrics']}")
    print(f"Artifacts written to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
