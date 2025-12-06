import os
import time
import glob

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    classification_report as sk_classification_report,
    brier_score_loss,
)
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression as SkLogReg
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.calibration import CalibratedClassifierCV

import kagglehub

from src.data_and_features import load_and_prepare_data
from src.metrics_and_utils import (
    evaluate_imbalance_metrics,
    pick_best_threshold,
    plot_training_curves,
    plot_confusion_heatmap,
)

# ============================================
#  FFNN Architectures & Loss Utilities
# ============================================

class MLP_Shallow(nn.Module):
    """
    2-hidden-layer FFNN used as our main shallow model.
    """
    def __init__(self, d_in, drop1=0.3, drop2=0.2):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(d_in, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(drop1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(drop2),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        return self.model(x)


class MLP_Deep(nn.Module):
    """
    Deeper 4-layer FFNN for Ablation 2.
    """
    def __init__(self, d_in, drop=0.3):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(d_in, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(drop),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(drop),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        return self.model(x)


def focal_loss(logits, targets, alpha=None, gamma=2.0):
    """
    Focal loss built on top of cross-entropy.
    logits: [N, 2]
    targets: [N] with values in {0,1}
    alpha: optional class-weight tensor [2]
    """
    ce = F.cross_entropy(logits, targets, reduction="none", weight=alpha)
    pt = torch.exp(-ce)             # predicted prob of the true class
    loss = ((1 - pt) ** gamma) * ce
    return loss.mean()


def calculate_class_weights(y_train, mode="inverse_freq"):
    """
    Returns a torch.float32 tensor of length 2 with class weights.
    Modes:
      - 'unweighted'
      - 'balanced'     (sklearn balanced formula)
      - 'inverse_freq' (1 / N_i, re-normalized)
      - 'focal'        (returns inverse_freq weights for focal)
    """
    classes = np.unique(y_train)

    if mode == "unweighted":
        weights_np = np.ones(len(classes), dtype=np.float64)
        print("Using UNWEIGHTED loss (weights = [1.0, 1.0])")

    elif mode == "balanced":
        weights_np = compute_class_weight(
            class_weight="balanced",
            classes=classes,
            y=y_train,
        )
        print(f"Using BALANCED weights (sklearn): {weights_np.round(4)}")

    elif mode == "inverse_freq":
        counts = np.bincount(y_train)
        # (1 / N_i) * (Total / n_classes), then re-normalize to n_classes
        weights_np = (1.0 / counts) * len(y_train) / len(classes)
        weights_np = weights_np / np.sum(weights_np) * len(classes)
        print(f"Using INVERSE FREQUENCY weights (re-normalized): {weights_np.round(4)}")

    elif mode == "focal":
        # For focal, we still need class weights; reuse inverse_freq scheme
        counts = np.bincount(y_train)
        weights_np = (1.0 / counts) * len(y_train) / len(classes)
        weights_np = weights_np / np.sum(weights_np) * len(classes)
        print(f"Using FOCAL loss with base weights: {weights_np.round(4)}")

    else:
        raise ValueError("Invalid mode. Choose 'unweighted', 'balanced', 'inverse_freq', or 'focal'.")

    return torch.tensor(weights_np, dtype=torch.float32)


def train_ffnn(
    model,
    X_train_proc,
    y_train,
    X_val_proc,
    y_val,
    weight_tensor=None,
    n_epochs=15,
    batch_size=256,
    lr=1e-3,
    use_focal=False,
):
    """
    Generic FFNN training loop for ablations.
    Returns:
      model, train_loss_history, val_f1_history
    """
    X_train_t = torch.tensor(X_train_proc, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t = torch.tensor(X_val_proc, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.long)

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(X_val_t, y_val_t),
        batch_size=batch_size,
        shuffle=False,
    )

    if not use_focal:
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    else:
        # focal_loss will be called manually
        criterion = None

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_loss_history, val_f1_history = [], []

    for epoch in range(n_epochs):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            if use_focal:
                loss = focal_loss(logits, yb, alpha=weight_tensor, gamma=2.0)
            else:
                loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / max(1, len(train_loader))
        train_loss_history.append(train_loss)

        # Validation Weighted F1
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                logits = model(xb)
                preds = logits.argmax(1)
                all_preds.extend(preds.tolist())
                all_targets.extend(yb.tolist())

        val_report = sk_classification_report(
            all_targets, all_preds, output_dict=True, zero_division=0
        )
        val_f1 = val_report["weighted avg"]["f1-score"]
        val_f1_history.append(val_f1)

        print(
            f"Epoch {epoch+1:02d} | "
            f"TrainLoss {train_loss:.4f} | ValF1 {val_f1:.4f}"
        )

    return model, train_loss_history, val_f1_history


# ============================================
#  Ablation 1: Class Weighting / Loss
# ============================================

def run_ablation1_loss_weighting(
    X_train_proc,
    y_train,
    X_val_proc,
    y_val,
    X_test_proc,
    y_test,
    results_dir="results",
):
    """
    Compare different loss / class-weight setups for the same shallow FFNN.
    Modes: unweighted, balanced, inverse_freq, focal
    """
    os.makedirs(results_dir, exist_ok=True)

    modes = ["unweighted", "balanced", "inverse_freq", "focal"]
    records = []

    for mode in modes:
        print("\n" + "=" * 60)
        print(f"Ablation 1 – Training FFNN with mode: {mode.upper()}")

        class_w = calculate_class_weights(y_train, mode="inverse_freq" if mode == "focal" else mode)

        d_in = X_train_proc.shape[1]
        model = MLP_Shallow(d_in, drop1=0.3, drop2=0.2)

        use_focal = mode == "focal"

        start = time.time()
        model, train_loss_hist, val_f1_hist = train_ffnn(
            model=model,
            X_train_proc=X_train_proc,
            y_train=y_train,
            X_val_proc=X_val_proc,
            y_val=y_val,
            weight_tensor=class_w,
            n_epochs=15,
            batch_size=256,
            lr=1e-3,
            use_focal=use_focal,
        )
        elapsed = time.time() - start
        print(f"Training time ({mode}): {elapsed:.2f} seconds")

        # Plot curves
        plot_training_curves(
            train_loss_history=train_loss_hist,
            val_metric_history=val_f1_hist,
            metric_name="Weighted F1",
            title_prefix=f"FFNN Ablation1 ({mode})",
            save_path=os.path.join(results_dir, f"ablation1_{mode}_curves.png"),
        )

        # Evaluate with threshold tuned on validation
        X_val_t = torch.tensor(X_val_proc, dtype=torch.float32)
        X_test_t = torch.tensor(X_test_proc, dtype=torch.float32)
        model.eval()
        with torch.no_grad():
            logits_val = model(X_val_t)
            probs_val = F.softmax(logits_val, dim=1)[:, 1].cpu().numpy()

        opt_th = pick_best_threshold(y_val, probs_val, mode="f1")

        with torch.no_grad():
            logits_test = model(X_test_t)
            probs_test = F.softmax(logits_test, dim=1)[:, 1].cpu().numpy()
            preds_test = (probs_test >= opt_th).astype(int)

        metrics = evaluate_imbalance_metrics(
            y_test,
            preds_test,
            probs_test,
            model_name=f"FFNN Ablation1 ({mode})",
        )

        # Brier for completeness
        brier = brier_score_loss(y_test, probs_test)

        metrics_row = {
            "mode": mode,
            "opt_threshold": float(opt_th),
            "brier": brier,
            "n_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "train_time_sec": elapsed,
        }
        metrics_row.update(metrics)
        records.append(metrics_row)

        # Confusion heatmap
        plot_confusion_heatmap(
            y_true=y_test,
            y_pred=preds_test,
            title=f"Confusion Matrix – Ablation1 ({mode})",
            save_path=os.path.join(results_dir, f"ablation1_{mode}_cm.png"),
        )

    df_ab1 = pd.DataFrame(records)
    out_csv = os.path.join(results_dir, "ablation1_ffnn_loss.csv")
    df_ab1.to_csv(out_csv, index=False)
    print(f"\nSaved Ablation 1 metrics to {out_csv}")
    print(df_ab1)


# ============================================
#  Ablation 2: Depth (Shallow vs Deep)
# ============================================

def run_ablation2_depth(
    X_train_proc,
    y_train,
    X_val_proc,
    y_val,
    X_test_proc,
    y_test,
    class_weights_mode="inverse_freq",
    results_dir="results",
):
    """
    Compare shallow vs deep FFNN under the same class-weighting scheme.
    """
    os.makedirs(results_dir, exist_ok=True)

    # Use the same weights (inverse_freq) for both
    class_w = calculate_class_weights(y_train, mode=class_weights_mode)

    d_in = X_train_proc.shape[1]

    configs = [
        ("shallow", lambda: MLP_Shallow(d_in, drop1=0.3, drop2=0.2)),
        ("deep",    lambda: MLP_Deep(d_in, drop=0.3)),
    ]

    records = []

    X_val_t = torch.tensor(X_val_proc, dtype=torch.float32)
    X_test_t = torch.tensor(X_test_proc, dtype=torch.float32)

    for depth_name, model_fn in configs:
        print("\n" + "=" * 60)
        print(f"Ablation 2 – Training {depth_name.upper()} Network")

        model = model_fn()

        start = time.time()
        model, train_loss_hist, val_f1_hist = train_ffnn(
            model=model,
            X_train_proc=X_train_proc,
            y_train=y_train,
            X_val_proc=X_val_proc,
            y_val=y_val,
            weight_tensor=class_w,
            n_epochs=15,
            batch_size=256,
            lr=1e-3,
            use_focal=False,
        )
        elapsed = time.time() - start
        print(f"Training Complete for {depth_name} in {elapsed:.2f} seconds.")

        # Curves
        plot_training_curves(
            train_loss_history=train_loss_hist,
            val_metric_history=val_f1_hist,
            metric_name="Weighted F1",
            title_prefix=f"FFNN Ablation2 ({depth_name})",
            save_path=os.path.join(results_dir, f"ablation2_{depth_name}_curves.png"),
        )

        # Evaluation
        model.eval()
        with torch.no_grad():
            logits_val = model(X_val_t)
            probs_val = F.softmax(logits_val, dim=1)[:, 1].cpu().numpy()

        opt_th = pick_best_threshold(y_val, probs_val, mode="f1")

        with torch.no_grad():
            logits_test = model(X_test_t)
            probs_test = F.softmax(logits_test, dim=1)[:, 1].cpu().numpy()
            preds_test = (probs_test >= opt_th).astype(int)

        metrics = evaluate_imbalance_metrics(
            y_test,
            preds_test,
            probs_test,
            model_name=f"FFNN Ablation2 ({depth_name})",
        )

        brier = brier_score_loss(y_test, probs_test)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        metrics_row = {
            "depth": depth_name,
            "opt_threshold": float(opt_th),
            "brier": brier,
            "n_params": n_params,
            "train_time_sec": elapsed,
        }
        metrics_row.update(metrics)
        records.append(metrics_row)

        plot_confusion_heatmap(
            y_true=y_test,
            y_pred=preds_test,
            title=f"Confusion Matrix – Ablation2 ({depth_name})",
            save_path=os.path.join(results_dir, f"ablation2_{depth_name}_cm.png"),
        )

    df_ab2 = pd.DataFrame(records)
    out_csv = os.path.join(results_dir, "ablation2_depth.csv")
    df_ab2.to_csv(out_csv, index=False)
    print(f"\nSaved Ablation 2 metrics to {out_csv}")
    print(df_ab2)


# ============================================
#  Ablation 3: Feature Group Minus-One (LogReg)
# ============================================

def build_engineered_df_for_ablation(sample_n=300_000, random_state=42):
    """
    Rebuilds the engineered dataframe needed for the Logistic Regression
    minus-one feature ablation, mirroring the notebook pipeline.
    """
    np.random.seed(random_state)

    path = kagglehub.dataset_download("sobhanmoosavi/us-accidents")
    print("Downloaded to:", path)

    csv_candidates = glob.glob(os.path.join(path, "**", "US_Accidents*.csv"), recursive=True)
    if not csv_candidates:
        csv_candidates = glob.glob(os.path.join(path, "**", "*.csv"), recursive=True)

    if not csv_candidates:
        raise FileNotFoundError("Couldn't find a CSV in the downloaded dataset.")

    csv_sizes = [(p, os.path.getsize(p)) for p in csv_candidates]
    csv_path = sorted(csv_sizes, key=lambda x: x[1], reverse=True)[0][0]
    print("Using CSV:", csv_path)

    use_cols = [
        "Severity", "Start_Time", "Weather_Condition",
        "Visibility(mi)", "Temperature(F)", "Wind_Speed(mph)",
        "Precipitation(in)", "Sunrise_Sunset",
    ]

    header_cols = pd.read_csv(csv_path, nrows=0).columns.tolist()
    present_cols = [c for c in use_cols if c in header_cols]

    if sample_n is None:
        df = pd.read_csv(csv_path, usecols=present_cols, low_memory=False)
    else:
        total_rows = sum(
            1 for _ in open(csv_path, "r", encoding="utf-8", errors="ignore")
        ) - 1
        keep_idx = set(
            np.random.choice(total_rows, size=min(sample_n, total_rows), replace=False)
        )

        def row_use(i):
            return (i - 1) in keep_idx

        df = pd.read_csv(
            csv_path,
            usecols=present_cols,
            low_memory=False,
            skiprows=lambda i: i > 0 and not row_use(i),
        )

    print("Raw sampled shape:", df.shape)

    # Binary target
    df = df[df["Severity"].isin([1, 2, 3, 4])].copy()
    df["target"] = df["Severity"].apply(lambda x: 0 if x in [1, 2] else 1)

    # Time features
    ts = pd.to_datetime(df["Start_Time"], errors="coerce")
    df["hour"] = ts.dt.hour
    df["dow"] = ts.dt.dayofweek
    df["is_weekend"] = (df["dow"] >= 5).astype(int)

    df["is_rush_hour"] = (
        ((df["hour"] >= 7) & (df["hour"] <= 9))
        | ((df["hour"] >= 16) & (df["hour"] <= 18))
    ).astype(int)

    # Weather simplification
    def simplify_weather(w):
        w = str(w).lower()
        if "thunder" in w:
            return "Thunderstorm"
        if "snow" in w or "sleet" in w or "blizzard" in w:
            return "Snow"
        if "rain" in w or "drizzle" in w or "shower" in w:
            return "Rain"
        if "fog" in w or "mist" in w or "haze" in w or "smoke" in w:
            return "Fog"
        if "cloud" in w or "overcast" in w:
            return "Cloudy"
        if "clear" in w or "fair" in w:
            return "Clear"
        return "Other"

    df["Weather_Simple"] = (
        df["Weather_Condition"].apply(simplify_weather)
        if "Weather_Condition" in df.columns
        else "Other"
    )

    df["Daylight"] = (
        df["Sunrise_Sunset"].map({"Day": 1, "Night": 0})
        if "Sunrise_Sunset" in df.columns
        else np.nan
    )

    if "Visibility(mi)" in df.columns:
        df["low_visibility"] = (df["Visibility(mi)"] < 3.0).astype(int)

    # Keep engineered set
    keep_cols = [
        "target",
        "Visibility(mi)", "Temperature(F)", "Wind_Speed(mph)", "Precipitation(in)",
        "hour", "is_weekend", "is_rush_hour", "low_visibility",
        "Weather_Simple", "Daylight",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].copy()
    print("Engineered df shape:", df.shape)
    return df


def run_logreg_minus_one_ablation(df, results_dir="results"):
    """
    Logistic Regression minus-one feature-group ablation, as in the notebook.
    Saves CSV:
        results/logreg_minus_one_ablation.csv
    """
    os.makedirs(results_dir, exist_ok=True)

    BASE_FEATURES = [
        "Temperature(F)", "Wind_Speed(mph)", "Weather_Simple",
    ]

    GROUP_VISIBILITY = ["Visibility(mi)", "low_visibility"]
    GROUP_PRECIP     = ["Precipitation(in)"]
    GROUP_TIME       = ["hour", "is_weekend", "is_rush_hour"]
    GROUP_DAYNIGHT   = ["Daylight"]

    ablation_variants = {
        "full": BASE_FEATURES + GROUP_VISIBILITY + GROUP_PRECIP + GROUP_TIME + GROUP_DAYNIGHT,
        "minus_visibility": BASE_FEATURES + GROUP_PRECIP + GROUP_TIME + GROUP_DAYNIGHT,
        "minus_precip": BASE_FEATURES + GROUP_VISIBILITY + GROUP_TIME + GROUP_DAYNIGHT,
        "minus_time": BASE_FEATURES + GROUP_VISIBILITY + GROUP_PRECIP + GROUP_DAYNIGHT,
        "minus_daynight": BASE_FEATURES + GROUP_VISIBILITY + GROUP_PRECIP + GROUP_TIME,
    }

    num_pipe = Pipeline(
        steps=[
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
        ]
    )
    cat_pipe = Pipeline(
        steps=[
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("oh", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    df_ab_source = df.copy()
    logreg_results = []
    param_grid = {"logreg__C": [0.01, 0.1, 1.0, 10.0]}

    for name, feat_list in ablation_variants.items():
        print("\n" + "=" * 60)
        print(f"LogReg minus-one ablation: {name}")

        sel_features = [c for c in feat_list if c in df_ab_source.columns]
        if len(sel_features) == 0:
            print(f"Skipping {name}: no features present.")
            continue

        df_ab = df_ab_source[["target"] + sel_features].copy()
        y_ab = df_ab["target"].values
        X_ab = df_ab.drop(columns=["target"])

        # Numeric & categorical for this ablation
        num_cols_ab = [
            c for c in [
                "Visibility(mi)", "Temperature(F)", "Wind_Speed(mph)",
                "Precipitation(in)", "hour", "is_weekend", "is_rush_hour",
                "low_visibility",
            ] if c in X_ab.columns
        ]
        cat_cols_ab = [
            c for c in ["Weather_Simple", "Daylight"] if c in X_ab.columns
        ]

        preproc_ab = ColumnTransformer(
            transformers=[
                ("num", num_pipe, num_cols_ab),
                ("cat", cat_pipe, cat_cols_ab),
            ]
        )

        # Train/val/test split
        X_train_ab, X_temp_ab, y_train_ab, y_temp_ab = train_test_split(
            X_ab, y_ab, test_size=0.2, stratify=y_ab, random_state=42
        )
        X_val_ab, X_test_ab, y_val_ab, y_test_ab = train_test_split(
            X_temp_ab, y_temp_ab, test_size=0.5, stratify=y_temp_ab, random_state=42
        )

        classes_ab = np.unique(y_train_ab)
        cw_arr = compute_class_weight(
            class_weight="balanced", classes=classes_ab, y=y_train_ab
        )
        class_weights_train_ab = {int(c): w for c, w in zip(classes_ab, cw_arr)}
        print("Class weights (train):", class_weights_train_ab)

        log_reg_pipeline = Pipeline(
            steps=[
                ("preprocessor", preproc_ab),
                ("logreg", SkLogReg(
                    solver="liblinear",
                    max_iter=500,
                    class_weight=class_weights_train_ab,
                )),
            ]
        )

        grid = GridSearchCV(
            log_reg_pipeline,
            param_grid,
            cv=3,
            scoring="f1",
            n_jobs=-1,
            verbose=0,
        )
        grid.fit(X_train_ab, y_train_ab)
        best_model_ab = grid.best_estimator_

        print(" Best params:", grid.best_params_)
        print(" Best CV f1:", grid.best_score_)

        # Validation: pick threshold
        y_val_proba_ab = best_model_ab.predict_proba(X_val_ab)[:, 1]
        opt_th_ab = pick_best_threshold(y_val_ab, y_val_proba_ab, mode="f1")

        y_val_pred_ab = (y_val_proba_ab >= opt_th_ab).astype(int)
        _ = evaluate_imbalance_metrics(
            y_val_ab,
            y_val_pred_ab,
            y_val_proba_ab,
            model_name=f"LogReg ({name}, val)",
        )

        # Test (raw)
        y_test_proba_ab = best_model_ab.predict_proba(X_test_ab)[:, 1]
        y_test_pred_ab = (y_test_proba_ab >= opt_th_ab).astype(int)

        eval_test = evaluate_imbalance_metrics(
            y_test_ab,
            y_test_pred_ab,
            y_test_proba_ab,
            model_name=f"LogReg ({name}, test)",
        )

        # Calibration (isotonic)
        calibrated = CalibratedClassifierCV(
            estimator=best_model_ab,
            method="isotonic",
            cv="prefit",
        )
        calibrated.fit(X_val_ab, y_val_ab)

        y_test_proba_cal = calibrated.predict_proba(X_test_ab)[:, 1]
        opt_th_cal_ab = pick_best_threshold(
            y_val_ab,
            calibrated.predict_proba(X_val_ab)[:, 1],
            mode="f1",
        )
        y_test_pred_cal = (y_test_proba_cal >= opt_th_cal_ab).astype(int)

        eval_test_cal = evaluate_imbalance_metrics(
            y_test_ab,
            y_test_pred_cal,
            y_test_proba_cal,
            model_name=f"LogReg ({name} + calibrated, test)",
        )

        logreg_results.append(
            {
                "ablation": name,
                "n_features": len(sel_features),
                "best_C": grid.best_params_["logreg__C"],
                "cv_f1": grid.best_score_,
                "val_opt_th": float(opt_th_ab),
                "test_severe_precision": eval_test["severe_precision"],
                "test_severe_recall": eval_test["severe_recall"],
                "test_severe_f1": eval_test["severe_f1"],
                "test_macro_f1": eval_test["macro_f1"],
                "test_weighted_f1": eval_test["weighted_f1"],
                "test_bal_acc": eval_test["balanced_accuracy"],
                "test_roc_auc": eval_test["roc_auc"],
                "test_pr_auc": eval_test["pr_auc"],
                "test_brier_raw": float(brier_score_loss(y_test_ab, y_test_proba_ab)),
                "test_brier_cal": float(brier_score_loss(y_test_ab, y_test_proba_cal)),
            }
        )

    logreg_metrics_df = pd.DataFrame(logreg_results)
    out_csv = os.path.join(results_dir, "logreg_minus_one_ablation.csv")
    logreg_metrics_df.to_csv(out_csv, index=False)
    print(f"\nSaved logistic minus-one ablation metrics to {out_csv}")
    print(logreg_metrics_df)


# ============================================
#  main(): run all ablations
# ============================================

def main():
    # -------------------------------
    # 1) Load data & preprocessing
    # -------------------------------
    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        preprocess,
        class_weights_train,
        num_cols,
        cat_cols,
    ) = load_data_and_features(sample_n=300_000, random_state=42)

    # Use same preprocessor to get numeric arrays
    X_train_proc = preprocess.fit_transform(X_train)
    X_val_proc = preprocess.transform(X_val)
    X_test_proc = preprocess.transform(X_test)

    # -------------------------------
    # 2) Ablation 1 – Loss / weighting
    # -------------------------------
    run_ablation1_loss_weighting(
        X_train_proc=X_train_proc,
        y_train=y_train,
        X_val_proc=X_val_proc,
        y_val=y_val,
        X_test_proc=X_test_proc,
        y_test=y_test,
        results_dir="results",
    )

    # -------------------------------
    # 3) Ablation 2 – Shallow vs Deep
    # -------------------------------
    run_ablation2_depth(
        X_train_proc=X_train_proc,
        y_train=y_train,
        X_val_proc=X_val_proc,
        y_val=y_val,
        X_test_proc=X_test_proc,
        y_test=y_test,
        class_weights_mode="inverse_freq",
        results_dir="results",
    )

    # -------------------------------
    # 4) Ablation 3 – Logistic minus-one
    # -------------------------------
    df_ablation = build_engineered_df_for_ablation(
        sample_n=300_000,
        random_state=42,
    )
    run_logreg_minus_one_ablation(df_ablation, results_dir="results")


if __name__ == "__main__":
    main()
