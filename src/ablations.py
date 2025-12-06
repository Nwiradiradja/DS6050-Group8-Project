import os
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split, GridSearchCV

from src.metrics_and_utils import (
    pick_best_threshold,
    evaluate_imbalance_metrics,
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

def run_ablation1_loss_weighting(
    X_train_proc,
    y_train,
    X_val_proc,
    y_val,
    X_test_proc,
    y_test,
    results_dir="results",
):
    print("\n[ABLATION 1] Loss Weighting")

    modes = ["unweighted", "balanced", "inverse_freq"]
    results = []

    for mode in modes:
        print(f"\n--- Running mode: {mode} ---")

        # Compute weights
        if mode == "unweighted":
            weights = torch.tensor([1.0, 1.0], dtype=torch.float32)
        elif mode == "balanced":
            classes = np.unique(y_train)
            cw = compute_class_weight(
                class_weight="balanced",
                classes=classes,
                y=y_train,
            )
            weights = torch.tensor(cw, dtype=torch.float32)
        elif mode == "inverse_freq":
            counts = np.bincount(y_train)
            inv = 1.0 / counts
            weights = torch.tensor(inv / inv.sum() * 2, dtype=torch.float32)

        # PyTorch setup
        X_train_t = torch.tensor(X_train_proc, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.long)
        X_val_t = torch.tensor(X_val_proc, dtype=torch.float32)
        y_val_t = torch.tensor(y_val, dtype=torch.long)
        X_test_t = torch.tensor(X_test_proc, dtype=torch.float32)

        train_loader = DataLoader(
            TensorDataset(X_train_t, y_train_t), batch_size=256, shuffle=True
        )

        d_in = X_train_proc.shape[1]

        model = nn.Sequential(
            nn.Linear(d_in, 64), nn.ReLU(), nn.BatchNorm1d(64), nn.Dropout(0.3),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, 2)
        )

        criterion = nn.CrossEntropyLoss(weight=weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Training
        for epoch in range(10):
            model.train()
            for xb, yb in train_loader:
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()

        # Evaluate on validation
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_probs = F.softmax(val_logits, dim=1)[:, 1].numpy()

        best_th = pick_best_threshold(y_val, val_probs, mode="f1")

        with torch.no_grad():
            test_logits = model(X_test_t)
            test_probs = F.softmax(test_logits, dim=1)[:, 1].numpy()
            test_preds = (test_probs >= best_th).astype(int)

        metrics = evaluate_imbalance_metrics(
            y_test, test_preds, test_probs,
            model_name=f"FFNN Ablation1 ({mode})"
        )
        metrics["mode"] = mode
        metrics["threshold"] = best_th
        results.append(metrics)

    # Save CSV
    out_path = os.path.join(results_dir, "ablation1_loss_weighting.csv")
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"\nSaved Ablation 1 → {out_path}")

def run_ablation2_depth(
    X_train_proc,
    y_train,
    X_val_proc,
    y_val,
    X_test_proc,
    y_test,
    results_dir="results",
):
    print("\n[ABLATION 2] Model Depth")

    variants = {
        "shallow": [64, 32],
        "deep": [128, 64, 32]
    }

    results = []

    for name, layers in variants.items():
        print(f"\n--- Running: {name.upper()} ---")

        d_in = X_train_proc.shape[1]

        # Build model
        modules = [nn.Linear(d_in, layers[0]), nn.ReLU(), nn.BatchNorm1d(layers[0]), nn.Dropout(0.3)]
        for i in range(1, len(layers)):
            modules.extend([nn.Linear(layers[i-1], layers[i]), nn.ReLU(), nn.Dropout(0.2)])
        modules.append(nn.Linear(layers[-1], 2))
        model = nn.Sequential(*modules)

        # Data
        X_train_t = torch.tensor(X_train_proc, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.long)
        X_val_t = torch.tensor(X_val_proc, dtype=torch.float32)
        y_val_t = torch.tensor(y_val, dtype=torch.long)
        X_test_t = torch.tensor(X_test_proc, dtype=torch.float32)

        train_loader = DataLoader(
            TensorDataset(X_train_t, y_train_t), batch_size=256, shuffle=True
        )

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Train
        for epoch in range(10):
            model.train()
            for xb, yb in train_loader:
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()

        # Validation threshold
        model.eval()
        with torch.no_grad():
            val_probs = F.softmax(model(X_val_t), dim=1)[:, 1].numpy()

        best_th = pick_best_threshold(y_val, val_probs, mode="f1")

        # Test
        with torch.no_grad():
            test_probs = F.softmax(model(X_test_t), dim=1)[:, 1].numpy()
            test_preds = (test_probs >= best_th).astype(int)

        metrics = evaluate_imbalance_metrics(
            y_test, test_preds, test_probs,
            model_name=f"FFNN Ablation2 ({name})"
        )
        metrics["depth"] = name
        metrics["threshold"] = best_th
        results.append(metrics)

    out_path = os.path.join(results_dir, "ablation2_depth.csv")
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"\nSaved Ablation 2 → {out_path}")

def build_engineered_df_for_ablation(sample_n=300000, random_state=42):
    import kagglehub, glob, os

    path = kagglehub.dataset_download("sobhanmoosavi/us-accidents")

    csv_candidates = glob.glob(os.path.join(path, "**", "US_Accidents*.csv"), recursive=True)
    if not csv_candidates:
        csv_candidates = glob.glob(os.path.join(path, "**", "*.csv"), recursive=True)

    csv_path = sorted(csv_candidates, key=lambda p: os.path.getsize(p), reverse=True)[0]

    use_cols = [
        "Severity", "Start_Time", "Weather_Condition",
        "Visibility(mi)", "Temperature(F)", "Wind_Speed(mph)",
        "Precipitation(in)", "Sunrise_Sunset"
    ]

    header = pd.read_csv(csv_path, nrows=0).columns.tolist()
    present = [c for c in use_cols if c in header]

    if sample_n:
        total = sum(1 for _ in open(csv_path, "r", encoding="utf-8", errors="ignore")) - 1
        keep = set(np.random.choice(total, size=min(sample_n, total), replace=False))
        df = pd.read_csv(csv_path, usecols=present, skiprows=lambda i: i > 0 and (i-1) not in keep)
    else:
        df = pd.read_csv(csv_path, usecols=present)

    # Cleaning + feature engineering identical to notebook
    df = df[df["Severity"].isin([1, 2, 3, 4])]
    df["target"] = df["Severity"].apply(lambda x: 0 if x in [1,2] else 1)

    ts = pd.to_datetime(df["Start_Time"], errors="coerce")
    df["hour"] = ts.dt.hour
    df["dow"] = ts.dt.dayofweek
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["is_rush_hour"] = (((df["hour"] >= 7)&(df["hour"]<=9)) | ((df["hour"]>=16)&(df["hour"]<=18))).astype(int)

    def simplify(w):
        w = str(w).lower()
        if "thunder" in w: return "Thunderstorm"
        if "snow" in w: return "Snow"
        if "rain" in w: return "Rain"
        if "fog" in w or "mist" in w: return "Fog"
        if "cloud" in w: return "Cloudy"
        if "clear" in w: return "Clear"
        return "Other"

    df["Weather_Simple"] = df["Weather_Condition"].apply(simplify)
    df["Daylight"] = df["Sunrise_Sunset"].map({"Day":1,"Night":0})
    df["low_visibility"] = df["Visibility(mi)"] < 3.0

    keep = [
        "target","Visibility(mi)","Temperature(F)","Wind_Speed(mph)","Precipitation(in)",
        "hour","is_weekend","is_rush_hour","low_visibility",
        "Weather_Simple","Daylight"
    ]

    return df[keep].copy()

def run_logreg_minus_one_ablation(df, results_dir="results"):
    print("\n[ABLATION 3] Minus-One Logistic Regression")

    BASE = ["Temperature(F)", "Wind_Speed(mph)", "Weather_Simple"]
    VIZ  = ["Visibility(mi)", "low_visibility"]
    PREC = ["Precipitation(in)"]
    TIME = ["hour","is_weekend","is_rush_hour"]
    DAYN = ["Daylight"]

    variants = {
        "full": BASE + VIZ + PREC + TIME + DAYN,
        "minus_visibility": BASE + PREC + TIME + DAYN,
        "minus_precip":    BASE + VIZ + TIME + DAYN,
        "minus_time":      BASE + VIZ + PREC + DAYN,
        "minus_daynight":  BASE + VIZ + PREC + TIME,
    }

    num_candidates = [
        "Visibility(mi)","Temperature(F)","Wind_Speed(mph)","Precipitation(in)",
        "hour","is_weekend","is_rush_hour","low_visibility"
    ]
    cat_candidates = ["Weather_Simple","Daylight"]

    results = []

    for name, feats in variants.items():
        print(f"\n--- Variant: {name} ---")

        cols = ["target"] + [c for c in feats if c in df.columns]
        df_sub = df[cols].copy()
        y = df_sub["target"].values
        X = df_sub.drop(columns=["target"])

        num_cols = [c for c in num_candidates if c in X.columns]
        cat_cols = [c for c in cat_candidates if c in X.columns]

        pre = ColumnTransformer([
            ("num", SimpleImputer(strategy="median"), num_cols),
            ("cat", Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("oh", OneHotEncoder(handle_unknown="ignore"))
            ]), cat_cols)
        ])

        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
        )

        classes = np.unique(y_train)
        cw = compute_class_weight(
            class_weight="balanced",
            classes=classes,
            y=y_train,
        )
        cw_dict = {int(c): float(w) for c, w in zip(classes, cw)}

        logreg = LogisticRegression(
            solver="liblinear",
            max_iter=500,
            class_weight=cw_dict,
        )

        pipe = Pipeline([
            ("pre", pre),
            ("logreg", logreg)
        ])

        grid = GridSearchCV(
            pipe,
            {"logreg__C": [0.01,0.1,1.0,10.0]},
            cv=3,
            scoring="f1",
            n_jobs=-1,
        )
        grid.fit(X_train, y_train)
        best = grid.best_estimator_

        y_val_proba = best.predict_proba(X_val)[:,1]
        th = pick_best_threshold(y_val, y_val_proba, mode="f1")

        y_test_proba = best.predict_proba(X_test)[:,1]
        y_test_pred  = (y_test_proba >= th).astype(int)

        raw_metrics = evaluate_imbalance_metrics(
            y_test, y_test_pred, y_test_proba,
            model_name=f"LogReg_minus1 ({name})"
        )

        calib = CalibratedClassifierCV(best, method="isotonic", cv="prefit")
        calib.fit(X_val, y_val)
        y_test_proba_cal = calib.predict_proba(X_test)[:,1]

        y_test_pred_cal = y_test_proba_cal >= th

        cal_metrics = evaluate_imbalance_metrics(
            y_test, y_test_pred_cal, y_test_proba_cal,
            model_name=f"LogReg_minus1 ({name}, calibrated)"
        )

        out = {
            "variant": name,
            "n_features": len(cols)-1,
            "best_C": grid.best_params_["logreg__C"],
        }
        out.update({
            k+"_raw": v for k,v in raw_metrics.items()
            if k != "model"
        })
        out.update({
            k+"_cal": v for k,v in cal_metrics.items()
            if k != "model"
        })

        results.append(out)

    out_path = os.path.join(results_dir, "logreg_minus_one_ablation.csv")
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"\nSaved Ablation 3 → {out_path}")
