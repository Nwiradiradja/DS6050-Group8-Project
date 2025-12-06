import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, brier_score_loss

import matplotlib.pyplot as plt
import seaborn as sns

from data_and_features import load_data_and_features
from metrics_and_utils import (
    evaluate_imbalance_metrics,
    pick_best_threshold,
    plot_training_curves,
    plot_confusion_heatmap,
)


# ---------------------------
# Simple shallow FFNN
# ---------------------------
class MLP(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(d_in, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 2),  # logits for classes {0,1}
        )

    def forward(self, x):
        return self.model(x)


def train_shallow_ffnn(
    X_train_proc,
    y_train,
    X_val_proc,
    y_val,
    class_weights_train,
    n_epochs=15,
    batch_size=256,
    lr=1e-3,
):
    """
    Train shallow FFNN with class-weighted cross-entropy.
    Returns:
      model, (train_loss_history, val_f1_history)
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

    d_in = X_train_proc.shape[1]
    model = MLP(d_in)

    # class_weights_train is a dict like {0: w0, 1: w1}
    weight_tensor = torch.tensor(
        [class_weights_train[0], class_weights_train[1]],
        dtype=torch.float32,
    )
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    from sklearn.metrics import classification_report as sk_clf_report

    train_loss_history, val_f1_history = [], []

    for epoch in range(n_epochs):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
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

        val_report = sk_clf_report(
            all_targets, all_preds, output_dict=True, zero_division=0
        )
        val_f1 = val_report["weighted avg"]["f1-score"]
        val_f1_history.append(val_f1)

        print(
            f"Epoch {epoch+1:02d} | "
            f"TrainLoss {train_loss:.4f} | ValF1 {val_f1:.4f}"
        )

    return model, (train_loss_history, val_f1_history)


def save_logreg_feature_importance(
    best_model, num_cols, cat_cols, out_path="G08/fig1.png", top_k=10
):
    """
    Build and save logistic regression feature-importance barplot.
    """
    pre = best_model.named_steps["preprocessor"]
    logreg_inner = best_model.named_steps["logreg"]

    num_features = num_cols

    cat_encoder = pre.named_transformers_["cat"].named_steps["oh"]
    cat_features = list(cat_encoder.get_feature_names_out(cat_cols))

    feature_names = np.array(num_features + cat_features)
    coefs = logreg_inner.coef_[0]

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "coef": coefs,
            "abs_coef": np.abs(coefs),
        }
    ).sort_values("abs_coef", ascending=False)

    print("\nTop 10 most important features (logreg, by |coef|):")
    print(importance_df.head(10))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    top_df = importance_df.head(top_k).iloc[::-1]  # reverse for plotting
    plt.figure(figsize=(6, 4))
    plt.barh(top_df["feature"], top_df["coef"])
    plt.title("Logistic Regression Feature Importance (Top 10)")
    plt.xlabel("Coefficient (for class=Severe)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"Saved feature importance figure to {out_path}")
    return importance_df


def main():
    # ---------------------------
    # 1) Load data & preprocessor
    # ---------------------------
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

    # ---------------------------
    # 2) Logistic Regression Baseline
    # ---------------------------
    print("\n=== Training Logistic Regression Baseline ===")

    log_reg = LogisticRegression(
        solver="liblinear",
        max_iter=500,
        class_weight=class_weights_train,
    )

    from sklearn.pipeline import Pipeline

    clf = Pipeline(
        steps=[
            ("preprocessor", preprocess),
            ("logreg", log_reg),
        ]
    )

    param_grid = {
        "logreg__C": [0.01, 0.1, 1.0, 10.0],
    }

    grid_search = GridSearchCV(
        clf,
        param_grid,
        cv=3,
        scoring="f1",
        n_jobs=-1,
        verbose=2,
    )

    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    print("Best params:", grid_search.best_params_)
    print("Best CV score:", grid_search.best_score_)

    # Validation threshold tuning
    y_val_proba = best_model.predict_proba(X_val)[:, 1]
    opt_th_log = pick_best_threshold(y_val, y_val_proba, mode="f1")
    y_val_pred = (y_val_proba >= opt_th_log).astype(int)

    eval_logreg_val = evaluate_imbalance_metrics(
        y_val,
        y_val_pred,
        y_val_proba,
        model_name="LogReg (weighted, val)",
    )

    # Test metrics using same threshold
    y_test_proba = best_model.predict_proba(X_test)[:, 1]
    y_test_pred = (y_test_proba >= opt_th_log).astype(int)

    eval_logreg_test = evaluate_imbalance_metrics(
        y_test,
        y_test_pred,
        y_test_proba,
        model_name="LogReg (weighted, test)",
    )

    from sklearn.metrics import classification_report as sk_clf_report

    print("\nFull classification report (LogReg, test):")
    print(sk_clf_report(y_test, y_test_pred, digits=4))

    # Calibration (isotonic)
    calibrated_clf = CalibratedClassifierCV(
        estimator=best_model,
        method="isotonic",
        cv="prefit",
    )
    calibrated_clf.fit(X_val, y_val)

    y_test_proba_raw = y_test_proba
    y_test_proba_cal = calibrated_clf.predict_proba(X_test)[:, 1]

    brier_raw = brier_score_loss(y_test, y_test_proba_raw)
    brier_cal = brier_score_loss(y_test, y_test_proba_cal)

    print("\nBrier score (LogReg, lower is better)")
    print(f"Uncalibrated: {brier_raw:.4f}")
    print(f"Calibrated  : {brier_cal:.4f}")

    y_test_pred_cal = (y_test_proba_cal >= 0.5).astype(int)
    eval_logreg_cal_test = evaluate_imbalance_metrics(
        y_test,
        y_test_pred_cal,
        y_test_proba_cal,
        model_name="LogReg (weighted + calibrated, test)",
    )

    # Feature importance figure (Figure 1)
    importance_df = save_logreg_feature_importance(
        best_model,
        num_cols=num_cols,
        cat_cols=cat_cols,
        out_path="G08/fig1.png",
        top_k=10,
    )

    # ---------------------------
    # 3) Shallow FFNN Baseline
    # ---------------------------
    print("\n=== Training Shallow FFNN Baseline ===")

    # Use the fitted preprocessor from best_model
    fitted_pre = best_model.named_steps["preprocessor"]
    X_train_proc = fitted_pre.transform(X_train)
    X_val_proc = fitted_pre.transform(X_val)
    X_test_proc = fitted_pre.transform(X_test)

    model_ffnn, (train_loss_hist, val_f1_hist) = train_shallow_ffnn(
        X_train_proc=X_train_proc,
        y_train=y_train,
        X_val_proc=X_val_proc,
        y_val=y_val,
        class_weights_train=class_weights_train,
        n_epochs=15,
        batch_size=256,
        lr=1e-3,
    )

    # Plot training curves for FFNN
    plot_training_curves(
        train_loss_history=train_loss_hist,
        val_metric_history=val_f1_hist,
        metric_name="Weighted F1",
        title_prefix="FFNN (Shallow, Weighted)",
    )

    # Threshold tuning on validation probs
    X_val_t = torch.tensor(X_val_proc, dtype=torch.float32)
    with torch.no_grad():
        logits_val = model_ffnn(X_val_t)
        probs_val = F.softmax(logits_val, dim=1)[:, 1].cpu().numpy()

    opt_th_ffnn = pick_best_threshold(y_val, probs_val, mode="f1")

    # Test set evaluation
    X_test_t = torch.tensor(X_test_proc, dtype=torch.float32)
    with torch.no_grad():
        logits_test = model_ffnn(X_test_t)
        probs_test = F.softmax(logits_test, dim=1)[:, 1].cpu().numpy()
        preds_test = (probs_test >= opt_th_ffnn).astype(int)

    eval_ffnn_test = evaluate_imbalance_metrics(
        y_test,
        preds_test,
        probs_test,
        model_name="FFNN (shallow weighted, test)",
    )

    print("\nFull classification report (FFNN, test):")
    print(sk_clf_report(y_test, preds_test, digits=4))

    # Confusion matrix heatmap
    plot_confusion_heatmap(
        y_true=y_test,
        y_pred=preds_test,
        title="Confusion Matrix – FFNN (shallow weighted)",
    )

    # ---------------------------
    # 4) FFNN calibration (Platt)
    # ---------------------------
    print("\n=== Calibrating FFNN with Platt Scaling ===")

    with torch.no_grad():
        val_logits = model_ffnn(X_val_t).cpu().numpy()
        test_logits = model_ffnn(X_test_t).cpu().numpy()

    # 1D score = logit difference
    val_scores = (val_logits[:, 1] - val_logits[:, 0]).reshape(-1, 1)
    test_scores = (test_logits[:, 1] - test_logits[:, 0]).reshape(-1, 1)

    from sklearn.linear_model import LogisticRegression as PlattLogReg

    platt_calibrator = PlattLogReg()
    platt_calibrator.fit(val_scores, y_val)

    # Calibrated probs on validation to re-pick threshold
    ffnn_val_proba_cal = platt_calibrator.predict_proba(val_scores)[:, 1]
    opt_th_cal = pick_best_threshold(y_val, ffnn_val_proba_cal, mode="f1")

    # Raw FFNN probs on test
    with torch.no_grad():
        raw_test_probs = F.softmax(
            torch.tensor(test_logits), dim=1
        )[:, 1].numpy()

    # Calibrated probs on test
    ffnn_test_proba_cal = platt_calibrator.predict_proba(test_scores)[:, 1]
    ffnn_test_pred_cal = (ffnn_test_proba_cal >= opt_th_cal).astype(int)

    brier_ffnn_raw = brier_score_loss(y_test, raw_test_probs)
    brier_ffnn_cal = brier_score_loss(y_test, ffnn_test_proba_cal)

    print("\nFFNN Brier score (lower is better)")
    print(f"Uncalibrated: {brier_ffnn_raw:.4f}")
    print(f"Calibrated  : {brier_ffnn_cal:.4f}")

    eval_ffnn_cal_test = evaluate_imbalance_metrics(
        y_test,
        ffnn_test_pred_cal,
        ffnn_test_proba_cal,
        model_name="FFNN (shallow weighted + calibrated, test)",
    )

    # ---------------------------
    # 5) Save metrics to CSV
    # ---------------------------
    results = [
        {**eval_logreg_test, "model": "LogReg (weighted, test)"},
        {
            **eval_logreg_cal_test,
            "model": "LogReg (weighted + calibrated, test)",
        },
        {**eval_ffnn_test, "model": "FFNN (shallow weighted, test)"},
        {
            **eval_ffnn_cal_test,
            "model": "FFNN (shallow weighted + calibrated, test)",
        },
    ]

    metrics_df = pd.DataFrame(results)
    os.makedirs("results", exist_ok=True)
    out_csv = "results/model_metrics.csv"
    metrics_df.to_csv(out_csv, index=False)

    print(f"\nSaved metrics to {out_csv}")
    print(metrics_df)


if __name__ == "__main__":
    main()
