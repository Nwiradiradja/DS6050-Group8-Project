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
from sklearn.metrics import classification_report as sk_clf_report
from sklearn.metrics import brier_score_loss
from sklearn.pipeline import Pipeline

import matplotlib.pyplot as plt
import seaborn as sns

# --- Project utilities ---
from src.metrics_and_utils import (
    evaluate_imbalance_metrics,
    pick_best_threshold,
    plot_training_curves,
    plot_confusion_heatmap,
)


# ============================================================
#  SHALLOW FFNN MODEL
# ============================================================
class MLP_Shallow(nn.Module):
    """Matches the architecture used in the notebook baseline."""
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


# ============================================================
#  LOGISTIC REGRESSION BASELINE
# ============================================================
def train_logistic_regression(X_train, X_val, X_test, y_train, y_val, y_test, preprocess):
    """
    Trains the weighted Logistic Regression baseline exactly as in the notebook.
    Returns:
        best_model,
        eval_logreg_val,
        eval_logreg_test,
        eval_logreg_cal_test,
        feature_importance_df
    """

    print("\n=== Training Logistic Regression Baseline ===")

    # Weighted LR
    class_weights = {0: (len(y_train) / (2 * np.sum(y_train == 0))),
                     1: (len(y_train) / (2 * np.sum(y_train == 1)))}

    log_reg = LogisticRegression(
        solver="liblinear",
        max_iter=500,
        class_weight=class_weights,
    )

    clf = Pipeline([
        ("preprocessor", preprocess),
        ("logreg", log_reg),
    ])

    param_grid = {"logreg__C": [0.01, 0.1, 1.0, 10.0]}

    grid = GridSearchCV(
        clf,
        param_grid,
        cv=3,
        scoring="f1",
        n_jobs=-1,
        verbose=0,
    )
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    # ------------------------------
    # Validation threshold
    # ------------------------------
    y_val_proba = best_model.predict_proba(X_val)[:, 1]
    opt_th = pick_best_threshold(y_val, y_val_proba, mode="f1")

    y_val_pred = (y_val_proba >= opt_th).astype(int)
    eval_logreg_val = evaluate_imbalance_metrics(
        y_val, y_val_pred, y_val_proba, model_name="LogReg (val)"
    )

    # ------------------------------
    # Test set
    # ------------------------------
    y_test_proba = best_model.predict_proba(X_test)[:, 1]
    y_test_pred = (y_test_proba >= opt_th).astype(int)

    eval_logreg_test = evaluate_imbalance_metrics(
        y_test, y_test_pred, y_test_proba, model_name="LogReg (test)"
    )

    # ------------------------------
    # Calibration (Isotonic)
    # ------------------------------
    calibrated = CalibratedClassifierCV(
        estimator=best_model,
        method="isotonic",
        cv="prefit",
    )
    calibrated.fit(X_val, y_val)

    y_test_proba_cal = calibrated.predict_proba(X_test)[:, 1]
    y_test_pred_cal = (y_test_proba_cal >= opt_th).astype(int)

    eval_logreg_cal_test = evaluate_imbalance_metrics(
        y_test, y_test_pred_cal, y_test_proba_cal,
        model_name="LogReg (calibrated test)"
    )

    # ------------------------------
    # Feature Importance
    # ------------------------------
    pre = best_model.named_steps["preprocessor"]
    lr_inner = best_model.named_steps["logreg"]

    num_cols = pre.transformers_[0][2]
    cat_encoder = pre.named_transformers_["cat"].named_steps["oh"]
    cat_cols = list(cat_encoder.get_feature_names_out(pre.transformers_[1][2]))

    feature_names = np.array(list(num_cols) + cat_cols)
    coefs = lr_inner.coef_[0]

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "coef": coefs,
        "abs_coef": np.abs(coefs),
    }).sort_values("abs_coef", ascending=False)

    return (
        best_model,
        eval_logreg_val,
        eval_logreg_test,
        eval_logreg_cal_test,
        importance_df,
    )


# ============================================================
#  SHALLOW FFNN BASELINE
# ============================================================
def train_ffnn(X_train, X_val, X_test, y_train, y_val, y_test, preprocess):
    """
    Train shallow weighted FFNN baseline.
    Returns:
        eval_ffnn_test,
        eval_ffnn_cal_test
    """

    print("\n=== Training Shallow FFNN Baseline ===")

    # Preprocess into numeric arrays
    X_train_proc = preprocess.transform(X_train)
    X_val_proc = preprocess.transform(X_val)
    X_test_proc = preprocess.transform(X_test)

    X_train_t = torch.tensor(X_train_proc, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t = torch.tensor(X_val_proc, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.long)
    X_test_t = torch.tensor(X_test_proc, dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=256, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=256, shuffle=False)

    # Shallow FFNN
    d_in = X_train_proc.shape[1]
    model = MLP_Shallow(d_in)

    # Balanced weights
    class_counts = np.bincount(y_train)
    class_weights = torch.tensor([
        len(y_train) / (2 * class_counts[0]),
        len(y_train) / (2 * class_counts[1]),
    ], dtype=torch.float32)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_loss_history = []
    val_f1_history = []

    # ------------------------------
    # Train FFNN
    # ------------------------------
    for epoch in range(15):
        model.train()
        running_loss = 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss_history.append(running_loss / len(train_loader))

        # Validation weighted F1
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                logits = model(xb)
                preds = logits.argmax(1)
                all_preds.extend(preds.tolist())
                all_targets.extend(yb.tolist())

        val_report = sk_clf_report(all_targets, all_preds, output_dict=True, zero_division=0)
        val_f1 = val_report["weighted avg"]["f1-score"]
        val_f1_history.append(val_f1)

        print(f"Epoch {epoch+1} | TrainLoss={train_loss_history[-1]:.4f} | ValF1={val_f1:.4f}")

    # Plot training curves
    plot_training_curves(
        train_loss_history,
        val_f1_history,
        metric_name="Weighted F1",
        title_prefix="FFNN Baseline",
    )

    # ------------------------------
    # Threshold selection
    # ------------------------------
    with torch.no_grad():
        logits_val = model(X_val_t)
        probs_val = F.softmax(logits_val, dim=1)[:, 1].cpu().numpy()

    opt_th = pick_best_threshold(y_val, probs_val, mode="f1")

    # ------------------------------
    # Test evaluation
    # ------------------------------
    with torch.no_grad():
        logits_test = model(X_test_t)
        probs_test = F.softmax(logits_test, dim=1)[:, 1].cpu().numpy()
        preds_test = (probs_test >= opt_th).astype(int)

    eval_ffnn_test = evaluate_imbalance_metrics(
        y_test, preds_test, probs_test, model_name="FFNN (test)"
    )

    # Confusion matrix
    plot_confusion_heatmap(y_test, preds_test, "FFNN Confusion Matrix")

    # ------------------------------
    # Calibration (Platt scaling)
    # ------------------------------
    with torch.no_grad():
        val_logits = model(X_val_t).cpu().numpy()
        test_logits = model(X_test_t).cpu().numpy()

    # Use logit difference
    val_scores = (val_logits[:, 1] - val_logits[:, 0]).reshape(-1, 1)
    test_scores = (test_logits[:, 1] - test_logits[:, 0]).reshape(-1, 1)

    from sklearn.linear_model import LogisticRegression as PlattLR
    platt = PlattLR()
    platt.fit(val_scores, y_val)

    ffnn_val_proba_cal = platt.predict_proba(val_scores)[:, 1]
    opt_th_cal = pick_best_threshold(y_val, ffnn_val_proba_cal, mode="f1")

    ffnn_test_proba_cal = platt.predict_proba(test_scores)[:, 1]
    ffnn_test_pred_cal = (ffnn_test_proba_cal >= opt_th_cal).astype(int)

    eval_ffnn_cal_test = evaluate_imbalance_metrics(
        y_test, ffnn_test_pred_cal, ffnn_test_proba_cal,
        model_name="FFNN (calibrated test)"
    )

    return eval_ffnn_test, eval_ffnn_cal_test
