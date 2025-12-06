import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from src.metrics_and_utils import (
    pick_best_threshold,
    evaluate_imbalance_metrics,
)

# ============================================================
# SHALLOW FFNN MODEL
# ============================================================
class MLP_Shallow(nn.Module):
    def __init__(self, d_in, drop1=0.3, drop2=0.2):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(d_in, 64), nn.ReLU(), nn.BatchNorm1d(64), nn.Dropout(drop1),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(drop2),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        return self.model(x)

# ============================================================
# LOGISTIC REGRESSION BASELINE (correct signature)
# ============================================================
def train_logistic_regression(
    X_train, X_val, X_test,
    y_train, y_val, y_test,
    preprocess,
):
    """
    EXACT interface expected by run_final.py.
    Returns:
        best_model,
        eval_logreg_val,
        eval_logreg_test,
        eval_logreg_cal_test,
        feature_importance_df
    """

    print("\n=== Training Logistic Regression Baseline ===")

    log_reg = LogisticRegression(
        solver="liblinear",
        max_iter=500,
        class_weight="balanced",
    )

    pipe = Pipeline([
        ("preprocessor", preprocess),
        ("logreg", log_reg),
    ])

    grid = GridSearchCV(
        pipe,
        {"logreg__C": [0.01, 0.1, 1.0, 10.0]},
        cv=3,
        scoring="f1",
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    # -------- Validation --------
    y_val_proba = best_model.predict_proba(X_val)[:, 1]
    best_th = pick_best_threshold(y_val, y_val_proba, mode="f1")
    y_val_pred = (y_val_proba >= best_th).astype(int)

    eval_logreg_val = evaluate_imbalance_metrics(
        y_val, y_val_pred, y_val_proba, "LogReg (val)"
    )

    # -------- Test raw --------
    y_test_proba = best_model.predict_proba(X_test)[:, 1]
    y_test_pred = (y_test_proba >= best_th).astype(int)

    eval_logreg_test = evaluate_imbalance_metrics(
        y_test, y_test_pred, y_test_proba, "LogReg (test)"
    )

    # -------- Calibration --------
    calibrated = CalibratedClassifierCV(
        estimator=best_model,
        method="isotonic",
        cv="prefit",
    )
    calibrated.fit(X_val, y_val)

    y_test_proba_cal = calibrated.predict_proba(X_test)[:, 1]
    y_test_pred_cal = (y_test_proba_cal >= 0.5).astype(int)

    eval_logreg_cal_test = evaluate_imbalance_metrics(
        y_test, y_test_pred_cal, y_test_proba_cal, "LogReg (calibrated)"
    )

    # -------- Feature Importance --------
    pre = best_model.named_steps["preprocessor"]
    lr = best_model.named_steps["logreg"]

    num_cols = pre.transformers_[0][2]
    cat_encoder = pre.named_transformers_["cat"].named_steps["oh"]
    cat_cols = list(cat_encoder.get_feature_names_out(pre.transformers_[1][2]))

    feature_names = np.array(list(num_cols) + cat_cols)
    coefs = lr.coef_[0]

    feature_df = pd.DataFrame({
        "feature": feature_names,
        "coef": coefs,
        "abs_coef": np.abs(coefs),
    }).sort_values("abs_coef", ascending=False)

    return (
        best_model,
        eval_logreg_val,
        eval_logreg_test,
        eval_logreg_cal_test,
        feature_df,
    )

# ============================================================
# SHALLOW FFNN BASELINE
# ============================================================
def train_ffnn(X_train, X_val, X_test, y_train, y_val, y_test, preprocess):

    print("\n=== Training Shallow FFNN Baseline ===")

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

    d_in = X_train_proc.shape[1]
    model = MLP_Shallow(d_in)

    class_counts = np.bincount(y_train)
    weights = torch.tensor([
        len(y_train) / (2 * class_counts[0]),
        len(y_train) / (2 * class_counts[1]),
    ], dtype=torch.float32)

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # -------------------- Train --------------------
    for epoch in range(15):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

    # -------------------- Validation --------------------
    with torch.no_grad():
        val_logits = model(X_val_t)
        val_probs = F.softmax(val_logits, dim=1)[:, 1].numpy()

    best_th = pick_best_threshold(y_val, val_probs, mode="f1")

    # -------------------- Test --------------------
    with torch.no_grad():
        test_logits = model(X_test_t)
        test_probs = F.softmax(test_logits, dim=1)[:, 1].numpy()
        test_preds = (test_probs >= best_th).astype(int)

    eval_ffnn_test = evaluate_imbalance_metrics(
        y_test, test_preds, test_probs, "FFNN (test)"
    )

    # -------------------- Calibration --------------------
    with torch.no_grad():
        val_logits = model(X_val_t).numpy()
        test_logits = model(X_test_t).numpy()

    val_scores = (val_logits[:, 1] - val_logits[:, 0]).reshape(-1, 1)
    test_scores = (test_logits[:, 1] - test_logits[:, 0]).reshape(-1, 1)

    from sklearn.linear_model import LogisticRegression as PlattLR
    platt = PlattLR()
    platt.fit(val_scores, y_val)

    test_probs_cal = platt.predict_proba(test_scores)[:, 1]
    test_preds_cal = (test_probs_cal >= best_th).astype(int)

    eval_ffnn_cal_test = evaluate_imbalance_metrics(
        y_test, test_preds_cal, test_probs_cal, "FFNN (calibrated)"
    )

    return eval_ffnn_test, eval_ffnn_cal_test
