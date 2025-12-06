"""
Main entry point for DS6050 Group 8 Project.
This script orchestrates the full pipeline:

1. Load & preprocess data
2. Train Logistic Regression baseline
3. Train Shallow FFNN baseline
4. Run Ablations: Loss, Depth, and Minus-One Features
5. Save all metrics into results/
"""

import os
import pandas as pd

# Local project imports
from src.data_and_features import load_and_prepare_data
from src.metrics_and_utils import evaluate_imbalance_metrics, pick_best_threshold
from src.baseline import train_logistic_regression, train_ffnn
from src.ablations import (
    run_ablation1_loss_weighting,
    run_ablation2_depth,
    run_logreg_minus_one_ablation,
)

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def main():

    print("\n==============================")
    print(" STEP 1 — LOAD & PREPROCESS DATA")
    print("==============================")

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        preprocess,
        df_full,
        class_weights_train,
        num_cols,
        cat_cols,
    ) = load_and_prepare_data(sample_n=300_000, random_state=42)

    # --------------------------------------------------
    # Step 2 — Logistic Regression Baseline
    # --------------------------------------------------
    print("\n==============================")
    print(" STEP 2 — LOGISTIC REGRESSION BASELINE")
    print("==============================")

    (
        best_lr_model,
        eval_logreg_val,
        eval_logreg_test,
        eval_logreg_cal_test,
        logreg_feature_df,
    ) = train_logistic_regression(
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        preprocess,
        num_cols,
        cat_cols,
    )

    # Save LR feature importances
    lr_feat_path = os.path.join(RESULTS_DIR, "logreg_feature_importance.csv")
    logreg_feature_df.to_csv(lr_feat_path, index=False)
    print(f"Saved logistic regression feature importance → {lr_feat_path}")

    # --------------------------------------------------
    # Step 3 — Shallow FFNN Baseline
    # --------------------------------------------------
    print("\n==============================")
    print(" STEP 3 — SHALLOW FFNN BASELINE")
    print("==============================")

    eval_ffnn_test, eval_ffnn_cal_test = train_ffnn(
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        preprocess,
        class_weights_train,
    )

    # --------------------------------------------------
    # Step 4 — Ablation Studies
    # --------------------------------------------------
    print("\n==============================")
    print(" STEP 4 — ABLATION STUDIES")
    print("==============================")

    # 1) Loss Weighting Ablation
    print("\n--- Ablation 1: Loss Weighting ---")
    run_ablation1_loss_weighting(
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        preprocess,
        results_dir=RESULTS_DIR,
    )

    # 2) Model Depth Ablation
    print("\n--- Ablation 2: Depth Variation (Shallow vs Deep FFNN) ---")
    run_ablation2_depth(
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        preprocess,
        class_weights_train,
        results_dir=RESULTS_DIR,
    )

    # 3) Minus-One Logistic Regression Ablation
    print("\n--- Ablation 3: Minus-One Feature Groups (LogReg) ---")
    run_logreg_minus_one_ablation(
        df_full,
        results_dir=RESULTS_DIR,
    )

    # --------------------------------------------------
    # Step 5 — Save Master Comparison Table
    # --------------------------------------------------
    print("\n==============================")
    print(" STEP 5 — SAVE MASTER METRICS TABLE")
    print("==============================")

    master_results = pd.DataFrame([
        {**eval_logreg_test,        "model": "LogReg"},
        {**eval_logreg_cal_test,    "model": "LogReg (Calibrated)"},
        {**eval_ffnn_test,          "model": "FFNN"},
        {**eval_ffnn_cal_test,      "model": "FFNN (Calibrated)"},
    ])

    master_path = os.path.join(RESULTS_DIR, "model_comparison.csv")
    master_results.to_csv(master_path, index=False)

    print(f"\nSaved master comparison metrics → {master_path}")
    print("\n🎉 All tasks complete! Check the results/ folder.\n")


if __name__ == "__main__":
    main()
