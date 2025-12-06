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

# --- Import project modules ---
from src.data_and_features import load_and_prepare_data
from src.metrics_and_utils import (
    pick_best_threshold,
    evaluate_imbalance_metrics,
    prepare_dataloaders,
)
from src.baseline import (
    train_logistic_regression,
    train_ffnn,
)
from src.ablations import (
    run_loss_ablation,
    run_depth_ablation,
    run_minus_one_ablation,
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
    ) = load_and_prepare_data()

    # ----------------------------------------
    # Step 2 — Logistic Regression Baseline
    # ----------------------------------------
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
        X_train, X_val, X_test, y_train, y_val, y_test, preprocess
    )

    print("\nLogReg Test Metrics:")
    print(eval_logreg_test)

    # Save LR feature importances
    lr_feat_out = os.path.join(RESULTS_DIR, "logreg_feature_importance.csv")
    logreg_feature_df.to_csv(lr_feat_out, index=False)
    print(f"Saved LR feature importance → {lr_feat_out}")

    # ----------------------------------------
    # Step 3 — Shallow FFNN Baseline
    # ----------------------------------------
    print("\n==============================")
    print(" STEP 3 — SHALLOW FFNN BASELINE")
    print("==============================")

    eval_ffnn_test, eval_ffnn_cal_test = train_ffnn(
        X_train, X_val, X_test, y_train, y_val, y_test, preprocess
    )

    print("\nFFNN Test Metrics:")
    print(eval_ffnn_test)

    # ----------------------------------------
    # Step 4 — Ablation Studies
    # ----------------------------------------
    print("\n==============================")
    print(" STEP 4 — ABLATION STUDIES")
    print("==============================")

    # 1) Loss Ablation
    print("\n--- Ablation 1: Loss Functions ---")
    loss_results = run_loss_ablation(
        X_train, X_val, X_test, y_train, y_val, y_test, preprocess
    )
    pd.DataFrame(loss_results).to_csv(
        os.path.join(RESULTS_DIR, "ablation_loss.csv"), index=False
    )

    # 2) Depth Ablation
    print("\n--- Ablation 2: Model Depth ---")
    depth_results = run_depth_ablation(
        X_train, X_val, X_test, y_train, y_val, y_test, preprocess
    )
    pd.DataFrame(depth_results).to_csv(
        os.path.join(RESULTS_DIR, "ablation_depth.csv"), index=False
    )

    # 3) Minus-One Logistic Regression
    print("\n--- Ablation 3: Minus-One Features (LogReg) ---")
    minus_one_results = run_minus_one_ablation()
    minus_one_results.to_csv(
        os.path.join(RESULTS_DIR, "logreg_minus_one.csv"), index=False
    )

    # ----------------------------------------
    # Step 5 — Save master comparison table
    # ----------------------------------------
    print("\n==============================")
    print(" STEP 5 — SAVE MASTER METRICS TABLE")
    print("==============================")

    master_results = pd.DataFrame(
        [
            {**eval_logreg_test, "model": "LogReg"},
            {**eval_logreg_cal_test, "model": "LogReg (Calibrated)"},
            {**eval_ffnn_test, "model": "FFNN"},
            {**eval_ffnn_cal_test, "model": "FFNN (Calibrated)"},
        ]
    )

    master_path = os.path.join(RESULTS_DIR, "model_comparison.csv")
    master_results.to_csv(master_path, index=False)

    print(f"Saved master comparison → {master_path}")
    print("\n🎉 All tasks complete! Your results are in the results/ folder.\n")


if __name__ == "__main__":
    main()
