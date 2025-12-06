# Is It Safe to Drive? — Predicting Accident Severity from Weather & Road Conditions

This project explores whether machine learning and shallow neural networks can predict accident severity using historical U.S. accident and weather data. We classify crashes into:

* Low severity (1–2)
* High severity (3–4)

using environmental and temporal features including visibility, temperature, precipitation, weather conditions, and time-of-day indicators.

# Project Goals

* Understand how weather and temporal conditions influence driving risk
* Build a baseline model for predicting crash severity
* Evaluate model performance under strong class imbalance (80:20)
* Identify which modeling choices matter through a structured ablation study

# Datasets

All datasets used are public on Kaggle:

* [US Accidents (2016–2023)](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents)
* [US Weather Events (2016–2022)](https://www.kaggle.com/datasets/sobhanmoosavi/us-weather-events)
* [USA Car Accidents Severity Prediction](https://www.kaggle.com/code/jingzongwang/usa-car-accidents-severity-prediction)

# Preprocessing Summary
* Converted severity labels to binary:
 * 1–2 → 0 (low severity)
 * 3–4 → 1 (high severity)
* Extracted hour of day, weekday vs weekend, and rush-hour indicators
* Simplified weather into categories:
Clear, Cloudy, Rain, Snow, Fog, Thunderstorm, Other
* Converted Sunrise/Sunset into a Day/Night binary
* Standardized numerical features and one-hot-encoded categorical features

# Baseline Model: Logistic Regression (Class-Weighted)
Our first model was a weighted Logistic Regression using a balanced class-weighting scheme.
Results on the test set:

* Weighted F1: 0.39
* Recall (Severe class): 0.81
* PR-AUC: 0.23
Despite a high severe-class recall at a tuned threshold, the model showed poor ranking ability (low PR-AUC), confirming the difficulty of learning patterns for rare events.

# Ablation Study Overview
To understand which modeling choices matter most, we ran three controlled ablations:
## Ablation 1 — Loss Function
* Compared: Unweighted CE, Balanced CE, Inverse Frequency Weighting, Focal Loss
* Winner: Inverse Frequency Weighting

## Ablation 2 — Model Depth
* Shallow FFNN (2 layers)
* Deep FFNN (4 layers)
* Winner: Shallow model (better generalization, fewer parameters)

## Ablation 3 — Feature Groups
Minus-one study across:
* Visibility
* Precipitation
* Time-of-Day features
* Day/Night indicator
* Weather category

Finding:
Removing visibility or precipitation caused the largest drop, confirming their predictive importance.

# Best Model: Shallow FFNN + Inverse Frequency Loss + All Features
* 126% improvement in Severe F1 over Logistic Regression
* Balanced recall across classes
* Best minority-class detection without overfitting

# Error Analysis 
* Logistic Regression: High overall accuracy but very low recall on severe crashes; misses most high-severity events, especially under rare weather types.
* Best FFNN (Shallow + Inverse Frequency): Greatly improves Severe F1 and recall but introduces more false positives during heavy rain and peak traffic, flagging some minor crashes as severe.
* Common Patterns: Misclassified severe crashes often occur in transitional conditions (light rain, dawn/dusk) where features look similar to minor crashes, suggesting the need for richer features.

# Next Steps
* Feature Engineering Improvements
* Advanced Loss Function
* Hyperparameter Expansion
* Final Comparative Study
