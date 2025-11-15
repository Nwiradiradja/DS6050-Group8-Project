# Is It Safe to Drive? — Predicting Accident Severity from Weather & Road Conditions

This project explores whether deep learning models can predict vehicle accident severity using historical U.S. accident and weather data. We build a baseline feed-forward neural network (FFNN) to classify accidents as low severity (1–2) or high severity (3–4) using environmental and temporal features such as visibility, temperature, precipitation, weather condition, and time of day.

# Project Goals
* Understand how weather and temporal conditions influence driving safety
* Build a baseline deep-learning model to predict accident severity
* Evaluate performance and identify challenges (class imbalance)
* Establish groundwork for improved models in the future

# Datasets

All datasets used are public on Kaggle:

* [US Accidents (2016–2023)](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents)
* [US Weather Events (2016–2022)](https://www.kaggle.com/datasets/sobhanmoosavi/us-weather-events)
* [USA Car Accidents Severity Prediction](https://www.kaggle.com/code/jingzongwang/usa-car-accidents-severity-prediction)

# Preprocessing Summary
* Converted severity labels to binary targets
  * 1-2 -> 0 (Low severity)
  * 3-4 -> 1 (High severity)
* Parse timestamps into hour of day and weekday/weekend indicator
* Map raw weather descriptions to simplified categories such as Clear, Cloudy, Rain, Snow, Fog, Thunderstorm, Other
* Convert day/night indicator to binary

# Preliminary Experiments & Abalation Study
The goal of our preliminary experiments was to address the extreme 80:20 class imbalance between minor and severe accidents, and to identify the best-performing configuration of our Feed-Forward Neural Network (FFNN).

### Baseline Model: Logistic Regression
Our initial baseline was a weighted Logistic Regression model.
* Weighted F1: 0.73
* Recall (Severe Class): 0.11
* Failed to detect most severe crashes

This baseline confirmed the problem: traditional linear models cannot learn minority-class patterns under heavy imbalance.

# Ablation Study Overview
* Ablation 1: Loss Function: Inverse Frequency Weighting
* Ablation 2: Model Depth: Shallow vs. Deep FFNN
* Ablation 3: Feature Groups

* Please see PDF docs for more info on Ablation

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
