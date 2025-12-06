# Instructions
## How To Run

### Create environment & install dependencies
```
pip install -r requirements.txt
```
### Run the Full Pipeline
```
python run_final.py
```

### The Script:
* Loads and preprocesses the dataset
* Trains the Logsitic Regression Baseline
* Trains the Shallow FFNN
* Runs all ablations (loss, depth, minus-one features)
* Saves metrics, CSVs, and plots into the "results/" folder

### Output files saved to:
```
results/
    model_comparison.csv
    logreg_feature_importance.csv
    ablation_loss.csv
    ablation_depth.csv
    logreg_minus_one.csv
```
