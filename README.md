# Instructions
## How To Run

### Clone the repo to your local machine
```
git clone <Paste the link>
```
* Alternative method is to simply download the folder

### Open Command Prompt and go to the folder the repo is located in
```
# For an example if the repo is in a folder called test located in my documents folder
cd Documents/test

# Check if repo is there
ls
<DS6050-Group8-Project>

# Access the folder
cd DS6050-Group8-Project
```

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
    ablation1_loss_weighting.csv
    ablation2_depth.csv
    logreg_minus_one.csv
```
