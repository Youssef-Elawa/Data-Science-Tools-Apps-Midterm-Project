# Data-Science-Tools-Apps-Midterm-Project

# House Prices Prediction - Kaggle ML Project

End-to-end machine learning project for the Kaggle **House Prices: Advanced Regression Techniques** competition.

This project focuses on predicting residential sale prices using feature engineering, preprocessing pipelines, hyperparameter optimization, and ensemble modeling. Multiple model families were explored, including gradient boosting, linear models, kernel methods, and tabular foundation models.

## Project Goal

The goal of this project is to maximize predictive performance on the Kaggle House Prices competition leaderboard.

Competition link:  
https://www.kaggle.com/c/house-prices-advanced-regression-techniques

## Models Used

The project experiments with several model families:

- LightGBM
- CatBoost
- XGBoost
- Lasso Regression
- ElasticNet
- Kernel Ridge Regression
- TabPFN
- Weighted ensemble / blending

## Main Techniques

### Data preprocessing
- Median imputation for numerical features
- Constant `"None"` imputation for categorical features
- One-hot encoding for categorical variables
- Standard scaling for linear-model pipelines

### Feature engineering
Engineered features include:
- `TotalSF`
- `TotalBathrooms`
- `HouseAge`
- `RemodAge`

In addition, `MSSubClass` was converted from numeric to categorical.

### Outlier handling
A small number of extreme outliers were removed, specifically houses with unusually large living area but relatively low sale prices.

### Hyperparameter optimization
Models were tuned using:
- Optuna
- cross-validation
- experiment tracking with Weights & Biases (W&B)

## Evaluation

Local validation was performed using **5-fold cross-validation** with RMSE on the log-transformed target:

- Target transformation: `log1p(SalePrice)`
- Evaluation metric: RMSE on log-transformed prices

This closely matches the Kaggle competition metric.

## Best Results

Replace these with your final numbers:

- Best local CV RMSE: `0.09197` (TabPFN)
- Best Kaggle public score: `0.12108` (ensemble)
  <img width="775" height="252" alt="image" src="https://github.com/user-attachments/assets/2df121b1-17a8-4e4e-85d9-7f20ea1871fb" />
- Other notable local CV results:
  - ElasticNet: `0.1072`
  - Lasso : `0.1072 `
  - Kernel Ridge: `0.1082`
  - LightGBM: `0.1157`

## Repository Contents

- `DSTA_Midterm.ipynb` — main notebook
- `requirements.txt` — required Python packages
- `README.md` — project overview

## 🚀 Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/Youssef-Elawa/Data-Science-Tools-Apps-Midterm-Project.git
cd Data-Science-Tools-Apps-Midterm-Project
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download Kaggle competition data

Download the following files from Kaggle and place them in the project root directory:

* `train.csv`
* `test.csv`

*(Tip: You can use the Kaggle API or download manually from the competition page.)*

### 4. Authenticate with Hugging Face

This project requires access to models from Hugging Face.

**Option A (recommended):**

```bash
huggingface-cli login
```

**Option B (in notebook):**
You will be prompted to enter your token when running the notebook.

### 5. Run the notebook

Open the notebook in:

* Google Colab, or
* Jupyter / VS Code locally

Then run all cells in order.


## Notes

This project was built as a practical machine learning exercise in:
- tabular data preprocessing
- feature engineering
- cross-validation
- hyperparameter tuning
- ensemble modeling
- Kaggle-style evaluation

## Future Improvements

Potential next steps:
- stacking with a meta-model
- more advanced feature interactions
- optimized ensemble weight search
- better handling of leaderboard/generalization gap

## Author

Youssef Ahmed

If you'd like, you can also connect with me here:
- LinkedIn: linkedin.com/in/youssef-ahmed-e
- Kaggle: kaggle.com/youssefahmedelawa
