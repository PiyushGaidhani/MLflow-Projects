# 02 - California Housing Price Prediction with MLflow

Random Forest regression model with hyperparameter tuning using GridSearchCV and MLflow tracking.

## 📋 Project Overview

This project demonstrates:
- Hyperparameter tuning with GridSearchCV
- MLflow experiment tracking for regression models
- Model registry and versioning
- Performance evaluation with MSE metric

## 🛠️ Tech Stack

- **Python** 3.13
- **MLflow** - Experiment tracking and model registry
- **Scikit-learn** - Random Forest Regressor & GridSearchCV
- **Pandas** - Data manipulation
- **NumPy** - Numerical operations

## 📊 Dataset

**California Housing Dataset** (sklearn)
- 20,640 samples
- 8 features: MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude
- Target: Median house value (in $100,000s)
- Split: 80% training, 20% testing

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
 

2. Start MLflow UI
mlflow ui --port 5000
Open your browser to http://127.0.0.1:5000

3. Run the Notebook

jupyter notebook housepricepredict.ipynb

🔧 Hyperparameter Tuning
Grid Search Parameters:

n_estimators: [100, 200]

max_depth: [5, 10, None]

min_samples_split: [2, 5]

min_samples_leaf: [1, 2]

Total combinations tested: 24 models

📈 Results
Best Model: Random Forest Regressor

Evaluation Metric: Mean Squared Error (MSE)

Best Parameters: Logged in MLflow

Registered Model: Best_Random_Forest_California

🔍 MLflow Features Implemented
Experiment Tracking
mlflow.set_tracking_uri() - Set MLflow tracking server

mlflow.set_experiment() - Create experiment for California housing

mlflow.start_run() - Start tracking context

Logging
mlflow.log_param() - Log individual hyperparameters

mlflow.log_metric() - Log MSE metric

Model signature inference for deployment

Model Registry
mlflow.sklearn.log_model() - Log Random Forest model

Automatic model versioning

Model signature for production deployment

Registered model name: "Best_Random_Forest_California"

📂 Project Structure

02-california-housing-prediction/
├── housepricepredict.ipynb   # Main notebook
├── README.md                  # This file
└── requirements.txt           # Python dependencies
💡 Key Learnings
GridSearchCV Integration - Systematic hyperparameter optimization

MLflow with sklearn - Tracking RandomForest experiments

Parameter Logging - Individual parameter logging with log_param()

Model Signature - Preparing models for production deployment

Regression Metrics - Using MSE for model evaluation

🎯 Next Steps
 Add RMSE and R² metrics

 Experiment with other algorithms (XGBoost, LightGBM)

 Feature engineering and selection

 Cross-validation tracking

 Model deployment with MLflow Models

📚 References
MLflow Documentation: https://mlflow.org/docs/latest/index.html

Scikit-learn Random Forest: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html


California Housing Dataset: https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset




