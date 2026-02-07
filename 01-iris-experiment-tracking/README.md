# 01 - Iris Experiment Tracking with MLflow

MLflow experiment tracking implementation using the classic Iris dataset and Logistic Regression classifier.

## 📋 Project Overview

This project demonstrates MLflow's core experiment tracking capabilities including:
- Parameter logging
- Metrics tracking
- Model registration and versioning
- Experiment comparison

## 🛠️ Tech Stack

- **Python** 3.13
- **MLflow** - Experiment tracking and model registry
- **Scikit-learn** - Machine learning model (Logistic Regression)
- **Pandas** - Data manipulation

## 📊 Dataset

**Iris Dataset** (UCI Machine Learning Repository)
- 150 samples
- 4 features: sepal length, sepal width, petal length, petal width
- 3 classes: Setosa, Versicolor, Virginica
- Split: 80% training, 20% testing

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install mlflow scikit-learn pandas

2. Start MLflow UI

mlflow ui --port 5000

Open your browser to http://127.0.0.1:5000

3. Run the Notebook

Open your browser to http://127.0.0.1:5000

3. Run the Notebook

jupyter notebook gettingstarted.ipynb

📈 Results
Algorithm: Logistic Regression

Solver: lbfgs, newton-cg

Max Iterations: 1000

Accuracy: ~96.67%

🔍 MLflow Features Implemented
Experiment Tracking
mlflow.set_tracking_uri() - Set MLflow tracking server

mlflow.set_experiment() - Create/select experiment

mlflow.start_run() - Start tracking context

Logging
mlflow.log_params() - Log model hyperparameters

mlflow.log_metrics() - Log performance metrics (accuracy)

mlflow.set_tag() - Add metadata tags

Model Registry
mlflow.sklearn.log_model() - Log scikit-learn model

Model versioning (versions 4, 5 created)

Model signature inference

Input example logging

📂 Project Structure

01-iris-experiment-tracking/
├── gettingstarted.ipynb    # Main notebook
├── README.md               # This file
└── requirements.txt        # Python dependencies

💡 Key Learnings
MLflow Tracking Server: Setting up local tracking URI for experiment management

Parameter Comparison: Testing different solvers (lbfgs vs newton-cg)

Model Versioning: Automatic versioning with model registry

Experiment Organization: Using experiments to group related runs

🔗 MLflow UI Views
After running the notebook, check the MLflow UI for:

Experiment runs comparison

Parameter and metric visualizations

Model artifacts and signatures

Registered model versions

📝 Notes
MLflow tracking server runs locally at http://127.0.0.1:5000

Experiment name: "IrisClassificationExperiment"

Registered model name: "tracking-quickstart"

Models saved in pickle format (scikit-learn default)

📚 References
https://mlflow.org/docs/latest/index.html

https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html



***

## Also create a `requirements.txt` file:

```txt
mlflow>=2.10.0
scikit-learn>=1.4.0
pandas>=2.1.0
numpy>=1.24.0

