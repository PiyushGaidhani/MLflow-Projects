# Wine Quality Prediction with MLflow & Streamlit

Predict white wine quality using machine learning, track experiments with **MLflow**, serve the best model as a REST API, and interact with it via a **Streamlit** web app.[web:86][web:101]

---

## 1. Project overview

This project uses the UCI Wine Quality dataset (white wine) to predict a quality score based on 11 physicochemical features (acidity, sugar, sulphates, alcohol, etc.).[web:101]

It demonstrates an end‑to‑end MLOps workflow:

- Data preparation and model training in a notebook.
- Hyperparameter tuning and experiment tracking with **MLflow**.
- Model registration and serving via **MLflow Model Registry**.[web:107]
- A simple REST client (`test_wine_quality_api.py`) that calls the `/invocations` endpoint.
- A **Streamlit** UI that lets users move sliders and see predicted wine quality in real time.[web:97][web:99]

---

## 2. Tech stack

- Python, NumPy, Pandas
- Scikit‑learn (**RandomForestRegressor** as deployed model)
- TensorFlow / Keras for ANN experiments (not deployed due to Keras 3 SavedModel issues)
- MLflow for tracking, model registry, and serving
- Streamlit for the front‑end web app[web:86][web:101]

---

## 3. Repository structure

```text
MLflow-Projects/
└── 03-wine-quality-mlflow/
    ├── wine_quality_mlflow.ipynb     # Notebook: EDA, training, MLflow logging
    ├── test_wine_quality_api.py      # Python client to call MLflow REST API
    ├── app.py                        # Streamlit UI for predictions
    ├── requirements.txt              # Project dependencies
    ├── README.md                     # This file
    └── mlruns/                       # Local MLflow tracking directory (gitignored)
```

## 4. Setup
Clone the repository and navigate to the project folder:

git clone <your_repo_url>
cd MLflow-Projects/02-wine-quality-mlflow

Create a virtual environment and install dependencies:
pip install -r requirements.txt
 or, minimally:
pip install mlflow scikit-learn pandas numpy streamlit requests

(Optional) Start MLflow UI to explore experiments:
mlflow ui
 Open http://127.0.0.1:5000

## 5. Training and experiment tracking
Model development is done in Starter.ipynb:

Load the white wine quality dataset.

Split into train/validation/test sets.

Train an ANN with Hyperopt for hyperparameter tuning (learning rate, momentum) and log each run in MLflow.

Train a RandomForestRegressor as a simpler, fast baseline.

Log evaluation metrics (e.g., RMSE) and models to the /wine-quality MLflow experiment.

Example of logging the RandomForest model (simplified):
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

rf_model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
rf_model.fit(train_x, train_y)

rf_pred = rf_model.predict(valid_x)
mse = mean_squared_error(valid_y, rf_pred)
rf_rmse = np.sqrt(mse)

mlflow.set_experiment("/wine-quality")

with mlflow.start_run(run_name="wine_rf_serving"):
    mlflow.log_metric("eval_rmse", rf_rmse)
    mlflow.sklearn.log_model(rf_model, "model")

In MLflow UI, you sort runs by eval_rmse ascending and pick the run with the lowest error as the production candidate.

6. Model registry and serving
After selecting the best run in MLflow UI:

Register the model in MLflow Model Registry (e.g., name wine-quality-model) and assign a stage (e.g., Production).​

Serve the registered model locally:
 From the project directory
mlflow models serve \
  -m "models:/wine-quality-model/Production" \
  -p 5001 --no-conda

  This starts a local REST API on http://127.0.0.1:5001/invocations.

## 7. Testing the REST API (/invocations)
You can test the endpoint either with curl or the provided Python script.

Curl example
curl -X POST http://127.0.0.1:5001/invocations \
  -H "Content-Type: application/json" \
  -d '{
    "dataframe_split": {
      "columns": [
        "fixed acidity","volatile acidity","citric acid","residual sugar",
        "chlorides","free sulfur dioxide","total sulfur dioxide",
        "density","pH","sulphates","alcohol"
      ],
      "data": [
        [7.0,0.27,0.36,20.7,0.045,45.0,170.0,1.001,3.0,0.45,8.8]
      ]
    }
  }'

Typical response:
{"predictions": [5.925]}


Python test client
test_wine_quality_api.py sends the same payload programmatically and prints the status and predictions.

python test_wine_quality_api.py

Example output:

🍷 Testing Wine Quality Prediction API...
Status: 200
Body: {"predictions": [5.925]}
Prediction JSON: {'predictions': [5.925]}

## 8. Streamlit web app
The Streamlit app (app.py) provides a simple UI to interact with the MLflow model without touching curl or Python scripts.​​

Key features:

Sliders for all 11 physicochemical features.

A “Predict quality” button that sends a request to http://127.0.0.1:5001/invocations.

Displays the predicted wine quality score in the UI.

Run the app:

# Ensure the MLflow server is already running on port 5001
streamlit run app.py
Then open the URL shown in the terminal (usually http://localhost:8501) and interact with the UI.

## 9. Possible extensions
Ideas for future improvements:

Containerize the model server and Streamlit app with Docker for portability.

Deploy to a cloud platform (e.g., AWS EC2, Azure, or Kubernetes) using the MLflow Docker image.

Add authentication or request logging middleware in front of the MLflow endpoint.​

Extend the Streamlit app with dataset exploration plots and feature importance visualizations.





