import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score

MODEL_FILE = "aqi_model.pkl"
PIPELINE_FILE = "aqi_pipeline.pkl"

# BUILD PIPELINE FUNCTION
def build_pipeline(num_attribs, cat_attribs):

    # Numerical Pipeline
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Categorical Pipeline
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    # Full Pipeline
    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", cat_pipeline, cat_attribs)
    ])

    return full_pipeline

# TRAIN MODEL IF NOT EXISTS
if not os.path.exists(MODEL_FILE):

    print("Training model...")

    # Load Dataset
    df = pd.read_csv("mydata.csv")

    # Drop unwanted column
    if "Unnamed: 0" in df.columns:
        df.drop("Unnamed: 0", axis=1, inplace=True)

    # Convert Date column
    df["Date"] = pd.to_datetime(df["Date"])

    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day

    # Drop original Date column
    df.drop("Date", axis=1, inplace=True)

    # Predicting AQI
    y = df["AQI"].copy()

    # Drop target columns
    X = df.drop(["AQI", "AQI_Bucket"], axis=1)

    # TRAIN TEST SPLIT
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Save test input for inference
    X_test.to_csv("input.csv", index=False)

    num_attribs = X_train.select_dtypes(include=np.number).columns.tolist()

    cat_attribs = X_train.select_dtypes(include="object").columns.tolist()

    # CREATE PIPELINE
    pipeline = build_pipeline(num_attribs, cat_attribs)

    # Transform Data
    X_train_prepared = pipeline.fit_transform(X_train)

   
    # MODEL TRAINING
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train_prepared, y_train)

    X_test_prepared = pipeline.transform(X_test)

    predictions = model.predict(X_test_prepared)

    rmse = root_mean_squared_error(y_test, predictions)

    r2 = r2_score(y_test, predictions)

    print(f"RMSE : {rmse}")
    print(f"R2 Score : {r2}")

    # CROSS VALIDATION
  
    scores = cross_val_score(
        model,
        X_train_prepared,
        y_train,
        scoring="neg_root_mean_squared_error",
        cv=5
    )

    rmse_scores = -scores

    print("Cross Validation RMSE Scores:")
    print(rmse_scores)

    print("Average RMSE:", rmse_scores.mean())

    joblib.dump(model, MODEL_FILE)

    joblib.dump(pipeline, PIPELINE_FILE)

    print("Model training complete!")
    print("Model and pipeline saved successfully!")
else:

    print("Loading trained model...")

    # Load model and pipeline
    model = joblib.load(MODEL_FILE)

    pipeline = joblib.load(PIPELINE_FILE)

    # Load new input data
    input_data = pd.read_csv("input.csv")

    # Transform input
    transformed_input = pipeline.transform(input_data)

    # Predict
    predictions = model.predict(transformed_input)

    # Save predictions
    input_data["Predicted_AQI"] = predictions

    input_data.to_csv("output.csv", index=False)

    print("Inference complete!")
    print("Results saved to output.csv")