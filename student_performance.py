import numpy as np
import pandas as pd
import streamlit as st

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


# Load Dataset
df = pd.read_csv("StudentsPerformance.csv")

st.title("Student Performance Prediction")

st.write("Dataset Preview")
st.dataframe(df.head())


# Feature Engineering
df["TotalScore"] = df["math score"] + df["reading score"] + df["writing score"]


# Encoding categorical data
encoder = LabelEncoder()

df["gender"] = encoder.fit_transform(df["gender"])
df["lunch"] = encoder.fit_transform(df["lunch"])
df["test preparation course"] = encoder.fit_transform(df["test preparation course"])
df["parental level of education"] = encoder.fit_transform(df["parental level of education"])


# Features and Target
X = df[
    [
        "gender",
        "lunch",
        "test preparation course",
        "parental level of education",
        "reading score",
        "writing score",
    ]
]

y = df["math score"]


# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(),
    "Gradient Boosting": GradientBoostingRegressor(),
}


st.header("Model Performance")

for name, model in models.items():

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)

    st.write(f"### {name}")
    st.write("RMSE:", rmse)
    st.write("R2 Score:", r2)
