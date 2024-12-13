import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import os

# Add debugging lines to help understand file loading
st.write("Current working directory:", os.getcwd())
st.write("Files in current directory:", os.listdir())

# Try reading the file with error handling
try:
    s = pd.read_csv("social_media_usage.csv")
    st.write("File loaded successfully!")
except Exception as e:
    st.error(f"Error loading file: {str(e)}")

# Define function clean_sm that takes one input x and uses np.where to return 0 or 1
def clean_sm(x):
    return np.where(x == 1, 1, 0)

# Select and prepare the data
columns_ss = ["income", "educ2", "par", "marital", "age", "gender", "web1h"]
ss = s.loc[:, columns_ss]

# Create the final dataframe with cleaned data
ss = pd.DataFrame({
    "income": np.where(ss["income"] > 10, np.nan, ss["income"]),
    "educ2": np.where(ss["educ2"] > 9,  np.nan, ss["educ2"]),
    "par": np.where(ss["par"] == 1, 1, 0),  # Parent binary
    "marital": np.where(ss["marital"] == 1, 1, 0),  # Marital binary
    "gender": np.where(ss["gender"] == 2, 1, 0),  # gender binary
    "age": np.where(ss["age"] > 98, np.nan, ss["age"]),
    "sm_li": clean_sm(ss["web1h"])
})

# Drop rows with missing values
ss = ss.dropna()

# Display the cleaned DataFrame in Streamlit
st.write("Cleaned DataFrame SS:")
st.write(ss.head())

# Create target vector and feature set
y = ss["sm_li"]
X = ss[["income", "educ2", "par", "marital", "gender", "age"]]

# Display shapes in Streamlit
st.write("Shape of Feature Set (X):", X.shape)
st.write("Shape of Target Vector (y):", y.shape)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    stratify=y,
    test_size=0.2,
    random_state=487
)

# Initialize and train the model
lr = LogisticRegression(class_weight="balanced", random_state=487)
lr.fit(X_train, y_train)

# Evaluate the model
accuracy = lr.score(X_test, y_test)
st.write(f"Model Accuracy: {accuracy:.2f}")

# Make predictions
y_pred = lr.predict(X_test)