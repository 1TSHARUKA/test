import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# Function to clean social media data
def clean_sm(x):
    return np.where(x == 1, 1, 0)


# Load and process data
@st.cache_data
def load_data():
    s = pd.read_csv("social_media_usage.csv")
    columns_ss = ["income", "educ2", "par", "marital", "age", "gender", "web1h"]
    ss = s.loc[:, columns_ss]

    ss = pd.DataFrame({
        "income": np.where(ss["income"] > 10, np.nan, ss["income"]),
        "educ2": np.where(ss["educ2"] > 9, np.nan, ss["educ2"]),
        "par": np.where(ss["par"] == 1, 1, 0),
        "marital": np.where(ss["marital"] == 1, 1, 0),
        "gender": np.where(ss["gender"] == 2, 1, 0),
        "age": np.where(ss["age"] > 98, np.nan, ss["age"]),
        "sm_li": clean_sm(ss["web1h"])
    })
    return ss.dropna()


# Title
st.title("LinkedIn User Prediction")

# Load data and train model
ss = load_data()
y = ss["sm_li"]
X = ss[["income", "educ2", "par", "marital", "gender", "age"]]

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=487)
lr = LogisticRegression(class_weight="balanced", random_state=487)
lr.fit(X_train, y_train)

# User inputs
income = st.slider("Income Level (1-9)", 1, 9, 5)
education = st.slider("Education Level (1-8)", 1, 8, 4)
parent = st.checkbox("Parent")
married = st.checkbox("Married")
female = st.checkbox("Female")
age = st.number_input("Age", 18, 98, 30)

# Make prediction
if st.button("Predict"):
    # Create input array
    user_input = np.array([[income, education, parent, married, female, age]])

    # Get prediction and probability
    prediction = lr.predict(user_input)[0]
    probability = lr.predict_proba(user_input)[0][1]

    # Display results
    st.write("### Prediction Results:")
    st.write(f"Prediction: {'LinkedIn User' if prediction == 1 else 'Not a LinkedIn User'}")
    st.write(f"Probability of being a LinkedIn user: {probability:.2%}")