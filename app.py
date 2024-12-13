import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

import os
os.system("pip install notebook")

# Load and clean the dataset
data = pd.read_csv('social_media_usage.csv')
data_filtered = data[['income', 'educ2', 'web1h', 'par', 'marital', 'age']]
data_filtered = data_filtered[
    (data_filtered['income'] <= 9) &
    (data_filtered['educ2'] <= 8) &
    (data_filtered['web1h'] <= 2) &
    (data_filtered['par'] <= 2) &
    (data_filtered['marital'] <= 6) &
    (data_filtered['age'] <= 98)
].dropna()

# Rename target column
data_filtered.rename(columns={'web1h': 'sm_li'}, inplace=True)

# Define features (X) and target (y)
X = data_filtered[['income', 'educ2', 'par', 'marital', 'age']]
y = data_filtered['sm_li']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=987
)

# Train logistic regression model
model = LogisticRegression(class_weight='balanced', random_state=987)
model.fit(X_train, y_train)

# Streamlit app for predictions
st.title("LinkedIn User Prediction App")

st.write("### Data Definitions")
st.write("""
- **Age**: Between 1 and 97  
- **Education**:  
  1: Less than high school  
  2: High school incomplete  
  3: High school graduate  
  4: Some college, no degree  
  5: Two-year associate degree  
  6: Four-year college graduate  
  7: Some postgraduate or professional schooling  
  8: Postgraduate or professional degree  

- **Parent**:  
  1: Yes  
  2: No  

- **Marital Status**:  
  1: Married  
  2: Living with a partner  
  3: Divorced  
  4: Separated  
  5: Widowed  
  6: Never been married  

- **Income**: 1 (lowest income) to 9 (highest income)
""")

# Input fields for user data
income = st.slider("Income (1-9)", 1, 9, step=1)
education = st.slider("Education Level (1-8)", 1, 8, step=1)
parent = st.selectbox("Parent (1: Yes, 2: No)", [1, 2])
marital = st.selectbox(
    "Marital Status",
    [1, 2, 3, 4, 5, 6],
    format_func=lambda x: [
        "Married", "Living with a partner", "Divorced",
        "Separated", "Widowed", "Never been married"
    ][x - 1]
)
age = st.number_input("Age (1-97)", min_value=1, max_value=97, step=1)

if st.button("Predict"):
    # Prepare input for prediction
    user_input = np.array([[income, education, parent, marital, age]])
    
    # Make predictions
    prediction = model.predict(user_input)
    prob = model.predict_proba(user_input)[:, 1]  # Probability for LinkedIn user
    
    # Display results
    if prediction[0] == 1:
        st.success(f"This individual is likely a LinkedIn user. Probability: {prob[0]:.2f}")
    else:
        st.warning(f"This individual is unlikely to be a LinkedIn user. Probability: {prob[0]:.2f}")


# In[ ]:




