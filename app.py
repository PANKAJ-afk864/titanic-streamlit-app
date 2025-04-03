
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model
model = joblib.load("titanic_model.pkl")

st.title("Titanic Survival Prediction")

# User inputs
Pclass = st.selectbox("Passenger Class", [1, 2, 3])
Sex = st.selectbox("Sex", ["Male", "Female"])
Age = st.number_input("Age", min_value=0, max_value=100, value=30)
SibSp = st.number_input("Siblings/Spouses Aboard", min_value=0, max_value=10, value=0)
Parch = st.number_input("Parents/Children Aboard", min_value=0, max_value=10, value=0)
Fare = st.number_input("Fare", min_value=0.0, max_value=500.0, value=50.0)
Embarked = st.selectbox("Embarked Port", ["C", "Q", "S"])

# Convert categorical inputs to numerical values
Sex = 0 if Sex == "Male" else 1
Embarked = {"C": 0, "Q": 1, "S": 2}[Embarked]

# Prediction
if st.button("Predict Survival"):
    features = np.array([[Pclass, Sex, Age, SibSp, Parch, Fare, Embarked]])
    prediction = model.predict(features)[0]

    if prediction == 1:
        st.success("The passenger is likely to survive. ✅")
    else:
        st.error("The passenger is unlikely to survive. ❌")
