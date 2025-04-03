import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load Titanic dataset
df = pd.read_csv("Titanic_train.csv")

# Data preprocessing (No Scaler)
df = df[['Survived', 'Pclass', 'Sex', 'Age', 'Fare']].dropna()
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})  # Convert categorical

X = df[['Pclass', 'Sex', 'Age', 'Fare']]
y = df['Survived']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "titanic_model.pkl")

print("Model training complete. titanic_model.pkl saved!")
