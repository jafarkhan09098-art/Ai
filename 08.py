
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load dataset
df = pd.read_csv("online_shoppers_intention.csv")

# Display first five records
print("First Five Records")
print(df.head())

# Encode categorical columns
le = LabelEncoder()

df["Month"] = le.fit_transform(df["Month"])
df["VisitorType"] = le.fit_transform(df["VisitorType"])
df["Weekend"] = le.fit_transform(df["Weekend"])
df["Revenue"] = le.fit_transform(df["Revenue"])

# Split features and target
X = df.drop("Revenue", axis=1)
y = df["Revenue"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Weak Classifier
weak_model = LogisticRegression(max_iter=200, random_state=42)
weak_model.fit(X_train, y_train)

weak_pred = weak_model.predict(X_test)
weak_accuracy = accuracy_score(y_test, weak_pred)

print("\nWeak Classifier Accuracy =", round(weak_accuracy * 100, 2), "%")

# AdaBoost Ensemble Model
ada_model = AdaBoostClassifier(
    n_estimators=50,
    learning_rate=1.0,
    random_state=42
)

ada_model.fit(X_train, y_train)

ada_pred = ada_model.predict(X_test)

print("\nModel Trained Successfully")

ada_accuracy = accuracy_score(y_test, ada_pred)

print("\nAdaBoost Accuracy =", round(ada_accuracy * 100, 2), "%")

# Compare models
print("\nComparison of Models")

print("Weak Classifier Accuracy =", round(weak_accuracy * 100, 2), "%")

print("AdaBoost Ensemble Accuracy =", round(ada_accuracy * 100, 2), "%")
