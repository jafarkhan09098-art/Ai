
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv("online_shoppers_intention.csv")

print("First Five Records")
print(df.head())

print("\nDataset Information")
print(df.info())

print("\nShape")
print(df.shape)

print("\nColumns")
print(df.columns.tolist())

encoder = LabelEncoder()

categorical_columns = [
    "Month",
    "VisitorType",
    "Weekend"
]

for column in categorical_columns:
    df[column] = encoder.fit_transform(df[column])

df["Revenue"] = encoder.fit_transform(df["Revenue"])

print("\nEncoded Dataset")
print(df.head())

x = df.drop("Revenue", axis=1)
y = df["Revenue"]

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Shape")
print(x_train.shape)

print("\nTesting Shape")
print(x_test.shape)

scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(x_train, y_train)

print("\nModel Trained Successfully")

y_pred = knn.predict(x_test)

result = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})

print("\nFirst 20 Predictions")
print(result.head(20))

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy =", round(accuracy * 100, 2), "%")

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix")
print(cm)

print("\nClassification Report")
print(classification_report(y_test, y_pred))

plt.figure(figsize=(6, 4))

df["Revenue"].value_counts().plot(kind="bar")

plt.title("Revenue Distribution")
plt.xlabel("Revenue")
plt.ylabel("Count")

plt.show()
