
import pandas as pd
import numpy as np

np.random.seed(42)

n = 12330

df = pd.DataFrame()

df["Administrative"] = np.random.poisson(2, n)
df["Administrative_Duration"] = np.random.uniform(0, 300, n)

df["Informational"] = np.random.poisson(1, n)
df["Informational_Duration"] = np.random.uniform(0, 200, n)

df["ProductRelated"] = np.random.poisson(20, n)
df["ProductRelated_Duration"] = np.random.uniform(0, 1500, n)

df["BounceRates"] = np.random.uniform(0, 0.2, n)
df["ExitRates"] = np.random.uniform(0, 0.2, n)

df["SpecialDay"] = np.random.choice(
    [0, 0.2, 0.4, 0.6, 0.8, 1], n
)

df["Month"] = np.random.choice(
    ["Jan", "Feb", "Mar", "Apr", "May", "June",
     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], n
)

df["OperatingSystems"] = np.random.randint(1, 9, n)
df["Browser"] = np.random.randint(1, 14, n)
df["Region"] = np.random.randint(1, 10, n)
df["TrafficType"] = np.random.randint(1, 21, n)

df["VisitorType"] = np.random.choice(
    ["Returning_Visitor", "New_Visitor", "Other"],
    n,
    p=[0.85, 0.14, 0.01]
)

df["Weekend"] = np.random.choice([True, False], n)

# Generate Revenue values
score = (
    df["ProductRelated"] * 0.08
    + df["Administrative"] * 0.05
    - df["BounceRates"] * 10
    - df["ExitRates"] * 8
    + np.random.normal(0, 2, n)
)

probability = 1 / (1 + np.exp(-(score - 2.5)))

df["Revenue"] = np.random.random(n) < probability

# Add PageValues
df["PageValues"] = np.where(
    df["Revenue"],
    np.random.uniform(5, 100, n),
    np.random.uniform(0, 5, n)
)

# Save CSV file
df.to_csv("online_shoppers_intention.csv", index=False)

print("CSV Dataset Created Successfully!")
print("Total Records:", len(df))
print("Total Columns:", len(df.columns))
print("File Name: online_shoppers_intention.csv")
