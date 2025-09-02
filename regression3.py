import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

file_path = "1) iris.csv"
df = pd.read_csv(file_path)

X = df[["sepal_length"]]
y = df["petal_length"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

print("Intercept (b0):", model.intercept_)
print("Coefficient (b1):", model.coef_[0])

y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("R-squared:", r2)
print("Mean Squared Error:", mse)

plt.figure(figsize=(8,6))
# Sort X_test and corresponding predictions
sorted_idx = X_test.squeeze().argsort()
X_sorted = X_test.squeeze().iloc[sorted_idx]
y_pred_sorted = y_pred[sorted_idx]

plt.figure(figsize=(8,6))
sns.scatterplot(x=X_test.squeeze(), y=y_test, color="blue", label="Actual")
sns.lineplot(x=X_sorted, y=y_pred_sorted, color="red", label="Predicted")
plt.title("Linear Regression: Actual vs Predicted", fontsize=14)
plt.xlabel("Sepal Length", fontsize=12)
plt.ylabel("Petal Length", fontsize=12)
plt.legend()
plt.show()

