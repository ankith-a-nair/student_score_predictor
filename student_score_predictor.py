# student_score_predictor.py

# -----------------------------
# Step 1: Import required libraries
# -----------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# -----------------------------
# Step 2: Load the dataset
# -----------------------------
print("Loading dataset...")
data = pd.read_csv("student_scores.csv")
print("Data loaded successfully!\n")

print("First 5 rows of data:")
print(data.head())

# -----------------------------
# Step 3: Visualize the data
# -----------------------------
plt.scatter(data['Hours'], data['Score'])
plt.xlabel("Hours Studied")
plt.ylabel("Score")
plt.title("Hours vs Score")
plt.grid(True)
plt.show()

# -----------------------------
# Step 4: Prepare data
# -----------------------------
X = data[['Hours']].values   # Feature (must be 2D)
y = data['Score'].values     # Target

# Split into training and testing parts (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Step 5: Train the model
# -----------------------------
model = LinearRegression()
model.fit(X_train, y_train)

print("\nModel trained successfully!")
print(f"Intercept (b): {model.intercept_:.2f}")
print(f"Coefficient (m): {model.coef_[0]:.2f}")

# -----------------------------
# Step 6: Evaluate model
# -----------------------------
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Square Error (RMSE): {rmse:.2f}")
print(f"R^2 Score: {r2:.2f}")

# -----------------------------
# Step 7: Visualize regression line
# -----------------------------
plt.scatter(X, y, color='blue', label='Actual data')
plt.plot(X, model.predict(X), color='red', label='Regression line')
plt.xlabel("Hours Studied")
plt.ylabel("Score")
plt.title("Linear Regression Fit")
plt.legend()
plt.grid(True)
plt.show()

# -----------------------------
# Step 8: Predict for a new input
# -----------------------------
hours = float(input("\nEnter number of study hours to predict score: "))
predicted_score = model.predict(np.array([[hours]]))[0]
print(f"Predicted score for studying {hours} hours = {predicted_score:.2f}")

# -----------------------------
# Step 9: Save the trained model
# -----------------------------
joblib.dump(model, "student_score_model.joblib")
print("\nModel saved as student_score_model.joblib")
