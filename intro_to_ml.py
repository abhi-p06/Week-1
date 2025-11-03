import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Training data: hours studied vs marks obtained
X = np.array([[1], [2], [3], [4], [5], [6]])
y = np.array([35, 45, 50, 60, 70, 80])

model = LinearRegression()
model.fit(X, y)

# Predict marks
pred = model.predict(X)

plt.scatter(X, y, color='blue', label='Actual')
plt.plot(X, pred, color='red', label='Predicted')
plt.xlabel("Hours Studied")
plt.ylabel("Marks Obtained")
plt.title("Simple Linear Regression")
plt.legend()
plt.show()

print("Predictions:", np.round(pred, 2))
