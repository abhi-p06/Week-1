import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

arr = np.array([10, 20, 30, 40])
print("Mean:", arr.mean())
print("Squared:", arr**2)

data = pd.DataFrame({
    "City": ["Delhi", "Mumbai", "Chennai"],
    "Temperature": [32, 35, 31]
})
print(data)
print("Average temp:", data["Temperature"].mean())

plt.bar(data["City"], data["Temperature"], color='green')
plt.title("City vs Temperature")
plt.show()
