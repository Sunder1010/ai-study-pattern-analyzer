import pandas as pd
from sklearn.linear_model import LinearRegression

# Load dataset
data = pd.read_csv("study_data.csv")

# Prepare data
X = data.index.values.reshape(-1,1)
y = data["hours"]

# Train model
model = LinearRegression()
model.fit(X,y)

# Predict next day study hours
prediction = model.predict([[len(data)]])

print("Predicted study hours for next day:", round(prediction[0],2))
