import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load data from CSV
data = pd.read_csv('csv_files\\building_data.csv')

# Features and target
X = data[['occupancy', 'temperature']]
y = data['energy_consumption']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")

# Predict energy for new conditions
new_data = pd.DataFrame({'occupancy': [35], 'temperature': [31]})
prediction = model.predict(new_data)
print(f"Predicted energy consumption: {prediction[0]} kWh")