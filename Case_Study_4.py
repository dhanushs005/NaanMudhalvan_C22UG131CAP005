import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load data from CSV
data = pd.read_csv('csv_files\\air_quality_data.csv')

# Features and target
X = data[['PM2.5', 'traffic_volume', 'temperature']]
y = data['AQI']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Gradient Boosting model
model = GradientBoostingRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")

# Predict AQI for new conditions
new_data = pd.DataFrame({'PM2.5': [75], 'traffic_volume': [1600], 'temperature': [27]})
prediction = model.predict(new_data)
print(f"Predicted AQI: {prediction[0]}")