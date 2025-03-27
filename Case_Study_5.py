import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load data from CSV
data = pd.read_csv('csv_files\\crop_data.csv')

# Features and target
X = data[['soil_quality', 'rainfall', 'temperature']]
y = data['yield']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")

# Predict yield for new conditions
new_data = pd.DataFrame({'soil_quality': [7.2], 'rainfall': [110], 'temperature': [26]})
prediction = model.predict(new_data)
print(f"Predicted yield: {prediction[0]} kg/hectare")