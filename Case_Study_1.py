import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load data from CSV
data = pd.read_csv('csv_files\\solar_data.csv')

# Features and target
X = data[['temperature', 'voltage', 'current']]
y = data['failure']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

# Predict failure for new data
new_data = pd.DataFrame({'temperature': [38], 'voltage': [212], 'current': [4.2]})
prediction = model.predict(new_data)
print(f"Predicted failure: {prediction[0]}")