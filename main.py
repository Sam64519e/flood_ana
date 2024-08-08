import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load the data
data = pd.read_csv('FloodPrediction.csv')

# Print column names to debug
print("Columns in the dataset:")
print(data.columns)

# Check for the presence of the 'Flood?' column
if 'Flood?' not in data.columns:
    print("The 'Flood?' column is not found in the dataset. Please check the column name.")
    # Optionally, print the first few rows to inspect the data
    print(data.head())
    exit()

# Preprocess the data
# Compute average temperature
data['Avg_Temp'] = (data['Max_Temp'] + data['Min_Temp']) / 2

# Remove rows with missing target values
data = data.dropna(subset=['Flood?'])

# Select relevant features
features = ['Avg_Temp', 'Rainfall', 'Relative_Humidity', 'Wind_Speed']
X = data[features]
y = data['Flood?']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print('Accuracy:', accuracy_score(y_test, y_pred))
print('Classification Report:')
print(classification_report(y_test, y_pred))

# Save the model for future use (optional)
import joblib

joblib.dump(model, 'flood_predictor_model.pkl')
