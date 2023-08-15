import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load the CSV data into a pandas DataFrame
data = pd.read_csv('pose_keypoints.csv')

# Extract features (keypoints) and labels (class_name)
features = data.iloc[:, 1:]  # Assuming keypoints start from column 1
labels = data.iloc[:, 0]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.01, shuffle=True, random_state=42)

# Apply SMOTE to balance the classes
smote = SMOTE(k_neighbors=3, random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Train a Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=200, random_state=42)
rf_classifier.fit(X_train_resampled, y_train_resampled)

# Predict using the trained model
y_pred = rf_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Save the trained model to a file
joblib.dump(rf_classifier, 'model.joblib')