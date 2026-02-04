import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib # For saving the model

# 1. Load the data
df_focus = pd.read_csv('data_focused.csv')
df_distract = pd.read_csv('data_distracted.csv')

# 2. Combine and Clean
# We drop any rows that might have NaN values if MediaPipe missed a frame
df = pd.concat([df_focus, df_distract], axis=0).dropna()

# Convert labels to numbers: Focused = 1, Distracted = 0
df['label'] = df['label'].map({'focused': 1, 'distracted': 0})

# 3. Split Features and Labels
X = df.drop('label', axis=1)
y = df['label']

# Split into Training and Testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train the "Brain"
print("Training the Random Forest model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Evaluate
predictions = model.predict(X_test)
acc = accuracy_score(y_test, predictions)
print(f"Model Accuracy: {acc * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, predictions))

# 6. Save the model for Phase 3
joblib.dump(model, 'focus_model.pkl')
print("Model saved as focus_model.pkl")