import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split


df_focus = pd.read_csv("data_focused.csv")
df_distract = pd.read_csv("data_distracted.csv")

df = pd.concat([df_focus, df_distract], axis=0).dropna()

df["label"] = df["label"].map({"focused": 1, "distracted": 0})

X = df.drop("label", axis=1)
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

print("Training the Random Forest model...")
print(f"  Samples: {len(df)} (focused={int((y == 1).sum())}, distracted={int((y == 0).sum())})")
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced",
    min_samples_leaf=2,
    max_depth=None,
    n_jobs=-1,
)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
acc = accuracy_score(y_test, predictions)
bacc = balanced_accuracy_score(y_test, predictions)
print(f"\nAccuracy: {acc * 100:.2f}%")
print(f"Balanced accuracy (better for uneven classes): {bacc * 100:.2f}%")
print("\nConfusion matrix [rows=true: distracted=0, focused=1] [cols=pred]:")
print(confusion_matrix(y_test, predictions))
print("\nClassification report (0=distracted, 1=focused):")
print(classification_report(y_test, predictions, digits=3))

joblib.dump(model, "focus_model.pkl")
print("\nModel saved as focus_model.pkl")
