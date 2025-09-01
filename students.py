# Student Performance Analyzer - ML Version
# Roshni Sharma Project Code
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

# -------------------------------
# 1. Load Datasets
# -------------------------------
# UCI student performance dataset (CSV with grades & student attributes)
# Replace with your dataset path
data = pd.read_csv("C:/Users/biswajeet/Downloads/student-mat.csv", sep=";")

# -------------------------------
# 2. Preprocessing
# -------------------------------
# Encode categorical variables
label_encoders = {}
for col in data.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Target Variable: Pass/Fail (if final grade G3 >= 10 → Pass else Fail)
data['performance'] = data['G3'].apply(lambda x: 1 if x >= 10 else 0)

X = data.drop(columns=["G1", "G2", "G3", "performance"])  # Features
y = data["performance"]  # Target

# Scale numerical features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# 3. Model Training
# -------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(kernel="rbf", probability=True, random_state=42)
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results[name] = classification_report(y_test, y_pred, output_dict=True)

    print(f"\n{name} Results:")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

# -------------------------------
# 4. Feature Importance (Random Forest)
# -------------------------------
rf = models["Random Forest"]
importances = rf.feature_importances_
feature_names = data.drop(columns=["G1", "G2", "G3", "performance"]).columns

# Plot Feature Importance
feat_imp = pd.DataFrame({"Feature": feature_names, "Importance": importances})
feat_imp = feat_imp.sort_values(by="Importance", ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x="Importance", y="Feature", data=feat_imp)
plt.title("Feature Importance (Random Forest)")
plt.show()