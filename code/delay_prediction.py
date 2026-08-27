import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

df = pd.read_csv("../data/logistics_data.csv")

features = [
    "Warehouse", "Customer_Location", "Delivery_Distance_km",
    "Transportation_Mode", "Package_Weight_kg", "Delivery_Priority",
    "Weather_Condition", "Traffic_Level",
    "Warehouse_Processing_Time_min", "Planned_Delivery_Time_hr"
]

X = df[features]
y = df["Delivery_Delay"]

categorical_features = [
    "Warehouse", "Customer_Location", "Transportation_Mode",
    "Delivery_Priority", "Weather_Condition", "Traffic_Level"
]
numerical_features = [
    "Delivery_Distance_km", "Package_Weight_kg",
    "Warehouse_Processing_Time_min", "Planned_Delivery_Time_hr"
]

preprocessor = ColumnTransformer([
    ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ("numerical", "passthrough", numerical_features)
])

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

pipeline.fit(X_train, y_train)

pred = pipeline.predict(X_test)
prob = pipeline.predict_proba(X_test)[:, 1]

print("Accuracy:", accuracy_score(y_test, pred))
print("Precision:", precision_score(y_test, pred, zero_division=0))
print("Recall:", recall_score(y_test, pred, zero_division=0))
print("F1:", f1_score(y_test, pred, zero_division=0))
print("ROC-AUC:", roc_auc_score(y_test, prob))
print("Confusion Matrix:\n", confusion_matrix(y_test, pred))
