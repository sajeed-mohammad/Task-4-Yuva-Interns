import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv("../data/logistics_data.csv")

print("=" * 80)
print("LOGISTICS DELIVERY DELAY - PREDICTIVE OPTIMIZATION")
print("=" * 80)

print("\nDataset loaded successfully!")
print("Total deliveries:", len(df))


# ============================================================
# 2. FEATURES AND TARGET
# ============================================================

features = [
    "Warehouse",
    "Customer_Location",
    "Delivery_Distance_km",
    "Transportation_Mode",
    "Package_Weight_kg",
    "Delivery_Priority",
    "Weather_Condition",
    "Traffic_Level",
    "Warehouse_Processing_Time_min",
    "Planned_Delivery_Time_hr"
]

X = df[features]
y = df["Delivery_Delay"]


# ============================================================
# 3. PREPROCESSING
# ============================================================

categorical_features = [
    "Warehouse",
    "Customer_Location",
    "Transportation_Mode",
    "Delivery_Priority",
    "Weather_Condition",
    "Traffic_Level"
]

numerical_features = [
    "Delivery_Distance_km",
    "Package_Weight_kg",
    "Warehouse_Processing_Time_min",
    "Planned_Delivery_Time_hr"
]

preprocessor = ColumnTransformer([
    (
        "categorical",
        OneHotEncoder(handle_unknown="ignore"),
        categorical_features
    ),
    (
        "numerical",
        "passthrough",
        numerical_features
    )
])


# ============================================================
# 4. RANDOM FOREST MODEL
# ============================================================

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])


# ============================================================
# 5. TRAIN MODEL
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

pipeline.fit(X_train, y_train)

print("\nModel trained successfully!")


# ============================================================
# 6. PREDICT DELAY PROBABILITY
# ============================================================

df["Delay_Probability"] = pipeline.predict_proba(X)[:, 1]


# ============================================================
# 7. OPTIMIZATION / DECISION RULES
# ============================================================

def recommend_action(row):

    probability = row["Delay_Probability"]
    traffic = row["Traffic_Level"]
    weather = row["Weather_Condition"]
    warehouse_time = row["Warehouse_Processing_Time_min"]

    actions = []

    # ----------------------------
    # Risk level
    # ----------------------------

    if probability >= 0.80:

        risk = "High Risk"

        actions.append(
            "Priority intervention: review route, ETA and dispatch resources"
        )

    elif probability >= 0.60:

        risk = "Medium Risk"

        actions.append(
            "Monitor closely and consider a delivery-time buffer"
        )

    else:

        risk = "Low Risk"

        actions.append(
            "Standard monitoring"
        )


    # ----------------------------
    # Traffic optimization
    # ----------------------------

    if traffic == "High":

        actions.append(
            "Use traffic-aware alternate routing"
        )


    # ----------------------------
    # Weather optimization
    # ----------------------------

    if weather in ["Rain", "Storm"]:

        actions.append(
            "Add weather-related time buffer"
        )


    # ----------------------------
    # Warehouse optimization
    # ----------------------------

    if warehouse_time > 45:

        actions.append(
            "Review warehouse processing bottleneck"
        )


    return risk, " | ".join(actions)


# ============================================================
# 8. APPLY OPTIMIZATION TO EVERY DELIVERY
# ============================================================

recommendations = df.apply(
    recommend_action,
    axis=1
)

df["Risk_Level"] = [
    result[0] for result in recommendations
]

df["Recommended_Action"] = [
    result[1] for result in recommendations
]


# ============================================================
# 9. SORT BY HIGHEST RISK
# ============================================================

results = df.sort_values(
    by="Delay_Probability",
    ascending=False
).copy()


# ============================================================
# 10. DISPLAY SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("OPTIMIZATION SUMMARY")
print("=" * 80)

print(
    "High-risk deliveries   :",
    (results["Risk_Level"] == "High Risk").sum()
)

print(
    "Medium-risk deliveries :",
    (results["Risk_Level"] == "Medium Risk").sum()
)

print(
    "Low-risk deliveries    :",
    (results["Risk_Level"] == "Low Risk").sum()
)


# ============================================================
# 11. TOP 10 HIGH-RISK DELIVERIES
# ============================================================

print("\n" + "=" * 80)
print("TOP 10 HIGH-RISK DELIVERIES")
print("=" * 80)

top10 = results[
    [
        "Warehouse",
        "Customer_Location",
        "Delivery_Distance_km",
        "Transportation_Mode",
        "Weather_Condition",
        "Traffic_Level",
        "Warehouse_Processing_Time_min",
        "Delay_Probability",
        "Risk_Level",
        "Recommended_Action"
    ]
].head(10).copy()


# Convert probability into percentage
top10["Delay_Probability"] = (
    top10["Delay_Probability"] * 100
).round(2)


print(top10.to_string(index=False))


# ============================================================
# 12. SAVE COMPLETE OPTIMIZATION RESULTS
# ============================================================

output_file = "../data/optimization_results.csv"

results.to_csv(
    output_file,
    index=False
)


# ============================================================
# 13. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 80)
print("OPTIMIZATION COMPLETED SUCCESSFULLY!")
print("=" * 80)

print(
    "Complete results saved to:",
    output_file
)

print("\nThe output contains:")
print("- Delay Probability")
print("- Risk Level")
print("- Traffic condition")
print("- Weather condition")
print("- Warehouse processing time")
print("- Recommended Action")

print("=" * 80)