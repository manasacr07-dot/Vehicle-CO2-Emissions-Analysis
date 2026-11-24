from pathlib import Path
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "co2_emissions.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

TARGET = "CO2 Emissions(g/km)"
FEATURES = [
    "Make", "Model", "Vehicle Class", "Engine Size(L)", "Cylinders",
    "Transmission", "Fuel Type", "Fuel Consumption City (L/100 km)",
    "Fuel Consumption Hwy (L/100 km)", "Fuel Consumption Comb (L/100 km)",
    "Fuel Consumption Comb (mpg)"
]
NUMERIC = [
    "Engine Size(L)", "Cylinders", "Fuel Consumption City (L/100 km)",
    "Fuel Consumption Hwy (L/100 km)", "Fuel Consumption Comb (L/100 km)",
    "Fuel Consumption Comb (mpg)"
]
CATEGORICAL = [c for c in FEATURES if c not in NUMERIC]

def make_pipeline(model):
    prep = ColumnTransformer([
        ("numeric", "passthrough", NUMERIC),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)
    ])
    return Pipeline([("preprocess", prep), ("model", model)])

def main():
    df = pd.read_csv(DATA)
    df = df.drop_duplicates().copy()
    df["Model"] = df["Model"].astype(str).str.strip().str.upper()
    df = df.dropna(subset=FEATURES + [TARGET])
    X = df[FEATURES]
    y = df[TARGET]
    y_tier = pd.cut(y, bins=[-float("inf"), 220, 273, float("inf")],
                    labels=["Low", "Medium", "High"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    reg = make_pipeline(RandomForestRegressor(
        n_estimators=200, random_state=42, n_jobs=-1, min_samples_leaf=1
    ))
    reg.fit(X_train, y_train)
    pred = reg.predict(X_test)

    Xc_train, Xc_test, yc_train, yc_test = train_test_split(
        X, y_tier, test_size=0.20, random_state=42, stratify=y_tier
    )
    clf = make_pipeline(RandomForestClassifier(
        n_estimators=200, random_state=42, n_jobs=-1, class_weight="balanced"
    ))
    clf.fit(Xc_train, yc_train)
    class_pred = clf.predict(Xc_test)

    joblib.dump(reg, MODEL_DIR / "co2_regressor.joblib")
    joblib.dump(clf, MODEL_DIR / "co2_classifier.joblib")
    joblib.dump(FEATURES, MODEL_DIR / "feature_columns.joblib")

    print(f"Rows used after removing exact duplicates: {len(df):,}")
    print(f"Random Forest R2: {r2_score(y_test, pred):.4f}")
    print(f"Random Forest MAE: {mean_absolute_error(y_test, pred):.2f} g/km")
    print(f"Emission category accuracy: {accuracy_score(yc_test, class_pred):.2%}")
    print("Models saved in the models/ folder.")

if __name__ == "__main__":
    main()
