"""
train_model.py
--------------
Trains and compares two models (Linear Regression and Random Forest Regressor)
on the UCI Student Performance dataset, then saves the better model as model.pkl.

Run:  python train_model.py
"""

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------------------------------------------------
# 1. Load the dataset
#    sep=";" because this CSV uses semicolons instead of commas
# ---------------------------------------------------------------
df = pd.read_csv("data/student-mat.csv", sep=";")
print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# ---------------------------------------------------------------
# 2. Select features (X) and target (y)
#
#    WHY these features?
#    - G1, G2     : previous grades -> strongest predictors of G3
#    - studytime  : hours of weekly study
#    - failures   : number of past class failures
#    - absences   : school absences
#    - Medu, Fedu : parents' education (known to affect performance)
#    - age        : student age
#    - sex, address, internet, romantic : background/lifestyle factors
#
#    NOTE ON DATA LEAKAGE:
#    G1/G2 are only safe to use if we predict G3 DURING the school year.
#    If we predict at the START of the year (before any test), G1/G2
#    do not exist yet, and using them would be data leakage.
# ---------------------------------------------------------------
NUMERIC_FEATURES = ["age", "Medu", "Fedu", "studytime", "failures",
                    "absences", "G1", "G2"]
CATEGORICAL_FEATURES = ["sex", "address", "internet", "romantic"]

X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
y = df["G3"]  # final grade (0-20) -> regression target

# ---------------------------------------------------------------
# 3. Train / test split (80% / 20%)
#    Training data  -> model learns patterns from it
#    Testing data   -> kept hidden during training, used ONLY for
#                      final evaluation (simulates new students)
#    random_state=42 -> same split every run (reproducibility)
# ---------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Training samples: {len(X_train)} | Testing samples: {len(X_test)}")

# ---------------------------------------------------------------
# 4. Preprocessing
#    - Numeric    : StandardScaler (centers values around 0 -> helps
#                   Linear Regression converge; trees don't need it
#                   but it does no harm)
#    - Categorical: OneHotEncoder (text -> 0/1 dummy columns,
#                   e.g. "internet" -> internet_yes / internet_no)
#    ColumnTransformer applies each step to the right columns only.
# ---------------------------------------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ]
)

# ---------------------------------------------------------------
# 5. Build two models as pipelines
#    Pipeline = preprocessing + model in one object, so the SAME
#    transformations are applied at training and prediction time.
# ---------------------------------------------------------------
linear_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", LinearRegression()),
])

forest_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(n_estimators=200, random_state=42)),
])

# ---------------------------------------------------------------
# 6. Train + evaluate both models
# ---------------------------------------------------------------
def evaluate(model, X_test, y_test):
    """Return MAE, RMSE and R2 for a trained model."""
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    return mae, rmse, r2


results = {}
for name, model in [("Linear Regression", linear_model),
                    ("Random Forest", forest_model)]:
    model.fit(X_train, y_train)
    mae, rmse, r2 = evaluate(model, X_test, y_test)
    results[name] = {"model": model, "MAE": mae, "RMSE": rmse, "R2": r2}
    print(f"\n{name}:")
    print(f"  MAE  = {mae:.3f}  (avg. absolute error, in grade points)")
    print(f"  RMSE = {rmse:.3f}  (penalizes big errors more)")
    print(f"  R2   = {r2:.3f}  (1.0 = perfect, 0 = no better than average)")

# ---------------------------------------------------------------
# 7. Pick the better model (lowest RMSE on the TEST set) and save it
# ---------------------------------------------------------------
best_name = min(results, key=lambda k: results[k]["RMSE"])
best_model = results[best_name]["model"]
print(f"\n>>> Best model: {best_name} (RMSE = {results[best_name]['RMSE']:.3f})")

joblib.dump(best_model, "model.pkl")

# Also save the metrics + feature names (useful for the Streamlit app)
meta = {
    "model_name": best_name,
    "metrics": {k: {m: results[k][m] for m in ["MAE", "RMSE", "R2"]}
                for k in results},
    "numeric_features": NUMERIC_FEATURES,
    "categorical_features": CATEGORICAL_FEATURES,
}
joblib.dump(meta, "model_meta.pkl")
print("Saved: model.pkl, model_meta.pkl")
