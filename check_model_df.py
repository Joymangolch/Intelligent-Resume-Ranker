import joblib, pandas as pd, numpy as np
from pathlib import Path

ART = Path("d:/resume_ranker_app/artifacts")
model = joblib.load(ART / "linear_regression_model.joblib")
cols = joblib.load(ART / "model_columns.joblib")

X = pd.DataFrame(np.zeros((1, len(cols))), columns=list(cols))
print("model.n_features_in_:", getattr(model, "n_features_in_", None))
print("X.shape:", X.shape)
print("columns match:", list(X.columns) == list(cols))
print("predict:", model.predict(X))
print("coef mean / intercept:", float(np.mean(model.coef_)), float(getattr(model, "intercept_", 0.0)))