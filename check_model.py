import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from utils import prepare_features_for_single_resume

ART = Path("d:/resume_ranker_app/artifacts")
model = joblib.load(ART / "linear_regression_model.joblib")
cols = joblib.load(ART / "model_columns.joblib")

print("model type:", type(model))
print("model.n_features_in_:", getattr(model, "n_features_in_", None))
print("len(model_columns):", len(cols))

# Predict with numpy zeros (no feature names)
X = np.zeros((1, len(cols)))
print("predicting (no names):", model.predict(X))

# Predict with DataFrame using exact column names
X_df = pd.DataFrame(np.zeros((1, len(cols))), columns=list(cols))
print("columns match:", list(X_df.columns) == list(cols))
print("predicting (with names):", model.predict(X_df))

# Show basic model info
try:
    import numpy as _np
    print("coef mean / intercept:", float(_np.mean(model.coef_)), float(getattr(model, "intercept_", 0.0)))
except Exception:
    pass

# Build artifacts dict and run feature-prep debug
artifacts = {
    "model": model,
    "model_columns": cols,
    # add any other vectorizers you have in artifacts folder if needed
}
fv = prepare_features_for_single_resume("sample resume text", "sample job desc", "Engineering", artifacts)
print("fv.shape:", fv.shape)
print("fv.columns match model_columns:", list(fv.columns) == list(cols))
print("fv sample values (first 10):", fv.iloc[0].astype(float).round(6).head(10).to_dict())