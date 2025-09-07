import joblib
from pathlib import Path
from utils import prepare_features_for_single_resume

ART = Path("d:/resume_ranker_app/artifacts")
artifacts = {
    "model": joblib.load(ART / "linear_regression_model.joblib"),
    "model_columns": joblib.load(ART / "model_columns.joblib"),
}

fv = prepare_features_for_single_resume("sample resume text", "sample job desc", "Engineering", artifacts)
print("fv.shape:", fv.shape)
print("fv.columns match model_columns:", list(fv.columns) == list(artifacts["model_columns"]))
print(fv.head().to_dict())