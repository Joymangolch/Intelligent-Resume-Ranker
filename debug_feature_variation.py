import joblib, pandas as pd
from pathlib import Path
from utils import prepare_features_for_single_resume
ART = Path("d:/resume_ranker_app/artifacts")
artifacts = {
    "model": joblib.load(ART/"linear_regression_model.joblib"),
    "model_columns": joblib.load(ART/"model_columns.joblib"),
    # add vectorizers if present: "tfidf_vectorizer_cosine": joblib.load(...), ...
}
# load resume files from a test folder or artifacts
import glob
files = glob.glob("d:/resume_ranker_app/test_resumes/*.pdf")  # adjust path
rows = []
names = []
for f in files:
    with open(f, "rb") as fh:
        text = fh.read().decode("utf-8", errors="ignore")  # or use read_pdf() if implemented
    fv = prepare_features_for_single_resume(text, "sample job desc", "Engineering", artifacts)
    rows.append(fv.iloc[0].astype(float))
    names.append(Path(f).name)

df = pd.DataFrame(rows, index=names)
print("shape:", df.shape)
print("nunique per column (first 20):")
print(df.nunique().sort_values(ascending=False).head(20))
print("Any identical rows? ->", df.duplicated().any())
print(df.head().T.iloc[:20])  # show first 20 features