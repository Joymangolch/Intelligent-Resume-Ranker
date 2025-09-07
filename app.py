import os
import re
import io
from pathlib import Path

import streamlit as st
import joblib
import pandas as pd
import requests

# Optional PDF backends
try:
    import fitz  # PyMuPDF
    PDF_BACKEND = "fitz"
except Exception:
    fitz = None
    try:
        import PyPDF2
        PDF_BACKEND = "pypdf2"
    except Exception:
        PyPDF2 = None
        PDF_BACKEND = None

from utils import prepare_features_for_single_resume

st.set_page_config(page_title="Intelligent Resume Ranker", layout="wide")

ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True)


@st.cache_resource
def download_artifact(url: str, dest: Path) -> Path:
    if not dest.exists():
        r = requests.get(url, stream=True)
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return dest


@st.cache_resource
def load_artifacts():
    """Load required joblib artifacts from artifacts directory."""
    artifacts = {}
    artifacts_dir = ARTIFACTS_DIR

    required_files = [
        "linear_regression_model.joblib",
        "tfidf_vectorizer_cosine.joblib",
        "skills_vectorizer.joblib",
        "edu_vectorizer.joblib",
        "exp_vectorizer.joblib",
        "model_columns.joblib",
        "unique_categories.joblib",
    ]

    missing = [f for f in required_files if not (artifacts_dir / f).exists()]
    if missing:
        st.error(f"Missing artifact files: {', '.join(missing)}. Place them in the 'artifacts' folder.")
        st.stop()

    try:
        artifacts["model"] = joblib.load(artifacts_dir / "linear_regression_model.joblib")
        artifacts["tfidf_vectorizer_cosine"] = joblib.load(artifacts_dir / "tfidf_vectorizer_cosine.joblib")
        artifacts["skills_vectorizer"] = joblib.load(artifacts_dir / "skills_vectorizer.joblib")
        artifacts["edu_vectorizer"] = joblib.load(artifacts_dir / "edu_vectorizer.joblib")
        artifacts["exp_vectorizer"] = joblib.load(artifacts_dir / "exp_vectorizer.joblib")
        artifacts["model_columns"] = joblib.load(artifacts_dir / "model_columns.joblib")
        artifacts["unique_categories"] = joblib.load(artifacts_dir / "unique_categories.joblib")
        return artifacts
    except Exception as e:
        st.error(f"Error loading artifacts: {e}")
        st.stop()


def read_pdf(uploaded_file) -> str | None:
    try:
        data = uploaded_file.read()
        if PDF_BACKEND == "fitz":
            doc = fitz.open(stream=data, filetype="pdf")
            text = "".join(page.get_text() for page in doc)
            doc.close()
            return text
        elif PDF_BACKEND == "pypdf2":
            reader = PyPDF2.PdfReader(io.BytesIO(data))
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            return text
        else:
            st.error("No PDF library installed. Install pymupdf or PyPDF2.")
            return None
    except Exception as e:
        st.error(f"Error reading PDF {getattr(uploaded_file, 'name', '')}: {e}")
        return None


def read_txt(uploaded_file) -> str | None:
    try:
        data = uploaded_file.read()
        if isinstance(data, bytes):
            return data.decode("utf-8", errors="ignore")
        return str(data)
    except Exception as e:
        st.error(f"Error reading TXT {getattr(uploaded_file, 'name', '')}: {e}")
        return None


def extract_skills(text: str) -> str:
    match = re.search(r"skills\s*[:\-]?\s*([^.\n]*)", text, re.IGNORECASE)
    return match.group(1).strip() if match else ""


# --- Main app UI ---
st.title("Intelligent Resume Ranker")
st.write("Upload a job description and resumes to rank them based on a predictive model.")

# Optionally download artifacts from secrets (if provided)
if "model" in st.secrets and "url" in st.secrets["model"]:
    try:
        download_artifact(st.secrets["model"]["url"], ARTIFACTS_DIR / "linear_regression_model.joblib")
    except Exception:
        # don't block if download fails; load_artifacts will report missing files
        pass

artifacts = load_artifacts()
resume_categories = artifacts["unique_categories"]

col1, col2 = st.columns([1, 1])

with col1:
    st.header("Step 1: Job Details")
    job_category = st.selectbox("Select Job Category", options=resume_categories)
    job_description = st.text_area(
        "Paste Job Description Here",
        height=300,
        placeholder="E.g., We are looking for a Data Scientist...",
    )

with col2:
    st.header("Step 2: Upload Resumes")
    uploaded_files = st.file_uploader(
        "Choose resume files (PDF or TXT)",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        help="Upload one or more PDF or TXT resume files",
    )

if st.button("Rank Resumes"):
    if not uploaded_files:
        st.warning("Please upload at least one resume.")
    elif not job_description or job_description.strip() == "":
        st.warning("Please provide a job description.")
    else:
        results = []
        for uploaded in uploaded_files:
            try:
                if uploaded.type == "application/pdf" or uploaded.name.lower().endswith(".pdf"):
                    text = read_pdf(uploaded)
                else:
                    text = read_txt(uploaded)

                if not text or not text.strip():
                    st.warning(f"Could not extract text from {uploaded.name}")
                    continue

                # prepare features and predict
                feature_vector = prepare_features_for_single_resume(text, job_description, job_category, artifacts)

                # DEBUG: show feature vector for this resume (first 20 features)
                try:
                    st.write(f"FEATURES for {uploaded.name}")
                    st.write(feature_vector.iloc[0].astype(float).round(6).head(20).to_dict())
                except Exception:
                    st.write(feature_vector.head())

                # model may expect 2D array-like
                try:
                    predicted_score = artifacts["model"].predict(feature_vector)[0]
                except Exception as e:
                    st.error(f"Model prediction failed for {uploaded.name}: {e}")
                    continue

                try:
                    score = float(predicted_score)
                except Exception:
                    score = 0.0

                # clamp and format
                score = max(0.0, min(100.0, score))
                results.append({"Filename": uploaded.name, "Predicted Score": f"{score:.2f}%"})
            except Exception as e:
                st.error(f"Error processing {uploaded.name}: {e}")

        if results:
            ranked_df = pd.DataFrame(results)
            ranked_df["Score_float"] = ranked_df["Predicted Score"].str.replace("%", "").astype(float)
            ranked_df = ranked_df.sort_values(by="Score_float", ascending=False).reset_index(drop=True)

            st.header("🏆 Top Candidate")
            top = ranked_df.iloc[[0]][["Filename", "Predicted Score"]]
            st.table(top)  # small table showing only the top result

            # optional: still show full ranked table below
            st.header("All Ranked Results")
            st.dataframe(ranked_df.drop(columns=["Score_float"]), use_container_width=True)

            scores = [float(s.replace("%", "")) for s in ranked_df["Predicted Score"].tolist()]
            st.subheader("Summary Statistics")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Average Score", f"{(sum(scores) / len(scores)):.2f}%")
            with c2:
                st.metric("Highest Score", f"{max(scores):.2f}%")
            with c3:
                st.metric("Lowest Score", f"{min(scores):.2f}%")
        else:
            st.error("Could not process any of the uploaded files. Please check file formats and content.")