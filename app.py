import streamlit as st
import joblib
import os
import re         # ADDED: Essential for regex operations
import fitz       # CORRECTED: Changed from PyMuPDF to fitz
import pandas as pd # ADDED: Essential for DataFrame operations
from utils import prepare_features_for_single_resume
# --- Page Configuration ---
st.set_page_config(
    page_title="Intelligent Resume Ranker",
    layout="wide"
)

# --- Caching Loaded Artifacts ---
@st.cache_resource
def load_artifacts():
    """Loads all the necessary artifacts for the model from the 'artifacts' directory."""
    artifacts = {}
    artifacts_dir = 'artifacts'
    
    # List of required files
    required_files = [
        'linear_regression_model.joblib', 'tfidf_vectorizer_cosine.joblib',
        'skills_vectorizer.joblib', 'edu_vectorizer.joblib', 'exp_vectorizer.joblib',
        'model_columns.joblib', 'unique_categories.joblib'
    ]
    
    # Check if all files exist
    for filename in required_files:
        path = os.path.join(artifacts_dir, filename)
        if not os.path.exists(path):
            st.error(f"Missing artifact file: {filename}. Please ensure all required .joblib files are in the 'artifacts' folder.")
            st.stop()

    # Load all artifacts
    artifacts['model'] = joblib.load(os.path.join(artifacts_dir, 'linear_regression_model.joblib'))
    artifacts['tfidf_vectorizer_cosine'] = joblib.load(os.path.join(artifacts_dir, 'tfidf_vectorizer_cosine.joblib'))
    artifacts['skills_vectorizer'] = joblib.load(os.path.join(artifacts_dir, 'skills_vectorizer.joblib'))
    artifacts['edu_vectorizer'] = joblib.load(os.path.join(artifacts_dir, 'edu_vectorizer.joblib'))
    artifacts['exp_vectorizer'] = joblib.load(os.path.join(artifacts_dir, 'exp_vectorizer.joblib'))
    artifacts['model_columns'] = joblib.load(os.path.join(artifacts_dir, 'model_columns.joblib'))
    artifacts['unique_categories'] = joblib.load(os.path.join(artifacts_dir, 'unique_categories.joblib'))
    return artifacts

# --- Helper functions to read uploaded files ---
def read_pdf(file):
    try:
        doc = fitz.open(stream=file.read(), filetype="pdf")
        text = "".join(page.get_text() for page in doc)
        return text
    except Exception as e:
        st.error(f"Error reading PDF {file.name}: {e}")
        return None

def read_txt(file):
    try:
        return file.getvalue().decode("utf-8")
    except Exception as e:
        st.error(f"Error reading TXT {file.name}: {e}")
        return None

# --- Main Application ---
st.title("Intelligent Resume Ranker")
st.write("Upload a job description and resumes to rank them based on a predictive model.")

# Load all artifacts and handle potential errors
artifacts = load_artifacts()
resume_categories = artifacts['unique_categories']

# --- UI Layout ---
col1, col2 = st.columns([1, 1])

with col1:
    st.header("Step 1: Job Details")
    job_category = st.selectbox("Select Job Category", options=resume_categories)
    job_description = st.text_area("Paste Job Description Here", height=300, placeholder="E.g., We are looking for a Data Scientist...")

with col2:
    st.header("Step 2: Upload Resumes")
    uploaded_files = st.file_uploader(
        "Choose resume files (PDF or TXT)",
        type=['pdf', 'txt'],
        accept_multiple_files=True
    )

# --- Processing and Ranking Button ---
if st.button("Rank Resumes", type="primary", use_container_width=True):
    if not job_description.strip():
        st.warning("Please provide a job description.")
    elif not uploaded_files:
        st.warning("Please upload at least one resume.")
    else:
        with st.spinner('Analyzing resumes... This may take a moment.'):
            results = []
            for file in uploaded_files:
                resume_text = ""
                if file.type == "application/pdf":
                    resume_text = read_pdf(file)
                elif file.type == "text/plain":
                    resume_text = read_txt(file)
                
                if resume_text:
                    feature_vector = prepare_features_for_single_resume(
                        resume_text, job_description, job_category, artifacts
                    )
                    predicted_score = artifacts['model'].predict(feature_vector)[0]
                    
                    results.append({
                        'Filename': file.name,
                        'Predicted Score': f"{max(0, min(100, predicted_score)):.2f}%"
                    })
            
            if results:
                st.header("🏆 Ranked Results")
                # Sort dataframe by score (as a float, not a string)
                ranked_df = pd.DataFrame(results)
                ranked_df['Score_float'] = ranked_df['Predicted Score'].str.replace('%', '').astype(float)
                ranked_df = ranked_df.sort_values(by='Score_float', ascending=False).drop(columns=['Score_float'])
                ranked_df = ranked_df.reset_index(drop=True)
                ranked_df.index += 1 # Start index from 1
                st.dataframe(ranked_df, use_container_width=True)
            else:
                st.error("Could not process any of the uploaded files.")
def extract_skills(text):
    match = re.search(r'skills\s*([^.]*)', text, re.IGNORECASE)
    return match.group(1).strip() if match else ""