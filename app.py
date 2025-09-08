# ======================================================================================
# app.py - Main Streamlit Application
#
# This script creates the user interface and orchestrates the resume ranking process.
# ======================================================================================

import streamlit as st
import joblib
import os
import fitz       # PyMuPDF is imported under the name 'fitz'
import pandas as pd
from utils import prepare_features_for_single_resume # All backend logic is in utils.py

# --- Page Configuration ---
# This should be the first Streamlit command in your script.
st.set_page_config(
    page_title="Intelligent Resume Ranker",
    page_icon="🤖",
    layout="wide"
)

# --- Caching Loaded Artifacts ---
@st.cache_resource
def load_artifacts():
    """
    Loads all the necessary machine learning artifacts from the 'artifacts' directory.
    This function is cached to ensure models are loaded only once, improving app performance.
    """
    artifacts = {}
    artifacts_dir = 'artifacts'
    
    # List of all required files for the application to function.
    required_files = [
        'linear_regression_model.joblib', 'tfidf_vectorizer_cosine.joblib',
        'skills_vectorizer.joblib', 'edu_vectorizer.joblib', 'exp_vectorizer.joblib',
        'model_columns.joblib', 'unique_categories.joblib'
    ]
    
    # This loop checks if all necessary files are present before loading them.
    # It provides a clear error message if a file is missing during deployment.
    for filename in required_files:
        path = os.path.join(artifacts_dir, filename)
        if not os.path.exists(path):
            st.error(f"Missing artifact file: {filename}. Please ensure all required .joblib files are in the 'artifacts' folder and have been pushed to your GitHub repository.")
            st.stop() # Halts the app if a critical file is missing.

    # Load all artifacts into a dictionary.
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
    """Reads and extracts text from an uploaded PDF file."""
    try:
        # Open the PDF file from the uploaded file's stream
        doc = fitz.open(stream=file.read(), filetype="pdf")
        text = "".join(page.get_text() for page in doc)
        return text
    except Exception as e:
        st.error(f"Error reading PDF file '{file.name}': {e}")
        return None

def read_txt(file):
    """Reads and decodes text from an uploaded TXT file."""
    try:
        return file.getvalue().decode("utf-8")
    except Exception as e:
        st.error(f"Error reading TXT file '{file.name}': {e}")
        return None

# --- Main Application ---
st.title("🤖 Intelligent Resume Ranker")
st.write("Upload a job description and one or more resumes to automatically rank candidates based on their relevance.")

# Load all artifacts. The app will stop here if files are missing.
artifacts = load_artifacts()
# Load job categories from the artifact, not from a CSV file.
resume_categories = artifacts['unique_categories']

# --- UI Layout ---
# Use a two-column layout for a clean user interface.
col1, col2 = st.columns([1, 1])

with col1:
    st.header("Step 1: Job Details")
    job_category = st.selectbox("Select Job Category", options=resume_categories)
    job_description = st.text_area("Paste Job Description Here", height=300, placeholder="E.g., We are looking for a Data Scientist with experience in Python and Machine Learning...")

with col2:
    st.header("Step 2: Upload Resumes")
    uploaded_files = st.file_uploader(
        "Choose resume files (PDF or TXT)",
        type=['pdf', 'txt'],
        accept_multiple_files=True
    )

# --- Processing and Ranking Logic ---
# This block executes only when the user clicks the "Rank Resumes" button.
if st.button("Rank Resumes", type="primary", use_container_width=True):
    # Input validation
    if not job_description.strip():
        st.warning("Please provide a job description.")
    elif not uploaded_files:
        st.warning("Please upload at least one resume.")
    else:
        # Show a spinner while processing.
        with st.spinner('Analyzing resumes... This may take a moment.'):
            results = []
            # Process each uploaded file.
            for file in uploaded_files:
                resume_text = ""
                if file.type == "application/pdf":
                    resume_text = read_pdf(file)
                elif file.type == "text/plain":
                    resume_text = read_txt(file)
                
                if resume_text:
                    # Call the backend function from utils.py to get the feature vector.
                    feature_vector = prepare_features_for_single_resume(
                        resume_text, job_description, job_category, artifacts
                    )
                    # Use the loaded model to predict the score.
                    predicted_score = artifacts['model'].predict(feature_vector)[0]
                    
                    # Store the results.
                    results.append({
                        'Filename': file.name,
                        # Clamp the score between 0 and 100 for display purposes.
                        'Predicted Score': f"{max(0, min(100, predicted_score)):.2f}%"
                    })
            
            # Display the results.
            if results:
                st.header("🏆 Ranked Results")
                # Create and format the results DataFrame.
                ranked_df = pd.DataFrame(results)
                ranked_df['Score_float'] = ranked_df['Predicted Score'].str.replace('%', '').astype(float)
                ranked_df = ranked_df.sort_values(by='Score_float', ascending=False).drop(columns=['Score_float'])
                ranked_df = ranked_df.reset_index(drop=True)
                ranked_df.index += 1 # Start rank from 1 instead of 0.

                # Show the top candidate separately
                top_candidate = ranked_df.iloc[0]
                st.subheader("🎯 Top Candidate")
                st.markdown(f"**{top_candidate['Filename']}** — **Score:** {top_candidate['Predicted Score']}")

                # Show the full ranking table
                st.dataframe(ranked_df, width='stretch')
            else:
                st.error("Could not process any of the uploaded files. Please check the file formats and try again.")