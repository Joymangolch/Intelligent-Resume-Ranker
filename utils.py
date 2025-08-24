import spacy 
import re
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import joblib

# Load spaCy model once for efficiency
nlp = spacy.load('en_core_web_sm')

# --- Define the same text processing functions from your training ---

def preprocess_text(text):
    doc = nlp(text)
    tokens = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct and not token.is_space]
    return ' '.join(tokens)

def extract_skills(text):
    match = re.search(r'skills\s*([^.]*)', text, re.IGNORECASE)
    return match.group(1).strip() if match else ""

def extract_education(text):
    match = re.search(r'education\s*([^.]*)', text, re.IGNORECASE)
    return match.group(1).strip() if match else ""

def extract_experience(text):
    match = re.search(r'experience\s*([^.]*)', text, re.IGNORECASE)
    return match.group(1).strip() if match else ""


# --- The main prediction pipeline function ---

def prepare_features_for_single_resume(resume_text, job_desc_text, job_category, artifacts):
    """
    Takes raw resume text and other inputs, and returns a feature vector
    ready for the model to predict on, using pre-loaded artifacts.
    """
    # 1. Preprocess all text
    processed_resume = preprocess_text(resume_text)
    processed_job_desc = preprocess_text(job_desc_text)
    
    # 2. Calculate Cosine Similarity using the loaded vectorizer
    tfidf_vectorizer_cosine = artifacts['tfidf_vectorizer_cosine']
    # IMPORTANT: Use .transform() here, NOT .fit_transform()
    tfidf_matrix = tfidf_vectorizer_cosine.transform([processed_job_desc, processed_resume])
    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])[0][0] * 100
    
    # 3. Extract features from the resume text
    extracted_skills = extract_skills(processed_resume)
    extracted_education = extract_education(processed_resume)
    extracted_experience = extract_experience(processed_resume)

    # 4. Vectorize extracted features using loaded vectorizers
    skills_vec = artifacts['skills_vectorizer'].transform([extracted_skills]).toarray()
    edu_vec = artifacts['edu_vectorizer'].transform([extracted_education]).toarray()
    exp_vec = artifacts['exp_vectorizer'].transform([extracted_experience]).toarray()

    # 5. Handle One-Hot Encoded Category
    model_columns = artifacts['model_columns']
    # Create a DataFrame with a single row of zeros, with the same columns as the training data
    feature_df = pd.DataFrame(0, index=[0], columns=model_columns)
    
    # 6. Populate the DataFrame with calculated values
    feature_df['similarity_score'] = cosine_sim
    
    # Set the appropriate category column to 1
    selected_category_col = f'category_{job_category}'
    if selected_category_col in feature_df.columns:
        feature_df[selected_category_col] = 1

    # Populate the vectorized text features
    skill_cols = [col for col in model_columns if col.startswith('skill_')]
    edu_cols = [col for col in model_columns if col.startswith('edu_')]
    exp_cols = [col for col in model_columns if col.startswith('exp_')]

    feature_df[skill_cols] = skills_vec
    feature_df[edu_cols] = edu_vec
    feature_df[exp_cols] = exp_vec
    
    # 7. Drop the target variable and return the final feature vector
    # The order is guaranteed to be correct because we used model_columns to create the DataFrame
    return feature_df.drop('similarity_score', axis=1)

