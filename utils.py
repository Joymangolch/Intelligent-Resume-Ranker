import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import spacy

# Try to load spaCy model, with fallback options
def load_spacy_model():
    """Load spaCy model with fallback options"""
    try:
        # Try the standard model first
        return spacy.load('en_core_web_sm')
    except OSError:
        try:
            # Try alternative model names
            return spacy.load('en')
        except OSError:
            st.warning("spaCy English model not found. Using basic text processing.")
            # Return None if no model is available
            return None

nlp = load_spacy_model()

def basic_text_processing(text):
    """Basic text processing when spaCy is not available"""
    if not text:
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and extra spaces
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    
    # Basic stopword removal (simple list)
    stopwords = {'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 
                'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 
                'to', 'was', 'will', 'with', 'the', 'this', 'but', 'they', 'have', 
                'had', 'what', 'said', 'each', 'which', 'she', 'do', 'how', 'their',
                'if', 'up', 'out', 'many', 'then', 'them', 'these', 'so', 'some'}
    
    words = text.split()
    words = [word for word in words if word not in stopwords and len(word) > 2]
    
    return ' '.join(words)

def preprocess_text(text):
    """Preprocess text using spaCy if available, otherwise use basic processing"""
    if not text or not isinstance(text, str):
        return ""
    
    if nlp is not None:
        # Use spaCy processing
        doc = nlp(text.lower())
        tokens = [token.lemma_ for token in doc 
                 if not token.is_stop and not token.is_punct and len(token.text) > 2]
        return ' '.join(tokens)
    else:
        # Use basic processing
        return basic_text_processing(text)

def extract_skills(resume_text):
    """Extract skills from resume text"""
    if not resume_text:
        return ""
    
    # Look for skills section
    skills_pattern = r'(?:skills?|technical skills?|core competencies?|expertise)[:\s]+(.*?)(?:\n\n|\n[A-Z]|$)'
    match = re.search(skills_pattern, resume_text, re.IGNORECASE | re.DOTALL)
    
    if match:
        skills_text = match.group(1).strip()
        # Clean up the skills text
        skills_text = re.sub(r'[•\-\*]', ' ', skills_text)  # Remove bullet points
        skills_text = re.sub(r'\s+', ' ', skills_text)  # Normalize whitespace
        return skills_text[:500]  # Limit length
    
    return ""

def extract_education(resume_text):
    """Extract education information from resume text"""
    if not resume_text:
        return ""
    
    # Look for education section
    education_pattern = r'(?:education|academic|qualifications?|degrees?)[:\s]+(.*?)(?:\n\n|\n[A-Z]|$)'
    match = re.search(education_pattern, resume_text, re.IGNORECASE | re.DOTALL)
    
    if match:
        education_text = match.group(1).strip()
        # Clean up the education text
        education_text = re.sub(r'[•\-\*]', ' ', education_text)
        education_text = re.sub(r'\s+', ' ', education_text)
        return education_text[:300]
    
    return ""

def extract_experience(resume_text):
    """Extract experience information from resume text"""
    if not resume_text:
        return ""
    
    # Look for experience section
    experience_pattern = r'(?:experience|work experience|employment|professional experience)[:\s]+(.*?)(?:\n\n|\n[A-Z]|$)'
    match = re.search(experience_pattern, resume_text, re.IGNORECASE | re.DOTALL)
    
    if match:
        experience_text = match.group(1).strip()
        # Clean up the experience text
        experience_text = re.sub(r'[•\-\*]', ' ', experience_text)
        experience_text = re.sub(r'\s+', ' ', experience_text)
        return experience_text[:500]
    
    return ""

def prepare_features_for_single_resume(resume_text, job_description, job_category, artifacts):
    """
    Prepare features for a single resume to match the training format
    """
    try:
        # Extract sections
        resume_skills = extract_skills(resume_text)
        resume_education = extract_education(resume_text)
        resume_experience = extract_experience(resume_text)
        
        # Preprocess texts
        processed_resume = preprocess_text(resume_text)
        processed_job_desc = preprocess_text(job_description)
        processed_skills = preprocess_text(resume_skills)
        processed_education = preprocess_text(resume_education)
        processed_experience = preprocess_text(resume_experience)
        
        # Calculate cosine similarity
        try:
            if processed_resume and processed_job_desc:
                tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
                tfidf_matrix = tfidf.fit_transform([processed_resume, processed_job_desc])
                cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            else:
                cosine_sim = 0.0
        except:
            cosine_sim = 0.0
        
        # Transform text features using saved vectorizers
        try:
            skills_features = artifacts['skills_vectorizer'].transform([processed_skills]).toarray()
            edu_features = artifacts['edu_vectorizer'].transform([processed_education]).toarray()
            exp_features = artifacts['exp_vectorizer'].transform([processed_experience]).toarray()
        except Exception as e:
            st.warning(f"Warning in feature extraction: {e}")
            # Use zero arrays as fallback
            skills_features = np.zeros((1, 100))  # Adjust size as needed
            edu_features = np.zeros((1, 100))
            exp_features = np.zeros((1, 100))
        
        # Create category encoding
        unique_categories = artifacts['unique_categories']
        category_features = [1 if cat == job_category else 0 for cat in unique_categories]
        
        # Combine all features
        combined_features = np.concatenate([
            [cosine_sim],
            skills_features.flatten(),
            edu_features.flatten(), 
            exp_features.flatten(),
            category_features
        ])
        
        # Ensure we have the right number of features
        model_columns = artifacts['model_columns']
        if len(combined_features) != len(model_columns):
            # Pad or truncate to match expected size
            if len(combined_features) < len(model_columns):
                combined_features = np.pad(combined_features, (0, len(model_columns) - len(combined_features)))
            else:
                combined_features = combined_features[:len(model_columns)]
        
        return combined_features.reshape(1, -1)
        
    except Exception as e:
        st.error(f"Error in feature preparation: {e}")
        # Return a default feature vector
        model_columns = artifacts['model_columns']
        return np.zeros((1, len(model_columns)))