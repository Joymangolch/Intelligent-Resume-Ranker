import streamlit as st
import re
import functools
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Lazy-load spaCy to avoid importing compiled C extensions at module import time
@functools.lru_cache(maxsize=1)
def get_spacy_model():
    try:
        import spacy
        try:
            return spacy.load("en_core_web_sm")
        except Exception:
            return None
    except Exception:
        return None


def prepare_features_for_single_resume(text: str, job_description: str, job_category: str, artifacts: dict):
    """
    Safe feature-prep fallback. Uses artifacts['model_columns'] when available.
    Will use spaCy only if it can be imported successfully at runtime.
    """
    cols = artifacts.get("model_columns") if isinstance(artifacts, dict) else None

    # Normalize cols to a plain list to avoid ambiguous truth-value checks on Index/ndarray
    if isinstance(cols, (pd.Index, np.ndarray)):
        cols = list(cols)

    # Explicitly check for None or empty list
    if cols is None or len(cols) == 0:
        return pd.DataFrame([[0.0]], columns=["feature_0"])

    row = {c: 0.0 for c in cols}

    # Naive skills extraction
    m = re.search(r"skills\s*[:\-]?\s*([^.\n]*)", text, re.IGNORECASE)
    skills_text = m.group(1).strip() if m else ""
    skill_count = len(re.findall(r"\w+", skills_text)) if skills_text else 0
    if "skill_count" in row:
        row["skill_count"] = float(skill_count)

    # One-hot job category
    category_col = f"category_{job_category}"
    if category_col in row:
        row[category_col] = 1.0

    # Use spaCy only if available (lazy)
    NLP = get_spacy_model()
    if NLP is not None:
        try:
            doc = NLP(text)
            tokens = [t.lemma_.lower() for t in doc if not t.is_punct and not t.is_space]
            if "token_count" in row:
                row["token_count"] = float(len(tokens))
        except Exception:
            pass

    # example inside prepare_features_for_single_resume, after extracting text:
    tfidf_vec = artifacts.get("tfidf_vectorizer_cosine")
    skills_vec = artifacts.get("skills_vectorizer")
    if tfidf_vec is not None:
        tfidf_vals = tfidf_vec.transform([text]).toarray()[0]
        # map tfidf_vals into corresponding model columns (depends on how you saved columns)
    # similarly use skills_vec.transform(...)
    # then ensure the final DataFrame aligns with artifacts["model_columns"]

    fv = pd.DataFrame([row], columns=cols)

    def ensure_feature_alignment(df, cols):
        for c in cols:
            if c not in df.columns:
                df[c] = 0.0
        return df[list(cols)]

    fv = ensure_feature_alignment(fv, cols)
    return fv