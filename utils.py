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
    if not cols:
        # very small fallback if no column metadata available
        return pd.DataFrame([[0.0]], columns=["feature_0"])

    # Initialize zeros for all expected model columns
    row = {c: 0.0 for c in cols}

    # Small, safe heuristics to populate a couple of simple features
    # 1) skill_count from a naive "Skills: ..." section
    m = re.search(r"skills\s*[:\-]?\s*([^.\n]*)", text, re.IGNORECASE)
    skills_text = m.group(1).strip() if m else ""
    skill_count = len(re.findall(r"\w+", skills_text)) if skills_text else 0
    if "skill_count" in row:
        row["skill_count"] = float(skill_count)

    # 2) one-hot job category if a corresponding column exists (e.g. "category_Engineering")
    category_col = f"category_{job_category}"
    if category_col in row:
        row[category_col] = 1.0

    # Use spaCy only if available (lazy)
    NLP = get_spacy_model()
    if NLP is not None:
        try:
            doc = NLP(text)
            tokens = [t.lemma_.lower() for t in doc if not t.is_punct and not t.is_space]
            # example: populate a token_count column if present
            if "token_count" in row:
                row["token_count"] = float(len(tokens))
        except Exception:
            pass

    return pd.DataFrame([row], columns=cols)