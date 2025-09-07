import streamlit as st
import re
import functools
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import typing


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


def ensure_feature_alignment(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Ensure df has exactly the columns in cols (same order), filling missing with 0.0."""
    for c in cols:
        if c not in df.columns:
            df[c] = 0.0
    return df[cols]


def _get_vectorizer_feature_names(vec) -> list:
    """Return feature names for vectorizer supporting multiple sklearn versions."""
    try:
        return list(vec.get_feature_names_out())
    except Exception:
        try:
            return list(vec.get_feature_names())
        except Exception:
            return []


def merge_vectorizer_outputs_into_row(text: str, skills_text: str, artifacts: dict, cols: list) -> dict:
    """
    Use available vectorizers from artifacts to populate a feature dict keyed by cols.

    Behavior:
    - For each vectorizer found in artifacts (tfidf_vectorizer_cosine, skills_vectorizer, edu_vectorizer, exp_vectorizer),
      transform the appropriate input (text or skills_text) and try to map each vector element into matching model column names.
    - Mapping strategies tried (in this order) for each feature name 'f':
        1) exact match of f in cols
        2) prefixed with vectorizer name: f"{vec_key}__{f}"
        3) prefixed short name: f"{vec_key.split('_')[0]}__{f}"
    - If no per-feature column matches but the vector length equals len(cols), the entire vector is used in order.
    - Fallback: leave zeros for unmatched columns.
    """
    row = {c: 0.0 for c in cols}
    vec_keys = [
        ("tfidf_vectorizer_cosine", text),
        ("skills_vectorizer", skills_text),
        ("edu_vectorizer", text),
        ("exp_vectorizer", text),
    ]

    for vec_key, input_text in vec_keys:
        vec = artifacts.get(vec_key)
        if vec is None:
            continue
        try:
            X = vec.transform([input_text])
            # sparse -> dense
            try:
                arr = X.toarray()[0]
            except Exception:
                arr = X[0] if isinstance(X, (list, tuple)) else np.asarray(X).reshape(-1)
            feat_names = _get_vectorizer_feature_names(vec)
        except Exception:
            # if transform fails, skip this vectorizer
            continue

        # If feature names are available, try mapping by name
        if feat_names and len(feat_names) == len(arr):
            mapped_any = False
            for fname, val in zip(feat_names, arr):
                # Try exact match
                if fname in row:
                    row[fname] = float(val)
                    mapped_any = True
                    continue
                # Try full prefixed name with vectorizer key
                pref1 = f"{vec_key}__{fname}"
                if pref1 in row:
                    row[pref1] = float(val)
                    mapped_any = True
                    continue
                # Try short prefix (e.g., "tfidf__token")
                short = vec_key.split("_")[0]
                pref2 = f"{short}__{fname}"
                if pref2 in row:
                    row[pref2] = float(val)
                    mapped_any = True
                    continue
            # If nothing mapped, fall back below to size-based mapping
            if mapped_any:
                continue

        # If we reach here and vector length equals cols length, assign in order
        if len(arr) == len(cols):
            for i, c in enumerate(cols):
                row[c] = float(arr[i])
            continue

        # No mapping applied for this vectorizer; skip (leave zeros)
    return row


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

    # --- Build basic heuristic row (skills, category, simple counts) ---
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

    # --- Merge vectorizer outputs (TF-IDF, skills vectors, etc.) ---
    try:
        vectored_row = merge_vectorizer_outputs_into_row(text=text, skills_text=skills_text, artifacts=artifacts, cols=cols)
        # update base row with vectored values (vectored values overwrite zeros)
        row.update(vectored_row)
    except Exception:
        # if vector merging fails, keep the heuristic-only row
        pass

    fv = pd.DataFrame([row], columns=cols)
    # final alignment guarantee
    fv = ensure_feature_alignment(fv, cols)
    return fv