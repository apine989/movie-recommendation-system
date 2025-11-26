from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer

from recommender_data import load_or_build_dataset

# Simple in-memory cache so we only build things once
_df_cache: Optional[pd.DataFrame] = None
_mlb: Optional[MultiLabelBinarizer] = None
_tfidf: Optional[TfidfVectorizer] = None
_content_matrix = None
_text_matrix = None


def get_dataset() -> pd.DataFrame:
    """Load the movie dataset and normalize some columns."""
    global _df_cache
    if _df_cache is None:
        df = load_or_build_dataset()
        df = df.copy()
        df["overview"] = df["overview"].fillna("")
        df["genres"] = df["genres"].apply(lambda x: x if isinstance(x, list) else [])
        _df_cache = df.reset_index(drop=True)
    return _df_cache


def build_content_features():
    """Build a feature matrix using genres + rating + popularity."""
    global _mlb, _content_matrix
    df = get_dataset()

    _mlb = MultiLabelBinarizer()
    genre_features = _mlb.fit_transform(df["genres"])

    # Add numeric features (vote_average and popularity)
    extra = df[["vote_average", "popularity"]].fillna(0).to_numpy()

    _content_matrix = np.hstack([genre_features, extra])
    return _content_matrix


def build_text_features():
    """Build a TF-IDF matrix over the movie overviews."""
    global _tfidf, _text_matrix
    df = get_dataset()

    _tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
    _text_matrix = _tfidf.fit_transform(df["overview"])
    return _text_matrix

def apply_filters(
    df: pd.DataFrame,
    min_year: Optional[int] = None,
    max_year: Optional[int] = None,
    min_rating: Optional[float] = None,
    min_votes: Optional[int] = None,
    languages: Optional[List[str]] = None,
    required_genres: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Apply basic filters to a movie DataFrame."""
    out = df.copy()

    # Make a numeric year column for safe comparisons
    out["year_num"] = pd.to_numeric(out["year"], errors="coerce")

    if min_year is not None:
        out = out[out["year_num"] >= min_year]
    if max_year is not None:
        out = out[out["year_num"] <= max_year]
    if min_rating is not None:
        out = out[out["vote_average"] >= min_rating]
    if min_votes is not None:
        out = out[out["vote_count"] >= min_votes]
    if languages:
        out = out[out["original_language"].isin(languages)]
    if required_genres:
        # Keep movies that contain ALL of the selected genres
        out = out[out["genres"].apply(lambda gs: all(g in gs for g in required_genres))]

    # We don't need year_num in the final output
    out = out.drop(columns=["year_num"])

    return out



def _get_index_for_title(title: str) -> Optional[int]:
    df = get_dataset()
    matches = df.index[df["title"] == title].tolist()
    return matches[0] if matches else None


def recommend_by_content(
    seed_title: str,
    top_n: int = 10,
    **filter_kwargs,
) -> pd.DataFrame:
    """Content-based recommendations using genres + numeric features."""
    df = get_dataset()
    global _content_matrix
    if _content_matrix is None:
        build_content_features()

    idx = _get_index_for_title(seed_title)
    if idx is None:
        raise ValueError(f"Seed title not found: {seed_title}")

    sims = cosine_similarity(_content_matrix[idx : idx + 1], _content_matrix)[0]

    df = df.copy()
    df["similarity"] = sims
    df = df[df["title"] != seed_title]  # don’t recommend the seed itself
    df = df.sort_values("similarity", ascending=False)

    df = apply_filters(df, **filter_kwargs)
    return df.head(top_n)


def recommend_by_text(
    seed_title: str,
    top_n: int = 10,
    **filter_kwargs,
) -> pd.DataFrame:
    """NLP-based recommendations using TF-IDF overplot descriptions."""
    df = get_dataset()
    global _text_matrix
    if _text_matrix is None:
        build_text_features()

    idx = _get_index_for_title(seed_title)
    if idx is None:
        raise ValueError(f"Seed title not found: {seed_title}")

    sims = cosine_similarity(_text_matrix[idx : idx + 1], _text_matrix)[0]

    df = df.copy()
    df["similarity"] = sims
    df = df[df["title"] != seed_title]
    df = df.sort_values("similarity", ascending=False)

    df = apply_filters(df, **filter_kwargs)
    return df.head(top_n)


def get_title_list() -> List[str]:
    """Return all movie titles sorted alphabetically."""
    df = get_dataset()
    return df["title"].sort_values().tolist()
