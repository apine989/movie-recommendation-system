import os
from typing import Dict, List

import pandas as pd

from tmdb_client import _get  # reuse the low-level TMDB helper


DATA_PATH = "movies.csv"


def _get_genre_map() -> Dict[int, str]:
    """Fetch a mapping from genre id -> genre name."""
    data = _get("/genre/movie/list", params={"language": "en-US"})
    genres = data.get("genres", [])
    return {g["id"]: g["name"] for g in genres}


def _fetch_movies(pages: int = 5, min_vote_count: int = 200) -> pd.DataFrame:
    """
    Fetch movies from TMDB discover endpoint.

    pages: how many pages of results to grab (20 movies per page).
    min_vote_count: filter out movies with very few votes.
    """
    all_results: List[dict] = []

    for page in range(1, pages + 1):
        data = _get(
            "/discover/movie",
            params={
                "sort_by": "popularity.desc",
                "include_adult": False,
                "include_video": False,
                "language": "en-US",
                "page": page,
                "vote_count.gte": min_vote_count,
            },
        )
        all_results.extend(data.get("results", []))

    df = pd.DataFrame(all_results)

    if "release_date" in df.columns:
        df["year"] = df["release_date"].str[:4]
    else:
        df["year"] = None

    # Map genre IDs -> names
    genre_map = _get_genre_map()

    def ids_to_names(ids):
        if not isinstance(ids, list):
            return []
        return [genre_map.get(i, "Unknown") for i in ids]

    if "genre_ids" in df.columns:
        df["genres"] = df["genre_ids"].apply(ids_to_names)
    else:
        df["genres"] = [[] for _ in range(len(df))]

    # Keep only columns we care about for now
    keep_cols = [
        "id",
        "title",
        "overview",
        "year",
        "vote_average",
        "vote_count",
        "original_language",
        "popularity",
        "genres",
        "genre_ids",
        "poster_path",
    ]
    df = df[[c for c in keep_cols if c in df.columns]]

    return df


def load_or_build_dataset() -> pd.DataFrame:
    """
    If movies.csv exists, load it. Otherwise, fetch from TMDB and save.
    """
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH, converters={"genres": eval})

    df = _fetch_movies(pages=5, min_vote_count=200)
    df.to_csv(DATA_PATH, index=False)
    return df
