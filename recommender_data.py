import os
from typing import Dict, List

import pandas as pd

from tmdb_client import (
    _get,
    get_movie_details,
    get_movie_credits,
    get_movie_certification,
)


DATA_PATH = "movies.csv"


def _get_genre_map() -> Dict[int, str]:
    """Fetch a mapping from genre id -> genre name from TMDB."""
    data = _get("/genre/movie/list", params={"language": "en-US"})
    genres = data.get("genres", [])
    return {g["id"]: g["name"] for g in genres}


def _fetch_movies(pages: int = 5, min_vote_count: int = 200) -> pd.DataFrame:
    """
    Fetch movies from TMDB discover endpoint and enrich them with extra metadata.
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

    # Simple year column from release_date.
    if "release_date" in df.columns:
        df["year"] = df["release_date"].str[:4]
    else:
        df["year"] = None

    # Map genre IDs -> names.
    genre_map = _get_genre_map()

    def ids_to_names(ids):
        if not isinstance(ids, list):
            return []
        return [genre_map.get(i, "Unknown") for i in ids]

    if "genre_ids" in df.columns:
        df["genres"] = df["genre_ids"].apply(ids_to_names)
    else:
        df["genres"] = [[] for _ in range(len(df))]

    # Extra columns we’ll fill from detail/credits endpoints.
    df["runtime"] = pd.NA
    df["certification"] = pd.NA
    df["revenue"] = pd.NA
    df["cast_names"] = [[] for _ in range(len(df))]
    df["director_names"] = [[] for _ in range(len(df))]

    # Enrich each movie with runtime, certification, and personnel info.
    for idx, row in df.iterrows():
        movie_id = row.get("id")
        if pd.isna(movie_id):
            continue
        try:
            details = get_movie_details(movie_id)
            credits = get_movie_credits(movie_id)
            cert = get_movie_certification(movie_id)
        except Exception:
            # If anything fails, just skip enrichment for this movie.
            continue

        df.at[idx, "runtime"] = details.get("runtime")
        df.at[idx, "revenue"] = details.get("revenue")
        df.at[idx, "certification"] = cert

        cast_list = [
            c.get("name")
            for c in (credits.get("cast") or [])[:5]
            if c.get("name")
        ]
        director_list = [
            c.get("name")
            for c in (credits.get("crew") or [])
            if c.get("job") == "Director" and c.get("name")
        ]
        df.at[idx, "cast_names"] = cast_list
        df.at[idx, "director_names"] = director_list

    keep_cols = [
        "id",
        "title",
        "overview",
        "year",
        "runtime",
        "certification",
        "vote_average",
        "vote_count",
        "original_language",
        "popularity",
        "genres",
        "genre_ids",
        "cast_names",
        "director_names",
        "revenue",
        "poster_path",
    ]
    df = df[[c for c in keep_cols if c in df.columns]]

    return df


def load_or_build_dataset() -> pd.DataFrame:
    """
    If movies.csv exists, load it. Otherwise, fetch from TMDB and save.
    """
    if os.path.exists(DATA_PATH):
        return pd.read_csv(
            DATA_PATH,
            converters={
                "genres": eval,
                "cast_names": eval,
                "director_names": eval,
            },
        )

    df = _fetch_movies(pages=5, min_vote_count=200)
    df.to_csv(DATA_PATH, index=False)
    return df
