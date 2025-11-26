import os
import requests
from dotenv import load_dotenv

# Streamlit is optional here; we only use it to read secrets when available.
try:
    import streamlit as st
except ImportError:
    st = None

load_dotenv()


def _get_api_key() -> str:
    """
    Look for the TMDB API key in the environment first, then in Streamlit secrets.
    """
    key = os.getenv("TMDB_API_KEY")

    # When running on Streamlit Cloud, it's convenient to store the key in st.secrets
    if not key and st is not None:
        secrets = getattr(st, "secrets", None)
        if secrets is not None:
            key = secrets.get("TMDB_API_KEY")

    if not key:
        raise RuntimeError("TMDB_API_KEY is not set.")
    return key


API_KEY = _get_api_key()
BASE_URL = "https://api.themoviedb.org/3"


def _get(path: str, params=None):
    """
    Small helper around the TMDB GET endpoint.

    path: API path like "/discover/movie"
    params: extra query-string parameters (page, filters, etc.)
    """
    if params is None:
        params = {}

    params["api_key"] = API_KEY
    url = f"{BASE_URL}{path}"

    resp = requests.get(url, params=params)
    resp.raise_for_status()
    return resp.json()


def get_popular_movies(page: int = 1):
    """
    Convenience helper for the 'popular movies' endpoint.
    """
    data = _get("/movie/popular", params={"page": page})
    return data["results"]


def get_movie_details(movie_id: int):
    """
    Basic movie details including runtime and revenue.
    """
    return _get(f"/movie/{movie_id}", params={"language": "en-US"})


def get_movie_credits(movie_id: int):
    """
    Cast and crew for a movie.
    """
    return _get(f"/movie/{movie_id}/credits", params={"language": "en-US"})


def get_movie_certification(movie_id: int, region: str = "US"):
    """
    Return the first non-empty certification for the given region, e.g. 'PG-13'.
    """
    data = _get(f"/movie/{movie_id}/release_dates")
    for entry in data.get("results", []):
        if entry.get("iso_3166_1") == region:
            for rel in entry.get("release_dates", []):
                cert = rel.get("certification")
                if cert:
                    return cert
    return None
