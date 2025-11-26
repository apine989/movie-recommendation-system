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

    This keeps configuration out of the code and works both locally and in deployment.
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

    # TMDB always expects the API key as a query parameter
    params["api_key"] = API_KEY
    url = f"{BASE_URL}{path}"

    resp = requests.get(url, params=params)
    resp.raise_for_status()
    return resp.json()


def get_popular_movies(page: int = 1):
    """
    Convenience helper for the 'popular movies' endpoint.

    Used by the UI to show a small 'Trending now' section.
    """
    data = _get("/movie/popular", params={"page": page})
    return data["results"]
