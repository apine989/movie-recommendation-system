import os
import requests
from dotenv import load_dotenv

try:
    import streamlit as st
except ImportError:
    st = None

load_dotenv()


def _get_api_key() -> str:
    key = os.getenv("TMDB_API_KEY")
    if not key and st is not None:
        secrets = getattr(st, "secrets", None)
        if secrets is not None:
            key = secrets.get("TMDB_API_KEY")
    if not key:
        raise RuntimeError("TMDB_API_KEY is not set.")
    return key


API_KEY = _get_api_key()
BASE_URL = "https://api.themoviedb.org/3"


def _get(path, params=None):
    if params is None:
        params = {}
    params["api_key"] = API_KEY
    url = f"{BASE_URL}{path}"
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    return resp.json()


def get_popular_movies(page=1):
    data = _get("/movie/popular", params={"page": page})
    return data["results"]
