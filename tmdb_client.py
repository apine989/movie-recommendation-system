import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("TMDB_API_KEY")
BASE_URL = "https://api.themoviedb.org/3"


def _get(path, params=None):
    if params is None:
        params = {}
    if API_KEY is None:
        raise RuntimeError("API key is not set. Check .env file.")
    params["api_key"] = API_KEY
    url = f"{BASE_URL}{path}"
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    return resp.json()


def get_popular_movies(page=1):
    """Return a list of popular movies (basic test call)."""
    data = _get("/movie/popular", params={"page": page})
    return data["results"]