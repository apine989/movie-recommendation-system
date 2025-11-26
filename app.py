import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from recommender_core import (
    get_dataset,
    get_title_list,
    recommend_by_content,
    recommend_by_text,
    build_content_features,
    apply_filters,
    _get_index_for_title,
)
from tmdb_client import get_popular_movies


# Load data and basic options
df = get_dataset()
titles = get_title_list()

year_values = sorted(
    {int(y) for y in df["year"].dropna() if str(y).isdigit()}
) or [2000, 2025]
year_min_default = year_values[0]
year_max_default = year_values[-1]

# Runtime range (may have missing values).
if "runtime" in df.columns:
    runtime_values = sorted(
        {
            int(r)
            for r in df["runtime"].dropna()
            if str(r).isdigit() and int(r) > 0
        }
    )
else:
    runtime_values = []

if runtime_values:
    runtime_min_default = runtime_values[0]
    runtime_max_default = runtime_values[-1]
else:
    runtime_min_default = 60
    runtime_max_default = 240

language_options = sorted(df["original_language"].dropna().unique().tolist())

all_genres = set()
for gs in df["genres"]:
    if isinstance(gs, list):
        all_genres.update(gs)
genre_options = sorted(all_genres)

# Certifications
if "certification" in df.columns:
    cert_options = sorted(
        {c for c in df["certification"].dropna().unique().tolist() if c}
    )
else:
    cert_options = []

# Actor / director options
actor_options = []
if "cast_names" in df.columns:
    actor_set = set()
    for names in df["cast_names"]:
        if isinstance(names, list):
            actor_set.update(names)
    actor_options = sorted(actor_set)

director_options = []
if "director_names" in df.columns:
    director_set = set()
    for names in df["director_names"]:
        if isinstance(names, list):
            director_set.update(names)
    director_options = sorted(director_set)


st.set_page_config(page_title="Movie Recommendation System", layout="wide")
st.title("🎬 Movie Recommendation System")
st.write(
    "Pick a movie you like, tune the filters, and choose a recommendation method."
)

# Sidebar filters
st.sidebar.header("Filters")

year_range = st.sidebar.slider(
    "Release year range",
    min_value=year_min_default,
    max_value=year_max_default,
    value=(year_min_default, year_max_default),
    step=1,
)

runtime_range = st.sidebar.slider(
    "Runtime (minutes)",
    min_value=runtime_min_default,
    max_value=runtime_max_default,
    value=(runtime_min_default, runtime_max_default),
    step=5,
)

min_rating = st.sidebar.slider(
    "Minimum rating",
    min_value=0.0,
    max_value=10.0,
    value=6.0,
    step=0.5,
)

min_votes = st.sidebar.slider(
    "Minimum vote count",
    min_value=0,
    max_value=5000,
    value=200,
    step=50,
)

selected_languages = st.sidebar.multiselect(
    "Languages",
    options=language_options,
)

selected_genres = st.sidebar.multiselect(
    "Genres (match exact)",
    options=genre_options,
)

selected_certs = st.sidebar.multiselect(
    "Certification levels",
    options=cert_options,
)

selected_actors = st.sidebar.multiselect(
    "Preferred actors (any match)",
    options=actor_options,
)

selected_directors = st.sidebar.multiselect(
    "Preferred directors (any match)",
    options=director_options,
)

st.sidebar.header("Your watchlist")
watchlist_titles = st.sidebar.multiselect(
    "Add movies you like",
    options=titles,
)

# Main controls
col1, col2 = st.columns([2, 1])

with col1:
    seed_title = st.selectbox("Choose a movie you like (seed)", titles)

with col2:
    method = st.radio(
        "Recommendation method",
        [
            "Content-based (genres + rating + popularity)",
            "Text-based (plot description similarity)",
        ],
    )

top_n = st.slider("Number of recommendations", 5, 30, 10)

col_btn1, col_btn2 = st.columns(2)
with col_btn1:
    run_seed = st.button("Get recommendations from seed movie")
with col_btn2:
    run_watchlist = st.button("Get recommendations from my watchlist")


def build_filter_kwargs():
    """Bundle current sidebar values into a dict for the core functions."""
    return dict(
        min_year=year_range[0],
        max_year=year_range[1],
        min_rating=min_rating,
        min_votes=min_votes,
        languages=selected_languages or None,
        required_genres=selected_genres or None,
        min_runtime=runtime_range[0],
        max_runtime=runtime_range[1],
        certifications=selected_certs or None,
        actor_names=selected_actors or None,
        director_names=selected_directors or None,
    )


# Seed-based recommendations
if run_seed:
    filter_kwargs = build_filter_kwargs()

    try:
        if method.startswith("Content"):
            st.caption("Using genres + rating + popularity features.")
            recs = recommend_by_content(seed_title, top_n=top_n, **filter_kwargs)
            explanation = (
                "These movies share similar genres, rating, and popularity "
                f"patterns with **{seed_title}**."
            )
        else:
            st.caption("Using TF-IDF similarity over plot overviews.")
            recs = recommend_by_text(seed_title, top_n=top_n, **filter_kwargs)
            explanation = (
                "These movies have plot descriptions similar to "
                f"**{seed_title}** based on TF-IDF text similarity."
            )
    except ValueError as e:
        st.error(str(e))
        recs = None
        explanation = ""

    if recs is None or recs.empty:
        st.warning("No movies matched your filters. Try relaxing them.")
    else:
        st.subheader(f"Recommendations based on: {seed_title}")
        st.markdown(explanation)
        st.markdown("---")
        for _, row in recs.iterrows():
            st.markdown(f"### {row['title']} ({row['year']})")
            st.write(
                f"⭐ Rating: {row['vote_average']}  "
                f"(votes: {row['vote_count']})"
            )
            if "runtime" in row and not np.isnan(row["runtime"]):
                st.write(f"Runtime: {int(row['runtime'])} min")
            if isinstance(row.get("genres"), list):
                st.write("Genres:", ", ".join(row["genres"]))
            if row.get("certification"):
                st.write("Certification:", row["certification"])
            st.write(row["overview"])
            st.markdown(
                f"_Similarity score: {row['similarity']:.3f} "
                "(relative to the seed movie)._"
            )
            st.markdown("---")


# Watchlist-based recommendations
if run_watchlist:
    if not watchlist_titles:
        st.warning("Add at least one movie to your watchlist in the sidebar.")
    else:
        filter_kwargs = build_filter_kwargs()

        # Build content matrix and a simple profile vector.
        content_matrix = build_content_features()
        df_full = get_dataset()

        indices = [
            _get_index_for_title(t)
            for t in watchlist_titles
            if _get_index_for_title(t) is not None
        ]
        if not indices:
            st.warning("Could not find your watchlist titles in the dataset.")
        else:
            profile_vec = content_matrix[indices].mean(axis=0, keepdims=True)
            sims = cosine_similarity(profile_vec, content_matrix)[0]

            df_profile = df_full.copy()
            df_profile["similarity"] = sims
            df_profile = df_profile[~df_profile["title"].isin(watchlist_titles)]
            df_profile = df_profile.sort_values("similarity", ascending=False)

            df_profile = apply_filters(df_profile, **filter_kwargs)
            recs = df_profile.head(top_n)

            if recs.empty:
                st.warning(
                    "No matches after applying filters. Try relaxing them."
                )
            else:
                st.subheader("Recommendations from your watchlist profile")
                st.markdown(
                    "These movies are similar to the overall profile of your "
                    "watchlist based on genres, rating, and popularity."
                )
                st.markdown("---")
                for _, row in recs.iterrows():
                    st.markdown(f"### {row['title']} ({row['year']})")
                    st.write(
                        f"⭐ Rating: {row['vote_average']}  "
                        f"(votes: {row['vote_count']})"
                    )
                    if "runtime" in row and not np.isnan(row["runtime"]):
                        st.write(f"Runtime: {int(row['runtime'])} min")
                    if isinstance(row.get("genres"), list):
                        st.write("Genres:", ", ".join(row["genres"]))
                    if row.get("certification"):
                        st.write("Certification:", row["certification"])
                    st.write(row["overview"])
                    st.markdown(
                        f"_Similarity score: {row['similarity']:.3f} "
                        "(relative to your watchlist profile)._"
                    )
                    st.markdown("---")


# Popular / trending section
st.subheader("🔥 Popular right now (TMDB)")
try:
    popular = get_popular_movies(page=1)[:10]
    for movie in popular:
        title = movie.get("title")
        year = movie.get("release_date", "")[:4]
        rating = movie.get("vote_average")
        votes = movie.get("vote_count")
        st.markdown(f"**{title}** ({year}) — ⭐ {rating} ({votes} votes)")
except Exception as e:
    st.warning(f"Could not load popular movies: {e}")
