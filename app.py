import streamlit as st

from recommender_core import (
    get_dataset,
    get_title_list,
    recommend_by_content,
    recommend_by_text,
)
from tmdb_client import get_popular_movies


# Load data once
df = get_dataset()
titles = get_title_list()

# Pre-compute values for the year slider so we don't do this on every rerun.
year_values = sorted(
    {int(y) for y in df["year"].dropna() if str(y).isdigit()}
) or [2000, 2025]
year_min_default = year_values[0]
year_max_default = year_values[-1]

# Unique language codes in the dataset, used for the language filter.
language_options = sorted(df["original_language"].dropna().unique().tolist())

# Collect a flat list of all genre labels.
all_genres = set()
for gs in df["genres"]:
    if isinstance(gs, list):
        all_genres.update(gs)
genre_options = sorted(all_genres)


# Streamlit layout
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
    "Languages (original_language)",
    options=language_options,
)

selected_genres = st.sidebar.multiselect(
    "Required genres (movie must contain all)",
    options=genre_options,
)

# Simple watchlist that lives in the sidebar; used to build a user profile.
st.sidebar.header("Your watchlist")
watchlist_titles = st.sidebar.multiselect(
    "Add movies you like",
    options=titles,
)

# Main controls: pick a seed movie and recommendation method.
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
        # Most likely: the selected seed title wasn't found in the dataset.
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
            if isinstance(row["genres"], list):
                st.write("Genres:", ", ".join(row["genres"]))
            st.write(row["overview"])
            st.markdown(
                f"_Similarity score: {row['similarity']:.3f} "
                "(relative to the seed movie)._"
            )
            st.markdown("---")


# Watchlist-based recommendations (enhanced feature)
# The idea here is to build a "user profile" as the average of the
# content-based vectors of all movies in the user's watchlist.
from recommender_core import recommend_by_content  # reuse content method

if run_watchlist:
    if not watchlist_titles:
        st.warning("Add at least one movie to your watchlist in the sidebar.")
    else:
        filter_kwargs = build_filter_kwargs()
        from recommender_core import (
            recommend_by_content as _rec_content,  # noqa: F401 (kept for clarity)
        )
        from recommender_core import (
            get_dataset,
            _get_index_for_title,
            build_content_features,
        )
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity

        df_full = get_dataset()
        # Make sure the content matrix is initialised.
        build_content_features()
        from recommender_core import _content_matrix  # type: ignore

        # Collect indices of movies in the user's watchlist.
        indices = [
            _get_index_for_title(t)
            for t in watchlist_titles
            if _get_index_for_title(t) is not None
        ]
        if not indices:
            st.warning("Could not find your watchlist titles in the dataset.")
        else:
            # Average vector across watchlist titles => simple profile.
            profile_vec = _content_matrix[indices].mean(axis=0, keepdims=True)
            sims = cosine_similarity(profile_vec, _content_matrix)[0]

            df_profile = df_full.copy()
            df_profile["similarity"] = sims
            # Don't recommend movies the user already has in their watchlist.
            df_profile = df_profile[~df_profile["title"].isin(watchlist_titles)]
            df_profile = df_profile.sort_values("similarity", ascending=False)

            from recommender_core import apply_filters

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
                    if isinstance(row["genres"], list):
                        st.write("Genres:", ", ".join(row["genres"]))
                    st.write(row["overview"])
                    st.markdown(
                        f"_Similarity score: {row['similarity']:.3f} "
                        "(relative to your watchlist profile)._"
                    )
                    st.markdown("---")


# Popular / trending section (enhanced feature)
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
    # It's better for the app to keep working even if this call fails.
    st.warning(f"Could not load popular movies: {e}")
