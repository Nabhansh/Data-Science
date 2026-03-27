"""
Movie Recommender System
=========================
Content-based and collaborative filtering recommender using:
- TF-IDF on movie genres/tags (content-based)
- User-item cosine similarity (collaborative)
- Hybrid scoring
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

np.random.seed(42)

# ── Sample movie catalogue ────────────────────────────────────────────────────
MOVIES = [
    {"id": 1,  "title": "The Dark Knight",          "genres": "action crime thriller superhero batman"},
    {"id": 2,  "title": "Inception",                "genres": "action sci-fi thriller dream mind"},
    {"id": 3,  "title": "The Shawshank Redemption", "genres": "drama prison hope friendship"},
    {"id": 4,  "title": "Interstellar",             "genres": "sci-fi space adventure drama time"},
    {"id": 5,  "title": "Pulp Fiction",             "genres": "crime drama thriller nonlinear"},
    {"id": 6,  "title": "The Matrix",               "genres": "action sci-fi cyberpunk reality simulation"},
    {"id": 7,  "title": "Forrest Gump",             "genres": "drama romance comedy history"},
    {"id": 8,  "title": "The Godfather",            "genres": "crime drama mafia family"},
    {"id": 9,  "title": "Goodfellas",               "genres": "crime drama biography mafia"},
    {"id": 10, "title": "The Silence of the Lambs", "genres": "thriller crime horror psychology"},
    {"id": 11, "title": "Schindler's List",         "genres": "drama biography war history"},
    {"id": 12, "title": "Avengers: Endgame",        "genres": "action superhero sci-fi marvel adventure"},
    {"id": 13, "title": "Titanic",                  "genres": "romance drama history disaster"},
    {"id": 14, "title": "The Lion King",            "genres": "animation family drama adventure"},
    {"id": 15, "title": "Se7en",                    "genres": "crime thriller mystery psychology dark"},
]

movies_df = pd.DataFrame(MOVIES)

# ── Synthetic user ratings (1-5) ──────────────────────────────────────────────
def generate_ratings(n_users=50):
    rows = []
    for user_id in range(1, n_users + 1):
        # Each user rates 5-12 random movies
        n_rated  = np.random.randint(5, 13)
        rated_ids = np.random.choice(movies_df["id"], size=n_rated, replace=False)
        for mid in rated_ids:
            rows.append({"user_id": user_id, "movie_id": mid,
                         "rating": np.random.choice([1,2,3,4,5], p=[0.05,0.10,0.20,0.35,0.30])})
    return pd.DataFrame(rows)


ratings_df = generate_ratings()

print("=" * 60)
print("  MOVIE RECOMMENDER SYSTEM")
print("=" * 60)
print(f"\nMovies   : {len(movies_df)}")
print(f"Users    : {ratings_df['user_id'].nunique()}")
print(f"Ratings  : {len(ratings_df)}")

# ── 1. Content-based filtering ────────────────────────────────────────────────
print("\n[1] Content-Based Filtering")
tfidf    = TfidfVectorizer(stop_words="english")
tfidf_mx = tfidf.fit_transform(movies_df["genres"])
content_sim = cosine_similarity(tfidf_mx, tfidf_mx)

def content_recommend(title, n=5):
    idx = movies_df[movies_df["title"] == title].index[0]
    scores = list(enumerate(content_sim[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:n+1]
    return movies_df.iloc[[i for i, _ in scores]][["title", "genres"]]

print("\nMovies similar to 'Inception':")
print(content_recommend("Inception").to_string(index=False))

print("\nMovies similar to 'The Godfather':")
print(content_recommend("The Godfather").to_string(index=False))

# ── 2. Collaborative filtering ────────────────────────────────────────────────
print("\n[2] Collaborative Filtering (User-Item)")
user_item = ratings_df.pivot_table(index="user_id", columns="movie_id", values="rating").fillna(0)
user_sim  = cosine_similarity(user_item)
user_sim_df = pd.DataFrame(user_sim, index=user_item.index, columns=user_item.index)

def collab_recommend(user_id, n=5):
    sim_users  = user_sim_df[user_id].sort_values(ascending=False)[1:11].index
    rated_by_user = set(ratings_df[ratings_df["user_id"] == user_id]["movie_id"])
    scores = {}
    for sim_user in sim_users:
        weight = user_sim_df.loc[user_id, sim_user]
        sim_ratings = ratings_df[ratings_df["user_id"] == sim_user]
        for _, row in sim_ratings.iterrows():
            if row["movie_id"] not in rated_by_user:
                scores[row["movie_id"]] = scores.get(row["movie_id"], 0) + weight * row["rating"]
    top_ids = sorted(scores, key=scores.get, reverse=True)[:n]
    return movies_df[movies_df["id"].isin(top_ids)]["title"].tolist()

print(f"\nRecommendations for User 1:")
recs = collab_recommend(1)
for i, r in enumerate(recs, 1):
    print(f"  {i}. {r}")

# ── 3. Hybrid recommender ─────────────────────────────────────────────────────
print("\n[3] Hybrid Recommender (Content + Collaborative)")

def hybrid_recommend(user_id, liked_movie, n=5, alpha=0.5):
    content_scores = {}
    idx = movies_df[movies_df["title"] == liked_movie].index[0]
    for j, score in enumerate(content_sim[idx]):
        mid = movies_df.iloc[j]["id"]
        if movies_df.iloc[j]["title"] != liked_movie:
            content_scores[mid] = score

    collab_scores = {}
    sim_users = user_sim_df[user_id].sort_values(ascending=False)[1:11].index
    for su in sim_users:
        w = user_sim_df.loc[user_id, su]
        for _, row in ratings_df[ratings_df["user_id"] == su].iterrows():
            collab_scores[row["movie_id"]] = collab_scores.get(row["movie_id"], 0) + w * row["rating"]

    # Normalise collab scores
    max_c = max(collab_scores.values()) if collab_scores else 1
    collab_scores = {k: v / max_c for k, v in collab_scores.items()}

    all_ids = set(content_scores) | set(collab_scores)
    hybrid = {mid: alpha * content_scores.get(mid, 0) + (1 - alpha) * collab_scores.get(mid, 0)
              for mid in all_ids}
    top_ids = sorted(hybrid, key=hybrid.get, reverse=True)[:n]
    return movies_df[movies_df["id"].isin(top_ids)]["title"].tolist()

print(f"\nHybrid recommendations for User 2 (liked 'The Matrix'):")
for i, r in enumerate(hybrid_recommend(2, "The Matrix"), 1):
    print(f"  {i}. {r}")

print("\nRecommender system complete!")