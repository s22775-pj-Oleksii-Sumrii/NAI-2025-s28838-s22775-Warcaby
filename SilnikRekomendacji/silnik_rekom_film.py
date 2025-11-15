"""
   Projekt: Silnik rekomendacji do filmy

AUTORZY:
Oleksii Sumrii (s22775),
Oskar Szyszko (s28838),

Problem który rozwiązuje:
Silnik rekomendacji rozwiązuje problem nadmiaru informacji i trudności w wyborze. 
W dzisiejszych czasach mamy miliony filmów, książek, muzyki czy produktów online, a użytkownik nie jest w stanie samodzielnie przeglądać całej oferty.

INSTRUKCJA PRZYGOTOWANIA ŚRODOWISKA:
Zainstaluj Python 3.8 lub nowszy z https://python.org/,
Zainstaluj Visual Studio Code (VS Code) z https://code.visualstudio.com/,
Zainstaluj PyGame w terminale wpisz: pip install pandas numpy scikit-learn requests (ale po instalowanym Pythonie)
Zainstaluj rozszerzenie „Python” w VS Code:
Otwórz VS Code
Kliknij ikonę rozszerzeń po lewej stronie albo Ctrl+Shift+X
Wyszukaj Python i zainstaluj rozszerzenie od Microsoft,
,
,
,
Otworz folder w którym znajduje się plik silnik_rekom_film.py,
Uruchom aplikacje:
Otwórz plik silnik_rekom_film.py
Kliknij przycisk Run Python File albo Ctrl+F5
"""

from typing import List, Optional, Dict
import pandas as pd
import numpy as np
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time


class MovieRecommenderTMDb:
    def __init__(self, df: Optional[pd.DataFrame] = None, stop_words: str = "english"):
        self.df = None
        self.vectorizer = TfidfVectorizer(stop_words=stop_words)
        self.tfidf_matrix = None
        if df is not None:
            self.fit(df)
        self.api_key = "c8014e53a4da86d82b2af8bfcb00a032"  # klucz edukacyjny

    # ---------------- TMDb ----------------
    def fetch_tmdb_metadata(self, title: str) -> Dict:
        search_url = "https://api.themoviedb.org/3/search/movie"
        params = {"api_key": self.api_key, "query": title}
        try:
            response = requests.get(search_url, params=params)
            data = response.json()
            if not data.get("results"):
                return {"error": "Film nie znaleziony w TMDb."}

            movie = data["results"][0]
            movie_id = movie["id"]

            details_url = f"https://api.themoviedb.org/3/movie/{movie_id}"
            params = {"api_key": self.api_key}
            response2 = requests.get(details_url, params=params)
            details = response2.json()

            return {
                "title": details.get("title"),
                "year": (
                    details.get("release_date", "").split("-")[0]
                    if details.get("release_date")
                    else None
                ),
                "genres": [g["name"] for g in details.get("genres", [])],
                "description": details.get("overview"),
                "poster": (
                    f"https://image.tmdb.org/t/p/w500{details.get('poster_path')}"
                    if details.get("poster_path")
                    else None
                ),
                "rating": details.get("vote_average"),
                "popularity": details.get("popularity"),
            }

        except Exception as e:
            return {"error": str(e)}

    # ---------------- przygotowanie danych ----------------
    def _prepare_text(self, df: pd.DataFrame) -> pd.Series:
        if "genres" not in df.columns:
            df["genres"] = ""
        if "description" not in df.columns:
            df["description"] = ""
        return (
            df["genres"].apply(lambda x: " ".join(x) if isinstance(x, list) else x)
            + " "
            + df["description"].fillna("")
        ).astype(str)

    def fit(self, df: pd.DataFrame):
        if "title" not in df.columns:
            raise ValueError("DataFrame musi zawierać kolumnę 'title'")
        self.df = df.copy().reset_index(drop=True)
        self.df["doc_text"] = self._prepare_text(self.df)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df["doc_text"])

    # ---------------- profil użytkownika ----------------
    def _user_vector(self, liked_titles: List[str]) -> np.ndarray:
        if self.df is None:
            raise RuntimeError("Najpierw użyj fit(df)!")
        idxs = self.df[self.df["title"].isin(liked_titles)].index.tolist()
        if not idxs:
            return np.asarray(self.tfidf_matrix.mean(axis=0)).reshape(1, -1)
        return np.asarray(self.tfidf_matrix[idxs].mean(axis=0)).reshape(1, -1)

    # ---------------- rekomendacje ----------------
    def recommend(self, liked_titles: List[str], top_n: int = 5) -> pd.DataFrame:
        user_vec = self._user_vector(liked_titles)
        sims = cosine_similarity(self.tfidf_matrix, user_vec).flatten()
        sims_percent = (sims * 100).round(2)
        self.df["score"] = sims_percent

        mask = ~self.df["title"].isin(liked_titles)
        cols = ["title", "year", "genres", "description", "score"]
        result = self.df[mask].sort_values("score", ascending=False).head(top_n)[cols]
        result["score"] = result["score"].astype(str) + "%"
        return result

    def anti_recommend(
        self,
        liked_titles: List[str],
        top_n: int = 5,
        exclude_titles: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Antyrekomendacje: filmy najmniej podobne do ulubionych.
        exclude_titles: lista tytułów do wykluczenia (np. rekomendacje + liked)
        """
        user_vec = self._user_vector(liked_titles)
        sims = cosine_similarity(self.tfidf_matrix, user_vec).flatten()
        sims_percent = (sims * 100).round(2)
        self.df["score"] = sims_percent

        mask = ~self.df["title"].isin(liked_titles)
        if exclude_titles:
            mask = mask & ~self.df["title"].isin(exclude_titles)

        cols = ["title", "year", "genres", "description", "score"]
        result = self.df[mask].sort_values("score", ascending=True).head(top_n)[cols]
        result["score"] = result["score"].astype(str) + "%"
        return result

    # ---------------- pobieranie wielu filmów ----------------
    def build_dataset_from_tmdb(
        self, titles: List[str], sleep: float = 0.25
    ) -> pd.DataFrame:
        movies = []
        for t in titles:
            data = self.fetch_tmdb_metadata(t)
            if "error" not in data:
                movies.append(data)
            time.sleep(sleep)
        return pd.DataFrame(movies)


if __name__ == "__main__":

    titles = [
        "Inception",
        "The Matrix",
        "Shrek",
        "Toy Story",
        "John Wick",
        "Avengers: Endgame",
        "Interstellar",
        "Gladiator",
        "Frozen",
        "The Walking Dead",
        "The Lion King",
        "Forrest Gump",
        "Titanic",
        "The Godfather",
        "Star Wars: A New Hope",
        "Dexter",
    ]

recommender = MovieRecommenderTMDb()
df = recommender.build_dataset_from_tmdb(titles)
recommender.fit(df)

liked = ["Interstellar", "Titanic"]

# 5 rekomendacji
reco = recommender.recommend(liked, top_n=5)

# 5 antyrekomendacji (bez powtarzania rekomendacji ani liked)
exclude_titles = reco["title"].tolist() + liked
anti_reco = recommender.anti_recommend(liked, top_n=5, exclude_titles=exclude_titles)

print("\n===== 5 rekomendacji =====")
print(reco)

print("\n===== 5 antyrekomendacji =====")
print(anti_reco)
