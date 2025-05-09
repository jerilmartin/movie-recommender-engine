# Movie Recommender Engine 🎬

A simple content-based movie recommender system using Python and scikit-learn. It suggests similar movies based on keywords, cast, genres, and director.

## How it works
- Combines selected features into a single string
- Vectorizes them using `CountVectorizer`
- Calculates similarity with `cosine_similarity`

## Getting Started

### 1. Install dependencies
```
pip install pandas scikit-learn numpy
```

### 2. Run the script
Make sure `movie_dataset.csv` is in the same folder.
```
python movie_recommender_starter.py
