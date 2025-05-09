import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Helper: Get movie title from index
def get_title_from_index(index):
    return df[df.index == index]["title"].values[0]

# Helper: Get index from movie title
def get_index_from_title(title):
    return df[df.title == title]["index"].values[0]

# Helper: Match closest movie title from user input
def find_closest_title(user_input):
    user_input = user_input.lower()
    matches = df[df['title'].str.lower().str.contains(user_input)]
    if not matches.empty:
        return matches.iloc[0]['title']
    else:
        return None

# Step 1: Load dataset
df = pd.read_csv("movie_dataset.csv")

# Step 2: Select relevant text-based features
features = ['keywords', 'cast', 'genres', 'director']

# Step 3: Fill missing feature values with empty strings
for feature in features:
    df[feature] = df[feature].fillna('')

# Step 4: Combine features into a single string
def combine_features(row):
    return f"{row['keywords']} {row['cast']} {row['genres']} {row['director']}"

df["combined_features"] = df.apply(combine_features, axis=1)

# Step 5: Convert text to a matrix of token counts
cv = CountVectorizer()
count_matrix = cv.fit_transform(df["combined_features"])

# Step 6: Compute cosine similarity matrix
cosine_sim = cosine_similarity(count_matrix)

# Step 7: Get movie input from user
user_input = input("Enter a movie name: ")
matched_title = find_closest_title(user_input)

if matched_title:
    print(f"\nTop 50 movies similar to '{matched_title}':\n")
    movie_index = get_index_from_title(matched_title)

    # Step 8: Get similarity scores and sort
    similar_movies = list(enumerate(cosine_sim[movie_index]))
    sorted_similar_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)

    # Step 9: Display top 50 recommendations
    i = 0
    for movie in sorted_similar_movies:
        print(get_title_from_index(movie[0]))
        i += 1
        if i > 50:
            break
else:
    print("Sorry, no matching movie found.")
