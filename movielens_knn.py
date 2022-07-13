import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

# MovieID, Title, Genres
movie_data = pd.DataFrame([movie.replace('\n','').split('::') for movie in open('movies.txt', encoding="ISO-8859-1").readlines()],
                          columns=['MovieID', 'Title', 'Genres'])

# UserID, Gender, Age, Occupation, Zip-code
user_data = pd.DataFrame([user.replace('\n','').split('::') for user in open('users.txt', encoding="ISO-8859-1").readlines()],
                         columns=['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code']).drop(['Occupation', 'Zip-code'], axis=1)
user_data['Age'] = user_data['Age'].astype(int)
user_data.replace({1: 1, 18: 2, 25: 3, 35: 4, 45: 5, 50: 6, 56:  7})

# UserID, MovieID, Rating, Timestamp
rating_data = pd.DataFrame([rating.replace('\n','').split('::') for rating in open('ratings.txt', encoding="ISO-8859-1").readlines()],
                           columns=['UserID', 'MovieID', 'Rating', 'Timestamp']).drop(['Timestamp'], axis=1)
rating_data['Rating'] = rating_data['Rating'].astype(int)

# pivot table
rate_pivot = rating_data.pivot(index='MovieID', columns='UserID', values='Rating').fillna(0).reset_index()

movie_rating = rating_data.groupby('MovieID').agg(RatingCount = pd.NamedAgg(column='Rating', aggfunc='count'),
                                                  RatingAve = pd.NamedAgg(column='Rating', aggfunc='mean')).reset_index()
movie_rating['Popularity'] = movie_rating['RatingCount'] / movie_rating['RatingCount'].max() * 10
movie_rating['RatingAve'] *= 10
movie_rating = pd.merge(movie_data.drop(['Title', 'Genres'], axis=1), movie_rating.drop(['RatingCount'], axis=1), how='left', on='MovieID').fillna(0)

categories = ['Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
              'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
movie_genres = {category: [] for category in categories}
for genres in movie_data['Genres']:
    genre_dict = {category: 0 for category in categories}
    for genre in genres.split('|'):
        genre_dict[genre] += 1
    for key in movie_genres.keys():
        if genre_dict[key] == 1:
            movie_genres[key].append(5)
        else:
            movie_genres[key].append(0)
for key in movie_genres.keys():
    movie_rating[key] = movie_genres[key]

movie_info = pd.merge(movie_rating, rate_pivot, how='left', on='MovieID').fillna(0)
knn = NearestNeighbors(n_neighbors=5, algorithm = 'brute', metric = 'minkowski')
neighbors = knn.fit(movie_info)

# test
test_idx = np.random.choice(movie_info.shape[0])
dist, ind = knn.kneighbors(movie_info.iloc[test_idx, :].values.reshape(1, -1), n_neighbors = 6)
for i in range(0, len(dist.flatten())):
    if i == 0:
        print('Recommendations for {0}:'.format(movie_data.iloc[test_idx, 1]))
    else:
        print('{0}: {1}, distance score {2}:'.format(i, movie_data.iloc[ind.flatten()[i], 1], round(dist.flatten()[i], 2)))
