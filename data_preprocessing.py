import numpy as np
import pandas as pd

# Reading data
# MovieID, Title, Genres
movie_data = pd.DataFrame([movie.replace('\n','').split('::') for movie in open('movies.txt', encoding="ISO-8859-1").readlines()],
                          columns=['MovieID', 'Title', 'Genres'])

# UserID, Gender, Age, Occupation, Zip-code
user_data = pd.DataFrame([user.replace('\n','').split('::') for user in open('users.txt', encoding="ISO-8859-1").readlines()],
                         columns=['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code'])
user_data['Age'] = user_data['Age'].astype(int)

# UserID, MovieID, Rating, Timestamp
rating_data = pd.DataFrame([rating.replace('\n','').split('::') for rating in open('ratings.txt', encoding="ISO-8859-1").readlines()],
                           columns=['UserID', 'MovieID', 'Rating', 'Timestamp'])
rating_data['Rating'] = rating_data['Rating'].astype(int)
rating_data['Timestamp'] = pd.to_datetime(rating_data['Timestamp'], unit='s')


# Selecting Data
# validation set (5 users)
# user_validation = np.random.choice(rating_data['UserID'], 5)
user_validation = ['2484', '4448', '2106', '5702', '1018']
# dataset for training and testing
rating_data_selected = rating_data[~rating_data['UserID'].isin(user_validation)]
rating_data_selected['Rank_Latest'] = rating_data.groupby(['UserID', 'Rating'])['Timestamp'].rank(method='first',ascending=False)

# dataset 1: training dataset
training_rating = rating_data_selected[rating_data_selected['Rank_Latest'] != 1].drop(['Rank_Latest'], axis=1)
# dataset 2: testing dataset
testing_rating = rating_data_selected[rating_data_selected['Rank_Latest'] == 1].drop(['Rank_Latest'], axis=1)
# dataset 3: validation dataset (5 users)
validation_rating = rating_data[rating_data['UserID'].isin(user_validation)]
