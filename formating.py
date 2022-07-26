#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Formating
import os 
import numpy as np
import pandas as pd
import pandas_profiling as pp
import re
from pathlib import Path  
from datetime import datetime, date
from time import strftime


# ## Reformating 

# In[59]:


def getData(datAddress, columns):
    """Read in .dat file with '::' as sep"""
    with open(datAddress, encoding="ISO-8859-1") as f: 
        lists = [line.strip().split('::') for line in f.readlines()]
        df = pd.DataFrame(lists, columns = columns)
    return df 


# In[62]:


def saveData(name,df):
    """Create a 'cleaned' folder under the directory to save the csv files"""
    parentPathName = os.getcwd() + '/cleaned/' 
    csvName = name + '.csv'
    os.makedirs(parentPathName, exist_ok = True)
    filepath = os.path.join(parentPathName, csvName)    
    df.to_csv(filepath, index = False) 
    
    return None


# In[61]:


os.getcwd()
userFile = './ml-1m/users.dat'
movieFile = './ml-1m/movies.dat'
ratingFile = './ml-1m/ratings.dat'

userColumns = ['UserID','Gender','Age','Occupation','Zip-code']
movieColumns = ['MovieID','Title','Genres']
ratingColumns = ['UserID','MovieID','Rating','Timestamp']

users = getData(userFile,userColumns)
movies = getData(movieFile, movieColumns)
ratings = getData(ratingFile, ratingColumns)
# mv = pd.read_csv(file_m, sep='::',engine='python', header=None)
print(users.head(2))
print(movies.head(2))
print(ratings.head(2))


# ### Movies

# In[63]:


# For movies, want to create new columns w/ year - extract the number within ()
# *: matches zero or more occurrences of the pattern left to it
# .: any character
# .*: 0 or more of any characters
movies['year'] = movies['Title'].astype('str').str.extractall('.*\((\d+)\)*.').unstack().fillna('').sum(axis = 1).astype(int)

# split genres
movies = pd.concat([movies,movies.Genres.str.get_dummies(sep='|')], axis = 1)
movies.sort_values('year',ascending = False, inplace = True)
movies.drop(columns = 'Genres', inplace = True)
print(movies.head(2))


# ### Ratings

# In[64]:


ratings['Timestamp'] = pd.to_datetime(ratings['Timestamp'], unit = 's')


# ### Selecting Data

# In[ ]:


# validation set (5 users)
# user_validation = np.random.choice(rating_data['UserID'], 5)
user_validation = ['2484', '4448', '2106', '5702', '1018']
# dataset for training and testing
rating_data_selected = ratings[~ratings['UserID'].isin(user_validation)]
rating_data_selected['Rank_Latest'] = ratings.groupby(['UserID', 'Rating'])['Timestamp'].rank(method='first',ascending=False)

# dataset 1: training dataset
training_rating = rating_data_selected[rating_data_selected['Rank_Latest'] != 1].drop(['Rank_Latest'], axis=1)
# dataset 2: testing dataset
testing_rating = rating_data_selected[rating_data_selected['Rank_Latest'] == 1].drop(['Rank_Latest'], axis=1)
# dataset 3: validation dataset (5 users)
validation_rating = ratings[ratings['UserID'].isin(user_validation)]


# In[65]:


saveData('movies',movies)
saveData('ratings',ratings)
saveData('users',users)
saveData('training_rating',training_rating)
saveData('testing_rating',testing_rating)
saveData('validation_rating',validation_rating)


# ## Recommendation system website 
# - https://towardsdatascience.com/introduction-to-recommender-systems-6c66cf15ada 
# - https://towardsdatascience.com/a-complete-guide-to-recommender-system-tutorial-with-sklearn-surprise-keras-recommender-5e52e8ceace1
# - https://www.kaggle.com/search?q=movie+recommend 
# - MovieLens specific 
#     - https://github.com/topics/movie-lens
