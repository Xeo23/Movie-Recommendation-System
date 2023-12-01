import pandas as pd
import numpy as np
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle


# Importing datasets(movies & credits) and creating dataframes. 
movies = pd.read_csv('./datasets/tmdb_5000_movies.csv')
credit = pd.read_csv('./datasets/tmdb_5000_credits.csv')

# Merging the two datasets
movies2 = movies.merge(credit,on='title')

# Removing unrelatable data.
movies2 = movies2[['id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]

# Drooping or removing items with missing values
movies2.dropna(inplace=True)

# Function to convert strings into list and extracting only the names.
def convert(text):
    List = []
    for i in ast.literal_eval(text):
        List.append(i['name']) 
    return List

# Applying convert function on columns(genre, keywords).
movies2['genres'] = movies2['genres'].apply(convert)
movies2['keywords'] = movies2['keywords'].apply(convert)

# Function to convert first 3 casts from strings into list and extracting the names.
def convert_cast(text):
    List = []
    count = 0
    for i in ast.literal_eval(text):
        if count < 3:
            List.append(i['name'])
        count+=1
    return List

# Applying convert_cast function to cast column.
movies2['cast'] = movies2['cast'].apply(convert_cast)

# Function to convert and fetch only director name from crew.
def convert_director(text):
    List = []
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            List.append(i['name'])
    return List

# Applying convert_director function to crew column.
movies2['crew'] = movies2['crew'].apply(convert_director)

# Converting sequence of strings into list.
movies2['overview'] = movies2['overview'].apply(lambda x:x.split())

# Function to replace space with underline to take the (name & surname) or (two words such as sci fi) as single entity. 
def remove_space(text):
    List = []
    for i in text:
        List.append(i.replace(" ","_"))
    return List

# Applying remove_space function to cast, crew, genres and keywords columns.
movies2['cast'] = movies2['cast'].apply(remove_space)
movies2['crew'] = movies2['crew'].apply(remove_space)
movies2['genres'] = movies2['genres'].apply(remove_space)
movies2['keywords'] = movies2['keywords'].apply(remove_space)

# Creating tags column by concatenating overview, genres, keywords, cast and crew.
movies2['tags'] = movies2['overview'] + movies2['genres'] + movies2['keywords'] + movies2['cast'] + movies2['crew']

# Creating new dataframe by removing columns overview, genres, keywords, cast, crew from movies2.
movies_list = movies2.drop(columns=['overview','genres','keywords','cast','crew'])

# Converting the tags column from list into strings.
movies_list['tags'] = movies_list['tags'].apply(lambda x: " ".join(x))

# Converting tags into lowercase.
movies_list['tags'] = movies_list['tags'].apply(lambda x:x.lower())

# Creating object of CountVectorizer, intializing max features and filtering english stop words.
cv = CountVectorizer(max_features=10000,stop_words='english')

# Converting cv output into an array.
vectors = cv.fit_transform(movies_list['tags']).toarray()

# Calculating similarity value of vectors using cosine_similarity and storing it in a variable.
similarities = cosine_similarity(vectors)

movies_list[movies_list['title'] == 'The Lego Movie'].index[0]

# Creating function that will sort and print the 10 mmost similar movies.
def recommend(movie):
    index = movies_list[movies_list['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarities[index])),reverse=True,key = lambda x: x[1])
    for i in distances[1:11]:
        print(movies_list.iloc[i[0]].title)
        
# movie_name=input('Enter official title of a movie: ')

# if movie_name in movies_list['title'].values:
#     recommend(movie_name)
# else:
#     print('The title name is not available in the database, check the title name.')

recommend("Iron Man")

pickle.dump(movies_list,open('movie_list.pkl','wb'))
pickle.dump(similarities,open('similarity.pkl','wb'))
