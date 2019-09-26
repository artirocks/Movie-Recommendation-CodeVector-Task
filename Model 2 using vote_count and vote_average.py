 
#-------import Libraries---------------------------
import pandas as pd
import numpy as np
import re
import nltk
#---------import Datset--------------------------------------------------------

df = pd.read_csv("movies_metadata.csv")
list(df)
links_small = pd.read_csv('links_small.csv')
links_small = links_small[links_small['tmdbId'].notnull()]['tmdbId'].astype('int')
"""
for i in range(0,45463):
    if(df['id'][i]=='1997-08-20'):
        print(i)
"""
   
df = df.drop([19730, 29503, 35587])
 
df['id'] = df['id'].astype('int')
smd = df[df['id'].isin(links_small)]
smd = smd.iloc[0:6000]
smd.shape
 
#------Recommendation on the basis of features---------------------------------
features = ['title','genres','overview']

#filling all NaNs with blank string
for feature in features:
    smd[feature] = smd[feature].fillna('') 

#------To make corpus append features in a row---------------------------------   

def combine_features(row):
    return row['title']+" "+row['overview']+" "+row['genres'] 
smd["combined_features"] = smd.apply(combine_features,axis=1) 

#----------Data Cleaning-------------------------------------------------------

list(smd)
for i in range(0,6000):
    smd["combined_features"] = re.sub('[^a-z A-z]',' ',smd["combined_features"][i]) 
     
    
#---------------To convert text into meaningful representation of numbers---------------------------------------------
    
from sklearn.feature_extraction.text import TfidfVectorizer
tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 3),min_df=0, stop_words='english')
count_matrix = tf.fit_transform(smd["combined_features"])

#---------------To find Cosine Similarity--------------------------------------

from sklearn.metrics.pairwise import linear_kernel
cosine_sim = linear_kernel(count_matrix,count_matrix)

#---Function to get title and Index of movies for recommendation

def get_title_from_index(index):
    return smd[smd.index == index]["title"].values[0]
def get_index_from_title(title):
    return smd[smd.title == title]["index"].values[0]

smd = smd.reset_index()
titles = smd['title']
# finding indices of every title
indices = pd.Series(smd.index, index=titles)

#-------Recommendatio for the movie which client have watched recently---------

movie_user_likes = "Toy Story"
movie_index = get_index_from_title(movie_user_likes)
similar_movies = list(enumerate(cosine_sim[movie_index]))
sorted_similar_movies = sorted(similar_movies,key=lambda x:x[1],reverse=True)[1:]

print("We recommend you to see these movies \n ")
c= smd['vote_average'].mean()
m= smd['vote_count'].quantile(0.9)

def weighted_rating(x, m=m, c=c):
    v = x['vote_count']
    R = x['vote_average']
    # Calculation based on the IMDB formula
    return (v/(v+m) * R) + (m/(m+v) * c)
    
def improved_recommendations(title):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:26]
    movie_indices = [i[0] for i in sim_scores]
    
    movies = smd.iloc[movie_indices][['title', 'vote_count', 'vote_average']]
    vote_counts = movies[movies['vote_count'].notnull()]['vote_count'].astype('int')
    vote_averages = movies[movies['vote_average'].notnull()]['vote_average'].astype('int')
    C = vote_averages.mean()
    m = vote_counts.quantile(0.60)
    qualified = movies[(movies['vote_count'] >= m) & (movies['vote_count'].notnull()) & (movies['vote_average'].notnull())]
    qualified['vote_count'] = qualified['vote_count'].astype('int')
    qualified['vote_average'] = qualified['vote_average'].astype('int')
    qualified['wr'] = qualified.apply(weighted_rating, axis=1)
    qualified = qualified.sort_values('wr', ascending=False).head(10)
    return qualified
improved_recommendations('Toy Story')    
