 
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
     
    
#---------------To count each word---------------------------------------------
    
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

i=0
print("We recommend you to see these movies \n ")
for element in sorted_similar_movies:
    print(get_title_from_index(element[0]))
    i=i+1
    if i>9:
        break
