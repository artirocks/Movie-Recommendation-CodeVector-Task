#-----------------------import Libraries---------------------------------------------------------------

import pandas as pd
import numpy as np
import re
from tqdm import tqdm
from gensim.models import Word2Vec 
import random 
import warnings;
warnings.filterwarnings('ignore')

#---------import Datset---------------------------------------------------------------------------------

metadata = pd.read_csv("movies_metadata.csv")
list(metadata)
"""['adult','belongs_to_collection', 'budget', 'genres','homepage','id','imdb_id','original_language',
'original_title','overview','popularity','poster_path','production_companies','production_countries',
'release_date','revenue','runtime','spoken_languages','status','tagline','title','video',
 'vote_average', 'vote_count']"""

rating_data = pd.read_csv("ratings.csv")
list(rating_data)
"""['userId', 'movieId', 'rating', 'timestamp'] """

credit = pd.read_csv("credits.csv")
list(credit)
""" ['cast', 'crew', 'id']"""
links_small = pd.read_csv('links_small.csv')
links_small = links_small[links_small['tmdbId'].notnull()]['tmdbId'].astype('int')
metadata = metadata.drop([19730, 29503, 35587])
 
metadata['id'] = metadata['id'].astype('int')

#------------------------Fill--------------------------------------------------------------------------

for i in tqdm(range(0,45466)):
    if(metadata['id'][i]=='nan'):
        print(i)
        
for i in tqdm(range(0,26024289)):
    if(rating_data['movieId'][i]=='nan'):
        print(i)
        
metadata['id'] = metadata['id'].fillna(int(-1)) 
rating_data['movieId'] = rating_data['movieId'].fillna(int(-1)) 

for i in tqdm(range(0,45436)):
    if(metadata['id'][i]== -1):
        print(i)
        
for i in tqdm(range(0,26024289)):
    if(rating_data['movieId'][i]== -1):
        print(i)
""" It shows there is no nan in these IDs"""

metadata = metadata.rename(columns={'id': 'movieId'})
df = pd.merge(metadata,rating_data, on=movieId)
df = df.drop([19730, 29503, 35587])
df['id'] = df['id'].astype('int')
smd = df[df['id'].isin(links_small)]
smd = smd.iloc[0:6000]
smd.shape
         
#------Recommendation on the basis of features---------------------------------------------------------

features = ['title','genres','overview']

#filling all NaNs with blank string
for feature in features:
    metadata[feature] = metadata[feature].fillna('') 

#------To make corpus append features in a row---------------------------------------------------------   

def combine_features(row):
    return row['title']+" "+row['overview']+" "+row['genres'] 
df["combined_features"] = df.apply(combine_features,axis=1) 
#----------Data Cleaning-------------------------------------------------------------------------------
 
for i in range(0,45466):
    df["combined_features"] = re.sub('[^a-z A-z]',' ',df["combined_features"][i]) 
        
#-------------REcommendation using word2vec-------------------------------------------

customers = df["userId"].unique().tolist()
len(customers)
# shuffle customer ID's
random.shuffle(customers)

# extract 90% of customer ID's
customers_train = [customers[i] for i in range(round(0.9*len(customers)))]

# split data into train and validation set
train_df = df[df['userId'].isin(customers_train)]
validation_df = df[~df['userId'].isin(customers_train)] 
# list to capture purchase history of the customers
purchases_train = []

# populate the list with the product codes
for i in tqdm(customers_train):
    temp = train_df[train_df["userId"] == i]["movieId"].tolist()
    purchases_train.append(temp)

# list to capture purchase history of the customers
purchases_val = []

# populate the list with the product codes
for i in tqdm(validation_df['userId'].unique()):
    temp = validation_df[validation_df["userId"] == i][" movieId"].tolist()
    purchases_val.append(temp)
    
# train word2vec model
model = Word2Vec(window = 10, sg = 1, hs = 0,
                 negative = 10, # for negative sampling
                 alpha=0.03, min_alpha=0.0007,
                 seed = 14)

model.build_vocab(purchases_train, progress_per=200)

model.train(purchases_train, total_examples = model.corpus_count, 
            epochs=10, report_delay=1)

model.init_sims(replace=True)

# extract all vectors
X = model[model.wv.vocab]

X.shape

products = train_df[["movieId", "combined_features"]]

# remove duplicates
products.drop_duplicates(inplace=True, subset='movieId', keep="last")

# create product-ID and product-description dictionary
products_dict = products.groupby('movieId')['combined_features'].apply(list).to_dict()

# test the dictionary
products_dict['84029E']    

def similar_products(v, n = 6):
    
    # extract most similar products for the input vector
    ms = model.similar_by_vector(v, topn= n+1)[1:]
    
    # extract name and similarity score of the similar products
    new_ms = []
    for j in ms:
        pair = (products_dict[j[0]][0], j[1])
        new_ms.append(pair)
        
    return new_ms        

similar_products(model['90019A'])

def aggregate_vectors(products):
    product_vec = []
    for i in products:
        try:
            product_vec.append(model[i])
        except KeyError:
            continue
        
    return np.mean(product_vec, axis=0)

len(purchases_val[0])
aggregate_vectors(purchases_val[0]).shape
similar_products(aggregate_vectors(purchases_val[0]))
similar_products(aggregate_vectors(purchases_val[0][-10:]))


 