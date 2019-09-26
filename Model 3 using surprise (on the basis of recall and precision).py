#-----------------------import Libraries---------------------------------------------------------------

import pandas as pd
import numpy as np
import re
from tqdm import tqdm
from surprise import Dataset
from surprise import SVD
from surprise.model_selection import KFold
from collections import defaultdict
import io  
from surprise import KNNBaseline
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

#---------------------------------Rating for each Movie--------------------------
print(rating_data['movieId'][ 0])
print(rating_data['movieId'][1])

rating_data = rating_data.sort_values(by=['movieId']) 
rating_data = rating_data.reset_index(drop=True) 

print(rating_data['movieId'][0])
print(rating_data['movieId'][1])
  
count = 1
rate =  rating_data['rating'][0]
rating_list = []
movie_id_list = []
first = 0
print(rating_data['movieId'][26024288])
print(rating_data['movieId'][26024287])
 
for i in tqdm(range(0,26024288)):
    if(rating_data['movieId'][i]==rating_data['movieId'][i+1]):
       count = count + 1
       rate = rate + rating_data['rating'][i+1]
    elif(first==0):
        first = 1
        rating_list.append(rate/count)
        movie_id_list.append(rating_data['movieId'][i])
        count = 1
        rate = rating_data['rating'][i+1]
    else:
        rating_list.append(rate/count)
        movie_id_list.append(rating_data['movieId'][i])
        count = 1   
        rate = rating_data['rating'][i+1]
        
rating_list.append(rating_data['rating'][26024288])
movie_id_list.append(rating_data['movieId'][26024288])   

movie_id_list = np.asarray(movie_id_list)
rating_list =np.asarray(rating_list)

ratings_movie_id = rating_data.as_matrix(columns=rating_data.columns[1:2]) 
r_id = np.unique(ratings_movie_id)


for i in range(0,45115):
    if(r_id[i]!=movie_id_list[i]):
        print(i)
movie_id_list.reshape(45115,1)
rating_list.reshape(45115,1)

movieId = pd.DataFrame(movie_id_list,columns=['movieId'])
movie_rating = pd.DataFrame(rating_list,columns=['m_rating'])
list(movieId)
movie_rating_data = pd.concat([movieId, movie_rating], axis=1)
list(movie_rating_data)

#---------New Dataframe with Rating--------------------------------------------------------------------------------

metadata = metadata.rename(columns={'id': 'movieId'})
list(metadata)
df = pd.merge(metadata,movie_rating_data, on=movieId) 
list(df)
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
        
     

#------------------------Find Precision and Recall using Surprise----------------------------


def get_top_n(predictions, n=10):
    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))
        # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]
    return top_n


# First train an SVD algorithm on the movielens dataset.
data = Dataset.load_from_file('df')
trainset = data.build_full_trainset()
algo = SVD()
algo.fit(trainset)

# Than predict ratings for all pairs (u, i) that are NOT in the training set.
testset = trainset.build_anti_testset()
predictions = algo.test(testset)

top_n = get_top_n(predictions, n=10)

# Print the recommended items for each user
for uid, user_ratings in top_n.items():
    print(uid, [iid for (iid, _) in user_ratings])
    
#------------------- To compute precision@k and recall@k using surprise-----------------------------------
       
def precision_recall_at_k(predictions, k=10, threshold=3.5):
    '''Return precision and recall at k metrics for each user.'''

    # First map the predictions to each user.
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = dict()
    recalls = dict()
    for uid, user_ratings in user_est_true.items():

        # Sort user ratings by estimated value
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        # Number of relevant items
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

        # Number of recommended items in top k
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])

        # Number of relevant and recommended items in top k
        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold))
                              for (est, true_r) in user_ratings[:k])

        # Precision@K: Proportion of recommended items that are relevant
        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 1

        # Recall@K: Proportion of relevant items that are recommended
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 1

    return precisions, recalls


kf = KFold(n_splits=5)
algo = SVD()

for trainset, testset in kf.split(df):
    algo.fit(trainset)
    predictions = algo.test(testset)
    precisions, recalls = precision_recall_at_k(predictions, k=5, threshold=4)

    # Precision and recall can then be averaged over all users
    print(sum(prec for prec in precisions.values()) / len(precisions))
    print(sum(rec for rec in recalls.values()) / len(recalls))


#--------------------------To get the k nearest neighbors of a user using surprise ------------------------



def read_item_names():
    file_name =  df
    rid_to_name = {}
    name_to_rid = {}
    with io.open(file_name, 'r', encoding='ISO-8859-1') as f:
        for line in f:
            line = line.split('|')
            rid_to_name[line[0]] = line[1]
            name_to_rid[line[1]] = line[0]

    return rid_to_name, name_to_rid


# First, train the algortihm to compute the similarities between items
trainset = df.build_full_trainset()
sim_options = {'name': 'pearson_baseline', 'user_based': False}
algo = KNNBaseline(sim_options=sim_options)
algo.fit(trainset)

# Read the mappings raw id <-> movie name
rid_to_name, name_to_rid = read_item_names()

# Retrieve inner id of the movie Toy Story
#*********************Movie Recommended for Movie**********
toy_story_raw_id = name_to_rid['Toy Story']
toy_story_inner_id = algo.trainset.to_inner_iid(toy_story_raw_id)

# Retrieve inner ids of the nearest neighbors of Toy Story.
toy_story_neighbors = algo.get_neighbors(toy_story_inner_id, k=10)

# Convert inner ids of the neighbors into names.
toy_story_neighbors = (algo.trainset.to_raw_iid(inner_id)
                       for inner_id in toy_story_neighbors)
toy_story_neighbors = (rid_to_name[rid]
                       for rid in toy_story_neighbors)

print()
print('The 10 nearest neighbors of Toy Story are:')
for movie in toy_story_neighbors:
    print(movie)

 
 