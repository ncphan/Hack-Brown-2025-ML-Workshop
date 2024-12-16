# maybe make this into a jupyter notebook and have them work through it

# imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
 
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import TSNE
 
# don't want warnings
import warnings
warnings.filterwarnings('ignore')
 
# read in the csv file and create a dataframe. then look at the first 5 rows
tracks = pd.read_csv('dataset.csv')
print(tracks.head())

# print out dimensions of df (rows, columns)
print(tracks.shape)

# print out the columns to see what attributes each track has
print(tracks.columns)


''' 
As you may have noticed, the quality of our machine learning model is heavily reliant on our dataset. Therefore, we need to
make sure that our dataset is clean in order to produce high-quality results. This includes dealing with missing/NULL values, 
duplicate data, and irrelevant data in our dataset. 
'''

# print out if we have any duplicates in our dataset. Looks like there's not!
print(tracks.duplicated())

# see if there's missing data in any of the columns
print(tracks.info())

# Categorical columns
cat_col = [col for col in tracks.columns if tracks[col].dtype == 'object']
print('Categorical columns :',cat_col)
# Numerical columns
num_col = [col for col in tracks.columns if tracks[col].dtype != 'object']
print('Numerical columns :',num_col)

# we see that there are some rows with the same track ID. This means that we have duplicate songs!
print(tracks[cat_col].nunique())



duplicates = tracks['track_id'].duplicated()
print("Rows with duplicates in 'track_id' column:")
print((tracks[duplicates])['track_name'])

value_counts = tracks['track_id'].value_counts()
duplicates_count = value_counts[value_counts > 1]
print("\nDuplicate track IDs and their counts:")
print(duplicates_count)

# get rid of 'Unnamed: 0' row

tracks = tracks.drop('Unnamed: 0', axis=1)

print(tracks.iloc[1874] == tracks.iloc[1925])

print(tracks.info())

df = tracks.duplicated()
print(tracks[df])
print(df.sum())

# then we should remove the duplicates

tracks = tracks.drop_duplicates()
duplicates = tracks['track_id'].duplicated()
print("Rows with duplicates in 'track_id' column:")
print((tracks[duplicates])['track_name'])

''' still have some duplicates because some songs are remixes, so they have different track IDs. Thus, it makes it harder to
find these remixes and get rid of them. You could probably utilize regex's to do this, but for the purposes of this workshop, we'll just 
leave in the remixes. They shouldn't affect the model that much since there are over 100,000 songs in the dataset.
'''

# then, we want to clean our dataset before we do any ML. 

# check if there's any missing/NULL values. There are none! Go into what you should do if there are? 

print(tracks.isnull().sum())

 
tracks.dropna(inplace = True)



print(round((tracks.isnull().sum()/tracks.shape[0])*100,2))

# handle outliers


# remove unneeded columns (track name, album name)


