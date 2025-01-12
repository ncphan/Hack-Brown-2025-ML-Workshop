# maybe make this into a jupyter notebook and have them work through it

# import libraries
import pandas as pd
 
# scikit-learn imports
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
 

# read in the csv file and create a dataframe. then look at the first 5 rows
tracks = pd.read_csv('dataset.csv')
pd.set_option('display.max_columns', None)
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

unique_tracks = tracks.drop_duplicates().reset_index(drop=True)
duplicates = unique_tracks['track_id'].duplicated()
print("Rows with duplicates in 'track_id' column:")
print((unique_tracks[duplicates])['track_name'])

''' still have some duplicates because some songs are remixes, so they have different track IDs. Thus, it makes it harder to
find these remixes and get rid of them. You could probably utilize regex's to do this, but for the purposes of this workshop, we'll just 
leave in the remixes. They shouldn't affect the model that much since there are over 100,000 songs in the dataset.
'''

# check if there's any missing/NULL values. There are none! Go into what you should do if there are? 
 
unique_tracks.dropna(inplace = True)

print(unique_tracks.isnull().sum())


# drop some attributes that we don't want to select on
unique_tracks = unique_tracks.drop(['track_id'], axis = 1)

print(unique_tracks['track_name'].nunique(), unique_tracks.shape)



# need to not use non-numerical features or convert them to numbers
# also need to standardize values

# only use popularity, duration_ms, explicit (OHE), danceability, energy, key, loudness,
# mode, speechiness, acousticness, instrumentalness, liveliness, valence, tempo, time_signature



# Print the unique genres

unique_genres = unique_tracks["track_genre"].unique()
print(unique_genres)
label_encoder = LabelEncoder()

# Fit and transform the 'Genres' column to numerical values
unique_tracks['track_genre'] = label_encoder.fit_transform(unique_tracks['track_genre'])

# Show the DataFrame with the encoded genre
print(unique_tracks["track_genre"].unique())
print(unique_tracks["track_name"].duplicated())
non_unique_rows = unique_tracks[unique_tracks.duplicated(keep=False)]
print(non_unique_rows.head())
unique_tracks = unique_tracks.drop_duplicates(subset=['track_name'], keep='first').reset_index(drop=True)
non_unique_rows = unique_tracks[unique_tracks.duplicated(keep=False)]

print(len(non_unique_rows))


# each song is either explicit (true) or not explicit (false). Use one-hot-encoding to represent
# explicit songs as 1, otherwise 0
unique_tracks['explicit'] = unique_tracks['explicit'].astype(int)
print(unique_tracks['explicit'].unique())


def scale_columns():
    scaler = StandardScaler()

    # Fit the scaler to the data and transform it
    scaled_data = scaler.fit_transform(unique_tracks.drop(columns=['album_name', 'track_name', 'artists']))
    scaled_df = pd.DataFrame(scaled_data, columns=unique_tracks.columns.drop(['album_name', 'track_name', 'artists']))

    print(scaled_df.head())
    print(scaled_df.shape)
    return scaled_df
    

def setup_knn(scaled_df):
    knn = NearestNeighbors(n_neighbors=10, metric='euclidean')
    knn.fit(scaled_df)
    return knn

def recommend_songs(track, knn, scaled_data):
    if track not in unique_tracks['track_name'].values:
        print("Track not in dataset")
    else:

        row_index = unique_tracks[unique_tracks['track_name'] == track].index[0]

        distances, indices = knn.kneighbors(scaled_data.iloc[row_index].values.reshape(1, -1))

        ret_tracks = []
        print(unique_tracks.iloc[row_index])
        for i in indices:
            ret_tracks.append((unique_tracks.iloc[i]['track_name'], unique_tracks.iloc[i]['artists']))
        print(ret_tracks)
        

scaled_data = scale_columns()
knn = setup_knn(scaled_data)
input_track = input("Enter a track: ")
recommend_songs(input_track, knn, scaled_data)


