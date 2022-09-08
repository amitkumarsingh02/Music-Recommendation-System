"""
### Algorithm

1. Using Album, Artist, Genre [1-21] Ratings 
  - Using Album, Artist, Genre [1-21] Ratings with weights
  - Using Album, Artist, Genre [1-21] Ratings - Mean, Max, Sum

Method 1 : Using Album, Artist, Genre [1-21] Ratings
"""

import pandas as pd
import warnings
warnings.filterwarnings("ignore")
output = pd.read_csv('./data/output21.txt', sep='|', header=None)
output.columns = ['User ID', 'Track ID', 'Album Rating', 'Artist Rating','Genre1', 'Genre2', 'Genre3', 'Genre4', 'Genre5', 'Genre6', 'Genre7', 'Genre8', 'Genre9', 'Genre10','Genre11', 'Genre12', 'Genre13', 'Genre14', 'Genre15', 'Genre16', 'Genre17', 'Genre18', 'Genre19', 'Genre20', 'Genre21']
output = output[['User ID', 'Track ID', 'Album Rating', 'Artist Rating','Genre1', 'Genre2', 'Genre3', 'Genre4', 'Genre5', 'Genre6', 'Genre7', 'Genre8', 'Genre9', 'Genre10','Genre11', 'Genre12', 'Genre13', 'Genre14', 'Genre15', 'Genre16', 'Genre17', 'Genre18', 'Genre19', 'Genre20', 'Genre21']]

"""#### 1.1 : Using Album, Artist, Genre [1-21] Ratings with weights
#### Sum (Artist * 0.8 + Album * 0.15 + Mean (Genre [1-22]) * 0.05)
"""

# Sum (Artist * 0.8 + Album * 0.15 + Mean (Genre1-22) * 0.05)
output_2 = output.copy()
output_2['Genre Mean'] = output_2[['Genre1', 'Genre2', 'Genre3', 'Genre4', 'Genre5', 'Genre6', 'Genre7', 'Genre8', 'Genre9', 'Genre10','Genre11', 'Genre12', 'Genre13', 'Genre14', 'Genre15', 'Genre16', 'Genre17', 'Genre18', 'Genre19', 'Genre20', 'Genre21']].mean(axis=1)
output_2['Album Rating'] = output_2['Album Rating'] * 0.8
output_2['Artist Rating'] = output_2['Artist Rating'] * 0.15
output_2['Genre Mean'] =  output_2['Genre Mean']  * 0.05
output_2['score'] = output_2[['Album Rating', 'Artist Rating','Genre Mean']].sum(axis=1)
output_2[['User ID', 'Track ID', 'score']]

rating = output_2.copy()
rating['userid_trackid'] = rating['User ID'].astype(str) + '_' + rating['Track ID'].astype(str)
rating.drop(['Track ID','Album Rating', 'Artist Rating'],axis=1)
rating = rating[['User ID', 'userid_trackid', 'score']]
rating['rating'] = 0
rating = rating.sort_values(by = ['User ID', 'score'], ascending = [True, False])

users = rating['User ID'].unique()
df = pd.DataFrame()
for userId in users:
    frame_to_update = rating.loc[rating['User ID'] == userId]
    frame_to_update.head(3)['rating'] = 1
    df = df.append(frame_to_update, ignore_index = True)
rating_answer = rating[['userid_trackid', 'rating']]
rating_answer = rating_answer.rename(columns={'userid_trackid':'TrackID', 'rating':'Predictor'})

rating_answer.to_csv('./kaggle_submissions/Predictions_gmean_sum81505.csv', index=False)

"""##### 1.2 : Using Album, Artist, Genre [1-21] Ratings - Average

#####  Mean (Artist + Album + Genre [1-21] )
"""

# Mean (Artist + Album + Genre1-21)
output_2 = output.copy()
output_2['score'] = output_2[['Album Rating', 'Artist Rating','Genre1', 'Genre2', 'Genre3', 'Genre4', 'Genre5', 'Genre6', 'Genre7', 'Genre8', 'Genre9', 'Genre10','Genre11', 'Genre12', 'Genre13', 'Genre14', 'Genre15', 'Genre16', 'Genre17', 'Genre18', 'Genre19', 'Genre20', 'Genre21']].mean(axis=1)
output_2[['User ID', 'Track ID', 'score']]

rating = output_2.copy()
rating['userid_trackid'] = rating['User ID'].astype(str) + '_' + rating['Track ID'].astype(str)
rating.drop(['Track ID','Album Rating', 'Artist Rating'],axis=1)
rating = rating[['User ID', 'userid_trackid', 'score']]
rating['rating'] = 0
rating = rating.sort_values(by = ['User ID', 'score'], ascending = [True, False])

users = rating['User ID'].unique()
df = pd.DataFrame()
for userId in users:
    frame_to_update = rating.loc[rating['User ID'] == userId]
    frame_to_update.head(3)['rating'] = 1
    df = df.append(frame_to_update, ignore_index = True)
rating = df
rating_answer = rating[['userid_trackid', 'rating']]
rating_answer = rating_answer.rename(columns={'userid_trackid':'TrackID', 'rating':'Predictor'})

rating_answer.to_csv('./kaggle_submissions/Predictions_mean.csv', index=False)

"""#### 1.3 : Using Album, Artist, Genre [1-21] Ratings - Sum

##### Sum (Artist + Album + Genre [1-21])
"""

# Sum (Artist + Album + Genre1-22)
output_2 = output.copy()
output_2['score'] = output_2[['Album Rating', 'Artist Rating','Genre1', 'Genre2', 'Genre3', 'Genre4', 'Genre5', 'Genre6', 'Genre7', 'Genre8', 'Genre9', 'Genre10','Genre11', 'Genre12', 'Genre13', 'Genre14', 'Genre15', 'Genre16', 'Genre17', 'Genre18', 'Genre19', 'Genre20', 'Genre21']].sum(axis=1)
output_2[['User ID', 'Track ID', 'score']]

rating = output_2.copy()
rating['userid_trackid'] = rating['User ID'].astype(str) + '_' + rating['Track ID'].astype(str)
rating.drop(['Track ID','Album Rating', 'Artist Rating'],axis=1)
rating = rating[['User ID', 'userid_trackid', 'score']]
rating['rating'] = 0
rating = rating.sort_values(by = ['User ID', 'score'], ascending = [True, False])

users = rating['User ID'].unique()
df = pd.DataFrame()
for userId in users:
    frame_to_update = rating.loc[rating['User ID'] == userId]
    frame_to_update.head(3)['rating'] = 1
    df = df.append(frame_to_update, ignore_index = True)
rating = df
rating_answer = rating[['userid_trackid', 'rating']]
rating_answer = rating_answer.rename(columns={'userid_trackid':'TrackID', 'rating':'Predictor'})

rating_answer.to_csv('./kaggle_submissions/Predictions_sum.csv', index=False)

"""#### 1.4 : Using Album, Artist, Genre [1-21] Ratings - Max

##### Max (Artist + Album + Genre1-22)
"""

# Max (Artist + Album + Genre1-22)
output_2 = output.copy()
output_2['score'] = output_2[['Album Rating', 'Artist Rating','Genre1', 'Genre2', 'Genre3', 'Genre4', 'Genre5', 'Genre6', 'Genre7', 'Genre8', 'Genre9', 'Genre10','Genre11', 'Genre12', 'Genre13', 'Genre14', 'Genre15', 'Genre16', 'Genre17', 'Genre18', 'Genre19', 'Genre20', 'Genre21']].max(axis=1)
output_2[['User ID', 'Track ID', 'score']]

rating = output_2.copy()
rating['userid_trackid'] = rating['User ID'].astype(str) + '_' + rating['Track ID'].astype(str)
rating.drop(['Track ID','Album Rating', 'Artist Rating'],axis=1)
rating = rating[['User ID', 'userid_trackid', 'score']]
rating['rating'] = 0
rating = rating.sort_values(by = ['User ID', 'score'], ascending = [True, False])

users = rating['User ID'].unique()
df = pd.DataFrame()
for userId in users:
    frame_to_update = rating.loc[rating['User ID'] == userId]
    frame_to_update.head(3)['rating'] = 1
    df = df.append(frame_to_update, ignore_index = True)
rating = df
rating_answer = rating[['userid_trackid', 'rating']]
rating_answer = rating_answer.rename(columns={'userid_trackid':'TrackID', 'rating':'Predictor'})

rating_answer.to_csv('kaggle_submissions/Predictions_max.csv', index=False)

"""
#### 1.5 : Using Album, Artist, Genre [1-21] Ratings - Weighted Sum
#### Mean (Artist * 0.8 + Album * 0.15 + Sum (Genre1-22) * 0.05)

"""

# Mean (Artist * 0.8 + Album * 0.15 + Sum (Genre1-22) * 0.05)
output['Genre Mean'] = output[['Genre1', 'Genre2', 'Genre3', 'Genre4', 'Genre5', 'Genre6', 'Genre7', 'Genre8', 'Genre9', 'Genre10','Genre11', 'Genre12', 'Genre13', 'Genre14', 'Genre15', 'Genre16', 'Genre17', 'Genre18', 'Genre19', 'Genre20', 'Genre21']].sum(axis=1)
output['Album Rating'] = output['Album Rating'] * 0.8
output['Artist Rating'] = output['Artist Rating'] * 0.15
output['Genre Mean'] =  output['Genre Mean']  * 0.05
output['score'] = output[['Album Rating', 'Artist Rating','Genre Mean']].mean(axis=1)
output[['User ID', 'Track ID', 'score']]

rating = output_2.copy()
rating['userid_trackid'] = rating['User ID'].astype(str) + '_' + rating['Track ID'].astype(str)
rating.drop(['Track ID','Album Rating', 'Artist Rating'],axis=1)
rating = rating[['User ID', 'userid_trackid', 'score']]
rating['rating'] = 0
rating = rating.sort_values(by = ['User ID', 'score'], ascending = [True, False])

users = rating['User ID'].unique()
df = pd.DataFrame()
for userId in users:
    frame_to_update = rating.loc[rating['User ID'] == userId]
    frame_to_update.head(3)['rating'] = 1
    df = df.append(frame_to_update, ignore_index = True)
rating = df
rating_answer = rating[['userid_trackid', 'rating']]
rating_answer = rating_answer.rename(columns={'userid_trackid':'TrackID', 'rating':'Predictor'})

rating_answer.to_csv('kaggle_submissions/Predictions_gsum_mean81505.csv', index=False)

