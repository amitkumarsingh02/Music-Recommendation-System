
!pip install pyspark

"""### Method 2 : Matrix factorization"""

# load pyspark modules
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql.types import IntegerType
import pandas as pd

from pyspark.sql import SparkSession
spark = SparkSession.builder\
.master("local")\
.appName("Matrix_factorization")\
.getOrCreate()

training = spark.read.csv("./data/trainItem.data", header = False)
# 3.1 rename training dataFrame column names
training = training.withColumnRenamed("_c0", "userID").withColumnRenamed("_c1", "itemID").withColumnRenamed("_c2", "rating")
# 3.2 assign training dataFrame column data types (dataFrame by default herein assume 'string' ty
training = training.withColumn("userID", training["userID"].cast(IntegerType()))
training = training.withColumn("itemID", training["itemID"].cast(IntegerType()))
training = training.withColumn("rating", training["rating"].cast('float'))

# 4. configure the ALS model
# Create ALS model
als = ALS(
    maxIter= 20,
    rank = 20,
    regParam = 0.01,
    userCol="userID",
    itemCol="itemID",
    ratingCol="rating",
    nonnegative = True,
    implicitPrefs = False,
    coldStartStrategy="drop"
)

# 5. fit the ALS model using the training
model = als.fit(training)

# 6. load testing data file into pySpark dataFrame format
testing = spark.read.csv("./data/testItem.data", header = False)
testing = testing.withColumnRenamed("_c0", "userID").withColumnRenamed("_c1", "itemID").withColumnRenamed("_c2", "rating")
# 7. rename testing dataFrame column names
testing = testing.withColumn("userID", testing["userID"].cast(IntegerType()))
testing = testing.withColumn("itemID", testing["itemID"].cast(IntegerType()))
testing = testing.withColumn("rating", testing["rating"].cast('float'))

predictions = model.transform(testing)
predictions.coalesce(1).write.csv("predictions")
predictions.toPandas().to_csv('myprediction.csv')

import glob
filename = glob.glob('./predictions/part-00000*.csv')
rating = pd.read_csv(filename[0], header=None)
rating.columns = ['UserID', 'TrackID', 'Rating', 'Score']
rating['userid_trackid'] = rating['UserID'].astype(str)+'_'+rating['TrackID'].astype(str)
rating["Rating"] = rating["Rating"].astype('int64')
rating.drop(['TrackID'],axis=1)
rating = rating.sort_values(by = ['UserID', 'Score'], ascending = [True, False])
rating.head()

import warnings
warnings.filterwarnings("ignore")
users = rating['UserID'].unique()
df = pd.DataFrame()
for userId in users:
    frame_to_update = rating.loc[rating['UserID'] == userId]
    frame_to_update.head(3)['Rating'] = 1
    df = df.append(frame_to_update, ignore_index = True)
rating = df
rating_answer = rating[['userid_trackid', 'Rating']]
rating_answer = rating_answer.rename(columns={'userid_trackid':'TrackID', 'Rating':'Predictor'})

# Adding missing values
newID = rating_answer['TrackID'].values
rating_old = pd.read_csv('./kaggle_submissions/Predictions_mean.csv')
oldID = rating_old['TrackID'].values
missing_vals = list(set(oldID) - set(newID))
for values in missing_vals:
    temp = {'TrackID': values, 'Predictor': 0}
    rating_answer = rating_answer.append(temp, ignore_index = True)
rating_answer.to_csv('./kaggle_submissions/Predictions_matrix.csv', index=False)



