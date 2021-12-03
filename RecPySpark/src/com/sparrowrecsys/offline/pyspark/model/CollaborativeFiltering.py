from pyspark import SparkConf
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql import functions as F

if __name__ == '__main__':
    conf = SparkConf().setAppName('collaborativeFiltering').setMaster('local')
    spark = SparkSession.builder.config(conf=conf).getOrCreate()
    #/Users/zhewang/Workspace/SparrowRecSys/src/main/resources/webroot/modeldata
    file_path = 'file:///Users/zhewang/Workspace/SparrowRecSys/src/main/resources'
    ratingResourcesPath = file_path + '/webroot/sampledata/ratings.csv'
    ratingSamples = spark.read.format('csv').option('header', 'true').load(ratingResourcesPath) \
        .withColumn("userIdInt", F.col("userId").cast(IntegerType())) \
        .withColumn("movieIdInt", F.col("movieId").cast(IntegerType())) \
        .withColumn("ratingFloat", F.col("rating").cast(FloatType()))
    training, test = ratingSamples.randomSplit((0.8, 0.2))
    # Build the recommendation model using ALS on the training data
    # Note we set cold start strategy to 'drop' to ensure we don't get NaN evaluation metrics
    als = ALS(regParam=0.01, maxIter=5, userCol='userIdInt', itemCol='movieIdInt', ratingCol='ratingFloat',
              coldStartStrategy='drop')
    model = als.fit(training)
    # Evaluate the model by computing the RMSE on the test data
    predictions = model.transform(test)
    model.itemFactors.show(10, truncate=False)
    model.userFactors.show(10, truncate=False)
    evaluator = RegressionEvaluator(predictionCol="prediction", labelCol='ratingFloat', metricName='rmse')
    rmse = evaluator.evaluate(predictions)
    print("Root-mean-square error = {}".format(rmse))
    # Generate top 10 movie recommendations for each user
    userRecs = model.recommendForAllUsers(10)
    # Generate top 10 user recommendations for each movie
    movieRecs = model.recommendForAllItems(10)
    # Generate top 10 movie recommendations for a specified set of users
    users = ratingSamples.select(als.getUserCol()).distinct().limit(3)
    userSubsetRecs = model.recommendForUserSubset(users, 10)
    # Generate top 10 user recommendations for a specified set of movies
    movies = ratingSamples.select(als.getItemCol()).distinct().limit(3)
    movieSubSetRecs = model.recommendForItemSubset(movies, 10)
    userRecs.show(5, False)
    movieRecs.show(5, False)
    userSubsetRecs.show(5, False)
    movieSubSetRecs.show(5, False)
    paramGrid = ParamGridBuilder().addGrid(als.regParam, [0.01]).build()
    cv = CrossValidator(estimator=als, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=10)
    cvModel = cv.fit(test)
    avgMetrics = cvModel.avgMetrics
    spark.stop()
