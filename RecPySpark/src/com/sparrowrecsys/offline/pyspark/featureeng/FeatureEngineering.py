from pyspark import SparkConf
from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoderEstimator, StringIndexer, QuantileDiscretizer, MinMaxScaler
from pyspark.ml.linalg import VectorUDT, Vectors
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql import functions as F


def oneHotEncoderExample(movieSamples):
    samplesWithIdNumber = movieSamples.withColumn("movieIdNumber", F.col("movieId").cast(IntegerType()))
    encoder = OneHotEncoderEstimator(inputCols=["movieIdNumber"], outputCols=['movieIdVector'], dropLast=False)
    oneHotEncoderSamples = encoder.fit(samplesWithIdNumber).transform(samplesWithIdNumber)
    oneHotEncoderSamples.printSchema()
    oneHotEncoderSamples.show(10)


def array2vec(genreIndexes, indexSize):
    genreIndexes.sort()
    fill_list = [1.0 for _ in range(len(genreIndexes))]
    return Vectors.sparse(indexSize, genreIndexes, fill_list)


def multiHotEncoderExample(movieSamples):
    samplesWithGenre = movieSamples.select("movieId", "title", explode(
        split(F.col("genres"), "\\|").cast(ArrayType(StringType()))).alias('genre'))
    genreIndexer = StringIndexer(inputCol="genre", outputCol="genreIndex")
    StringIndexerModel = genreIndexer.fit(samplesWithGenre)
    genreIndexSamples = StringIndexerModel.transform(samplesWithGenre).withColumn("genreIndexInt",
                                                                                  F.col("genreIndex").cast(IntegerType()))
    indexSize = genreIndexSamples.agg(max(F.col("genreIndexInt"))).head()[0] + 1
    processedSamples = genreIndexSamples.groupBy('movieId').agg(
        F.collect_list('genreIndexInt').alias('genreIndexes')).withColumn("indexSize", F.lit(indexSize))
    finalSample = processedSamples.withColumn("vector",
                                              udf(array2vec, VectorUDT())(F.col("genreIndexes"), F.col("indexSize")))
    finalSample.printSchema()
    finalSample.show(10)


def ratingFeatures(ratingSamples):
    ratingSamples.printSchema()
    ratingSamples.show()
    # calculate average movie rating score and rating count
    movieFeatures = ratingSamples.groupBy('movieId').agg(F.count(F.lit(1)).alias('ratingCount'),
                                                         F.avg("rating").alias("avgRating"),
                                                         F.variance('rating').alias('ratingVar')) \
        .withColumn('avgRatingVec', udf(lambda x: Vectors.dense(x), VectorUDT())('avgRating'))
    movieFeatures.show(10)
    # bucketing
    ratingCountDiscretizer = QuantileDiscretizer(numBuckets=100, inputCol="ratingCount", outputCol="ratingCountBucket")
    # Normalization
    ratingScaler = MinMaxScaler(inputCol="avgRatingVec", outputCol="scaleAvgRating")
    pipelineStage = [ratingCountDiscretizer, ratingScaler]
    featurePipeline = Pipeline(stages=pipelineStage)
    movieProcessedFeatures = featurePipeline.fit(movieFeatures).transform(movieFeatures)
    movieProcessedFeatures.show(10)


if __name__ == '__main__':
    conf = SparkConf().setAppName('featureEngineering').setMaster('local')
    spark = SparkSession.builder.config(conf=conf).getOrCreate()
    file_path = 'file:///Users/zhewang/Workspace/SparrowRecSys/src/main/resources'
    movieResourcesPath = file_path + "/webroot/sampledata/movies.csv"
    movieSamples = spark.read.format('csv').option('header', 'true').load(movieResourcesPath)
    print("Raw Movie Samples:")
    movieSamples.show(10)
    movieSamples.printSchema()
    print("OneHotEncoder Example:")
    oneHotEncoderExample(movieSamples)
    print("MultiHotEncoder Example:")
    multiHotEncoderExample(movieSamples)
    print("Numerical features Example:")
    ratingsResourcesPath = file_path + "/webroot/sampledata/ratings.csv"
    ratingSamples = spark.read.format('csv').option('header', 'true').load(ratingsResourcesPath)
    ratingFeatures(ratingSamples)
