from pyspark import SparkConf
from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder, StringIndexer, QuantileDiscretizer, MinMaxScaler
from pyspark.ml.linalg import VectorUDT, Vectors
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql import functions as F


def one_hot_encoder_example(movie_samples):
    samples_with_id_number = movie_samples.withColumn("movieIdNumber", F.col("movieId").cast(IntegerType()))
    encoder = OneHotEncoder(inputCols=["movieIdNumber"], outputCols=['movieIdVector'], dropLast=True)
    one_hot_encoder_samples = encoder.fit(samples_with_id_number).transform(samples_with_id_number)
    one_hot_encoder_samples.printSchema()
    one_hot_encoder_samples.show(20)


def array2vec(genre_indexes, index_size):
    genre_indexes.sort()
    fill_list = [1.0 for _ in range(len(genre_indexes))]
    return Vectors.sparse(index_size, genre_indexes, fill_list)


def multi_hot_encoder_example(movie_samples):
    example = explode(split(F.col("genres"), "\\|").cast(ArrayType(StringType())))
    samples_with_genre = movie_samples.select("movieId", "title", "genres", example.alias('genre'))
    samples_with_genre.show(10)
    genre_indexer = StringIndexer(inputCol="genre", outputCol="genreIndex")
    string_indexer_model = genre_indexer.fit(samples_with_genre)
    genre_index_samples = string_indexer_model.transform(samples_with_genre).withColumn("genreIndexInt",
                                                                                        F.col("genreIndex").cast(
                                                                                            IntegerType()))
    index_size = genre_index_samples.agg(max(F.col("genreIndexInt"))).head()[0] + 1
    processed_samples = genre_index_samples.groupBy('movieId').agg(
        F.collect_list('genreIndexInt').alias('genreIndexes')).withColumn("indexSize", F.lit(index_size))
    final_sample = processed_samples.withColumn("vector",
                                                udf(array2vec, VectorUDT())(F.col("genreIndexes"), F.col("indexSize")))
    final_sample.printSchema()
    final_sample.show(30, 100)


def rating_features(rating_samples):
    rating_samples.printSchema()
    rating_samples.show()
    # calculate average movie rating score and rating count
    movie_features = rating_samples.groupBy('movieId').agg(F.count(F.lit(1)).alias('ratingCount'),
                                                           F.avg("rating").alias("avgRating"),
                                                           F.variance('rating').alias('ratingVar')) \
        .withColumn('avgRatingVec', udf(lambda x: Vectors.dense(x), VectorUDT())('avgRating'))
    movie_features.show(10)
    # bucketing
    rating_count_discretizer = QuantileDiscretizer(numBuckets=100, inputCol="ratingCount",
                                                   outputCol="ratingCountBucket")
    # Normalization
    rating_scaler = MinMaxScaler(inputCol="avgRatingVec", outputCol="scaleAvgRating")
    pipeline_stage = [rating_count_discretizer, rating_scaler]
    feature_pipeline = Pipeline(stages=pipeline_stage)
    movie_processed_features = feature_pipeline.fit(movie_features).transform(movie_features)
    movie_processed_features.show(101)


if __name__ == '__main__':
    conf = SparkConf().setAppName('featureEngineering').setMaster('local')
    spark = SparkSession.builder.config(conf=conf).getOrCreate()
    file_path = '/home/eleven/Documents'
    movieResourcesPath = file_path + "/Library/sampleData/movies.csv"
    movieSamples = spark.read.format('csv').option('header', 'true').load(movieResourcesPath)
    print("Raw Movie Samples:")
    movieSamples.show(10)
    movieSamples.printSchema()
    print("OneHotEncoder Example:")
    one_hot_encoder_example(movieSamples)
    print("MultiHotEncoder Example:")
    multi_hot_encoder_example(movieSamples)
    print("Numerical features Example:")
    ratingsResourcesPath = file_path + "/Library/sampleData/ratings.csv"
    ratingSamples = spark.read.format('csv').option('header', 'true').load(ratingsResourcesPath)
    rating_features(ratingSamples)
