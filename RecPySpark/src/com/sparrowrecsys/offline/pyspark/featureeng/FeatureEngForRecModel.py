from pyspark import SparkContext, SparkConf
import redis
from pyspark.sql import SparkSession
import pyspark.sql as sql
from pyspark.sql.functions import *
from pyspark.sql.types import *
from collections import defaultdict
from pyspark.sql import functions as F

NUMBER_PRECISION = 2
host = 'localhost'
port = 6379


def addSampleLabel(ratingSamples):
    ratingSamples.show(5, truncate=False)
    ratingSamples.printSchema()
    sampleCount = ratingSamples.count()
    ratingSamples.groupBy('rating').count().orderBy('rating').withColumn('percentage',
                                                                         F.col('count') / sampleCount).show()
    ratingSamples = ratingSamples.withColumn('label', when(F.col('rating') >= 3.5, 1).otherwise(0))
    return ratingSamples


def extractReleaseYearUdf(title):
    # add realease year
    if not title or len(title.strip()) < 6:
        return 1990
    else:
        yearStr = title.strip()[-5:-1]
    return int(yearStr)


def addMovieFeatures(movieSamples, ratingSamplesWithLabel):
    # add movie basic features
    samplesWithMovies1 = ratingSamplesWithLabel.join(movieSamples, on=['movieId'], how='left')
    # add releaseYear,title
    samplesWithMovies2 = samplesWithMovies1.withColumn('releaseYear',
                                                       udf(extractReleaseYearUdf, IntegerType())('title')) \
        .withColumn('title', udf(lambda x: x.strip()[:-6].strip(), StringType())('title')) \
        .drop('title')
    # split genres
    samplesWithMovies3 = samplesWithMovies2.withColumn('movieGenre1', split(F.col('genres'), "\\|")[0]) \
        .withColumn('movieGenre2', split(F.col('genres'), "\\|")[1]) \
        .withColumn('movieGenre3', split(F.col('genres'), "\\|")[2])
    # add rating features
    movieRatingFeatures = samplesWithMovies3.groupBy('movieId').agg(F.count(F.lit(1)).alias('movieRatingCount'),
                                                                    format_number(F.avg(F.col('rating')),
                                                                                  NUMBER_PRECISION).alias(
                                                                        'movieAvgRating'),
                                                                    F.stddev(F.col('rating')).alias(
                                                                        'movieRatingStddev')).fillna(0) \
        .withColumn('movieRatingStddev', format_number(F.col('movieRatingStddev'), NUMBER_PRECISION))
    # join movie rating features
    samplesWithMovies4 = samplesWithMovies3.join(movieRatingFeatures, on=['movieId'], how='left')
    samplesWithMovies4.printSchema()
    samplesWithMovies4.show(5, truncate=False)
    return samplesWithMovies4


def extractGenres(genres_list):
    '''
    pass in a list which format like ["Action|Adventure|Sci-Fi|Thriller", "Crime|Horror|Thriller"]
    count by each genre，return genre_list in reverse order
    eg:
    (('Thriller',2),('Action',1),('Sci-Fi',1),('Horror', 1), ('Adventure',1),('Crime',1))
    return:['Thriller','Action','Sci-Fi','Horror','Adventure','Crime']
    '''
    genres_dict = defaultdict(int)
    for genres in genres_list:
        for genre in genres.split('|'):
            genres_dict[genre] += 1
    sortedGenres = sorted(genres_dict.items(), key=lambda x: x[1], reverse=True)
    return [x[0] for x in sortedGenres]


def addUserFeatures(samplesWithMovieFeatures):
    extractGenresUdf = udf(extractGenres, ArrayType(StringType()))
    samplesWithUserFeatures = samplesWithMovieFeatures \
        .withColumn('userPositiveHistory',
                    F.collect_list(when(F.col('label') == 1, F.col('movieId')).otherwise(F.lit(None))).over(
                        sql.Window.partitionBy("userId").orderBy(F.col("timestamp")).rowsBetween(-100, -1))) \
        .withColumn("userPositiveHistory", reverse(F.col("userPositiveHistory"))) \
        .withColumn('userRatedMovie1', F.col('userPositiveHistory')[0]) \
        .withColumn('userRatedMovie2', F.col('userPositiveHistory')[1]) \
        .withColumn('userRatedMovie3', F.col('userPositiveHistory')[2]) \
        .withColumn('userRatedMovie4', F.col('userPositiveHistory')[3]) \
        .withColumn('userRatedMovie5', F.col('userPositiveHistory')[4]) \
        .withColumn('userRatingCount',
                    F.count(F.lit(1)).over(sql.Window.partitionBy('userId').orderBy('timestamp').rowsBetween(-100, -1))) \
        .withColumn('userAvgReleaseYear', F.avg(F.col('releaseYear')).over(
        sql.Window.partitionBy('userId').orderBy('timestamp').rowsBetween(-100, -1)).cast(IntegerType())) \
        .withColumn('userReleaseYearStddev', F.stddev(F.col("releaseYear")).over(
        sql.Window.partitionBy('userId').orderBy('timestamp').rowsBetween(-100, -1))) \
        .withColumn("userAvgRating", format_number(
        F.avg(F.col("rating")).over(sql.Window.partitionBy('userId').orderBy('timestamp').rowsBetween(-100, -1)),
        NUMBER_PRECISION)) \
        .withColumn("userRatingStddev", F.stddev(F.col("rating")).over(
        sql.Window.partitionBy('userId').orderBy('timestamp').rowsBetween(-100, -1))) \
        .withColumn("userGenres", extractGenresUdf(
        F.collect_list(when(F.col('label') == 1, F.col('genres')).otherwise(F.lit(None))).over(
            sql.Window.partitionBy('userId').orderBy('timestamp').rowsBetween(-100, -1)))) \
        .withColumn("userRatingStddev", format_number(F.col("userRatingStddev"), NUMBER_PRECISION)) \
        .withColumn("userReleaseYearStddev", format_number(F.col("userReleaseYearStddev"), NUMBER_PRECISION)) \
        .withColumn("userGenre1", F.col("userGenres")[0]) \
        .withColumn("userGenre2", F.col("userGenres")[1]) \
        .withColumn("userGenre3", F.col("userGenres")[2]) \
        .withColumn("userGenre4", F.col("userGenres")[3]) \
        .withColumn("userGenre5", F.col("userGenres")[4]) \
        .drop("genres", "userGenres", "userPositiveHistory") \
        .filter(F.col("userRatingCount") > 1)
    samplesWithUserFeatures.printSchema()
    # samplesWithUserFeatures.show(10)
    # samplesWithUserFeatures.filter(samplesWithMovieFeatures['userId'] == 1).orderBy(F.col('timestamp').asc()).show(
    #     truncate=False)
    return samplesWithUserFeatures


def splitAndSaveTrainingTestSamples(samplesWithUserFeatures, file_path):
    smallSamples = samplesWithUserFeatures.sample(0.1)
    training, test = smallSamples.randomSplit((0.8, 0.2))
    trainingSavePath = file_path + '/trainingSamples'
    testSavePath = file_path + '/testSamples'
    training.repartition(1).write.option("header", "true").mode('overwrite') \
        .csv(trainingSavePath)
    test.repartition(1).write.option("header", "true").mode('overwrite') \
        .csv(testSavePath)


def splitAndSaveTrainingTestSamplesByTimeStamp(samplesWithUserFeatures, file_path):
    smallSamples = samplesWithUserFeatures.sample(0.1).withColumn("timestampLong", F.col("timestamp").cast(LongType()))
    quantile = smallSamples.stat.approxQuantile("timestampLong", [0.8], 0.05)
    splitTimestamp = quantile[0]
    training = smallSamples.where(F.col("timestampLong") <= splitTimestamp).drop("timestampLong")
    test = smallSamples.where(F.col("timestampLong") > splitTimestamp).drop("timestampLong")
    trainingSavePath = file_path + '/trainingSamples'
    testSavePath = file_path + '/testSamples'
    training.repartition(1).write.option("header", "true").mode('overwrite') \
        .csv(trainingSavePath)
    test.repartition(1).write.option("header", "true").mode('overwrite') \
        .csv(testSavePath)

def extractAndSaveUserFeaturesToRedis(samples, re):
    userLatestSamples = samples.withColumn("userRowNum",
                                           F.count(F.lit(1)).over(
                                               sql.Window.partitionBy("userId").orderBy(F.desc(F.col("timestamp"))))) \
        .filter(F.col("userRowNum") == 1) \
        .select("userId", "userRatedMovie1", "userRatedMovie2", "userRatedMovie3", "userRatedMovie4",
                "userRatedMovie5", "userRatingCount", "userAvgReleaseYear", "userReleaseYearStddev",
                "userAvgRating", "userRatingStddev", "userGenre1", "userGenre2",
                "userGenre3", "userGenre4", "userGenre5") \
        .na.fill("")

    userFeaturePrefix = "uf:"

    sampleArray = userLatestSamples.collect()
    print("total movie size:" + sampleArray.length)
    # for sample <- sampleArray
    #     userKey = userFeaturePrefix + sample.getAs[String]("userId")
    #     valueMap = mutable.Map[String, String]()
    #     valueMap("userRatedMovie1") = sample.getAs[String]("userRatedMovie1")
    #     valueMap("userRatedMovie2") = sample.getAs[String]("userRatedMovie2")
    #     valueMap("userRatedMovie3") = sample.getAs[String]("userRatedMovie3")
    #     valueMap("userRatedMovie4") = sample.getAs[String]("userRatedMovie4")
    #     valueMap("userRatedMovie5") = sample.getAs[String]("userRatedMovie5")
    #     valueMap("userGenre1") = sample.getAs[String]("userGenre1")
    #     valueMap("userGenre2") = sample.getAs[String]("userGenre2")
    #     valueMap("userGenre3") = sample.getAs[String]("userGenre3")
    #     valueMap("userGenre4") = sample.getAs[String]("userGenre4")
    #     valueMap("userGenre5") = sample.getAs[String]("userGenre5")
    #     valueMap("userRatingCount") = sample.getAs[Long]("userRatingCount").toString
    #     valueMap("userAvgReleaseYear") = sample.getAs[Int]("userAvgReleaseYear").toString
    #     valueMap("userReleaseYearStddev") = sample.getAs[String]("userReleaseYearStddev")
    #     valueMap("userAvgRating") = sample.getAs[String]("userAvgRating")
    #     valueMap("userRatingStddev") = sample.getAs[String]("userRatingStddev")
    # re.hmset(userKey, JavaConversions.mapAsJavaMap(valueMap))


def extractAndSaveMovieFeaturesToRedis(samples, re):
    movieLatestSamples = samples.withColumn("movieRowNum", F.count(F.lit(1)).over(
        sql.Window.partitionBy("movieId").orderBy(F.desc(F.col("timestamp"))))) \
        .filter(col("movieRowNum") == 1) \
        .select("movieId", "releaseYear",
                "movieGenre1", "movieGenre2", "movieGenre3",
                "movieRatingCount", "movieAvgRating", "movieRatingStddev") \
        .na.fill("")
    movieFeaturePrefix = "mf:"
    sampleArray = movieLatestSamples.collect()
    print("total movie size:" + sampleArray.length)
    # for sample <- sampleArray
    #     movieKey = movieFeaturePrefix + sample.getAs[String]("movieId")
    #     valueMap = mutable.Map[String, String]()
    #     valueMap("movieGenre1") = sample.getAs[String]("movieGenre1")
    #     valueMap("movieGenre2") = sample.getAs[String]("movieGenre2")
    #     valueMap("movieGenre3") = sample.getAs[String]("movieGenre3")
    #     valueMap("movieRatingCount") = sample.getAs[Long]("movieRatingCount").toString
    #     valueMap("releaseYear") = sample.getAs[Int]("releaseYear").toString
    #     valueMap("movieAvgRating") = sample.getAs[String]("movieAvgRating")
    #     valueMap("movieRatingStddev") = sample.getAs[String]("movieRatingStddev")
    # re.hmset(movieKey, JavaConversions.mapAsJavaMap(valueMap))


if __name__ == '__main__':
    pool = redis.ConnectionPool(host=host, port=port)
    r = redis.Redis(connection_pool=pool)
    conf = SparkConf().setAppName('featureEngineering').setMaster('local')
    spark = SparkSession.builder.config(conf=conf).config(
        'spark.sql.debug.maxToStringFields', 5000).config('spark.debug.maxToStringFields', 5000).getOrCreate()
    file_path = '/home/eleven/Documents'
    movieResourcesPath = file_path + "/Library/sampleData/movies.csv"
    ratingsResourcesPath = file_path + "/Library/sampleData/ratings.csv"
    movieSamples = spark.read.format('csv').option('header', 'true').load(movieResourcesPath)
    ratingSamples = spark.read.format('csv').option('header', 'true').load(ratingsResourcesPath)
    ratingSamplesWithLabel = addSampleLabel(ratingSamples)
    ratingSamplesWithLabel.show(10, truncate=False)
    samplesWithMovieFeatures = addMovieFeatures(movieSamples, ratingSamplesWithLabel)
    samplesWithUserFeatures = addUserFeatures(samplesWithMovieFeatures)
    # save samples as csv format
    splitAndSaveTrainingTestSamples(samplesWithUserFeatures, file_path + "/Library/sampleData")
    # splitAndSaveTrainingTestSamplesByTimeStamp(samplesWithUserFeatures, file_path + "/webroot/sampledata")

    # extractAndSaveUserFeaturesToRedis(samplesWithUserFeatures, r)
    # extractAndSaveMovieFeaturesToRedis(samplesWithUserFeatures, r)
