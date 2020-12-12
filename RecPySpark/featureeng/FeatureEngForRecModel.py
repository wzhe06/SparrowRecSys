from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
import pyspark.sql as sql
from pyspark.sql.functions import *
from pyspark.sql.types import*
from collections import defaultdict
NUMBER_PRECISION=2

def addSampleLabel(ratingSamples):
    ratingSamples.show(5,truncate=False)
    ratingSamples.printSchema()
    sampleCount = ratingSamples.count()
    ratingSamples.groupBy('rating').count().orderBy('rating').withColumn('percentage',
                                                                         col('count') / sampleCount).show()
    ratingSamples = ratingSamples.withColumn('label', when(col('rating') >= 3.5, 1).otherwise(0))
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
    samplesWithMovies2 = samplesWithMovies1.withColumn('releaseYear',udf(extractReleaseYearUdf, IntegerType())('title')) \
        .withColumn('title', udf(lambda x: x.strip()[:-6].strip(), StringType())('title')) \
        .drop('title')
    # split genres
    samplesWithMovies3 = samplesWithMovies2.withColumn('movieGenre1', split(col('genres'), "\\|")[0]) \
        .withColumn('movieGenre2', split(col('genres'), "\\|")[1]) \
        .withColumn('movieGenre3', split(col('genres'), "\\|")[2])
    # add rating features
    movieRatingFeatures = samplesWithMovies3.groupBy('movieId').agg(count(lit(1)).alias('movieRatingCount'),
        format_number(avg(col('rating')),NUMBER_PRECISION).alias('movieAvgRating'),
        stddev(col('rating')).alias('movieRatingStddev')).fillna(0) \
        .withColumn('movieRatingStddev', format_number(col('movieRatingStddev'), NUMBER_PRECISION))
    #join movie rating features
    samplesWithMovies4 = samplesWithMovies3.join(movieRatingFeatures,on = ['movieId'],how='left')
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
        for genre in  genres.split('|'):
            genres_dict[genre] +=1
    sortedGenres = sorted(genres_dict.items(),key=lambda x:x[1],reverse=True)
    return [x[0] for x in sortedGenres]


def addUserFeatures(samplesWithMovieFeatures):
    extractGenresUdf = udf(extractGenres, ArrayType(StringType()))
    samplesWithUserFeatures = samplesWithMovieFeatures \
        .withColumn('userPositiveHistory',
                    collect_list(when(col('label') == 1, col('movieId')).otherwise(lit(None))).over(
                        sql.Window.partitionBy("userId").orderBy(col("timestamp")).rowsBetween(-100, -1))) \
        .withColumn("userPositiveHistory", reverse(col("userPositiveHistory"))) \
        .withColumn('userRatedMovie1', col('userPositiveHistory')[0]) \
        .withColumn('userRatedMovie2', col('userPositiveHistory')[1]) \
        .withColumn('userRatedMovie3', col('userPositiveHistory')[2]) \
        .withColumn('userRatedMovie4', col('userPositiveHistory')[3]) \
        .withColumn('userRatedMovie5', col('userPositiveHistory')[4]) \
        .withColumn('userRatingCount',
                    count(lit(1)).over(sql.Window.partitionBy('userId').orderBy('timestamp').rowsBetween(-100, -1))) \
        .withColumn('userAvgReleaseYear', avg(col('releaseYear')).over(
        sql.Window.partitionBy('userId').orderBy('timestamp').rowsBetween(-100, -1)).cast(IntegerType())) \
        .withColumn('userReleaseYearStddev', stddev(col("releaseYear")).over(
        sql.Window.partitionBy('userId').orderBy('timestamp').rowsBetween(-100, -1))) \
        .withColumn("userAvgRating", format_number(
        avg(col("rating")).over(sql.Window.partitionBy('userId').orderBy('timestamp').rowsBetween(-100, -1)),
        NUMBER_PRECISION)) \
        .withColumn("userRatingStddev", stddev(col("rating")).over(
        sql.Window.partitionBy('userId').orderBy('timestamp').rowsBetween(-100, -1))) \
        .withColumn("userGenres", extractGenresUdf(
        collect_list(when(col('label') == 1, col('genres')).otherwise(lit(None))).over(
            sql.Window.partitionBy('userId').orderBy('timestamp').rowsBetween(-100, -1)))) \
        .withColumn("userRatingStddev", format_number(col("userRatingStddev"), NUMBER_PRECISION)) \
        .withColumn("userReleaseYearStddev", format_number(col("userReleaseYearStddev"), NUMBER_PRECISION)) \
        .withColumn("userGenre1", col("userGenres")[0]) \
        .withColumn("userGenre2", col("userGenres")[1]) \
        .withColumn("userGenre3", col("userGenres")[2]) \
        .withColumn("userGenre4", col("userGenres")[3]) \
        .withColumn("userGenre5", col("userGenres")[4]) \
        .drop("genres", "userGenres", "userPositiveHistory") \
        .filter(col("userRatingCount") > 1)
    samplesWithUserFeatures.printSchema()
    samplesWithUserFeatures.show(10)
    samplesWithUserFeatures.filter(samplesWithMovieFeatures['userId'] == 1).orderBy(col('timestamp').asc()).show(truncate=False)
    return samplesWithUserFeatures


def splitAndSaveTrainingTestSamples(samplesWithUserFeatures, file_path):
    smallSamples = samplesWithUserFeatures.sample(0.1)
    training, test = smallSamples.randomSplit((0.8, 0.2))
    trainingSavePath = file_path +'/trainingSamples'
    testSavePath = file_path +'/testSamples'
    training.repartition(1).write.option("header", "true").mode('overwrite') \
        .csv(trainingSavePath)
    test.repartition(1).write.option("header", "true").mode('overwrite') \
        .csv(testSavePath)

def splitAndSaveTrainingTestSamplesByTimeStamp(samplesWithUserFeatures, file_path):
    smallSamples = samplesWithUserFeatures.sample(0.1).withColumn("timestampLong", col("timestamp").cast(LongType()))
    quantile = smallSamples.stat.approxQuantile("timestampLong", [0.8], 0.05)
    splitTimestamp = quantile[0]
    training = smallSamples.where(col("timestampLong") <= splitTimestamp).drop("timestampLong")
    test = smallSamples.where(col("timestampLong") > splitTimestamp).drop("timestampLong")
    trainingSavePath = file_path +'/trainingSamples'
    testSavePath = file_path +'/testSamples'
    training.repartition(1).write.option("header", "true").mode('overwrite') \
        .csv(trainingSavePath)
    test.repartition(1).write.option("header", "true").mode('overwrite') \
        .csv(testSavePath)


if __name__ =='__main__':
    conf = SparkConf().setAppName('featureEngineering').setMaster('local')
    spark = SparkSession.builder.config(conf=conf).getOrCreate()
    file_path = 'file:///home/hadoop/SparrowRecSys/src/main/resources'
    movieResourcesPath = file_path + "/webroot/sampledata/movies.csv"
    ratingsResourcesPath = file_path + "/webroot/sampledata/ratings.csv"
    movieSamples = spark.read.format('csv').option('header', 'true').load(movieResourcesPath)
    ratingSamples = spark.read.format('csv').option('header', 'true').load(ratingsResourcesPath)
    ratingSamplesWithLabel = addSampleLabel(ratingSamples)
    ratingSamplesWithLabel.show(10, truncate=False)
    samplesWithMovieFeatures = addMovieFeatures(movieSamples, ratingSamplesWithLabel)
    samplesWithUserFeatures = addUserFeatures(samplesWithMovieFeatures)
    #save samples as csv format
    splitAndSaveTrainingTestSamples(samplesWithUserFeatures, file_path + "/webroot/sampledata")
    #splitAndSaveTrainingTestSamplesByTimeStamp(samplesWithUserFeatures, file_path + "/webroot/sampledata")
