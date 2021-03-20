import os
from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.ml.feature import BucketedRandomProjectionLSH
from pyspark.mllib.feature import Word2Vec
from pyspark.ml.linalg import Vectors
import random
from collections import defaultdict
import numpy as np
from pyspark.sql import functions as F
import redis


class UdfFunction:
    @staticmethod
    def sortF(movie_list, timestamp_list):
        """
        Input parameter:
            movie_list: List[str], list of movieIds
            timestamp_list: List[str], list of timestamps

        Return:
            List[str]: list of sorted movieIds

        sort by time and return the corresponding movie sequence
        eg:
            input: movie_list:[1,2,3]
                   timestamp_list:[1112486027,1212546032,1012486033]
            return [3,1,2]
        """
        pairs = []
        for m, t in zip(movie_list, timestamp_list):
            pairs.append((m, t))
        # sort by time
        pairs = sorted(pairs, key=lambda x: int(x[1]))  # direct string comparison is dangerous, "11" < "9" is True
        return [x[0] for x in pairs]


redisHost = "localhost"
redisPort = 6379
TTL = 60 * 60 * 23  # 24 hours


def processItemSequence(spark, rawSampleDataPath):
    # rating data
    ratingSamples = spark.read.format("csv").option("header", "true").load(rawSampleDataPath)
    print("ratingSample and schema:")
    ratingSamples.show(5)
    ratingSamples.printSchema()
    sortUdf = udf(UdfFunction.sortF, ArrayType(StringType()))
    userSeq = ratingSamples \
        .where(F.col("rating") >= 3.5) \
        .groupBy("userId") \
        .agg(sortUdf(F.collect_list("movieId"), F.collect_list("timestamp")).alias('movieIds')) \
        .withColumn("movieIdStr", array_join(F.col("movieIds"), " "))  # movieId as string
    print("movieIdStr:")
    userSeq.select("userId", "movieIdStr", "movieIds").show(10, truncate=False)
    # userSeq.limit(100).agg(F.collect_list("movieIds")).show()
    # print(userSeq.select('movieIdStr').rdd.map(lambda x: x[0].split(' ')).take(10))
    return userSeq.select('movieIdStr').rdd.map(lambda x: x[0].split(' '))


def processItemSequence1(spark, rawSampleDataPath):  # refactor above code without using UDF
    # rating data
    customStruct = StructType([
        StructField("userId", StringType()),
        StructField("movieId", StringType()),
        StructField("rating", FloatType()),
        StructField("timestamp", IntegerType())
    ])
    ratingSamples = spark.read.format("csv").option("header", "true").schema(customStruct).load(rawSampleDataPath)
    print("ratingSample and schema:")
    ratingSamples.show(5)
    ratingSamples.printSchema()
    userSeq = ratingSamples \
        .where(F.col("rating") >= 3.5) \
        .groupBy("userId") \
        .agg(F.sort_array(F.collect_list(F.struct("timestamp", "movieId"))).alias("sortedTimeAndMovieId")) \
        .withColumn("sortedMovieId", F.col("sortedTimeAndMovieId.movieId")) \
        .drop("sortedTimeAndMovieId")
    print("movieIdStr:")
    flatedMovieList = userSeq.select("sortedMovieId").rdd.flatMap(lambda x: x)  # same as rdd.map(lambda x: x[0])
    # flatedMovieList = userSeq.agg(F.collect_list("sortedMovieId")).rdd.flatMap(lambda x: x[0])  # this also works
    print(flatedMovieList.take(10))
    return flatedMovieList


def embeddingLSH(spark, movieEmbMap):
    movieEmbSeq = []
    for key, embedding_list in movieEmbMap.items():
        embedding_list = [np.float64(embedding) for embedding in embedding_list]
        movieEmbSeq.append((key, Vectors.dense(embedding_list)))
    movieEmbDF = spark.createDataFrame(movieEmbSeq).toDF("movieId", "emb")
    bucketProjectionLSH = BucketedRandomProjectionLSH(inputCol="emb", outputCol="bucketId", bucketLength=0.1,
                                                      numHashTables=3)
    bucketModel = bucketProjectionLSH.fit(movieEmbDF)
    embBucketResult = bucketModel.transform(movieEmbDF)
    print("movieId, emb, bucketId schema:")
    embBucketResult.printSchema()
    print("movieId, emb, bucketId data result:")
    embBucketResult.show(10, truncate=False)
    print("Approximately searching for 5 nearest neighbors of the sample embedding:")
    sampleEmb = Vectors.dense(0.795, 0.583, 1.120, 0.850, 0.174, -0.839, -0.0633, 0.249, 0.673, -0.237)
    sampleDF = spark.createDataFrame([("2", sampleEmb)]).toDF("movieId", "emb")
    bucketModel.transform(sampleDF).show(10, truncate=False)
    bucketModel.approxNearestNeighbors(movieEmbDF, sampleEmb, 5).show(truncate=False)


def trainItem2vec(spark, samples, embLength, embOutputPath, saveToRedis, redisKeyPrefix):
    word2vec = Word2Vec().setVectorSize(embLength).setWindowSize(5).setNumIterations(10)
    model = word2vec.fit(samples)
    synonyms = model.findSynonyms("158", 20)
    print("similarMovieId, cosineSimilarity")
    for synonym, cosineSimilarity in synonyms:
        print(synonym, cosineSimilarity)
    embOutputDir = '/'.join(embOutputPath.split('/')[:-1])
    if not os.path.exists(embOutputDir):
        os.makedirs(embOutputDir)
    with open(embOutputPath, 'w') as f:
        for movie_id in model.getVectors():  # model.getVectors() -> {movie_id: List[movie_embedding]}
            vectors = " ".join([str(emb) for emb in model.getVectors()[movie_id]])
            f.write(movie_id + ":" + vectors + "\n")

    # save to Redis
    if saveToRedis:
        r = redis.Redis(host=redisHost, port=redisPort, db=0, decode_responses=True)
        for movieId in model.getVectors():
            r.set(redisKeyPrefix + ":" + movieId, " ".join([str(emb) for emb in model.getVectors()[movie_id]]), ex=TTL)

    embeddingLSH(spark, model.getVectors())
    return model


def generate_pair(x):
    # eg:
    # watch sequence:['858', '50', '593', '457']
    # return:[['858', '50'],['50', '593'],['593', '457']]
    pairSeq = []
    previousItem = ''
    for item in x:
        if not previousItem:
            previousItem = item
        else:
            pairSeq.append((previousItem, item))
            previousItem = item
    return pairSeq


def generateTransitionMatrix(samples):
    pairSamples = samples.flatMap(lambda x: generate_pair(x))
    pairCountMap = pairSamples.countByValue()  # return {value: count}
    pairTotalCount = 0
    transitionCountMatrix = defaultdict(dict)
    itemCountMap = defaultdict(int)
    for key, cnt in pairCountMap.items():
        key1, key2 = key
        transitionCountMatrix[key1][key2] = cnt
        itemCountMap[key1] += cnt
        pairTotalCount += cnt
    transitionMatrix = defaultdict(dict)
    itemDistribution = defaultdict(dict)
    for key1, transitionMap in transitionCountMatrix.items():
        for key2, cnt in transitionMap.items():
            # node transition probability
            transitionMatrix[key1][key2] = transitionCountMatrix[key1][key2] / itemCountMap[key1]
    for itemid, cnt in itemCountMap.items():
        itemDistribution[itemid] = cnt / pairTotalCount
    return transitionMatrix, itemDistribution


def oneRandomWalk(transitionMatrix, itemDistribution, sampleLength):
    sample = []
    # pick the first element
    randomDouble = random.random()
    firstItem = ""
    accumulateProb = 0.0
    for item, prob in itemDistribution.items():
        accumulateProb += prob
        if accumulateProb >= randomDouble:
            firstItem = item
            break
    sample.append(firstItem)
    curElement = firstItem
    i = 1
    while i < sampleLength:
        if (curElement not in itemDistribution) or (curElement not in transitionMatrix):
            break
        probDistribution = transitionMatrix[curElement]
        randomDouble = random.random()
        accumulateProb = 0.0
        for item, prob in probDistribution.items():
            accumulateProb += prob
            if accumulateProb >= randomDouble:
                curElement = item
                break
        sample.append(curElement)
        i += 1
    return sample


def randomWalk(transitionMatrix, itemDistribution, sampleCount, sampleLength):
    samples = []
    for i in range(sampleCount):
        samples.append(oneRandomWalk(transitionMatrix, itemDistribution, sampleLength))
    return samples


def graphEmb(samples, spark, embLength, embOutputFilename, saveToRedis, redisKeyPrefix):
    transitionMatrix, itemDistribution = generateTransitionMatrix(samples)
    sampleCount = 20000
    sampleLength = 10
    newSamples = randomWalk(transitionMatrix, itemDistribution, sampleCount, sampleLength)
    rddSamples = spark.sparkContext.parallelize(newSamples)
    return trainItem2vec(spark, rddSamples, embLength, embOutputFilename, saveToRedis, redisKeyPrefix)


def generateUserEmb(spark, rawSampleDataPath, model, embLength, embOutputPath, saveToRedis, redisKeyPrefix):
    customStruct = StructType([
        StructField("userId", StringType()),
        StructField("movieId", StringType()),
        StructField("rating", FloatType()),
        StructField("timestamp", IntegerType())
    ])
    ratingSamples = spark.read.format("csv").option("header", "true").schema(customStruct).load(rawSampleDataPath)
    Vectors_list = []
    for key, value in model.getVectors().items():
        Vectors_list.append((key, list(value)))
    fields = [
        StructField('movieId', StringType(), False),
        StructField('emb', ArrayType(FloatType()), False)
    ]
    schema = StructType(fields)
    Vectors_df = spark.createDataFrame(Vectors_list, schema=schema)
    ratingSamples = ratingSamples.join(Vectors_df, on='movieId', how='inner')
    # Method #1 to calculate userEmbedding by averaging movieEmbedding
    # result = ratingSamples.select('userId', 'emb').rdd.map(lambda x: (x[0], (x[1], 1))) \
    #     .reduceByKey(lambda a, b: ([a[0][i] + b[0][i] for i in range(embLength)], a[1]+b[1])) \
    #     .map(lambda x: (x[0], [x[1][0][i]/x[1][1] for i in range(embLength)])).collect()

    # Method #2 for userEmbedding calculation
    result = ratingSamples.select('userId', 'emb').groupBy('userId').agg(
        F.array(*[F.avg(F.col("emb")[i]) for i in range(embLength)]).alias('avgEmb')).rdd \
        .map(lambda x: (x[0], x[1])).collect()

    with open(embOutputPath, 'w') as f:
        for row in result:
            vectors = " ".join([str(emb) for emb in row[1]])
            f.write(row[0] + ":" + vectors + "\n")
    if saveToRedis:
        r = redis.Redis(host=redisHost, port=redisPort, db=0, decode_responses=True)
        for userId, embedding in result:
            r.set(redisKeyPrefix + ":" + userId, " ".join([str(emb) for emb in embedding]), ex=TTL)


if __name__ == '__main__':
    conf = SparkConf().setAppName('ctrModel').setMaster('local')
    spark = SparkSession.builder.config(conf=conf).getOrCreate()
    # Change to your own filepath
    file_path = '../../../../../../../src/main/resources/'
    rawSampleDataPath = file_path + "/webroot/sampledata/ratings.csv"
    outputDir = file_path + "/webroot/modeldata2"
    embLength = 10
    samples = processItemSequence1(spark, rawSampleDataPath)
    model = trainItem2vec(spark, samples, embLength,  # item2Vec embedding
                          embOutputPath=outputDir + "/item2vecEmb.csv", saveToRedis=True,
                          redisKeyPrefix="i2vEmb")
    graphEmb(samples, spark, embLength, embOutputFilename=outputDir + "/itemGraphEmb.csv",  # Deep-walk graph embedding
             saveToRedis=False, redisKeyPrefix="graphEmb")
    generateUserEmb(spark, rawSampleDataPath, model, embLength,
                    embOutputPath=outputDir + "/userEmb.csv", saveToRedis=True,
                    redisKeyPrefix="uEmb")
