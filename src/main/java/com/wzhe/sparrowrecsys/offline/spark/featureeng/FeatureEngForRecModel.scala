package com.wzhe.sparrowrecsys.offline.spark.featureeng

import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkConf
import org.apache.spark.sql.expressions.{UserDefinedFunction, Window}
import org.apache.spark.sql.functions.{format_number, _}
import org.apache.spark.sql.{DataFrame, SparkSession}
import redis.clients.jedis.Jedis
import redis.clients.jedis.params.SetParams

import scala.collection.immutable.ListMap
import scala.collection.{JavaConversions, mutable}

object FeatureEngForRecModel {

  val NUMBER_PRECISION = 2
  val redisEndpoint = "localhost"
  val redisPort = 6379

  def addSampleLabel(ratingSamples:DataFrame): DataFrame ={
    ratingSamples.show(10, truncate = false)
    ratingSamples.printSchema()
    val sampleCount = ratingSamples.count()
    ratingSamples.groupBy(col("rating")).count().orderBy(col("rating"))
      .withColumn("percentage", col("count")/sampleCount).show(100,truncate = false)

    ratingSamples.withColumn("label", when(col("rating") >= 3.5, 1).otherwise(0))
  }

  def addMovieFeatures(movieSamples:DataFrame, ratingSamples:DataFrame): DataFrame ={

    //add movie basic features
    val samplesWithMovies1 = ratingSamples.join(movieSamples, Seq("movieId"), "left")
    //add release year
    val extractReleaseYearUdf = udf({(title: String) => {
      if (null == title || title.trim.length < 6) {
        1990 // default value
      }
      else {
        val yearString = title.trim.substring(title.length - 5, title.length - 1)
        yearString.toInt
      }
    }})

    //add title
    val extractTitleUdf = udf({(title: String) => {title.trim.substring(0, title.trim.length - 6).trim}})

    val samplesWithMovies2 = samplesWithMovies1.withColumn("releaseYear", extractReleaseYearUdf(col("title")))
      .withColumn("title", extractTitleUdf(col("title")))
      .drop("title")  //title is useless currently

    //split genres
    val samplesWithMovies3 = samplesWithMovies2.withColumn("movieGenre1",split(col("genres"),"\\|").getItem(0))
      .withColumn("movieGenre2",split(col("genres"),"\\|").getItem(1))
      .withColumn("movieGenre3",split(col("genres"),"\\|").getItem(2))

    //add rating features
    val movieRatingFeatures = samplesWithMovies3.groupBy(col("movieId"))
      .agg(count(lit(1)).as("movieRatingCount"),
        format_number(avg(col("rating")), NUMBER_PRECISION).as("movieAvgRating"),
        stddev(col("rating")).as("movieRatingStddev"))
    .na.fill(0).withColumn("movieRatingStddev",format_number(col("movieRatingStddev"), NUMBER_PRECISION))


    //join movie rating features
    val samplesWithMovies4 = samplesWithMovies3.join(movieRatingFeatures, Seq("movieId"), "left")
    samplesWithMovies4.printSchema()
    samplesWithMovies4.show(10, truncate = false)

    samplesWithMovies4
  }

  val extractGenres: UserDefinedFunction = udf { (genreArray: Seq[String]) => {
    val genreMap = mutable.Map[String, Int]()
    genreArray.foreach((element:String) => {
      val genres = element.split("\\|")
      genres.foreach((oneGenre:String) => {
        genreMap(oneGenre) = genreMap.getOrElse[Int](oneGenre, 0)  + 1
      })
    })
    val sortedGenres = ListMap(genreMap.toSeq.sortWith(_._2 > _._2):_*)
    sortedGenres.keys.toSeq
  }}

  def addUserFeatures(ratingSamples:DataFrame): DataFrame ={
    val samplesWithUserFeatures = ratingSamples
      .withColumn("userPositiveHistory", collect_list(when(col("label") === 1, col("movieId")).otherwise(lit(null)))
        .over(Window.partitionBy("userId")
          .orderBy(col("timestamp")).rowsBetween(-100, -1)))
      .withColumn("userPositiveHistory", reverse(col("userPositiveHistory")))
      .withColumn("userRatedMovie1",col("userPositiveHistory").getItem(0))
      .withColumn("userRatedMovie2",col("userPositiveHistory").getItem(1))
      .withColumn("userRatedMovie3",col("userPositiveHistory").getItem(2))
      .withColumn("userRatedMovie4",col("userPositiveHistory").getItem(3))
      .withColumn("userRatedMovie5",col("userPositiveHistory").getItem(4))
      .withColumn("userRatingCount", count(lit(1))
        .over(Window.partitionBy("userId")
          .orderBy(col("timestamp")).rowsBetween(-100, -1)))
      .withColumn("userAvgReleaseYear", format_number(avg(col("releaseYear"))
        .over(Window.partitionBy("userId")
          .orderBy(col("timestamp")).rowsBetween(-100, -1)), NUMBER_PRECISION))
      .withColumn("userReleaseYearStddev", stddev(col("releaseYear"))
        .over(Window.partitionBy("userId")
          .orderBy(col("timestamp")).rowsBetween(-100, -1)))
      .withColumn("userAvgRating", format_number(avg(col("rating"))
        .over(Window.partitionBy("userId")
          .orderBy(col("timestamp")).rowsBetween(-100, -1)), NUMBER_PRECISION))
      .withColumn("userRatingStddev", stddev(col("rating"))
        .over(Window.partitionBy("userId")
          .orderBy(col("timestamp")).rowsBetween(-100, -1)))
      .withColumn("userGenres", extractGenres(collect_list(when(col("label") === 1, col("genres")).otherwise(lit(null)))
        .over(Window.partitionBy("userId")
          .orderBy(col("timestamp")).rowsBetween(-100, -1))))
      .na.fill(0)
      .withColumn("userRatingStddev",format_number(col("userRatingStddev"), NUMBER_PRECISION))
      .withColumn("userReleaseYearStddev",format_number(col("userReleaseYearStddev"), NUMBER_PRECISION))
      .withColumn("userGenre1",col("userGenres").getItem(0))
      .withColumn("userGenre2",col("userGenres").getItem(1))
      .withColumn("userGenre3",col("userGenres").getItem(2))
      .withColumn("userGenre4",col("userGenres").getItem(3))
      .withColumn("userGenre5",col("userGenres").getItem(4))
      .drop("genres", "userGenres", "userPositiveHistory")
      .filter(col("userRatingCount") > 1)

    samplesWithUserFeatures.printSchema()
    samplesWithUserFeatures.show(100, truncate = false)

    samplesWithUserFeatures
  }

  def extractAndSaveMovieFeaturesToRedis(samples:DataFrame): DataFrame = {
    val movieLatestSamples = samples.withColumn("movieRowNum", row_number()
      .over(Window.partitionBy("movieId")
        .orderBy(col("timestamp").desc)))
      .filter(col("movieRowNum") === 1)
      .select("movieId","releaseYear", "movieGenre1","movieGenre2","movieGenre3","movieRatingCount",
        "movieAvgRating", "movieRatingStddev")
      .na.fill("")

    movieLatestSamples.printSchema()
    movieLatestSamples.show(100, truncate = false)

    val movieFeaturePrefix = "mf:"

    val redisClient = new Jedis(redisEndpoint, redisPort)
    val params = SetParams.setParams()
    //set ttl to 24hs * 30
    params.ex(60 * 60 * 24 * 30)
    val sampleArray = movieLatestSamples.collect()
    println("total movie size:" + sampleArray.length)
    var insertedMovieNumber = 0
    val movieCount = sampleArray.length
    for (sample <- sampleArray){
      val movieKey = movieFeaturePrefix + sample.getAs[String]("movieId")
      val valueMap = mutable.Map[String, String]()
      valueMap("movieGenre1") = sample.getAs[String]("movieGenre1")
      valueMap("movieGenre2") = sample.getAs[String]("movieGenre2")
      valueMap("movieGenre3") = sample.getAs[String]("movieGenre3")
      valueMap("movieRatingCount") = sample.getAs[Long]("movieRatingCount").toString
      valueMap("releaseYear") = sample.getAs[Int]("releaseYear").toString
      valueMap("movieAvgRating") = sample.getAs[String]("movieAvgRating")
      valueMap("movieRatingStddev") = sample.getAs[String]("movieRatingStddev")

      redisClient.hset(movieKey, JavaConversions.mapAsJavaMap(valueMap))
      insertedMovieNumber += 1
      if (insertedMovieNumber % 100 ==0){
        println(insertedMovieNumber + "/" + movieCount + "...")
      }
    }

    redisClient.close()
    movieLatestSamples
  }


  def extractAndSaveUserFeaturesToRedis(samples:DataFrame): DataFrame = {
    val userLatestSamples = samples.withColumn("userRowNum", row_number()
      .over(Window.partitionBy("userId")
        .orderBy(col("timestamp").desc)))
      .filter(col("userRowNum") === 1)
      .select("userId","userRatedMovie1", "userRatedMovie2","userRatedMovie3","userRatedMovie4","userRatedMovie5",
        "userRatingCount", "userAvgReleaseYear", "userReleaseYearStddev", "userAvgRating", "userRatingStddev",
        "userGenre1", "userGenre2","userGenre3","userGenre4","userGenre5")
      .na.fill("")

    userLatestSamples.printSchema()
    userLatestSamples.show(100, truncate = false)

    val userFeaturePrefix = "uf:"

    val redisClient = new Jedis(redisEndpoint, redisPort)
    val params = SetParams.setParams()
    //set ttl to 24hs * 30
    params.ex(60 * 60 * 24 * 30)
    val sampleArray = userLatestSamples.collect()
    println("total user size:" + sampleArray.length)
    var insertedUserNumber = 0
    val userCount = sampleArray.length
    for (sample <- sampleArray){
      val userKey = userFeaturePrefix + sample.getAs[String]("userId")
      val valueMap = mutable.Map[String, String]()
      valueMap("userRatedMovie1") = sample.getAs[String]("userRatedMovie1")
      valueMap("userRatedMovie2") = sample.getAs[String]("userRatedMovie2")
      valueMap("userRatedMovie3") = sample.getAs[String]("userRatedMovie3")
      valueMap("userRatedMovie4") = sample.getAs[String]("userRatedMovie4")
      valueMap("userRatedMovie5") = sample.getAs[String]("userRatedMovie5")
      valueMap("userGenre1") = sample.getAs[String]("userGenre1")
      valueMap("userGenre2") = sample.getAs[String]("userGenre2")
      valueMap("userGenre3") = sample.getAs[String]("userGenre3")
      valueMap("userGenre4") = sample.getAs[String]("userGenre4")
      valueMap("userGenre5") = sample.getAs[String]("userGenre5")
      valueMap("userRatingCount") = sample.getAs[Long]("userRatingCount").toString
      valueMap("userAvgReleaseYear") = sample.getAs[String]("userAvgReleaseYear")
      valueMap("userReleaseYearStddev") = sample.getAs[String]("userReleaseYearStddev")
      valueMap("userAvgRating") = sample.getAs[String]("userAvgRating")
      valueMap("userRatingStddev") = sample.getAs[String]("userRatingStddev")

      redisClient.hset(userKey, JavaConversions.mapAsJavaMap(valueMap))
      insertedUserNumber += 1
      if (insertedUserNumber % 100 ==0){
        println(insertedUserNumber + "/" + userCount + "...")
      }
    }

    redisClient.close()
    userLatestSamples
  }

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)

    val conf = new SparkConf()
      .setMaster("local")
      .setAppName("featureEngineering")
      .set("spark.submit.deployMode", "client")

    val spark = SparkSession.builder.config(conf).getOrCreate()
    val movieResourcesPath = this.getClass.getResource("/webroot/sampledata/movies.csv")
    val movieSamples = spark.read.format("csv").option("header", "true").load(movieResourcesPath.getPath)

    val ratingsResourcesPath = this.getClass.getResource("/webroot/sampledata/ratings.csv")
    val ratingSamples = spark.read.format("csv").option("header", "true").load(ratingsResourcesPath.getPath)

    val ratingSamplesWithLabel = addSampleLabel(ratingSamples)
    ratingSamplesWithLabel.show(10, truncate = false)

    val samplesWithMovieFeatures = addMovieFeatures(movieSamples, ratingSamplesWithLabel)
    val samplesWithUserFeatures = addUserFeatures(samplesWithMovieFeatures)

    /* save samples as csv format
    val sampleResourcesPath = this.getClass.getResource("/webroot/sampledata")
    samplesWithUserFeatures.sample(0.1).repartition(1).write.option("header", "true")
      .csv(sampleResourcesPath+"/modelsamples")
     */


    //save user features and item features to redis for online inference
    extractAndSaveUserFeaturesToRedis(samplesWithUserFeatures)
    extractAndSaveMovieFeaturesToRedis(samplesWithUserFeatures)

  }

}
