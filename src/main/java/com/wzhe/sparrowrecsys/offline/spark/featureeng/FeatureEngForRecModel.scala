package com.wzhe.sparrowrecsys.offline.spark.featureeng

import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkConf
import org.apache.spark.sql.expressions.{UserDefinedFunction, Window}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, SparkSession}

import scala.collection.immutable.ListMap
import scala.collection.mutable

object FeatureEngForRecModel {
  def addSampleLabel(ratingSamples:DataFrame): DataFrame ={
    ratingSamples.show(10, truncate = false)
    ratingSamples.printSchema()
    val sampleCount = ratingSamples.count()
    ratingSamples.groupBy(col("rating")).count().orderBy(col("rating"))
      .withColumn("percentage", col("count")/sampleCount).show(100,false)

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

    //split genres
    val samplesWithMovies3 = samplesWithMovies2.withColumn("movieGenre1",split(col("genres"),"\\|").getItem(0))
      .withColumn("movieGenre2",split(col("genres"),"\\|").getItem(1))
      .withColumn("movieGenre3",split(col("genres"),"\\|").getItem(2))

    //add rating features
    val movieRatingFeatures = samplesWithMovies3.groupBy(col("movieId"))
      .agg(count(lit(1)).as("movieRatingCount"),
        avg(col("rating")).as("movieAvgRating"),
        stddev(col("rating")).as("movieRatingStddev"))

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
      .withColumn("userAvgReleaseYear", avg(col("releaseYear"))
        .over(Window.partitionBy("userId")
          .orderBy(col("timestamp")).rowsBetween(-100, -1)))
      .withColumn("userReleaseYearStddev", stddev(col("releaseYear"))
        .over(Window.partitionBy("userId")
          .orderBy(col("timestamp")).rowsBetween(-100, -1)))
      .withColumn("userAvgRating", avg(col("rating"))
        .over(Window.partitionBy("userId")
          .orderBy(col("timestamp")).rowsBetween(-100, -1)))
      .withColumn("userRatingStddev", stddev(col("rating"))
        .over(Window.partitionBy("userId")
          .orderBy(col("timestamp")).rowsBetween(-100, -1)))
      .withColumn("userGenres", extractGenres(collect_list(when(col("label") === 1, col("genres")).otherwise(lit(null)))
        .over(Window.partitionBy("userId")
          .orderBy(col("timestamp")).rowsBetween(-100, -1))))
      .withColumn("userGenre1",col("userGenres").getItem(0))
      .withColumn("userGenre2",col("userGenres").getItem(1))
      .withColumn("userGenre3",col("userGenres").getItem(2))
      .withColumn("userGenre4",col("userGenres").getItem(3))
      .withColumn("userGenre5",col("userGenres").getItem(4))
      .drop("genres", "userGenres", "userPositiveHistory")

    samplesWithUserFeatures.printSchema()
    samplesWithUserFeatures.show(100, truncate = false)

    samplesWithUserFeatures
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
    ratingSamplesWithLabel.show(10, false)

    val samplesWithMovieFeatures = addMovieFeatures(movieSamples, ratingSamplesWithLabel)
    val samplesWithUserFeatures = addUserFeatures(samplesWithMovieFeatures)
  }

}
