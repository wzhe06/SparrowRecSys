package com.wzhe.sparrowrecsys.offline.spark.featureeng

import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, sql}
import org.apache.spark.ml.{Pipeline, PipelineModel, PipelineStage}
import org.apache.spark.ml.feature._
import org.apache.spark.ml.linalg.{DenseVector, Vectors}
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._

object FeatureEngineering {

  //transform array(user_embedding, item_embedding) to vector for following vectorAssembler
  def transferArray2Vector(samples:DataFrame):DataFrame = {
    import samples.sparkSession.implicits._

    samples
      .map(row => {
        (row.getAs[Int]("user_id"),
          row.getAs[Int]("item_id"),
          row.getAs[Int]("category_id"),
          row.getAs[String]("content_type"),
          row.getAs[String]("timestamp"),
          row.getAs[Long]("user_item_click"),
          row.getAs[Double]("user_item_imp"),
          row.getAs[Double]("item_ctr"),
          row.getAs[Int]("is_new_user"),
          Vectors.dense(row.getAs[Seq[Double]]("user_embedding").toArray),
          Vectors.dense(row.getAs[Seq[Double]]("item_embedding").toArray),
          row.getAs[Int]("label")
        )
      }).toDF(
      "user_id",
      "item_id",
      "category_id",
      "content_type",
      "timestamp",
      "user_item_click",
      "user_item_imp",
      "item_ctr",
      "is_new_user",
      "user_embedding",
      "item_embedding",
      "label")
  }

  //calculate inner product between user embedding and item embedding
  def calculateEmbeddingInnerProduct(samples:DataFrame): DataFrame ={
    import samples.sparkSession.implicits._

    samples.map(row => {
      val userEmbedding = row.getAs[DenseVector]("user_embedding")
      val itemEmbedding = row.getAs[DenseVector]("item_embedding")
      var aSquare = 0.0
      var bSquare = 0.0
      var abProduct = 0.0

      for (i <-0 until userEmbedding.size){
        aSquare += userEmbedding(i) * userEmbedding(i)
        bSquare += itemEmbedding(i) * itemEmbedding(i)
        abProduct += userEmbedding(i) * itemEmbedding(i)
      }
      var innerProduct = 0.0
      if (aSquare == 0 || bSquare == 0){
        innerProduct = 0.0
      }else{
        innerProduct = abProduct / (Math.sqrt(aSquare) * Math.sqrt(bSquare))
      }

      (row.getAs[Int]("user_id"),
        row.getAs[Int]("item_id"),
        row.getAs[Int]("category_id"),
        row.getAs[String]("content_type"),
        row.getAs[String]("timestamp"),
        row.getAs[Long]("user_item_click"),
        row.getAs[Double]("user_item_imp"),
        row.getAs[Double]("item_ctr"),
        row.getAs[Int]("is_new_user"),
        innerProduct,
        row.getAs[Int]("label")
      )
    }).toDF(
      "user_id",
      "item_id",
      "category_id",
      "content_type",
      "timestamp",
      "user_item_click",
      "user_item_imp",
      "item_ctr",
      "is_new_user",
      "embedding_inner_product",
      "label")
  }

  //calculate outer product between user embedding and item embedding
  def calculateEmbeddingOuterProduct(samples:DataFrame): DataFrame ={
    import samples.sparkSession.implicits._

    samples.map(row => {
      val user_embedding = row.getAs[DenseVector]("user_embedding")
      val item_embedding = row.getAs[DenseVector]("item_embedding")

      val outerProductEmbedding:Array[Double] = Array.fill[Double](user_embedding.size)(0)

      for (i <-0 until user_embedding.size){
        outerProductEmbedding(i) = user_embedding(i) * item_embedding(i)
      }

      (row.getAs[Int]("user_id"),
        row.getAs[Int]("item_id"),
        row.getAs[Int]("category_id"),
        row.getAs[String]("content_type"),
        row.getAs[String]("timestamp"),
        row.getAs[Long]("user_item_click"),
        row.getAs[Double]("user_item_imp"),
        row.getAs[Double]("item_ctr"),
        row.getAs[Int]("is_new_user"),
        Vectors.dense(outerProductEmbedding),
        row.getAs[Int]("label")
      )
    }).toDF(
      "user_id",
      "item_id",
      "category_id",
      "content_type",
      "timestamp",
      "user_item_click",
      "user_item_imp",
      "item_ctr",
      "is_new_user",
      "embedding_outer_product",
      "label")
  }

  //pre process features to generate feature vector including embedding outer product
  def preProcessOuterProductSamples(samples:DataFrame):PipelineModel = {
    val contentTypeIndexer = new StringIndexer().setInputCol("content_type").setOutputCol("content_type_index")

    val oneHotEncoder = new OneHotEncoderEstimator()
      .setInputCols(Array("content_type_index"))
      .setOutputCols(Array("content_type_vector"))
      .setDropLast(false)

    val ctrDiscretizer = new QuantileDiscretizer()
      .setInputCol("item_ctr")
      .setOutputCol("ctr_bucket")
      .setNumBuckets(100)

    val vectorAsCols = Array("content_type_vector", "ctr_bucket", "user_item_click", "user_item_imp", "is_new_user", "embedding_outer_product")
    val vectorAssembler = new VectorAssembler().setInputCols(vectorAsCols).setOutputCol("vectorFeature")

    val scaler = new MinMaxScaler().setInputCol("vectorFeature").setOutputCol("scaledFeatures")

    /*
    val scaler = new StandardScaler()
      .setInputCol("vectorFeature")
      .setOutputCol("scaledFeatures")
      .setWithStd(true)
      .setWithMean(true)
    */

    val pipelineStage: Array[PipelineStage] = Array(contentTypeIndexer, oneHotEncoder, ctrDiscretizer, vectorAssembler, scaler)
    val featurePipeline = new Pipeline().setStages(pipelineStage)

    featurePipeline.fit(samples)
  }

  //normal pre process samples to generate feature vector
  def preProcessSamples(samples:DataFrame):PipelineModel = {
    val pipelineStage: Array[PipelineStage] = preProcessSamplesStages()
    val featurePipeline = new Pipeline().setStages(pipelineStage)
    featurePipeline.fit(samples)
  }

  def preProcessSamplesStages():Array[PipelineStage] = {
    val contentTypeIndexer = new StringIndexer().setInputCol("content_type").setOutputCol("content_type_index")

    val oneHotEncoder = new OneHotEncoderEstimator()
      .setInputCols(Array("content_type_index"))
      .setOutputCols(Array("content_type_vector"))
      .setDropLast(false)

    val ctrDiscretizer = new QuantileDiscretizer()
      .setInputCol("item_ctr")
      .setOutputCol("ctr_bucket")
      .setNumBuckets(100)

    val vectorAsCols = Array("content_type_vector", "ctr_bucket", "user_item_click", "user_item_imp", "is_new_user", "user_embedding", "item_embedding")
    val vectorAssembler = new VectorAssembler().setInputCols(vectorAsCols).setOutputCol("vectorFeature")

    val scaler = new MinMaxScaler().setInputCol("vectorFeature").setOutputCol("scaledFeatures")

    Array(contentTypeIndexer, oneHotEncoder, ctrDiscretizer, vectorAssembler, scaler)
  }

  //pre process features to generate feature vector including embedding inner product
  def preProcessInnerProductSamples(samples:DataFrame):PipelineModel = {

    val pipelineStage: Array[PipelineStage] = preProcessInnerProductSamplesStages()
    val featurePipeline = new Pipeline().setStages(pipelineStage)

    featurePipeline.fit(samples)
  }

  def preProcessInnerProductSamplesStages():Array[PipelineStage] = {
    val contentTypeIndexer = new StringIndexer().setInputCol("content_type").setOutputCol("content_type_index")

    val oneHotEncoder = new OneHotEncoderEstimator()
      .setInputCols(Array("content_type_index"))
      .setOutputCols(Array("content_type_vector"))
      .setDropLast(false)

    val ctrDiscretizer = new QuantileDiscretizer()
      .setInputCol("item_ctr")
      .setOutputCol("ctr_bucket")
      .setNumBuckets(100)

    val vectorAsCols = Array("content_type_vector", "ctr_bucket", "user_item_click", "user_item_imp", "is_new_user", "embedding_inner_product")
    val vectorAssembler = new VectorAssembler().setInputCols(vectorAsCols).setOutputCol("vectorFeature")

    val scaler = new MinMaxScaler().setInputCol("vectorFeature").setOutputCol("scaledFeatures")
   Array(contentTypeIndexer, oneHotEncoder, ctrDiscretizer, vectorAssembler, scaler)
  }

  def oneHotEncoderExample(samples:DataFrame): Unit ={

    val samplesWithIdNumber = samples.withColumn("movieIdNumber", col("movieId").cast(sql.types.IntegerType))

    val oneHotEncoder = new OneHotEncoderEstimator()
      .setInputCols(Array("movieIdNumber"))
      .setOutputCols(Array("movieIdVector"))
      .setDropLast(false)

    val pipelineStage: Array[PipelineStage] = Array(oneHotEncoder)
    val featurePipeline = new Pipeline().setStages(pipelineStage)

    val oneHotEncoderSamples = featurePipeline.fit(samplesWithIdNumber).transform(samplesWithIdNumber)
    oneHotEncoderSamples.show(10)
  }

  val array2vec: UserDefinedFunction = udf { (a: Seq[Int], length: Int) => org.apache.spark.ml.linalg.Vectors.sparse(length, a.sortWith(_ < _).toArray, Array.fill[Double](a.length)(1.0)) }

  def multiHotEncoderExample(samples:DataFrame): Unit ={

    val samplesWithGenre = samples.select(col("movieId"), col("title"),explode(split(col("genres"), "\\|").cast("array<string>")).as("genre"))

    samplesWithGenre.show(10)

    val genreIndexer = new StringIndexer().setInputCol("genre").setOutputCol("genreIndex")

    val stringIndexerModel : StringIndexerModel = genreIndexer.fit(samplesWithGenre)
    print(stringIndexerModel.extractParamMap().toString())

    val genreIndexSamples = stringIndexerModel.transform(samplesWithGenre)
      .withColumn("genreIndexInt", col("genreIndex").cast(sql.types.IntegerType))

    val indexSize = genreIndexSamples.agg(max(col("genreIndexInt"))).head().getAs[Int](0) + 1

    val processedSamples =  genreIndexSamples
      .groupBy(col("movieId")).agg(collect_list("genreIndexInt").as("genreIndexes"))
        .withColumn("indexSize", typedLit(indexSize))

    val finalSample = processedSamples.withColumn("vector", array2vec(col("genreIndexes"),col("indexSize")))
    finalSample.printSchema()
    finalSample.show(10)
    //oneHotEncoderSamples.show(10)
  }

  val double2vec: UserDefinedFunction = udf { (value: Double) => org.apache.spark.ml.linalg.Vectors.dense(value) }

  def ratingFeatures(samples:DataFrame): Unit ={

    samples.printSchema()
    samples.show(10)

    val movieFeatures = samples.groupBy(col("movieId"))
      .agg(count(lit(1)).as("ratingCount"),
        avg(col("rating")).as("avgRating"),
        variance(col("rating")).as("ratingVar"))
        .withColumn("avgRatingVec", double2vec(col("avgRating")))

    movieFeatures.select(col("avgRating")).show(1000)

    val userFeatures = samples.groupBy(col("userId"))
      .agg(count(lit(1)).as("ratingCount"),
        avg(col("rating")).as("avgRating"),
        variance(col("rating")).as("ratingVar"))


    val ratingCountDiscretizer = new QuantileDiscretizer()
      .setInputCol("ratingCount")
      .setOutputCol("ratingCountBucket")
      .setNumBuckets(100)

    val ratingScaler = new MinMaxScaler().setInputCol("avgRatingVec").setOutputCol("scaleAvgRating")

    val pipelineStage: Array[PipelineStage] = Array(ratingCountDiscretizer, ratingScaler)
    val featurePipeline = new Pipeline().setStages(pipelineStage)

    val movieProcessedFeatures = featurePipeline.fit(movieFeatures).transform(movieFeatures)

    movieProcessedFeatures.show(100)
  }

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)

    val conf = new SparkConf()
      .setMaster("local")
      .setAppName("ctrModel")
      .set("spark.submit.deployMode", "client")

    val spark = SparkSession.builder.config(conf).getOrCreate()

    /*
    val movieResourcesPath = this.getClass.getResource("/webroot/sampledata/movies.csv")
    val movieSamples = spark.read.format("csv").option("header", "true").load(movieResourcesPath.getPath)
    movieSamples.printSchema()
    movieSamples.show(10)

    //oneHotEncoderExample(rawSamples)
    multiHotEncoderExample(movieSamples)
     */


    val ratingsResourcesPath = this.getClass.getResource("/webroot/sampledata/ratings.csv")

    val ratingSamples = spark.read.format("csv").option("header", "true").load(ratingsResourcesPath.getPath)
    ratingFeatures(ratingSamples)

  }
}
