package com.wzhe.sparrowrecsys.offline.spark.featureeng
import java.io.{BufferedWriter, File, FileWriter}

import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkConf
import org.apache.spark.mllib.feature.Word2Vec
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions._
import scala.util.control.Breaks._

import scala.util.Random

object Embedding {
  def processItemSequence(sparkSession: SparkSession): RDD[Seq[String]] ={
    val ratingsResourcesPath = this.getClass.getResource("/webroot/sampledata/ratings.csv")

    val ratingSamples = sparkSession.read.format("csv").option("header", "true").load(ratingsResourcesPath.getPath)

    val sortUdf: UserDefinedFunction = udf((rows: Seq[Row]) => {
      rows.map { case Row(movieId: String, timestamp: String) => (movieId, timestamp) }
        .sortBy { case (movieId, timestamp) => timestamp }
        .map { case (movieId, timestamp) => movieId }
    })

    ratingSamples.show(10, false)
    ratingSamples.printSchema()
    val userSeq = ratingSamples.where(col("rating") >= 3.5).groupBy("userId").agg(sortUdf(collect_list(struct("movieId", "timestamp"))) as "movieIds")
        .withColumn("movieIdStr", array_join(col("movieIds"), " "))

    userSeq.printSchema()

    userSeq.select("movieIdStr").show(100, truncate = false)

    userSeq.select("movieIdStr").rdd.map(r => r.getAs[String]("movieIdStr").split(" ").toSeq)
  }

  def trainItem2vec(samples : RDD[Seq[String]]): Unit ={

    /*
    val conf = new SparkConf()
      .setMaster("local")
      .setAppName("ctrModel")
      .set("spark.submit.deployMode", "client")

    val sc = new SparkContext(conf)

    val wordsResourcesPath = this.getClass.getResource("/webroot/sampledata/item2vecdata2.txt")

    val input = sc.textFile(wordsResourcesPath.getPath).map(line => line.split(" ").toSeq)*/

    val word2vec = new Word2Vec()
      .setVectorSize(10)
      .setWindowSize(5)
      .setNumIterations(10)

    val model = word2vec.fit(samples)

    val synonyms = model.findSynonyms("592", 20)

    for((synonym, cosineSimilarity) <- synonyms) {
      println(s"$synonym $cosineSimilarity")
    }

    // Save and load model
    //model.save(sc, "myModelPath")

    //println(model.getVectors)

    val ratingsResourcesPath = this.getClass.getResource("/webroot/sampledata/")

    println(ratingsResourcesPath.getPath)
    val file = new File(ratingsResourcesPath.getPath + "graphEmbedding.txt")
    val bw = new BufferedWriter(new FileWriter(file))
    var id = 0
    for (movieId <- model.getVectors.keys){
      id+=1
      bw.write( movieId + ":" + model.getVectors(movieId).mkString(" ") + "\n")
    }
    bw.close()
    //val sameModel = Word2VecModel.load(sc, "myModelPath")
    // $example off$

    //sc.stop()
  }

  def oneRandomWalk(transferMatrix : scala.collection.mutable.Map[String, scala.collection.mutable.Map[String, Long]], itemCount : scala.collection.mutable.Map[String, Long], itemTotalCount:Long, sampleLength:Int): Seq[String] ={
    val sample = scala.collection.mutable.ListBuffer[String]()

    //pick the first element
    val randomDouble = Random.nextDouble()
    var firstElement = ""
    var culCount:Long = 0
    breakable { for ((item, count) <- itemCount) {
      culCount += count
      if (culCount >= randomDouble * itemTotalCount){
        firstElement = item
        break
      }
    }}

    sample.append(firstElement)
    var curElement = firstElement

    breakable { for( w <- 1 until sampleLength) {
      if (!itemCount.contains(curElement) || !transferMatrix.contains(curElement)){
        break
      }

      val probDistribution = transferMatrix(curElement)
      val curCount = itemCount(curElement)
      val randomDouble = Random.nextDouble()
      var culCount:Long = 0
      breakable { for ((item, count) <- probDistribution) {
        culCount += count
        if (culCount >= randomDouble * curCount){
          curElement = item
          break
        }
      }}
      sample.append(curElement)
    }}
    Seq(sample.toList : _*)
  }


  def randomWalk(transferMatrix : scala.collection.mutable.Map[String, scala.collection.mutable.Map[String, Long]], itemCount : scala.collection.mutable.Map[String, Long]): Seq[Seq[String]] ={
    val sampleCount = 20000
    val sampleLength = 10

    val samples = scala.collection.mutable.ListBuffer[Seq[String]]()

    var itemTotalCount:Long = 0
    for ((k,v) <- itemCount) itemTotalCount += v

    for( w <- 1 to sampleCount) {
      samples.append(oneRandomWalk(transferMatrix, itemCount, itemTotalCount, sampleLength))
    }

    Seq(samples.toList : _*)
  }

  def graphEmb(samples : RDD[Seq[String]], sparkSession: SparkSession): Unit ={
    val pairSamples = samples.flatMap[String]( sample => {
      var pairSeq = Seq[String]()
      var previousItem:String = null
      sample.foreach((element:String) => {
        if(previousItem != null){
          pairSeq = pairSeq :+ (previousItem + ":" + element)
        }
        previousItem = element
      })
      pairSeq
    })

    val pairCount = pairSamples.countByValue()
    val transferMatrix = scala.collection.mutable.Map[String, scala.collection.mutable.Map[String, Long]]()
    val itemCount = scala.collection.mutable.Map[String, Long]()
    println(pairCount.size)
    var lognumber = 0
    pairCount.foreach( pair => {
      val pairItems = pair._1.split(":")
      val count = pair._2
      lognumber = lognumber + 1
      println(lognumber, pair._1)

      if (pairItems.length == 2){
        val item1 = pairItems.apply(0)
        val item2 = pairItems.apply(1)
        if(!transferMatrix.contains(pairItems.apply(0))){
          transferMatrix(item1) = scala.collection.mutable.Map[String, Long]()
        }

        transferMatrix(item1)(item2) = count
        itemCount(item1) = itemCount.getOrElse[Long](item1, 0) + count
      }
    })

    val newSamples = randomWalk(transferMatrix, itemCount)

    val rddSamples = sparkSession.sparkContext.parallelize(newSamples)

    trainItem2vec(rddSamples)

  }

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)

    val conf = new SparkConf()
      .setMaster("local")
      .setAppName("ctrModel")
      .set("spark.submit.deployMode", "client")

    val spark = SparkSession.builder.config(conf).getOrCreate()
    val samples = processItemSequence(spark)
    graphEmb(samples, spark)
    //trainItem2vec(samples)
  }
}
