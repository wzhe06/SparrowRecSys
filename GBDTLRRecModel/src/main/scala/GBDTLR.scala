import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.linalg
import org.apache.spark.mllib.linalg.{DenseVector, SparseVector, Vector}
import org.apache.spark.sql.SparkSession
//import org.apache.spark.ml.linalg.{DenseVector, SparseVector}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.configuration.{BoostingStrategy, FeatureType, Strategy}
import org.apache.spark.mllib.tree.GradientBoostedTrees
import org.apache.spark.mllib.tree.model.{GradientBoostedTreesModel, Node}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.sql.DataFrame

/**
 * Author:
    lixiang, 183570397@qq.com
  GBDT and LR:
    reference: Practical Lessons from Predicting Clicks on Ads at Facebook
 */
object GBDTLR extends Serializable {


  def fit(train: RDD[LabeledPoint]) ={
    val numTrees = 40
    val boostingStrategy = BoostingStrategy.defaultParams("Classification")
    boostingStrategy.setNumIterations(numTrees)
    val treeStratery = Strategy.defaultStrategy("Classification")
    treeStratery.setMaxDepth(8)
    treeStratery.setNumClasses(2)
    //    treeStratery.setCategoricalFeaturesInfo(Map[Int, Int]())
    boostingStrategy.setTreeStrategy(treeStratery)
    var gbdtModel = GradientBoostedTrees.train(train, boostingStrategy)
    //    gbdtModel.save(sc, "/lx/model_gbdt")
    println("treeWeights>>>"+gbdtModel.treeWeights.toList.toString())

    val treeLeafArray = new Array[Array[Int]](numTrees)
    for (i <- 0.until(numTrees)) {
      treeLeafArray(i) = getLeafNodes(gbdtModel.trees(i).topNode)
    }
    for (i <- 0.until(numTrees)) {
      val tree = gbdtModel.trees(i)
      val topNode = tree.topNode
      println("第i棵树>>"+i+">>topnodeid>>"+topNode.id+">>numNodes>>"+tree.numNodes+">>"+treeLeafArray(i).length+">>"+treeLeafArray(i).mkString(","))
      println("叶子节点误差>>>", treeLeafArray(i).length, (gbdtModel.trees(i).numNodes + 1) / 2)
    }
    (gbdtModel, treeLeafArray)
  }

  def transform(features: Vector, gbdtModel: GradientBoostedTreesModel, treeLeafArray: Array[Array[Int]]) = {
    var newFeature = new Array[Double](0)
    val numTrees = gbdtModel.numTrees
    for (i <- 0.until(numTrees)) {
      val treePredict = predictModify(gbdtModel.trees(i).topNode, features)
      //gbdt tree is binary tree
      val treeArray = new Array[Double]((gbdtModel.trees(i).numNodes + 1) / 2)
      treeArray(treeLeafArray(i).indexOf(treePredict)) = gbdtModel.treeWeights(i) //设置为树的权重//i
      newFeature = newFeature ++ treeArray
    }
    import org.apache.spark.ml.linalg.DenseVector
    new DenseVector(newFeature).toSparse
  }

  //get decision tree leaf's nodes
  def getLeafNodes(node: Node): Array[Int] = {
    var treeLeafNodes = new Array[Int](0)
    if (node.isLeaf) {
      treeLeafNodes = treeLeafNodes.:+(node.id)
    } else {
      treeLeafNodes = treeLeafNodes ++ getLeafNodes(node.leftNode.get)
      treeLeafNodes = treeLeafNodes ++ getLeafNodes(node.rightNode.get)
    }
    treeLeafNodes
  }

  // predict decision tree leaf's node value
  def predictModify(node: Node, features: Vector): Int = {
    val split = node.split
    if (node.isLeaf) {
      node.id
    } else {
      if (split.get.featureType == FeatureType.Continuous) {
        if (features(split.get.feature) <= split.get.threshold) {
          //          println("Continuous left node")
          predictModify(node.leftNode.get, features)
        } else {
          //          println("Continuous right node")
          predictModify(node.rightNode.get, features)
        }
      } else {
        if (split.get.categories.contains(features(split.get.feature))) {
          //          println("Categorical left node")
          predictModify(node.leftNode.get, features)
        } else {
          //          println("Categorical right node")
          predictModify(node.rightNode.get, features)
        }
      }
    }
  }

  def evaluate(predictions: DataFrame): Unit ={
    val binaryClassificationEvaluator = new BinaryClassificationEvaluator()
      .setLabelCol("label")
      .setRawPredictionCol("rawPrediction")
      .setMetricName("areaUnderROC")
    val auc = binaryClassificationEvaluator.evaluate(predictions)
    println("auc>>>", auc)

  }

  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder().master("local[*]").appName("GbdtAndLr").getOrCreate()
    val sc = spark.sparkContext
    sc.setLogLevel("WARN")
    Logger.getLogger("app").setLevel(Level.WARN)
    import spark.implicits._

    val file_path = "file://" +
      "/Users/v_lixiang13/Desktop/SparrowRecSys/src/main/resources"
    val ratingResourcesPath = file_path + "/webroot/sampledata/trainingSamples.csv"
    val ratingSampes = spark.read.format("csv").option("header", true).load(ratingResourcesPath)
      .withColumn("userIdInt", col("userId").cast(DoubleType))
      .withColumn("movieIdInt", col("movieId").cast(DoubleType))
      .withColumn("ratingFloat", col("rating").cast(DoubleType))
      .withColumn("rating", col("rating").cast(DoubleType))
      .withColumn("label", col("label").cast(DoubleType))

    //printSchema
    ratingSampes.printSchema()
    println(ratingSampes.show(5))

    // 简单版特征工程，可以加上userEmb， itemEmb相关特征
    val features_names = List("movieAvgRating", "movieRatingStddev", "movieRatingCount", "userAvgRating", "userRatingStddev", "userRatingCount", "releaseYear")
    val train = ratingSampes.map(x=>{
      val label = x.getAs[Double]("label")
      val features = features_names.map(name => {
        x.getAs[String](name).toDouble
      })
      LabeledPoint(label, new DenseVector(features.toArray))
    })

    val tuple = fit(train.rdd)
    val gbdtModel = tuple._1
    val treeLeafArray = tuple._2
    val gbdt_b = sc.broadcast(gbdtModel)
    println("treeLeafArray: ", treeLeafArray.map(_.length).sum)

    // 获取叶子结点
    val trainForLr = train.map(p => {
      val leafNodes = transform(p.features, gbdt_b.value, treeLeafArray)
      org.apache.spark.ml.feature.LabeledPoint(label = p.label, features = new linalg.DenseVector(leafNodes.toArray).toSparse)
    }).toDF()
    println("leaf features")
    trainForLr.show()

    val lr = new LogisticRegression()
    val model = lr.fit(trainForLr)
    val predictions = model.transform(trainForLr)
    predictions.show()

    evaluate(predictions)
    sc.stop()

    println("done")
















  }


}
