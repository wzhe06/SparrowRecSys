package com.sparrowrecsys.offline.spark.evaluate

import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.sql.DataFrame

class Evaluator {
  def evaluate(predictions:DataFrame):Unit = {

    import  predictions.sparkSession.implicits._

    val scoreAndLabels = predictions.select("label", "probability").map { row =>
      (row.apply(1).asInstanceOf[DenseVector](1), row.getAs[Int]("label").toDouble)
    }

    val metrics = new BinaryClassificationMetrics(scoreAndLabels.rdd)

    println("AUC under PR = " + metrics.areaUnderPR())
    println("AUC under ROC = " + metrics.areaUnderROC())
  }
}
