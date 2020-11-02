package main

import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._

import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator

object Start {
  def main(args: Array[String]): Unit = {
    val spark: SparkSession = SparkSession.builder()
      .master("local[1]")
      .appName("Tokenizing-Text-in-Spark")
      .getOrCreate()

    import spark.implicits._

    spark.sparkContext.setLogLevel("ERROR")

    val df = spark.read.format("csv").option("inferSchema", "true").option("header", "true").load("src/main/resources/creditcard.csv")
    df.printSchema()
    df.show()

    val vectorAssembler = new VectorAssembler()
      .setInputCols(Array("V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10", "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28", "Amount"))
      .setOutputCol("features")
    val output = vectorAssembler.transform(df)
    val allData = output.select("features", "class")
    allData.show()

    val Array(trainingData, testData) = allData.randomSplit(Array(0.7, 0.3))

    val rf = new RandomForestClassifier()
      .setLabelCol("class")
      .setFeaturesCol("features")
      .setNumTrees(100)
    val model = rf.fit(trainingData)

    val predictions = model.transform(testData)
    predictions.select("prediction").show(5)

    val evaluator = new BinaryClassificationEvaluator()
      .setLabelCol("class")
      .setRawPredictionCol("prediction")
    val accuracy = evaluator.evaluate(predictions)
    println(s"Accuracy = ${(accuracy)}")

    println("Program has Finished")
    spark.stop()
  }
}