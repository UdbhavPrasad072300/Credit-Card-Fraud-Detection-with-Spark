package main

import org.apache.spark.sql.{SparkSession, DataFrame}

import org.apache.spark.sql.functions._

object Start {
  def main(args: Array[String]): Unit = {
    val spark: SparkSession = SparkSession.builder()
      .master("local[1]")
      .appName("Tokenizing-Text-in-Spark")
      .getOrCreate()

    import spark.implicits._

    spark.sparkContext.setLogLevel("ERROR")



    println("Program has Finished")
    spark.stop()
  }
}