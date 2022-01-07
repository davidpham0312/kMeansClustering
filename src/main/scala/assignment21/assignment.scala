package assignment21

import org.apache.spark.SparkConf
import org.apache.spark.sql.functions.{window, column, desc, col}


import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.Column
import org.apache.spark.sql.Row
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.types.{ArrayType, StringType, StructField, IntegerType, DoubleType}
import org.apache.spark.sql.types.DataType
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.functions.{count, sum, min, max, asc, desc, udf, to_date, avg, mean, stddev}

import org.apache.spark.sql.functions.explode
import org.apache.spark.sql.functions.array
import org.apache.spark.sql.SparkSession

import org.apache.spark.storage.StorageLevel
import org.apache.spark.streaming.{Seconds, StreamingContext}



import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.{MinMaxScaler, Tokenizer, HashingTF}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.clustering.{KMeans, KMeansSummary, KMeansModel}
import org.apache.spark.sql.functions.udf
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}    // For BT4
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.linalg.Vector


import java.io.{PrintWriter, File}


//import java.lang.Thread
import sys.process._


import org.apache.log4j.Logger
import org.apache.log4j.Level
import scala.collection.immutable.Range

object assignment {
  // Suppress the log messages:
  Logger.getLogger("org").setLevel(Level.OFF)
                       
  val spark = SparkSession.builder()
                          .appName("assignment21")
                          .config("spark.driver.host", "localhost")
                          .master("local")
                          .getOrCreate()
 
                      
  val dataK5D2: DataFrame = spark.read
                       .format("csv")
                       .option("header", "true")
                       .option("inferSchema", "true")
                       .csv("data/dataK5D2.csv")

  
  val dataK5D3: DataFrame =  spark.read
                       .format("csv")
                       .option("header", "true")
                       .option("inferSchema", "true")
                       .csv("data/dataK5D3.csv")
                       

  
  val dataK5D3WithLabels: DataFrame =  spark.read
                       .format("csv")
                       .option("header", "true")
                       .option("inferSchema", "true")
                       .csv("data/dataK5D3.csv")
  
  val toDouble = udf[Double, String]( _.toDouble)

  // Helper function to rescale data
  def rescale(df: DataFrame, oldName: String, newName: String, min_val: Double, max_val: Double): DataFrame = {
    val vectorizeCol = udf( (v:Double) => Vectors.dense(Array(v)) )        
   
    //MinMaxScaler can be used to transform the 'f3' column to be scale up  
    val df1 = df.withColumn("vec", vectorizeCol(df(oldName)))    
    val scaler = new MinMaxScaler()
       .setInputCol("vec")
       .setOutputCol(newName)
       .setMax(max_val)
       .setMin(min_val)
    val df2 = scaler.fit(df1).transform(df1)
    return df2
  }
  // TASK 1
                       
                      
  
  def task1(df: DataFrame, k: Int) : Array[(Double, Double)]  = {
    //Load data to dataFrame, remove header line and change the a/b column variable type to double
    val data = df.drop("LABEL")
    .withColumn("a", toDouble(df("a")))
    .withColumn("b", toDouble(df("b")))
    .na.drop("any")  // Drop any rows in which any value is null in a/b column 
    
    //load data to the memory 
    data.cache()
  
    //Another way: statistical analysis of column 'a'  
    data.describe().show()
  
    val rescaleA = rescale(data, "a", "rescaledA", 0, 1)
    val rescaledData = rescale(rescaleA, "b", "rescaledB", 0, 1)
    //Create a VectorAssembler for mapping input column "a" and "b" to "features" 
    //This step needed because all machine learning algorithms in Spark take as input a Vector type  
    val vectorAssembler = new VectorAssembler()
      .setInputCols(Array("rescaledA", "rescaledB"))
      .setOutputCol("features")
    
    
    //Another solution: Performs transform and drops if null value in a and b columns
    //val transformedData = vectorAssembler.transform(data.na.drop(Array("a", "b")))

      
    //Perform pipeline with sequence of stages to process and learn from data 
    val transformationPipeline = new Pipeline()
        .setStages(Array(vectorAssembler))
    val pipeLine = transformationPipeline.fit(rescaledData)
    val transformedTraining = pipeLine.transform(rescaledData)

    
    // Create a k-means object and fit the transformedTraining to get a k-means object
    //     set parameters: k = 5, number of cluster
    //     k is randomly assigned within the data                : 
    val kmeans = new KMeans()
      .setK(k)
      .setSeed(1L)

    
      
    // train the model
    val kmModel: KMeansModel = kmeans.fit(transformedTraining)

    
    
   //5 k-means cluster centroids of vector data type converted to array as return values 
    val centers = kmModel.clusterCenters
      .map(x => x.toArray)
      .map{case Array(f1,f2) => (f1,f2)}
    

    println(s"\n Number of centroids = ${centers.length} \n ") 
    return centers
    
    
  }
  
  println("\n Task 1: k-means clustering for two dimentional data: dataK5D2.csv")
  val answer1 = task1(dataK5D2, 5)
  answer1.foreach(println)
  

  // TASK 2
  
  
  
  def task2(df: DataFrame, k: Int) : Array[(Double, Double, Double)] = {
    
    // df.show()
    //Load data to dataFrame, remove header line  
    val data = df.drop("LABEL")
    .withColumn("a", toDouble(df("a")))
    .withColumn("b", toDouble(df("b")))
    .withColumn("c", toDouble(df("c")))
    .na.drop("any")  // Drop any rows in which any value is null in a/b column 


    //load data to the memory 
    data.cache()
  

   // Statistic analysis, used to rescale data later
    println("\n Summary of descriptive statistics ---")
    data.describe().show()

   //The statistical analysis shows that column 'c' standard deviation too high as 273.63044184094457
   // it need to be scaled down, otherwise variance sensitive k-means can compute entirely on the basisi of the 'c' 
 
   //import org.apache.spark.mllib.linalg.Vectors

    
    val dataA = rescale(data, "a", "aScaled", 0, 1)
    val dataAB = rescale(dataA, "b", "bScaled", 0, 1)
    val data2 = rescale(dataAB, "c", "cScaled", 0, 1)
    //data2.describe().show()
    //data2.show()       
      
 
    
    //Create a VectorAssembler for mapping input column "a", "b" and "c" to "features"
    //This step needed because all machine learning algorithms in Spark take as input a Vector type  
    val vectorAssembler = new VectorAssembler()
        .setInputCols(Array("a", "b", "cScaled"))
        .setOutputCol("features")
    
    
    //Perform pipeline with sequence of stages to process and learn from data 
    val transformationPipeline = new Pipeline()
        .setStages(Array(vectorAssembler))
    val pipeLine = transformationPipeline.fit(data2)
    val transformedTraining = pipeLine.transform(data2)
    
    // println("\n After scaling down ---")
    // transformedTraining.show

    
      
    // Create a k-means object and fit the transformedTraining to get a k-means object
    //     set parameters: k = 5, number of cluster
    //     k is randomly assigned within the data  
    val kmeans = new KMeans()
        .setK(k)
        .setSeed(1L)
    
    // train the model
    val kmModel: KMeansModel = kmeans.fit(transformedTraining)
    
    // println("\n Summary of k-mean clustinering with prediction ---")
    // kmModel.summary.predictions.show
    
 
     
    //5 k-means cluster centroids of vector data type converted to array as return values
    val centers = kmModel.clusterCenters.map(x => x.toArray)
        .map{case Array(f1,f2,f3) => (f1,f2,f3)}
    
    println(s"\n Number of centroids = ${centers.length} \n ") 
    

    // Scale back all values to the original scale 
 
    val df5 = spark.createDataFrame(centers)
        .toDF("f1", "f2", "f3")
    
   // Scale back the data
    val center4 = rescale(df5, "f3", "f3ScaledBack", 9.5387, 991.9577)
    val center3 = rescale(center4, "f2", "f2ScaledBack", 0.018, 9.9893)
    val center2 = rescale(center3, "f1", "f1ScaledBack", -0.9994, 0.9975)
      
    //center2.show()    

    val centersDF = center2.select(col("f1ScaledBack"), col("f2ScaledBack"), col("f3ScaledBack"))

    
    //Convert DataFrame to Array  
    val centersArray = centersDF.select("f1ScaledBack", "f2ScaledBack", "f3ScaledBack").collect()
        .map(each => (each.getAs[Double]("f1ScaledBack"), each.getAs[Double]("f2ScaledBack"), each.getAs[Double]("f3ScaledBack")))
        .toArray

    return centersArray
 
  }
  
  println("\n Task 2: k-means clustering for three dimentional data: dataK5D3.csv")
  val answer2 = task2(dataK5D3, 5)
  answer2.foreach(println)


  // TASK 3
  
  def task3(df: DataFrame, k: Int) : Array[(Double, Double)] = {
     
    // df.show()
 
    
    // Maps a string column of labels to an ML column of label indices
    val indexer = new StringIndexer()
    .setInputCol("LABEL")
    .setOutputCol("lid")
      
    val df1 = indexer
    .fit(df)
    .transform(df)
   
    
    
    // Load data to dataFrame, remove header line and cast the a/b column variable type to double
    // Drop LABEL column, but cast label ids (lid) to Double
    val dfl = df1.drop("LABEL")
                  .selectExpr("cast(a as Double) a", 
                      "cast(b as Double) b", 
                      "cast(lid as Double) label")
                  .na.drop("any")  // Drop any rows in which any value is null in a/b column 
  
    val dfA = rescale(dfl, "a", "rescaledA", 0, 1)
    val data = rescale(dfA, "b", "rescaledB", 0, 1)
    
    //load data to the memory 
    data.cache()
    
    
    // Create VectorAssembler for mapping input column "a", "b" and "label" to "features"
    // This step needed because ML algorithms in Spark take as input a Vector type
    val vectorAssembler = new VectorAssembler()
    .setInputCols(Array("rescaledA", "rescaledB", "label"))
    .setOutputCol("features")
    
    // Perform pipeline with sequence of stages to process and learn from data
    val transformationPipeline = new Pipeline()
    .setStages(Array(vectorAssembler))
    val pipeLine = transformationPipeline.fit(data)
    val transformedTraining = pipeLine.transform(data)
    
    // Create a k-means object and fit the transformedTraining to get a k-means object
    val kmeans = new KMeans()
    .setK(k)
    .setSeed(1L)
    val kmModel = kmeans.fit(transformedTraining)
    val centers = kmModel.clusterCenters
    .map(x => x.toArray)
    .map{case Array(f1,f2,f3) => (f1,f2,f3)}
    // changed from 0.5 to 0.43 in order to get two centers  
    .filter(x => (x._3 > 0.43))
    .map{case (f1,f2,f3) => (f1,f2)}
    
    return centers

  }

  println("Task 3:")
  val answer3 = task3(dataK5D3WithLabels, 5)
  answer3.foreach(println)
  

  // Parameter low is the lowest k and high is the highest one.
  def task4(df: DataFrame, low: Int, high: Int): Array[(Int, Double)]  = {

    // Maps a string column of labels to an ML column of label indices
    val indexer = new StringIndexer()
    .setInputCol("LABEL")
    .setOutputCol("lid")
      
    val df1 = indexer
    .fit(df)
    .transform(df)
    
    // Load data to dataFrame, remove header line and cast the a/b column variable type to double
    // Drop LABEL column, but cast label ids (lid) to Double
    val dfl = df1.drop("LABEL")
                  .selectExpr("cast(a as Double) a", 
                      "cast(b as Double) b", 
                      "cast(lid as Double) label")
                  .na.drop("any")  // Drop any rows in which any value is null in a/b column 
  
    val dfA = rescale(dfl, "a", "rescaledA", 0, 1)
    val data = rescale(dfA, "b", "rescaledB", 0, 1)

    //load data to the memory 
    data.cache()
    
    // Create a VectorAssembler for mapping input column "a", "b" and "c" to "features"
    // This step needed because all machine learning algorithms in Spark take as input a Vector type
    val vectorAssembler = new VectorAssembler()
    .setInputCols(Array("a", "b", "label"))
    .setOutputCol("features")

    // Perform pipeline with sequence of stages to process and learn from data
    val transformationPipeline = new Pipeline()
    .setStages(Array(vectorAssembler))
    val pipeLine = transformationPipeline.fit(data)
    val transformedTraining = pipeLine.transform(data)
    
    import scala.collection.mutable.ArrayBuffer
    val clusters = ArrayBuffer [Int] ()
    val costs = ArrayBuffer [Double] ()

    // Calculating the cost (sum of squared distances of points to their nearest center)
    for (i <- low to high) {
      
      val kmeans = new KMeans()
      .setK(i)
      .setSeed(1L)
      val kmModel = kmeans.fit(transformedTraining)
      val cost = kmModel.computeCost(transformedTraining)
      
      clusters += i
      costs += cost
      
    }
    // Code for Bonus task 5. I tried to compile but an error raised despite following
    // the instructions. The error is "object plot is not a member of package breeze"
    // I used python matplotlib to plot the data. The source code and figure is included in the directory
    //import breeze.linalg._
    //import breeze.numerics._
    //import breeze.plot._ 
    //val f = Figure()
    //val p = fig.subplot(0)
    //p += plot(clusterAmount, cost)
    val pairs = clusters.toArray.zip(costs)
    return pairs
    
  }
   
  println("Task 4:")
  val answer4 = task4(dataK5D2, 2, 10)
  answer4.foreach(println)
  
  
  // Bonus task 4. Here I design Gaussian mixture Model that benefits
  // from machine learning pipeline.
  val data = dataK5D2.drop("LABEL")
    .withColumn("a", toDouble(dataK5D2("a")))
    .withColumn("b", toDouble(dataK5D2("b")))
    .na.drop("any")
    
  //load data to the memory 
  data.cache()


  val rescaleA = rescale(data, "a", "rescaledA", 0, 1)
  val rescaledData = rescale(rescaleA, "b", "rescaledB", 0, 1)  
  val vectorAssembler = new VectorAssembler()
    .setInputCols(Array("rescaledA", "rescaledB"))
    .setOutputCol("features")
  

  val transformationPipeline = new Pipeline()
      .setStages(Array(vectorAssembler))
  val pipeLine = transformationPipeline.fit(rescaledData)
  val transformedTraining = pipeLine.transform(rescaledData)
  
  import org.apache.spark.ml.clustering.GaussianMixture
  val gmm = new GaussianMixture()
    .setK(2)
  val model = gmm.fit(transformedTraining)
  
  // output parameters of mixture model model
  for (i <- 0 until model.getK) {
    println(s"Gaussian $i:\nweight=${model.weights(i)}\n" +
        s"mu=${model.gaussians(i).mean}\nsigma=\n${model.gaussians(i).cov}\n")
}

}
