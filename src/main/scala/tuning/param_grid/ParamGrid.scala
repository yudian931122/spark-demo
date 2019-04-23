package tuning.param_grid

import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{HashingTF, Tokenizer}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}

/**
  * Created by yudian on 2019/4/22.
  * 模型参数调优
  * 使用网格搜索 + 交叉验证的方式
  */

object ParamGrid {
  def main(args: Array[String]): Unit = {
    // 设置Log级别
    Logger.getLogger("org").setLevel(Level.ERROR)
    // 初始化spark session
    val sparkConf = new SparkConf().setAppName("ParamGrid").setMaster("local")
    val ss = SparkSession.builder.config(sparkConf).getOrCreate()

    // 1.创建数据集
    val training = ss.createDataFrame(Seq(
      (0L, "a b c d e spark", 1.0),
      (1L, "d b", 0.0),
      (2L, "spark f g h", 1.0),
      (3L, "hadoop mapreduce", 0.0),
      (4L, "d spark who", 1.0),
      (5L, "g d a y", 0.0),
      (6L, "spark fly", 1.0),
      (7L, "was mapreduce", 0.0),
      (8L, "e spark program,", 1.0),
      (9L, "a e c l", 0.0),
      (10L, "spark compile", 1.0),
      (11L, "hadoop software", 0.0)
    )).toDF("id", "text", "label")

    // 2.构建pipeline模型
    // 2-1 分词器，transformer
    val tokenizer = new Tokenizer().setInputCol("text").setOutputCol("words")
    // 2-2 词频统计，transformer
    val hashingTF = new HashingTF().setInputCol(tokenizer.getOutputCol).setOutputCol("features")
    // 2-3 LR分类器，Estimator
    val LR = new LogisticRegression().setMaxIter(20)
    // 2-4 构建pipeline，Estimator
    val pipeline = new Pipeline().setStages(Array(tokenizer, hashingTF, LR))

    // 3.创建参数网格
    val paramGrid = new ParamGridBuilder()
      .addGrid(hashingTF.numFeatures, Array(10, 100, 1000))
      .addGrid(LR.regParam, Array(0.1, 0.01))
      .build()

    // 4.交叉验证，Estimator
    // Evaluator：评估器
    val cv = new CrossValidator().setEstimator(pipeline)
      .setEstimatorParamMaps(paramGrid)
      .setEvaluator(new BinaryClassificationEvaluator())
      .setNumFolds(2)

    // 5.使用交叉验证训练模型
    val cvModel = cv.fit(training)

    // 创建测试集
    val test = ss.createDataFrame(Seq(
      (12L, "spark i j h"),
      (13L, "a f c"),
      (14L, "spark mapreduce"),
      (15L, "apache hadoop")
    )).toDF("id", "text")

    // 6.测试
    cvModel.transform(test).select("id", "text", "probability", "prediction")
      .collect().foreach{
      case Row(id: Long, text: String, probability: Vector, prediction: Double) =>
        println(s"($id, $text) -> probability=$probability, prediction=$prediction")
    }
  }
}
