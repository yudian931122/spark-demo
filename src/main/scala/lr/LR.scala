package lr

import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.sql._
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkConf


/**
  * Created by yudian on 2019/4/21.
  * LR初探
  */
object LR {
  def main(args: Array[String]): Unit = {
    // 设置Log级别
    Logger.getLogger("org").setLevel(Level.ERROR)
    // 创建spark conf
    val sparkConf = new SparkConf().setAppName("LR").setMaster("local")
    // 创建spark session
    val ss = SparkSession.builder.config(conf = sparkConf).getOrCreate()

    val dataPath = ""

    // 1.创建带标签和特征的数据（DataFrame)
    val training = ss.createDataFrame(Seq(
      (1.0, Vectors.dense(2.0, 1.1, 0.1)),
      (0.0, Vectors.dense(0.0, 1.3, -1.1)),
      (0.0, Vectors.dense(0.0, 1.2, 1.0)),
      (1.0, Vectors.dense(2.0, 1.0, -0.5))
    )).toDF("label", "features")

//    println(training.collectAsList())

    // 2.创建lr模型，也就是构建一个Estimator
    val lr = new LogisticRegression()
    // 打印模型参数解释
    println(lr.explainParams())
    // 设置模型参数: 最大迭代次数和正则化参数
    lr.setMaxIter(10).setRegParam(0.01)

    // 3.训练模型，得到一个Transformer
    val model = lr.fit(training)

    // 查看当前训练好的模型使用的超参
    println(model.parent.extractParamMap())

    println("*" * 50)

    // 使用ParamMap指定模型参数
    val paramMap = ParamMap(lr.maxIter -> 20).put(lr.maxIter, 30).
      put(lr.regParam -> 0.1, lr.threshold -> 0.55)

    val model2 = lr.fit(training, paramMap)
    println(model2.parent.extractParamMap())

    // 4.测试模型
    val testing = ss.createDataFrame(Seq(
      (1.0, Vectors.dense(-1.0, 1.5, 0.3)),
      (0.0, Vectors.dense(3.0, 2.0, -0.1)),
      (1.0, Vectors.dense(0.0, 2.2, -1.5))
    )).toDF("label", "features")

    val pred = model2.transform(testing)

    println(pred.collectAsList())
  }
}
