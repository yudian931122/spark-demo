package tuning.train_split

import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkConf
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.sql.{Dataset, Row, SparkSession}

/**
  * Created by yudian on 2019/4/22.
  * 模型参数调优
  * 使用网格搜索 + 训练校验分离的形式
  */

object TrainSplitDemo {
  def main(args: Array[String]): Unit = {
    // 设置Log级别
    Logger.getLogger("org").setLevel(Level.ERROR)
    // 初始化spark session
    val sparkConf = new SparkConf().setAppName("ParamGrid").setMaster("local")
    val ss = SparkSession.builder.config(sparkConf).getOrCreate()

    val dataPath = "D:\\ML\\data\\sample_linear_regression_data.txt"
    // 需要指定一下特征的数量，不然无法识别
    val data = ss.read.format("libsvm").option("numFeatures", "10").load(dataPath)

    val Array(training, test) = data.randomSplit(Array(0.9, 0.1), seed = 12345)

    val lr = new LinearRegression().setMaxIter(20)

    val paramGrid = new ParamGridBuilder()
      .addGrid(lr.elasticNetParam, Array(0.0, 0.5, 1.0))
      .addGrid(lr.regParam, Array(0.1, 0.01))
      .addGrid(lr.fitIntercept)
      .build()

    val ts = new TrainValidationSplit()
      .setEstimator(lr).setEstimatorParamMaps(paramGrid)
      .setEvaluator(new RegressionEvaluator())
      .setTrainRatio(0.8)

    val tsModel = ts.fit(training)

    tsModel.transform(test).select("features", "label", "prediction").show()
  }
}
