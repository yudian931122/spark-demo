package pipline

import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkConf
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.ml.feature.{HashingTF, Tokenizer}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.linalg.Vector

/**
  * Created by yudian on 2019/4/21.
  * PipLine初探
  */
object PipeLineDemo {
  def main(args: Array[String]): Unit = {
    // 设置Log级别
    Logger.getLogger("org").setLevel(Level.ERROR)
    // 创建spark conf
    val sparkConf = new SparkConf().setAppName("LR").setMaster("local")
    // 创建spark session
    val ss = SparkSession.builder.config(conf = sparkConf).getOrCreate()

    // 1.创建训练集数据
    val training = ss.createDataFrame(Seq(
      (0L, "a b c d e spark", 1.0),
      (1L, "d b", 0.0),
      (2L, "spark f g h", 1.0),
      (3L, "hadoop mapreduce", 0.0)
    )).toDF("id", "text", "label")

    // 2.数据预处理
    // 2-1.创建分词器，并指定输入数据列名和输出数据的列名
    val tokenizer = new Tokenizer().setInputCol("text").setOutputCol("words")

    // 2-2.统计词频，输入是tokenizer的输出
    // setNumFeatures: 指定特征的数量，在这里也就是最终输出的feature中包含的词的数量
    val hashingTF = new HashingTF().setNumFeatures(1000).setInputCol(tokenizer.getOutputCol).setOutputCol("features")

    // 3.建立模型
    val lr = new LogisticRegression().setMaxIter(20).setRegParam(0.01)

    // 4.创建PipLine
    // 将整个流程串起来
    val pipeline = new Pipeline().setStages(
      Array(tokenizer, hashingTF, lr)
    )

    // 5.训练模型
//    val model = pipeline.fit(training)

    // 6.保存管道和模型
//    pipeline.save("/Users/yudian/IdeaProjects/spark-demo/src/main/scala/pipline/LRpipeline")
//    model.save("/Users/yudian/IdeaProjects/spark-demo/src/main/scala/pipline/LRmodel")

    val model1 = PipelineModel.load("/Users/yudian/IdeaProjects/spark-demo/src/main/scala/pipline/LRmodel")

    // 7.创建测试集
    val test = ss.createDataFrame(Seq(
      (4L, "spark i j h", 1.0),
      (5L, "l m n", 0.0),
      (6L, "spark mapreduce", 1.0),
      (7L, "hadoop apache", 0.0)
    )).toDF("id", "text", "label")

    // 使用测试数据进行测试
    val res = model1.transform(test)

    val rows: Array[Row] = res.select("id", "text", "probability", "prediction").collect()

    rows.foreach{case Row(id: Long, text: String, probability: Vector, prediction: Double) =>
      println(s"($id, $text) -> probability=$probability, prediction=$prediction")
    }

  }
}
