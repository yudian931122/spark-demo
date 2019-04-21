package word_count

import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by yudian on 2019/2/20.
  */
object WordCount {
  def main(args: Array[String]): Unit = {
    val sparkConf = new SparkConf().setAppName("WordCount").setMaster("local")
    val sc = new SparkContext(sparkConf)

    val dataPath = "/Users/yudian/study/datas/spark/words.txt"

    val textRdd = sc.textFile(dataPath)

    val wordCountRdd = textRdd.flatMap(_.split(" ")).map((_, 1)).reduceByKey((x, y) => x + y)

    wordCountRdd.collect().foreach(x => println(x._1 + " " + x._2))
  }
}
