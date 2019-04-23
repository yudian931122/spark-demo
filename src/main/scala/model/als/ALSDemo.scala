package model.als

import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.recommendation.{ALS, Rating}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession


/**
  * Created by yudian on 2019/4/23.
  * ALS初探
  */


case class Movie(movieId: Int, title: String, genres:Seq[String])
case class User(userId: Int, gender: String, age: Int, occupation: Int, zip: String)

object ALSDemo {

  def main(args: Array[String]): Unit = {
    // 设置Log级别
    Logger.getLogger("org").setLevel(Level.ERROR)
    // 初始化spark session
    val sparkConf = new SparkConf().setAppName("ParamGrid").setMaster("local")
    val sc = new SparkContext(sparkConf)
    val ss = SparkSession.builder.config(sparkConf).getOrCreate()

    // 1.加载数据
    val movieDataPath = "D:\\ML\\recommendation\\ml-1m\\movies.dat"
    val ratingDataPath = "D:\\ML\\recommendation\\ml-1m\\ratings.dat"
    val userDataPath = "D:\\ML\\recommendation\\ml-1m\\users.dat"

    // 1-1 加载评分数据，并cache住
    val ratingText = sc.textFile(ratingDataPath)

    val ratingRDD = ratingText.map(str => {
      val fields = str.split("::")
      assert(fields.size == 4)
      Rating(fields(0).toInt, fields(1).toInt, fields(2).toDouble)
    }).cache()

    // 1-2 加载用户信息和电影信息
    val userText = sc.textFile(userDataPath)

    val userRDD = userText.map(str => {
      val fields = str.split("::")
      assert(fields.size == 5)
      User(fields(0).toInt, fields(1), fields(2).toInt, fields(3).toInt, fields(4))
    }).cache()

    val movieText = sc.textFile(movieDataPath)

    val movieRDD = movieText.map(str => {
      val fields = str.split("::")
      assert(fields.size == 3)
      Movie(fields(0).toInt, fields(1), fields(2).split("\\|").toSeq)
    }).cache()

    // 1-3 查看统计信息
    println("Total number of ratings: " + ratingRDD.count())
    println("Total number of movies rated: " + ratingRDD.map(_.product).distinct().count())
    println("Total number of users who rated movies: " + ratingRDD.map(_.user).distinct().count())

    // 2.将RDD数据转换成DF，并注册成表
    import ss.implicits._
    val ratingDF = ratingRDD.toDF()
    val movieDF = movieRDD.toDF()
    val userDF = userRDD.toDF()
    ratingDF.createOrReplaceTempView("ratings")
    movieDF.createOrReplaceTempView("movies")
    userDF.createOrReplaceTempView("users")

    // 3.查看相关统计信息
    // 3-1 查看每个产品的最高评分和最低评分以及被评分次数，按照被评分次数排序
    val res = ss.sql(
      """
         select title, rmax, rmin, ucnt from
        (select product, max(rating) as rmax, min(rating) as rmin, count(distinct user) as ucnt
        from ratings group by product) ratingsCNT
        join movies on product=movieId
        order by ucnt desc
      """.stripMargin)
    res.show()

    // 3-2 查看前10个最活跃的用户（评分电影数最多）
    val mostActiveUser = ss.sql(
      """
        select user, count(*) as cnt
        from ratings group by user order by cnt desc
        limit 10
      """.stripMargin)
    mostActiveUser.show()

    // 3-3 查看最活跃用户的所有电影评分中，大于4分的电影
    ss.sql(
      """
        select distinct title, rating from
        ratings join movies on product=movieId
        where user=4196 and rating>4
      """.stripMargin)

    // 4.训练ALS模型
    // 4-1 切分数据集，实际上这里按照时间的先后顺序切分应该是更加合理的方式
    val Array(trainingRDD, testRDD) = ratingRDD.randomSplit(Array(0.8, 0.2), 0L)
    val trainingSet = trainingRDD.cache()
    val testSet = testRDD.cache()

    println("TrainingSet size: " + trainingSet.count())
    println("TestSet size: " + testSet.count())

    // 4-2 训练ALS模型
    // setIterations: 设置迭代次数
    // setRank: 设置隐因子的数量，也就是向量的维数
    val model = new ALS().setIterations(10).setRank(20).run(trainingSet)

    // 5. 模型使用
    // 5-1 为4196用户推荐5个电影
    val recomForTopUser = model.recommendProducts(4196, 5)
    val movieTitle = movieRDD.map(movie => (movie.movieId, movie.title)).collectAsMap()
    recomForTopUser.map{
      case Rating(_, product, rating) => (movieTitle(product), rating)
    }.foreach(println(_))

    // 6. 预测
    // 6-1 将数据处理成predict函数需要的格式
    val testUserProduct = testSet.map {
      case Rating(user, product, _) => (user, product)
    }

    // 6-2拿到预测结果
    model.predict(testUserProduct)
    val testUserProductPredict = model.predict(testUserProduct)

    // 6-3 匹配真实值和预测值
    val testSetPair = testSet.map {
      case Rating(user, product, rating) => ((user, product), rating)
    }
    val predictionPair = testUserProductPredict.map {
      case Rating(user, product, rating) => ((user, product), rating)
    }
    val joinTestPrediction: RDD[((Int, Int), (Double, Double))] = testSetPair.join(predictionPair)

    // 6-4 手动计算mse
    val mea = joinTestPrediction.map {
      case ((user, product), (ratingT, ratingP)) =>
        val err = ratingT - ratingP
        Math.abs(err)
    }.mean()
    println(s"model MEA: $mea")

    // 6-5 使用spark mllib中实现好的评估器对模型进行评估
    import org.apache.spark.mllib.evaluation.RegressionMetrics
    // 构造RegressionMetrics需要的数据格式
    val ratingTP = joinTestPrediction.map {
      case ((user, product), (ratingT, ratingP)) =>
        (ratingT, ratingP)
    }

    val metrics = new RegressionMetrics(ratingTP)
    println("model MEA: " + metrics.meanAbsoluteError)
    println("model RMEA: " + metrics.rootMeanSquaredError)
  }
}
