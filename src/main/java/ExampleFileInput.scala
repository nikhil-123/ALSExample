import java.io.{File, PrintWriter}

import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql._
import org.apache.spark.broadcast._
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.mllib.recommendation.{ALS, Rating}

import scala.util.Random

object ExampleFileInput extends Serializable {
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)

    val spark = SparkSession.builder().appName("appName").master("local[*]").config("spark.some.config.option", "some-value")
      .getOrCreate()
    import spark.implicits._

    val rawUserArtistData = spark.read.textFile("../DataSets/user_artist_data.txt").repartition(4)
    val rawArtistData = spark.read.textFile("../DataSets/artist_data.txt").repartition(4)
    val rawArtistAlias = spark.read.textFile("../DataSets/artist_alias.txt").repartition(4)

    //User and Artist IDs
    val userArtistDF = rawUserArtistData.map(parseUserArtistIDLines).toDF("user","artist")

    //Artist names and ID mapping
    val artistByID= rawArtistData.flatMap(parseArtistByIDLines).toDF("id","name")


    //Artist alias. Mapping artist IDs that may be mispelled or nonstandard
    val artistAlias = rawArtistAlias.flatMap(parseArtistAliasLines).collect().toMap

    val bArtistAlias = spark.sparkContext.broadcast(artistAlias)
    val trainData = buildCounts(rawUserArtistData,bArtistAlias,spark)
    trainData.cache()
    val Array(training, test) = trainData.randomSplit(Array(0.8, 0.2))


    val als = new ALS().
      setSeed(Random.nextLong()).
      setImplicitPrefs(true).
      setRank(10).
      setAlpha(1.0)
    val trainDataRdd = trainData.rdd
    //trainDataRdd.saveAsTextFile("new.txt")
    val trainDataRatingRdd = trainDataRdd.map(x=>x.toString).map(x=>x.substring(1,x.length-1).split(",")).
      map(x => Rating(x(0).toInt,x(1).toInt,x(2).toDouble))


    val model = ALS.train(trainDataRatingRdd,10,5)


    //val movieRecommendations = model.recommendForAllUsers(5)
   //val users = trainData.select(als.getUserCol).distinct().limit(3)
    //val userSubsetRecs = model.recommendForUserSubset(users, 10)
    //userSubsetRecs.persist()
    //userSubsetRecs.collect()
    //val userSubsetRecsWithName = userSubsetRecs.join(artistByID,userSubsetRecs("artistID")===artistByID("id"),"inner")
    //userSubsetRecsWithName.show()
    artistByID.show()
    val recommendations = model.recommendProducts(2096705, 10)
    for (recommendation <- recommendations) {
      //println( artistByID.select("name").where("id = " + "'" + recommendation.product.toInt + "'") + " score " + recommendation.rating )
      println(recommendation.product.toInt  + " score " + recommendation.rating )
    }

  }

  def parseUserArtistIDLines(line: String)={
    val fields= line.split(' ')
    (fields(0).toInt,fields(1).toInt)
  }

  def parseArtistByIDLines(line: String)={
    val (id,name) = line.span(_ != '\t')
    if(name.isEmpty){
      None
    }
    else {
      try {
        Some((id.toInt, name.trim))
      }
      catch{
        case _: NumberFormatException => None
      }
    }
  }

  def parseArtistAliasLines(line: String)={
    val (artist, alias) = line.span(_ != '\t')
    if(artist.isEmpty){
      None
    }
    else{
      try {
        Some((artist.toInt, alias.toInt))
      }
      catch{
        case _: NumberFormatException => None
      }
    }
  }

  def buildCounts(rawUserArtistData: Dataset[String], bArtistAlias: Broadcast[Map[Int,Int]], spark: SparkSession): DataFrame ={
    import spark.implicits._

    rawUserArtistData.map{line =>
      val Array(userID, artistID, count) = line.split(' ').map(_.toInt)
      val finalArtistID = bArtistAlias.value.getOrElse(artistID, artistID)
      (userID , finalArtistID, count)
    }.toDF("user","artist","count")
  }



  /*def makeRecommendations(model: ALSModel, userID: Int, howMany: Int): Dataframe={
    val toRecommend = model.itemFactors.select("id".as("artist"))
  }*/

}


