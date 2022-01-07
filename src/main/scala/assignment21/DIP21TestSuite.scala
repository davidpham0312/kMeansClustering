package assignment21

import org.scalatest.{FunSuite, BeforeAndAfterAll}
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.apache.spark.SparkContext._
import org.scalatest.Finders

@RunWith(classOf[JUnitRunner])
class DIP21TestSuite extends FunSuite with BeforeAndAfterAll {

  def initializeAssignment21(): Boolean =
    try {
      assignment
      true
    } catch {
      case ex: Throwable =>
        println(ex.getMessage)
        ex.printStackTrace()
        false
    }

  override def afterAll(): Unit = {
    assert(initializeAssignment21(), "Something is wrong with your assignment object")
    import assignment._
    spark.stop()
  }


  test("Simple test 1") {
    assert(initializeAssignment21(), "Something is wrong with your assignment object")
    import assignment._
    val v = task1(dataK5D2, 5)
    assert(v.length == 5, "Did not return five means")
  }

  test("Simple test 2") {
    assert(initializeAssignment21(), "Something is wrong with your assignment object")
    import assignment._
    val v = task2(dataK5D3, 5)
    assert(v.length == 5, "Did not return five means")
  }
  
  test("Label test") {
    assert(initializeAssignment21(), "Something is wrong with your assignment object")
    import assignment._
    val v = task3(dataK5D3WithLabels, 5)
    assert(v.length == 2, "Did not return two means")    
  }

  test("Elbow test") {
    assert(initializeAssignment21(), "Something is wrong with your assignment object")
    import assignment._
    val v = task4(dataK5D2, 2, 10)
    assert(v.length == 9, "Did not return 9 measures")
  }
}
