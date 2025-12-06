

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

def main():
    spark = SparkSession.builder \
        .appName("LinearRegressionRevenuePrediction") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")

    data = [
        (230.1, 37.8, 69.2, 22.1),
        (44.5, 39.3, 45.1, 10.4),
        (17.2, 45.9, 69.3, 9.3),
        (151.5, 41.3, 58.5, 18.5),
        (180.8, 10.8, 58.4, 12.9),
        (8.7, 48.9, 75.0, 7.2),
        (57.5, 32.8, 23.5, 11.8),
        (120.2, 19.6, 11.6, 13.2),
        (199.8, 2.6, 21.2, 10.6),
        (66.1, 5.8, 24.2, 8.6),
        (214.7, 24.0, 4.0, 17.4),
        (23.8, 35.1, 65.9, 9.2),
        (97.5, 7.6, 7.2, 9.7),
        (204.1, 32.9, 46.0, 19.0),
        (195.4, 47.7, 52.9, 22.4),
        (67.8, 36.6, 114.0, 12.5),
        (281.4, 39.6, 55.8, 24.4),
        (69.2, 20.5, 18.3, 11.3),
        (147.3, 23.9, 19.1, 14.6),
        (218.4, 27.7, 53.4, 18.0)
    ]

    columns = ["tv_spend", "radio_spend", "social_spend", "revenue"]
    df = spark.createDataFrame(data, columns)

    print("\n=== Raw Data ===")
    df.show(5)

    assembler = VectorAssembler(
        inputCols=["tv_spend", "radio_spend", "social_spend"],
        outputCol="features"
    )

    df_features = assembler.transform(df).select("features", "revenue")

    print("\n=== Data with Features Vector ===")
    df_features.show(5, truncate=False)

    train_data, test_data = df_features.randomSplit([0.8, 0.2], seed=42)

    print(f"\nTrain count: {train_data.count()}, Test count: {test_data.count()}")

    lr = LinearRegression(featuresCol="features", labelCol="revenue")

    lr_model = lr.fit(train_data)

    predictions = lr_model.transform(test_data)

    print("\n=== Predictions (revenue vs prediction) ===")
    predictions.select("revenue", "prediction").show(10)

    evaluator_rmse = RegressionEvaluator(
        labelCol="revenue",
        predictionCol="prediction",
        metricName="rmse"
    )
    rmse = evaluator_rmse.evaluate(predictions)

    evaluator_r2 = RegressionEvaluator(
        labelCol="revenue",
        predictionCol="prediction",
        metricName="r2"
    )
    r2 = evaluator_r2.evaluate(predictions)

    print("\n=== Model Evaluation ===")
    print(f"RMSE: {rmse:.4f}")
    print(f"R^2:  {r2:.4f}")

    print("\n=== Model Parameters ===")
    print("Coefficients:", lr_model.coefficients)
    print("Intercept:", lr_model.intercept)

    spark.stop()


if __name__ == "__main__":
    main()
