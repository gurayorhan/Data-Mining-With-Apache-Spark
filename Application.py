import pandas as pd
from pyspark.sql import SparkSession
from pyspark.ml.feature import OneHotEncoderEstimator, StringIndexer, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import DecisionTreeClassifier

spark = SparkSession.builder.appName('ml-heart').getOrCreate()
df = spark.read.csv('../heart.csv', header = True, inferSchema = True)
df.printSchema()


pd.DataFrame(df.take(5), columns=df.columns).transpose()

numeric_features = [t[0] for t in df.dtypes if t[1] == 'int']
df.select(numeric_features).describe().toPandas().transpose()

categoricalColumns = []

stages = []

for categoricalCol in categoricalColumns:
    stringIndexer = StringIndexer(inputCol = categoricalCol, outputCol = categoricalCol + 'Index')
    encoder = OneHotEncoderEstimator(inputCols=[stringIndexer.getOutputCol()], outputCols=[categoricalCol + "classVec"])
    stages += [stringIndexer, encoder]
    
label_stringIdx = StringIndexer(inputCol = 'target', outputCol = 'label')

stages += [label_stringIdx]

numericCols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs','restecg', 'thalach', 'exang','oldpeak', 'slope', 'ca', 'thal']

assemblerInputs = [c + "classVec" for c in categoricalColumns] + numericCols

assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")

stages += [assembler]

pipeline = Pipeline(stages = stages)
pipelineModel = pipeline.fit(df)
df = pipelineModel.transform(df)
selectedCols = ['label', 'features'] + cols
df = df.select(selectedCols)
df.printSchema()

pd.DataFrame(df.take(5), columns=df.columns).transpose()

train, test = df.randomSplit([0.7, 0.3])
print("Train Data Count: " + str(train.count()))
print("Test Data Count: " + str(test.count()))

lr = LogisticRegression(featuresCol = 'features', labelCol = 'label', maxIter=20)
lrModel = lr.fit(train)
predictions = lrModel.transform(test)
evaluator = BinaryClassificationEvaluator()
print('Performance rate: ' + str(evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})*100))

predictions = lrModel.transform(test)
predictions.select('label','prediction','probability').show(5)

dt = DecisionTreeClassifier(featuresCol = 'features', labelCol = 'label', maxDepth = 3)
dtModel = dt.fit(train)
predictions = dtModel.transform(test)
evaluator = BinaryClassificationEvaluator()
print('Performance rate: ' + str(evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})*100))

predictions.select('label','prediction','probability').show(5)

