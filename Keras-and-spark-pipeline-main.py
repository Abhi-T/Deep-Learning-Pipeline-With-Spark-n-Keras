# Spark Session, Pipeline, Functions, and Metrics
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
from pyspark.ml.feature import OneHotEncoderEstimator, StringIndexer, StandardScaler, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.sql.functions import rand
from pyspark.mllib.evaluation import MulticlassMetrics

#keras/ Deep Learning
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras import optimizers, regularizers
from keras.optimizers import Adam

#Elephas for deep learning on spark
from elephas.ml_model import ElephasEstimator

#spark context
conf=SparkConf()
conf.set('spark.executor.memory','1g')
conf.set('spark.core.max','2')
conf.setAppName('Spark_DL_Pipeline')
sc=SparkContext('local',conf=conf)
sql_context=SQLContext(sc)

# Load Data to Spark Dataframe
df = sql_context.read.csv('bank.csv',header=True,inferSchema=True)

#view schema
# print(df.printSchema())

# Preview Dataframe (Pandas Preview is Cleaner)
# print(df.limit(5).toPandas())

# Drop Unnessary Features (Day and Month)
df = df.drop('day', 'month')

# Preview Dataframe
print(df.limit(5).toPandas())

# Helper function to select features to scale given their skew
def select_features_to_scale(df=df, lower_skew=-2, upper_skew=2, dtypes='int32', drop_cols=['']):
    #Empty selected feature list for output
    selected_features=[]

    #select features  to scale based on inputs ('in32' type, drop 'ID' columns or others, skew bounds)
    feature_list=list(df.toPandas().select_dtypes(include=[dtypes]).columns.drop(drop_cols))

    #loop through 'feature_list' to select features based on skew/kurtosis (Kurtosis is a measure of whether the data are heavy-tailed or light-tailed relative to a normal distribution.)
    for feature in feature_list:
        if df.toPandas()[feature].kurtosis()<-2 or df.toPandas()[feature].kurtosis()>2:
            selected_features.append(feature)
    #return feature list to scale
    return selected_features


# Spark Pipeline
cat_features = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'poutcome']
num_features = ['age','balance','duration','campaign','pdays','previous']
label='deposit'

#Pipeline stages list
stages=[]

#loop for string indexer and OHE for categorical variables
for features in cat_features:
    #index categorical features
    string_indexer=StringIndexer(inputCol=features, outputCol=features+"_index")

    #one hot encode categorical features
    encoder=OneHotEncoderEstimator(inputCols=[string_indexer.getOutputCol()], outputCols=[features+"_class_vec"])

    #append pipeline stages
    stages+= [string_indexer, encoder]

#index label feature
label_str_index=StringIndexer(inputCol=label, outputCol="label_index")

#scale feature: select the features to scale using helper " select_features_to_scale: function above and standardize
unscaled_features=select_features_to_scale(df=df, lower_skew=-2, upper_skew=2, dtypes='int32', drop_cols=['id'])

unscaled_assembler =VectorAssembler(inputCols=unscaled_features, outputCol="unscaled_features")
scaler=StandardScaler(inputCol="unscaled_features", outputCol="scaled_features")

stages+= [unscaled_assembler, scaler]

# Create list of Numeric Features that Are Not Being Scaled
num_unscaled_diff_list = list(set(num_features) - set(unscaled_features))
# Assemble or Concat the Categorical Features and Numeric Features
assembler_inputs = [feature + "_class_vec" for feature in cat_features] + num_unscaled_diff_list
assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="assembled_inputs")
stages += [label_str_index, assembler]
# Assemble Final Training Data of Scaled, Numeric, and Categorical Engineered Features
assembler_final = VectorAssembler(inputCols=["scaled_features","assembled_inputs"], outputCol="features")
stages += [assembler_final]

# print(stages)

# Set Pipeline
pipeline = Pipeline(stages=stages)

# Fit Pipeline to Data
pipeline_model = pipeline.fit(df)

# Transform Data using Fitted Pipeline
df_transform = pipeline_model.transform(df)

# Preview Newly Transformed Data
# print(df_transform.limit(5).toPandas())
# Data Structure Type is a PySpark Dataframe
type(df_transform)

# Select only 'features' and 'label_index' for Final Dataframe
df_transform_fin = df_transform.select('features','label_index')
# print(df_transform_fin.limit(5).toPandas())

# Shuffle Data
df_transform_fin = df_transform_fin.orderBy(rand())

# Split Data into Train / Test Sets
train_data, test_data = df_transform_fin.randomSplit([.8, .2],seed=1234)

#Deep learning part
# Number of Classes
nb_classes = train_data.select("label_index").distinct().count()
# Number of Inputs or Input Dimensions
input_dim = len(train_data.select("features").first()[0])

# Set up Deep Learning Model / Architecture
model = Sequential()
model.add(Dense(256, input_shape=(input_dim,), activity_regularizer=regularizers.l2(0.01)))
model.add(Activation('relu'))
model.add(Dropout(rate=0.3))
model.add(Dense(256, activity_regularizer=regularizers.l2(0.01)))
model.add(Activation('relu'))
model.add(Dropout(rate=0.3))
model.add(Dense(nb_classes))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam')

# Model Summary
print(model.summary())

#Distributed Deep Learning
# Set and Serialize Optimizer
optimizer_conf = optimizers.Adam(lr=0.01)
opt_conf = optimizers.serialize(optimizer_conf)

# Initialize SparkML Estimator and Get Settings
estimator = ElephasEstimator()
estimator.setFeaturesCol("features")
estimator.setLabelCol("label_index")
estimator.set_keras_model_config(model.to_yaml())
estimator.set_categorical_labels(True)
estimator.set_nb_classes(nb_classes)
estimator.set_num_workers(1)
estimator.set_epochs(25)
estimator.set_batch_size(64)
estimator.set_verbosity(1)
estimator.set_validation_split(0.10)
estimator.set_optimizer_config(opt_conf)
estimator.set_mode("synchronous")
estimator.set_loss("binary_crossentropy")
estimator.set_metrics(['acc'])

# Create Deep Learning Pipeline
dl_pipeline = Pipeline(stages=[estimator])


def dl_pipeline_fit_score_results(dl_pipeline=dl_pipeline,
                                  train_data=train_data,
                                  test_data=test_data,
                                  label='label_index'):
    fit_dl_pipeline = dl_pipeline.fit(train_data)
    pred_train = fit_dl_pipeline.transform(train_data)
    pred_test = fit_dl_pipeline.transform(test_data)

    pnl_train = pred_train.select(label, "prediction")
    pnl_test = pred_test.select(label, "prediction")

    pred_and_label_train = pnl_train.rdd.map(lambda row: (row[label], row['prediction']))
    pred_and_label_test = pnl_test.rdd.map(lambda row: (row[label], row['prediction']))

    metrics_train = MulticlassMetrics(pred_and_label_train)
    metrics_test = MulticlassMetrics(pred_and_label_test)

    print("Training Data Accuracy: {}".format(round(metrics_train.precision(), 4)))
    print("Training Data Confusion Matrix")
    print(pnl_train.crosstab('label_index', 'prediction').toPandas())

    print("\nTest Data Accuracy: {}".format(round(metrics_test.precision(), 4)))
    print("Test Data Confusion Matrix")
    print(pnl_test.crosstab('label_index', 'prediction').toPandas())

dl_pipeline_fit_score_results(dl_pipeline=dl_pipeline,
                              train_data=train_data,
                              test_data=test_data,
                              label='label_index')

import time
time.sleep(200)
